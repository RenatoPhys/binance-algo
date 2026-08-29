"""Sleeve and cross-strategy netted accounting on verified OOS curves."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal

import orjson
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.costs import explicit_cost_rates
from binance_algo.research.strategy_portfolio.compatibility import CompatibilityReport
from binance_algo.research.strategy_portfolio.loader import LoadedStrategyComponent

PORTFOLIO_ACCOUNTING_TOLERANCE = 1.0e-10


@dataclass(frozen=True, slots=True)
class PortfolioAccounting:
    sleeve_curve: pl.DataFrame
    netted_curve: pl.DataFrame
    aggregate_weights: tuple[dict[str, float], ...]
    aggregate_trade_weights: tuple[dict[str, float], ...]
    component_weights: tuple[float, ...]
    aligned_row_indices: tuple[tuple[int, ...], ...]


def _aligned_indices(
    component: LoadedStrategyComponent,
    decision_times: tuple[int, ...],
) -> tuple[int, ...]:
    lookup = {
        int(value): index
        for index, value in enumerate(component.oos_curve["decision_time_ms"].to_list())
    }
    try:
        return tuple(lookup[value] for value in decision_times)
    except KeyError as exc:
        raise ResearchError(
            f"aligned timestamp is absent from {component.run.experiment_id}"
        ) from exc


def _weighted_value(
    components: tuple[LoadedStrategyComponent, ...],
    indices: tuple[tuple[int, ...], ...],
    weights: tuple[float, ...],
    *,
    period: int,
    column: str,
) -> float:
    return math.fsum(
        weight * float(component.oos_curve[column][component_indices[period]])
        for component, component_indices, weight in zip(
            components,
            indices,
            weights,
            strict=True,
        )
    )


def build_portfolio_accounting(
    components: tuple[LoadedStrategyComponent, ...],
    capital_weights: tuple[Decimal, ...],
    compatibility: CompatibilityReport,
) -> PortfolioAccounting:
    """Build both views; callers choose which accounting mode to foreground."""

    if not components or len(components) != len(capital_weights):
        raise ResearchError("portfolio accounting requires one capital weight per component")
    if not compatibility.valid:
        raise ResearchError(
            "cannot account an incompatible strategy portfolio: " + "; ".join(compatibility.issues)
        )
    weights = tuple(float(value) for value in capital_weights)
    if not math.isclose(math.fsum(weights), 1.0, abs_tol=1.0e-15):
        raise ResearchError("portfolio accounting capital weights must sum to one")
    indices = tuple(
        _aligned_indices(component, compatibility.decision_times) for component in components
    )
    first = components[0]
    symbols = first.symbols
    previous = dict.fromkeys(symbols, 0.0)
    previous_fold: int | None = None
    sleeve_rows: list[dict[str, object]] = []
    netted_rows: list[dict[str, object]] = []
    aggregate_weight_rows: list[dict[str, float]] = []
    aggregate_trade_rows: list[dict[str, float]] = []
    period_count = len(compatibility.decision_times)
    for period, decision_time in enumerate(compatibility.decision_times):
        source_index = indices[0][period]
        fold = int(first.oos_curve["fold"][source_index])
        execution_time = int(first.oos_curve["execution_time_ms"][source_index])
        if previous_fold != fold:
            previous = dict.fromkeys(symbols, 0.0)
            previous_fold = fold
        target = {
            symbol: math.fsum(
                weight * component.weights[component_indices[period]][symbol]
                for component, component_indices, weight in zip(
                    components,
                    indices,
                    weights,
                    strict=True,
                )
            )
            for symbol in symbols
        }
        is_last_in_fold = period == period_count - 1
        if not is_last_in_fold:
            next_index = indices[0][period + 1]
            is_last_in_fold = int(first.oos_curve["fold"][next_index]) != fold
        trade_weights = {
            symbol: abs(target[symbol] - previous[symbol])
            + (abs(target[symbol]) if is_last_in_fold else 0.0)
            for symbol in symbols
        }
        netted_turnover = math.fsum(trade_weights.values())
        sleeve_turnover = _weighted_value(
            components,
            indices,
            weights,
            period=period,
            column="turnover",
        )
        price_pnl = _weighted_value(
            components,
            indices,
            weights,
            period=period,
            column="price_pnl",
        )
        funding_pnl = _weighted_value(
            components,
            indices,
            weights,
            period=period,
            column="funding_pnl",
        )
        sleeve_fees = _weighted_value(
            components,
            indices,
            weights,
            period=period,
            column="trading_fees",
        )
        sleeve_spread = _weighted_value(
            components,
            indices,
            weights,
            period=period,
            column="spread_cost",
        )
        sleeve_slippage = _weighted_value(
            components,
            indices,
            weights,
            period=period,
            column="slippage_cost",
        )
        sleeve_net_return = _weighted_value(
            components,
            indices,
            weights,
            period=period,
            column="net_return",
        )
        rates = explicit_cost_rates(first.spec.cost_model, execution_time)
        netted_fees = netted_turnover * rates.fee_rate
        netted_spread = netted_turnover * rates.half_spread_rate
        netted_slippage = netted_turnover * rates.slippage_rate
        netted_net_return = price_pnl + funding_pnl - netted_fees - netted_spread - netted_slippage
        if 1.0 + sleeve_net_return <= 0 or 1.0 + netted_net_return <= 0:
            raise ResearchError("strategy portfolio equity became non-positive")
        unnetted_gross = _weighted_value(
            components,
            indices,
            weights,
            period=period,
            column="gross_exposure",
        )
        netted_gross = math.fsum(abs(value) for value in target.values())
        offset_ratio = 1.0 - netted_gross / unnetted_gross if unnetted_gross else 0.0
        net_exposure = math.fsum(target.values())
        beta_exposure = _weighted_value(
            components,
            indices,
            weights,
            period=period,
            column="beta_exposure",
        )
        regime = float(first.oos_curve["market_volatility_regime"][source_index])
        sleeve_error = sleeve_net_return - (
            price_pnl + funding_pnl - sleeve_fees - sleeve_spread - sleeve_slippage
        )
        netted_error = netted_net_return - (
            price_pnl + funding_pnl - netted_fees - netted_spread - netted_slippage
        )
        if max(abs(sleeve_error), abs(netted_error)) > PORTFOLIO_ACCOUNTING_TOLERANCE:
            raise ResearchError("strategy portfolio accounting reconciliation exceeded tolerance")
        common = {
            "fold": fold,
            "decision_time_ms": decision_time,
            "execution_time_ms": execution_time,
            "price_pnl": price_pnl,
            "funding_pnl": funding_pnl,
            "market_volatility_regime": regime,
            "beta_exposure": beta_exposure,
        }
        sleeve_rows.append(
            {
                **common,
                "trading_fees": sleeve_fees,
                "spread_cost": sleeve_spread,
                "slippage_cost": sleeve_slippage,
                "net_return": sleeve_net_return,
                "turnover": sleeve_turnover,
                "gross_exposure": unnetted_gross,
                "net_exposure": _weighted_value(
                    components,
                    indices,
                    weights,
                    period=period,
                    column="net_exposure",
                ),
                "accounting_error": sleeve_error,
            }
        )
        sleeve_cost = sleeve_fees + sleeve_spread + sleeve_slippage
        netted_cost = netted_fees + netted_spread + netted_slippage
        netted_rows.append(
            {
                **common,
                "trading_fees": netted_fees,
                "spread_cost": netted_spread,
                "slippage_cost": netted_slippage,
                "net_return": netted_net_return,
                "turnover": netted_turnover,
                "gross_exposure": netted_gross,
                "unnetted_gross_exposure": unnetted_gross,
                "net_exposure": net_exposure,
                "offset_ratio": offset_ratio,
                "netting_savings": sleeve_cost - netted_cost,
                "accounting_error": netted_error,
                "weights_json": orjson.dumps(target, option=orjson.OPT_SORT_KEYS).decode(),
            }
        )
        aggregate_weight_rows.append(target)
        aggregate_trade_rows.append(trade_weights)
        previous = target
    sleeve_curve = pl.DataFrame(sleeve_rows).with_columns(
        (pl.col("net_return") + 1.0).cum_prod().alias("oos_equity")
    )
    netted_curve = pl.DataFrame(netted_rows).with_columns(
        (pl.col("net_return") + 1.0).cum_prod().alias("oos_equity"),
        pl.col("netting_savings").cum_sum().alias("cumulative_netting_savings"),
    )
    return PortfolioAccounting(
        sleeve_curve=sleeve_curve,
        netted_curve=netted_curve,
        aggregate_weights=tuple(aggregate_weight_rows),
        aggregate_trade_weights=tuple(aggregate_trade_rows),
        component_weights=weights,
        aligned_row_indices=indices,
    )


__all__ = [
    "PORTFOLIO_ACCOUNTING_TOLERANCE",
    "PortfolioAccounting",
    "build_portfolio_accounting",
]
