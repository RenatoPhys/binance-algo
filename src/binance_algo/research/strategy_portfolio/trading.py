"""Reconstruct explicitly simulated rebalance events and trade legs from target weights."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

import numpy as np

from binance_algo.research.strategy_portfolio.accounting import PortfolioAccounting
from binance_algo.research.strategy_portfolio.compatibility import CompatibilityReport
from binance_algo.research.strategy_portfolio.loader import LoadedStrategyComponent


def _event_type(previous: float, target: float, epsilon: float) -> str:
    previous_active = abs(previous) > epsilon
    target_active = abs(target) > epsilon
    if not previous_active and target_active:
        return "entry"
    if previous_active and not target_active:
        return "exit"
    if previous * target < 0:
        return "flip"
    if abs(target) > abs(previous):
        return "increase"
    return "reduce"


def _summarize_legs(
    legs: list[dict[str, Any]],
    *,
    periods: int,
) -> dict[str, Any]:
    weights = np.asarray([float(item["trade_weight"]) for item in legs], dtype=np.float64)
    event_times = sorted({int(item["decision_time_ms"]) for item in legs})
    gaps = np.diff(np.asarray(event_times, dtype=np.float64)) / 3_600_000
    type_counts = {
        name: sum(item["event_type"] == name for item in legs)
        for name in (
            "entry",
            "exit",
            "increase",
            "reduce",
            "flip",
            "forced_fold_close",
        )
    }
    return {
        "rebalance_events": len(event_times),
        "trade_legs": len(legs),
        "entries": type_counts["entry"],
        "exits": type_counts["exit"],
        "increases": type_counts["increase"],
        "reductions": type_counts["reduce"],
        "flips": type_counts["flip"],
        "forced_fold_closes": type_counts["forced_fold_close"],
        "simulated_traded_weight": float(np.sum(weights)) if len(weights) else 0.0,
        "mean_trade_weight": float(np.mean(weights)) if len(weights) else 0.0,
        "median_trade_weight": float(np.median(weights)) if len(weights) else 0.0,
        "p90_trade_weight": float(np.quantile(weights, 0.9)) if len(weights) else 0.0,
        "maximum_trade_weight": float(np.max(weights)) if len(weights) else 0.0,
        "mean_hours_between_rebalances": float(np.mean(gaps)) if len(gaps) else 0.0,
        "periods_without_trading_fraction": (
            max(0.0, 1.0 - len(event_times) / periods) if periods else 0.0
        ),
        "long_trade_weight": math.fsum(
            float(item["trade_weight"]) for item in legs if item["side"] == "long"
        ),
        "short_trade_weight": math.fsum(
            float(item["trade_weight"]) for item in legs if item["side"] == "short"
        ),
    }


def _group_legs(legs: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    totals: dict[object, tuple[int, float]] = {}
    for item in legs:
        key = item[field]
        count, weight = totals.get(key, (0, 0.0))
        totals[key] = (count + 1, weight + float(item["trade_weight"]))
    return [
        {"key": key, "trade_legs": count, "simulated_traded_weight": weight}
        for key, (count, weight) in sorted(totals.items(), key=lambda value: str(value[0]))
    ]


def _book_legs(
    weights: tuple[dict[str, float], ...],
    *,
    folds: tuple[int, ...],
    decision_times: tuple[int, ...],
    execution_times: tuple[int, ...],
    epsilon: float,
) -> list[dict[str, Any]]:
    symbols = tuple(weights[0])
    previous = dict.fromkeys(symbols, 0.0)
    previous_fold: int | None = None
    legs: list[dict[str, Any]] = []
    for period, target in enumerate(weights):
        fold = folds[period]
        if fold != previous_fold:
            previous = dict.fromkeys(symbols, 0.0)
            previous_fold = fold
        is_last = period == len(weights) - 1 or folds[period + 1] != fold
        timestamp = datetime.fromtimestamp(execution_times[period] / 1_000, tz=UTC)
        for symbol in symbols:
            delta = target[symbol] - previous[symbol]
            if abs(delta) > epsilon:
                legs.append(
                    {
                        "decision_time_ms": decision_times[period],
                        "execution_time_ms": execution_times[period],
                        "month": timestamp.strftime("%Y-%m"),
                        "weekday_utc": timestamp.weekday(),
                        "hour_utc": timestamp.hour,
                        "fold": fold,
                        "symbol": symbol,
                        "previous_weight": previous[symbol],
                        "target_weight": target[symbol],
                        "delta_weight": delta,
                        "trade_weight": abs(delta),
                        "side": "long" if delta > 0 else "short",
                        "event_type": _event_type(previous[symbol], target[symbol], epsilon),
                        "forced_fold_close": False,
                    }
                )
            if is_last and abs(target[symbol]) > epsilon:
                legs.append(
                    {
                        "decision_time_ms": decision_times[period],
                        "execution_time_ms": execution_times[period],
                        "month": timestamp.strftime("%Y-%m"),
                        "weekday_utc": timestamp.weekday(),
                        "hour_utc": timestamp.hour,
                        "fold": fold,
                        "symbol": symbol,
                        "previous_weight": target[symbol],
                        "target_weight": 0.0,
                        "delta_weight": -target[symbol],
                        "trade_weight": abs(target[symbol]),
                        "side": "long" if -target[symbol] > 0 else "short",
                        "event_type": "forced_fold_close",
                        "forced_fold_close": True,
                    }
                )
        previous = target
    return legs


def reconstruct_trading(
    accounting: PortfolioAccounting,
    components: tuple[LoadedStrategyComponent, ...],
    compatibility: CompatibilityReport,
    *,
    epsilon: float,
) -> dict[str, Any]:
    first = components[0]
    indices = accounting.aligned_row_indices[0]
    folds = tuple(int(first.oos_curve["fold"][index]) for index in indices)
    executions = tuple(int(first.oos_curve["execution_time_ms"][index]) for index in indices)
    aggregate_legs = _book_legs(
        accounting.aggregate_weights,
        folds=folds,
        decision_times=compatibility.decision_times,
        execution_times=executions,
        epsilon=epsilon,
    )
    component_legs: list[list[dict[str, Any]]] = []
    for component, component_indices, capital_weight in zip(
        components,
        accounting.aligned_row_indices,
        accounting.component_weights,
        strict=True,
    ):
        scaled = tuple(
            {
                symbol: capital_weight * component.weights[index][symbol]
                for symbol in component.symbols
            }
            for index in component_indices
        )
        component_legs.append(
            _book_legs(
                scaled,
                folds=folds,
                decision_times=compatibility.decision_times,
                execution_times=executions,
                epsilon=epsilon,
            )
        )
    contribution_lookup: dict[tuple[int, str, bool], dict[str, float]] = defaultdict(dict)
    for component, legs in zip(components, component_legs, strict=True):
        for leg in legs:
            key = (
                int(leg["decision_time_ms"]),
                str(leg["symbol"]),
                bool(leg["forced_fold_close"]),
            )
            contribution_lookup[key][component.declaration.label] = float(leg["delta_weight"])
    for leg in aggregate_legs:
        key = (
            int(leg["decision_time_ms"]),
            str(leg["symbol"]),
            bool(leg["forced_fold_close"]),
        )
        leg["component_contributions"] = {
            name: contribution_lookup[key][name] for name in sorted(contribution_lookup[key])
        }
    summary = _summarize_legs(aggregate_legs, periods=len(compatibility.decision_times))
    unnetted_weight = math.fsum(
        float(_summarize_legs(legs, periods=len(folds))["simulated_traded_weight"])
        for legs in component_legs
    )
    netted_weight = float(summary["simulated_traded_weight"])
    summary.update(
        {
            "unnetted_simulated_traded_weight": unnetted_weight,
            "netted_simulated_traded_weight": netted_weight,
            "netted_trade_weight_reduction": unnetted_weight - netted_weight,
            "turnover_reduction_fraction": (
                1.0 - netted_weight / unnetted_weight if unnetted_weight else 0.0
            ),
        }
    )
    by_symbol = _group_legs(aggregate_legs, "symbol")
    total_activity = math.fsum(float(item["simulated_traded_weight"]) for item in by_symbol)
    shares = [
        float(item["simulated_traded_weight"]) / total_activity
        for item in by_symbol
        if total_activity
    ]
    weekday_hour: dict[tuple[int, int], float] = defaultdict(float)
    for leg in aggregate_legs:
        weekday_hour[(int(leg["weekday_utc"]), int(leg["hour_utc"]))] += float(leg["trade_weight"])
    top_events = sorted(
        aggregate_legs,
        key=lambda item: (
            -float(item["trade_weight"]),
            int(item["execution_time_ms"]),
            str(item["symbol"]),
            str(item["event_type"]),
        ),
    )[:50]
    return {
        "summary": summary,
        "activity_hhi_by_symbol": math.fsum(value * value for value in shares),
        "top_symbol_activity_share": max(shares, default=0.0),
        "by_symbol": by_symbol,
        "by_month": _group_legs(aggregate_legs, "month"),
        "by_fold": _group_legs(aggregate_legs, "fold"),
        "by_weekday_utc": _group_legs(aggregate_legs, "weekday_utc"),
        "by_hour_utc": _group_legs(aggregate_legs, "hour_utc"),
        "by_side": _group_legs(aggregate_legs, "side"),
        "by_event_type": _group_legs(aggregate_legs, "event_type"),
        "weekday_hour_utc": [
            {"weekday": weekday, "hour": hour, "simulated_traded_weight": value}
            for (weekday, hour), value in sorted(weekday_hour.items())
        ],
        "largest_events": top_events,
        "components": [
            {
                "label": component.declaration.label,
                **_summarize_legs(legs, periods=len(folds)),
            }
            for component, legs in zip(components, component_legs, strict=True)
        ],
    }


__all__ = ["reconstruct_trading"]
