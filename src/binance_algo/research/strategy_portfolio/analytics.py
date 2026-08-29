"""Transparent portfolio performance, drawdown, diversification, and attribution analytics."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import numpy as np
import polars as pl

from binance_algo.research.performance import HOURS_PER_YEAR, calculate_return_statistics
from binance_algo.research.strategy_portfolio.accounting import PortfolioAccounting
from binance_algo.research.strategy_portfolio.compatibility import CompatibilityReport
from binance_algo.research.strategy_portfolio.loader import LoadedStrategyComponent
from binance_algo.research.strategy_portfolio.trading import reconstruct_trading
from binance_algo.research.validation.robustness import effective_strategy_count


def _returns(frame: pl.DataFrame, column: str = "net_return") -> np.ndarray[Any, Any]:
    return np.asarray(frame[column].to_numpy(), dtype=np.float64)


def _segment_rows(
    frame: pl.DataFrame,
    keys: Sequence[object],
    *,
    key_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[object, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        grouped[key].append(index)
    rows: list[dict[str, Any]] = []
    values = _returns(frame)
    for key in sorted(grouped, key=str):
        selected = values[np.asarray(grouped[key], dtype=np.int64)]
        statistics = calculate_return_statistics(selected)
        rows.append({key_name: key, **asdict(statistics)})
    return rows


def monthly_metrics(frame: pl.DataFrame) -> list[dict[str, Any]]:
    months = [
        datetime.fromtimestamp(int(value) / 1_000, tz=UTC).strftime("%Y-%m")
        for value in frame["decision_time_ms"]
    ]
    return _segment_rows(frame, months, key_name="month")


def fold_metrics(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return _segment_rows(frame, frame["fold"].to_list(), key_name="fold")


def regime_metrics(frame: pl.DataFrame) -> list[dict[str, Any]]:
    values = np.asarray(frame["market_volatility_regime"].to_numpy(), dtype=np.float64)
    low, high = np.quantile(values, [1 / 3, 2 / 3])
    regimes = ["low" if value <= low else "middle" if value <= high else "high" for value in values]
    return _segment_rows(frame, regimes, key_name="regime")


def performance_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    statistics = calculate_return_statistics(_returns(frame))
    months = monthly_metrics(frame)
    folds = fold_metrics(frame)
    costs = {
        name: float(frame[name].sum()) for name in ("trading_fees", "spread_cost", "slippage_cost")
    }
    return {
        **asdict(statistics),
        "price_pnl": float(frame["price_pnl"].sum()),
        "funding_pnl": float(frame["funding_pnl"].sum()),
        **costs,
        "net_pnl": float(frame["net_return"].sum()),
        "turnover": float(frame["turnover"].sum()),
        "annualized_turnover": float(frame["turnover"].sum()) * HOURS_PER_YEAR / frame.height,
        "average_gross_exposure": float(frame.select(pl.col("gross_exposure").mean()).item()),
        "maximum_gross_exposure": float(frame.select(pl.col("gross_exposure").max()).item()),
        "average_net_exposure": float(frame.select(pl.col("net_exposure").mean()).item()),
        "maximum_absolute_beta_exposure": float(
            frame.select(pl.col("beta_exposure").abs().max()).item()
        ),
        "accounting_error_max": float(frame.select(pl.col("accounting_error").abs().max()).item()),
        "positive_months": sum(float(item["total_return"]) > 0 for item in months),
        "total_months": len(months),
        "best_month": max((float(item["total_return"]) for item in months), default=0.0),
        "worst_month": min((float(item["total_return"]) for item in months), default=0.0),
        "worst_fold": min((float(item["total_return"]) for item in folds), default=0.0),
    }


def drawdown_analysis(frame: pl.DataFrame) -> dict[str, Any]:
    times = [int(value) for value in frame["decision_time_ms"]]
    equity = np.cumprod(1.0 + _returns(frame))
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    episodes: list[dict[str, Any]] = []
    start: int | None = None
    trough: int | None = None
    running_peak_index = 0
    episode_peak_index = 0
    for index, drawdown in enumerate(drawdowns):
        if equity[index] >= equity[running_peak_index]:
            running_peak_index = index
        if drawdown < 0 and start is None:
            start = index
            episode_peak_index = running_peak_index
            trough = index
        elif start is not None:
            assert trough is not None
            if drawdown < drawdowns[trough]:
                trough = index
            if drawdown >= -1.0e-15:
                episodes.append(
                    {
                        "start_time_ms": times[episode_peak_index],
                        "trough_time_ms": times[trough],
                        "recovery_time_ms": times[index],
                        "depth": float(drawdowns[trough]),
                        "duration_hours": (times[index] - times[episode_peak_index]) / 3_600_000,
                        "recovered": True,
                    }
                )
                start = None
                trough = None
    if start is not None:
        assert trough is not None
        episodes.append(
            {
                "start_time_ms": times[episode_peak_index],
                "trough_time_ms": times[trough],
                "recovery_time_ms": None,
                "depth": float(drawdowns[trough]),
                "duration_hours": (times[-1] - times[episode_peak_index]) / 3_600_000,
                "recovered": False,
            }
        )
    top = sorted(
        episodes,
        key=lambda item: (
            float(item["depth"]),
            int(item["start_time_ms"]),
        ),
    )[:5]
    active = next((item for item in reversed(episodes) if not item["recovered"]), None)
    daily_drawdown: dict[str, tuple[int, float]] = {}
    for time, value in zip(times, drawdowns, strict=True):
        day = datetime.fromtimestamp(time / 1_000, tz=UTC).strftime("%Y-%m-%d")
        current = daily_drawdown.get(day)
        if current is None or value < current[1]:
            daily_drawdown[day] = (time, float(value))
    return {
        "current_drawdown": float(drawdowns[-1]),
        "maximum_drawdown": float(np.min(drawdowns)),
        "maximum_duration_hours": max(
            (float(item["duration_hours"]) for item in episodes),
            default=0.0,
        ),
        "current_underwater_duration_hours": (
            float(active["duration_hours"]) if active is not None else 0.0
        ),
        "unrecovered": active is not None,
        "episodes": top,
        "series": [
            {"date": day, "decision_time_ms": value[0], "drawdown": value[1]}
            for day, value in sorted(daily_drawdown.items())
        ],
    }


def _safe_correlation(left: np.ndarray[Any, Any], right: np.ndarray[Any, Any]) -> float | None:
    if len(left) < 2 or float(np.std(left)) <= 1.0e-18 or float(np.std(right)) <= 1.0e-18:
        return None
    value = float(np.corrcoef(left, right)[0, 1])
    return value if math.isfinite(value) else None


def _correlation_matrix(
    values: tuple[np.ndarray[Any, Any], ...],
    *,
    masks: tuple[np.ndarray[Any, Any], ...] | None = None,
) -> tuple[list[list[float | None]], list[list[int]]]:
    correlations: list[list[float | None]] = []
    observations: list[list[int]] = []
    for left_index, left in enumerate(values):
        correlation_row: list[float | None] = []
        observation_row: list[int] = []
        for right_index, right in enumerate(values):
            mask = (
                masks[left_index] & masks[right_index]
                if masks is not None
                else np.ones(len(left), dtype=np.bool_)
            )
            count = int(np.sum(mask))
            correlation_row.append(_safe_correlation(left[mask], right[mask]))
            observation_row.append(count)
        correlations.append(correlation_row)
        observations.append(observation_row)
    return correlations, observations


def _daily_values(
    times: tuple[int, ...],
    values: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], tuple[str, ...]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for time, value in zip(times, values, strict=True):
        day = datetime.fromtimestamp(time / 1_000, tz=UTC).strftime("%Y-%m-%d")
        grouped[day].append(float(value))
    days = tuple(sorted(grouped))
    return (
        np.asarray(
            [math.prod(1.0 + value for value in grouped[day]) - 1.0 for day in days],
            dtype=np.float64,
        ),
        days,
    )


def correlation_analytics(
    components: tuple[LoadedStrategyComponent, ...],
    accounting: PortfolioAccounting,
    compatibility: CompatibilityReport,
    *,
    epsilon: float,
) -> dict[str, Any]:
    labels = [component.declaration.label for component in components]
    net_returns = tuple(
        np.asarray(
            [float(component.oos_curve["net_return"][index]) for index in indices],
            dtype=np.float64,
        )
        for component, indices in zip(
            components,
            accounting.aligned_row_indices,
            strict=True,
        )
    )
    gross_returns = tuple(
        np.asarray(
            [
                float(component.oos_curve["price_pnl"][index])
                + float(component.oos_curve["funding_pnl"][index])
                for index in indices
            ],
            dtype=np.float64,
        )
        for component, indices in zip(
            components,
            accounting.aligned_row_indices,
            strict=True,
        )
    )
    active_masks = tuple(
        np.asarray(
            [float(component.oos_curve["gross_exposure"][index]) > epsilon for index in indices],
            dtype=np.bool_,
        )
        for component, indices in zip(
            components,
            accounting.aligned_row_indices,
            strict=True,
        )
    )
    hourly, hourly_observations = _correlation_matrix(net_returns)
    active, active_observations = _correlation_matrix(net_returns, masks=active_masks)
    daily_net = tuple(
        _daily_values(compatibility.decision_times, values)[0] for values in net_returns
    )
    daily_gross = tuple(
        _daily_values(compatibility.decision_times, values)[0] for values in gross_returns
    )
    daily, daily_observations = _correlation_matrix(daily_net)
    daily_gross_matrix, daily_gross_observations = _correlation_matrix(daily_gross)
    matrix = np.column_stack(daily_net)
    effective = effective_strategy_count(matrix)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw_correlation = np.corrcoef(matrix, rowvar=False)
    correlation_numeric = np.nan_to_num(raw_correlation, nan=0.0)
    rank = int(np.linalg.matrix_rank(correlation_numeric)) if len(components) > 1 else 1
    warnings: list[str] = []
    if rank < len(components):
        warnings.append("daily return correlation matrix is singular or nearly redundant")
    if any(value is None for row in daily for value in row):
        warnings.append("at least one daily correlation is unavailable because a sleeve is flat")
    return {
        "labels": labels,
        "hourly_net": {"values": hourly, "observations": hourly_observations},
        "daily_net": {"values": daily, "observations": daily_observations},
        "daily_gross": {
            "values": daily_gross_matrix,
            "observations": daily_gross_observations,
        },
        "active_only": {"values": active, "observations": active_observations},
        "nominal_components": len(components),
        "effective_independent_strategies": effective,
        "warnings": warnings,
    }


def position_similarity(
    components: tuple[LoadedStrategyComponent, ...],
    accounting: PortfolioAccounting,
    *,
    epsilon: float,
) -> dict[str, Any]:
    labels = [component.declaration.label for component in components]
    matrix: list[list[float | None]] = [[None for _ in components] for _ in components]
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(components):
        for right_index in range(left_index, len(components)):
            right = components[right_index]
            cosine_values: list[float] = []
            same_values: list[float] = []
            opposite_values: list[float] = []
            overlap_values: list[float] = []
            both_active = 0
            for period in range(len(accounting.aggregate_weights)):
                left_weights = left.weights[accounting.aligned_row_indices[left_index][period]]
                right_weights = right.weights[accounting.aligned_row_indices[right_index][period]]
                left_array = np.asarray(
                    [left_weights[symbol] for symbol in left.symbols], dtype=np.float64
                )
                right_array = np.asarray(
                    [right_weights[symbol] for symbol in left.symbols], dtype=np.float64
                )
                left_norm = float(np.linalg.norm(left_array))
                right_norm = float(np.linalg.norm(right_array))
                denominator = min(
                    float(np.sum(np.abs(left_array))),
                    float(np.sum(np.abs(right_array))),
                )
                if left_norm <= epsilon or right_norm <= epsilon or denominator <= epsilon:
                    continue
                both_active += 1
                cosine_values.append(
                    float(np.dot(left_array, right_array) / (left_norm * right_norm))
                )
                overlap = np.minimum(np.abs(left_array), np.abs(right_array))
                same = float(np.sum(overlap[left_array * right_array > 0]))
                opposite = float(np.sum(overlap[left_array * right_array < 0]))
                same_values.append(same / denominator)
                opposite_values.append(opposite / denominator)
                overlap_values.append((same + opposite) / denominator)
            mean_cosine = float(np.mean(cosine_values)) if cosine_values else None
            matrix[left_index][right_index] = mean_cosine
            matrix[right_index][left_index] = mean_cosine
            if right_index > left_index:
                mean_same = float(np.mean(same_values)) if same_values else 0.0
                mean_opposite = float(np.mean(opposite_values)) if opposite_values else 0.0
                pairs.append(
                    {
                        "left": labels[left_index],
                        "right": labels[right_index],
                        "observations": both_active,
                        "mean_cosine_similarity": mean_cosine,
                        "median_cosine_similarity": (
                            float(np.median(cosine_values)) if cosine_values else None
                        ),
                        "same_direction_exposure": mean_same,
                        "opposite_direction_exposure": mean_opposite,
                        "active_overlap_ratio": (
                            float(np.mean(overlap_values)) if overlap_values else 0.0
                        ),
                        "conflict_offset_ratio": (
                            mean_opposite / (mean_same + mean_opposite)
                            if mean_same + mean_opposite > 0
                            else 0.0
                        ),
                        "both_active_fraction": both_active / len(accounting.aggregate_weights),
                    }
                )
    return {"labels": labels, "matrix": matrix, "pairs": pairs}


def attribution_and_concentration(
    components: tuple[LoadedStrategyComponent, ...],
    accounting: PortfolioAccounting,
    trading: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    component_rows = []
    for component, indices, capital_weight in zip(
        components,
        accounting.aligned_row_indices,
        accounting.component_weights,
        strict=True,
    ):
        selected_sums = {
            column: math.fsum(float(component.oos_curve[column][index]) for index in indices)
            for column in (
                "price_pnl",
                "funding_pnl",
                "net_return",
                "trading_fees",
                "spread_cost",
                "slippage_cost",
            )
        }

        component_rows.append(
            {
                "label": component.declaration.label,
                "experiment_id": component.run.experiment_id,
                "run_id": component.run.run_id,
                "capital_weight": capital_weight,
                "gross_contribution": capital_weight
                * (selected_sums["price_pnl"] + selected_sums["funding_pnl"]),
                "net_contribution": capital_weight * selected_sums["net_return"],
                "price_pnl": capital_weight * selected_sums["price_pnl"],
                "funding_pnl": capital_weight * selected_sums["funding_pnl"],
                "explicit_cost": capital_weight
                * (
                    selected_sums["trading_fees"]
                    + selected_sums["spread_cost"]
                    + selected_sums["slippage_cost"]
                ),
            }
        )
    symbols = components[0].symbols
    symbol_rows: list[dict[str, Any]] = []
    for symbol in symbols:
        values = {
            name: math.fsum(
                capital_weight
                * float(component.symbol_metrics.filter(pl.col("symbol") == symbol)[name].item())
                for component, capital_weight in zip(
                    components,
                    accounting.component_weights,
                    strict=True,
                )
            )
            for name in (
                "price_pnl",
                "funding_pnl",
                "trading_fees",
                "spread_cost",
                "slippage_cost",
                "net_pnl",
                "turnover",
            )
        }
        symbol_rows.append(
            {
                "symbol": symbol,
                **values,
                "netting_cost_allocation": None,
                "note": "netting savings are not allocated by symbol",
            }
        )
    absolute_pnl = math.fsum(abs(float(row["net_pnl"])) for row in symbol_rows)
    pnl_shares = (
        [abs(float(row["net_pnl"])) / absolute_pnl for row in symbol_rows] if absolute_pnl else []
    )
    activity = trading["by_symbol"]
    top_activity = max(
        activity,
        key=lambda item: float(item["simulated_traded_weight"]),
        default=None,
    )
    maximum_symbol_weight = max(
        (abs(value) for row in accounting.aggregate_weights for value in row.values()),
        default=0.0,
    )
    concentration = {
        "capital_weight_hhi": math.fsum(value * value for value in accounting.component_weights),
        "maximum_component_weight": max(accounting.component_weights),
        "maximum_absolute_symbol_weight": maximum_symbol_weight,
        "activity_hhi_by_symbol": trading["activity_hhi_by_symbol"],
        "top_activity_symbol": top_activity["key"] if top_activity is not None else None,
        "top_activity_share": trading["top_symbol_activity_share"],
        "top_pnl_symbol": (
            symbol_rows[int(np.argmax(np.asarray(pnl_shares)))]["symbol"] if pnl_shares else None
        ),
        "top_pnl_share": max(pnl_shares, default=0.0),
        "volume_participation": None,
    }
    return component_rows, symbol_rows, concentration


def _rolling_series(
    frame: pl.DataFrame,
    windows: tuple[int, ...],
) -> dict[str, list[dict[str, Any]]]:
    returns = _returns(frame)
    times = [int(value) for value in frame["decision_time_ms"]]
    display_ends = [
        index
        for index, time in enumerate(times)
        if index == len(times) - 1
        or datetime.fromtimestamp(time / 1_000, tz=UTC).date()
        != datetime.fromtimestamp(times[index + 1] / 1_000, tz=UTC).date()
    ]
    output: dict[str, list[dict[str, Any]]] = {}
    for window in windows:
        rows: list[dict[str, Any]] = []
        for end in (index for index in display_ends if index >= window - 1):
            selected = returns[end - window + 1 : end + 1]
            deviation = float(np.std(selected, ddof=1)) if len(selected) > 1 else 0.0
            rows.append(
                {
                    "decision_time_ms": times[end],
                    "return": float(np.prod(1.0 + selected) - 1.0),
                    "annualized_volatility": deviation * math.sqrt(HOURS_PER_YEAR),
                    "sharpe": (
                        float(np.mean(selected)) / deviation * math.sqrt(HOURS_PER_YEAR)
                        if deviation
                        else 0.0
                    ),
                }
            )
        output[str(window)] = rows
    return output


def chart_series(
    accounting: PortfolioAccounting,
    components: tuple[LoadedStrategyComponent, ...],
    compatibility: CompatibilityReport,
    *,
    rolling_windows: tuple[int, ...],
) -> dict[str, Any]:
    sleeve_equity = np.cumprod(1.0 + _returns(accounting.sleeve_curve))
    netted_equity = np.cumprod(1.0 + _returns(accounting.netted_curve))
    sleeve_turnover = np.cumsum(_returns(accounting.sleeve_curve, "turnover"))
    netted_turnover = np.cumsum(_returns(accounting.netted_curve, "turnover"))
    netting_savings = np.cumsum(_returns(accounting.netted_curve, "netting_savings"))
    drawdowns = netted_equity / np.maximum.accumulate(netted_equity) - 1.0
    component_equities = []
    for component, indices in zip(
        components,
        accounting.aligned_row_indices,
        strict=True,
    ):
        returns = np.asarray(
            [float(component.oos_curve["net_return"][index]) for index in indices],
            dtype=np.float64,
        )
        component_equities.append(np.cumprod(1.0 + returns))
    by_day: dict[str, list[int]] = defaultdict(list)
    for index, time in enumerate(compatibility.decision_times):
        day = datetime.fromtimestamp(time / 1_000, tz=UTC).strftime("%Y-%m-%d")
        by_day[day].append(index)
    daily = []
    for day in sorted(by_day):
        day_indices = by_day[day]
        last = day_indices[-1]
        daily.append(
            {
                "date": day,
                "decision_time_ms": compatibility.decision_times[last],
                "sleeve_equity": float(sleeve_equity[last]),
                "netted_equity": float(netted_equity[last]),
                "drawdown": float(np.min(drawdowns[np.asarray(day_indices, dtype=np.int64)])),
                "sleeve_cumulative_turnover": float(sleeve_turnover[last]),
                "netted_cumulative_turnover": float(netted_turnover[last]),
                "cumulative_netting_savings": float(netting_savings[last]),
                "components": {
                    component.declaration.label: float(values[last])
                    for component, values in zip(components, component_equities, strict=True)
                },
            }
        )
    return {
        "daily": daily,
        "rolling": _rolling_series(accounting.netted_curve, rolling_windows),
        "fold_boundaries": [
            {
                "fold": int(row[0]),
                "decision_time_ms": int(row[1]),
            }
            for row in accounting.netted_curve.group_by("fold", maintain_order=True)
            .first()
            .select("fold", "decision_time_ms")
            .iter_rows()
        ],
    }


def analyze_portfolio(
    components: tuple[LoadedStrategyComponent, ...],
    accounting: PortfolioAccounting,
    compatibility: CompatibilityReport,
    *,
    trade_epsilon: float,
    rolling_windows: tuple[int, ...],
) -> dict[str, Any]:
    trading = reconstruct_trading(
        accounting,
        components,
        compatibility,
        epsilon=trade_epsilon,
    )
    component_attribution, symbol_attribution, concentration = attribution_and_concentration(
        components,
        accounting,
        trading,
    )
    return {
        "sleeve_metrics": performance_metrics(accounting.sleeve_curve),
        "netted_metrics": performance_metrics(accounting.netted_curve),
        "drawdown": drawdown_analysis(accounting.netted_curve),
        "correlations": correlation_analytics(
            components,
            accounting,
            compatibility,
            epsilon=trade_epsilon,
        ),
        "position_similarity": position_similarity(
            components,
            accounting,
            epsilon=trade_epsilon,
        ),
        "trading": trading,
        "component_attribution": component_attribution,
        "symbol_attribution": symbol_attribution,
        "concentration": concentration,
        "fold_metrics": fold_metrics(accounting.netted_curve),
        "regime_metrics": regime_metrics(accounting.netted_curve),
        "monthly_metrics": monthly_metrics(accounting.netted_curve),
        "chart_series": chart_series(
            accounting,
            components,
            compatibility,
            rolling_windows=rolling_windows,
        ),
    }


__all__ = [
    "analyze_portfolio",
    "attribution_and_concentration",
    "chart_series",
    "correlation_analytics",
    "drawdown_analysis",
    "fold_metrics",
    "monthly_metrics",
    "performance_metrics",
    "position_similarity",
    "regime_metrics",
]
