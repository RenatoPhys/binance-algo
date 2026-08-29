"""Causal multi-horizon time-series trend vote."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext, StrategyScores, TrainingDataset
from binance_algo.research.panel import PanelData, matrix_to_long_frame
from binance_algo.research.strategies.fixed import projected_panel, validate_fixed_panel_fit

MULTI_HORIZON_TREND_FEATURES = ("log_return_1h",)
HOUR_MS = 3_600_000


@dataclass(frozen=True, slots=True)
class MultiHorizonTrendParameters:
    short_lookback_hours: int
    medium_lookback_hours: int
    long_lookback_hours: int
    short_weight: float
    medium_weight: float
    long_weight: float

    def __post_init__(self) -> None:
        horizons = (
            self.short_lookback_hours,
            self.medium_lookback_hours,
            self.long_lookback_hours,
        )
        if not 24 <= horizons[0] < horizons[1] < horizons[2] <= 24 * 90:
            raise ResearchError(
                "trend lookbacks must be strictly increasing between one and 90 days"
            )
        weights = (self.short_weight, self.medium_weight, self.long_weight)
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in weights):
            raise ResearchError("trend weights must be finite and in [0, 1]")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ResearchError("trend weights must sum to one")


def causal_rolling_return(
    panel: PanelData,
    *,
    lookback_hours: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    hourly_returns = np.asarray(panel.matrix("log_return_1h"), dtype=np.float64)
    trend = np.full_like(hourly_returns, np.nan)
    breaks = np.flatnonzero(np.diff(panel.times) != HOUR_MS) + 1
    starts = np.concatenate((np.asarray([0]), breaks))
    ends = np.concatenate((breaks, np.asarray([len(panel.times)])))
    for start, end in zip(starts, ends, strict=True):
        segment = hourly_returns[start:end]
        if len(segment) < lookback_hours:
            continue
        cumulative = np.vstack(
            (
                np.zeros((1, len(panel.symbols)), dtype=np.float64),
                np.cumsum(segment, axis=0),
            )
        )
        offsets = np.arange(lookback_hours - 1, len(segment))
        right = offsets + 1
        left = right - lookback_hours
        trend[start + offsets] = cumulative[right] - cumulative[left]
    last_valid = np.full(len(panel.symbols), np.nan, dtype=np.float64)
    for row in range(len(trend)):
        valid = np.isfinite(trend[row])
        last_valid[valid] = trend[row, valid]
        trend[row, ~valid] = last_valid[~valid]
    return trend


def _score_panel(
    panel: PanelData,
    *,
    parameters: MultiHorizonTrendParameters,
    context: FoldContext,
) -> StrategyScores:
    panel.require_complete_range(
        context.test_start_ms,
        context.test_end_ms,
        role="multi-horizon trend scoring",
    )
    time_slice = panel.time_slice(context.test_start_ms, context.test_end_ms)
    components = (
        (
            parameters.short_weight,
            causal_rolling_return(
                panel,
                lookback_hours=parameters.short_lookback_hours,
            )[time_slice],
        ),
        (
            parameters.medium_weight,
            causal_rolling_return(
                panel,
                lookback_hours=parameters.medium_lookback_hours,
            )[time_slice],
        ),
        (
            parameters.long_weight,
            causal_rolling_return(
                panel,
                lookback_hours=parameters.long_lookback_hours,
            )[time_slice],
        ),
    )
    if any(np.any(~np.isfinite(values)) for _, values in components):
        raise ResearchError("multi-horizon trend scoring lacks causal return history")
    score = np.zeros_like(components[0][1])
    for weight, values in components:
        score += weight * np.sign(values)
    return StrategyScores(
        matrix_to_long_frame(
            times=panel.times[time_slice],
            symbols=panel.symbols,
            value_name="score",
            values=score,
        )
    )


@dataclass(frozen=True, slots=True)
class FittedMultiHorizonTrendStrategy:
    parameters: MultiHorizonTrendParameters

    def score(self, features: pl.DataFrame, *, context: FoldContext) -> StrategyScores:
        panel = projected_panel(
            features,
            required_features=MULTI_HORIZON_TREND_FEATURES,
            start_ms=context.test_start_ms,
            end_ms=context.test_end_ms,
            role="multi-horizon trend scoring",
        )
        return _score_panel(panel, parameters=self.parameters, context=context)

    def score_panel(self, features: PanelData, *, context: FoldContext) -> StrategyScores:
        return _score_panel(features, parameters=self.parameters, context=context)


@dataclass(frozen=True, slots=True)
class MultiHorizonTrendStrategy:
    parameters: MultiHorizonTrendParameters
    strategy_id: str = field(default="multi_horizon_trend", init=False)
    strategy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return MULTI_HORIZON_TREND_FEATURES

    def target_column(self) -> str | None:
        return None

    def fit(
        self,
        train: TrainingDataset,
        *,
        context: FoldContext,
    ) -> FittedMultiHorizonTrendStrategy:
        if train.target is not None:
            raise ResearchError("multi-horizon trend is non-trainable and rejects a target")
        projected_panel(
            train.features,
            required_features=self.required_features(),
            start_ms=context.train_start_ms,
            end_ms=context.train_end_ms,
            role="multi-horizon trend training",
        )
        return FittedMultiHorizonTrendStrategy(self.parameters)

    def fit_panel(
        self,
        train: PanelData,
        *,
        target: pl.DataFrame | None,
        context: FoldContext,
    ) -> FittedMultiHorizonTrendStrategy:
        validate_fixed_panel_fit(
            train,
            required_features=self.required_features(),
            target=target,
            context=context,
            role="multi-horizon trend",
        )
        return FittedMultiHorizonTrendStrategy(self.parameters)


__all__ = [
    "MULTI_HORIZON_TREND_FEATURES",
    "FittedMultiHorizonTrendStrategy",
    "MultiHorizonTrendParameters",
    "MultiHorizonTrendStrategy",
    "causal_rolling_return",
]
