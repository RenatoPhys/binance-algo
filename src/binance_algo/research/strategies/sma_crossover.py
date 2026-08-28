"""Fixed causal crossover between two simple moving averages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import (
    FoldContext,
    StrategyScores,
    TrainingDataset,
    select_feature_view,
)
from binance_algo.research.panel import PanelData, matrix_to_long_frame
from binance_algo.research.strategies.fixed import (
    cross_sectional_zscore,
    projected_panel,
    validate_fixed_panel_fit,
)

SMA_CROSSOVER_FEATURES = ("log_return_1h",)
HOUR_MS = 3_600_000


def _moving_average(
    cumulative_prices: np.ndarray[Any, np.dtype[np.float64]],
    offsets: np.ndarray[Any, np.dtype[np.int64]],
    window: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    right = offsets + 1
    left = right - window
    return np.asarray(
        (cumulative_prices[right] - cumulative_prices[left]) / window,
        dtype=np.float64,
    )


@dataclass(frozen=True, slots=True)
class SmaCrossoverParameters:
    fast_window_hours: int
    slow_window_hours: int

    def __post_init__(self) -> None:
        if not 2 <= self.fast_window_hours <= 72:
            raise ResearchError("fast SMA window must be between 2 and 72 hours")
        if not 4 <= self.slow_window_hours <= 720:
            raise ResearchError("slow SMA window must be between 4 and 720 hours")
        if self.fast_window_hours >= self.slow_window_hours:
            raise ResearchError("fast SMA window must be shorter than slow SMA window")


def causal_sma_crossover(
    panel: PanelData,
    *,
    parameters: SmaCrossoverParameters,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Return the raw causal log fast/slow SMA ratio for the complete panel."""

    hourly_returns = np.asarray(
        panel.matrix("log_return_1h"),
        dtype=np.float64,
    )
    crossover = np.full_like(hourly_returns, np.nan)
    breaks = np.flatnonzero(np.diff(panel.times) != HOUR_MS) + 1
    starts = np.concatenate((np.asarray([0]), breaks))
    ends = np.concatenate((breaks, np.asarray([len(panel.times)])))
    for start, end in zip(starts, ends, strict=True):
        segment = hourly_returns[start:end]
        if len(segment) < parameters.slow_window_hours:
            continue
        relative_log_prices = np.cumsum(segment, axis=0)
        relative_prices = np.exp(relative_log_prices - relative_log_prices[0])
        cumulative_prices = np.vstack(
            (
                np.zeros((1, len(panel.symbols)), dtype=np.float64),
                np.cumsum(relative_prices, axis=0),
            )
        )
        offsets = np.arange(parameters.slow_window_hours - 1, len(segment))
        fast = _moving_average(cumulative_prices, offsets, parameters.fast_window_hours)
        slow = _moving_average(cumulative_prices, offsets, parameters.slow_window_hours)
        crossover[start + offsets] = np.log(fast / slow)
    last_valid = np.full(len(panel.symbols), np.nan, dtype=np.float64)
    for row in range(len(crossover)):
        valid = np.isfinite(crossover[row])
        last_valid[valid] = crossover[row, valid]
        crossover[row, ~valid] = last_valid[~valid]
    return crossover


def _score_panel(
    panel: PanelData,
    *,
    parameters: SmaCrossoverParameters,
    context: FoldContext,
) -> StrategyScores:
    test_slice = panel.time_slice(context.test_start_ms, context.test_end_ms)
    panel.require_complete_range(
        context.test_start_ms,
        context.test_end_ms,
        role="SMA crossover scoring",
    )
    crossover = causal_sma_crossover(panel, parameters=parameters)
    test_crossover = crossover[test_slice]
    if np.any(~np.isfinite(test_crossover)):
        raise ResearchError(
            "SMA crossover scoring lacks the causal history required by its slow window"
        )
    scores = cross_sectional_zscore(test_crossover)
    return StrategyScores(
        matrix_to_long_frame(
            times=panel.times[test_slice],
            symbols=panel.symbols,
            value_name="score",
            values=scores,
        )
    )


@dataclass(frozen=True, slots=True)
class FittedSmaCrossoverStrategy:
    parameters: SmaCrossoverParameters

    def score(self, features: pl.DataFrame, *, context: FoldContext) -> StrategyScores:
        projected = select_feature_view(
            features,
            required_features=SMA_CROSSOVER_FEATURES,
        )
        panel = PanelData.from_frame(projected, feature_columns=SMA_CROSSOVER_FEATURES)
        return _score_panel(panel, parameters=self.parameters, context=context)

    def score_panel(self, features: PanelData, *, context: FoldContext) -> StrategyScores:
        return _score_panel(features, parameters=self.parameters, context=context)


@dataclass(frozen=True, slots=True)
class SmaCrossoverStrategy:
    parameters: SmaCrossoverParameters
    strategy_id: str = field(default="sma_crossover", init=False)
    strategy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return SMA_CROSSOVER_FEATURES

    def target_column(self) -> str | None:
        return None

    def fit(
        self,
        train: TrainingDataset,
        *,
        context: FoldContext,
    ) -> FittedSmaCrossoverStrategy:
        if train.target is not None:
            raise ResearchError("SMA crossover is non-trainable and does not accept a target")
        projected_panel(
            train.features,
            required_features=self.required_features(),
            start_ms=context.train_start_ms,
            end_ms=context.train_end_ms,
            role="SMA crossover training",
        )
        return FittedSmaCrossoverStrategy(self.parameters)

    def fit_panel(
        self,
        train: PanelData,
        *,
        target: pl.DataFrame | None,
        context: FoldContext,
    ) -> FittedSmaCrossoverStrategy:
        validate_fixed_panel_fit(
            train,
            required_features=self.required_features(),
            target=target,
            context=context,
            role="SMA crossover",
        )
        return FittedSmaCrossoverStrategy(self.parameters)


__all__ = [
    "SMA_CROSSOVER_FEATURES",
    "FittedSmaCrossoverStrategy",
    "SmaCrossoverParameters",
    "SmaCrossoverStrategy",
    "causal_sma_crossover",
]
