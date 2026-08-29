"""Causal multi-day cross-sectional relative-strength strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext, StrategyScores, TrainingDataset
from binance_algo.research.panel import PanelData, matrix_to_long_frame
from binance_algo.research.strategies.fixed import (
    cross_sectional_zscore,
    projected_panel,
    validate_fixed_panel_fit,
)

RELATIVE_STRENGTH_FEATURES = ("log_return_1h",)
HOUR_MS = 3_600_000


@dataclass(frozen=True, slots=True)
class RelativeStrengthParameters:
    lookback_hours: int

    def __post_init__(self) -> None:
        if not 24 <= self.lookback_hours <= 24 * 90:
            raise ResearchError("relative-strength lookback must be between one and 90 days")


def causal_relative_strength(
    panel: PanelData,
    *,
    parameters: RelativeStrengthParameters,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    hourly_returns = np.asarray(panel.matrix("log_return_1h"), dtype=np.float64)
    strength = np.full_like(hourly_returns, np.nan)
    breaks = np.flatnonzero(np.diff(panel.times) != HOUR_MS) + 1
    starts = np.concatenate((np.asarray([0]), breaks))
    ends = np.concatenate((breaks, np.asarray([len(panel.times)])))
    window = parameters.lookback_hours
    for start, end in zip(starts, ends, strict=True):
        segment = hourly_returns[start:end]
        if len(segment) < window:
            continue
        cumulative = np.vstack(
            (
                np.zeros((1, len(panel.symbols)), dtype=np.float64),
                np.cumsum(segment, axis=0),
            )
        )
        offsets = np.arange(window - 1, len(segment))
        right = offsets + 1
        left = right - window
        strength[start + offsets] = cumulative[right] - cumulative[left]
    last_valid = np.full(len(panel.symbols), np.nan, dtype=np.float64)
    for row in range(len(strength)):
        valid = np.isfinite(strength[row])
        last_valid[valid] = strength[row, valid]
        strength[row, ~valid] = last_valid[~valid]
    return strength


def _score_panel(
    panel: PanelData,
    *,
    parameters: RelativeStrengthParameters,
    context: FoldContext,
) -> StrategyScores:
    panel.require_complete_range(
        context.test_start_ms,
        context.test_end_ms,
        role="relative-strength scoring",
    )
    time_slice = panel.time_slice(context.test_start_ms, context.test_end_ms)
    raw = causal_relative_strength(panel, parameters=parameters)[time_slice]
    if np.any(~np.isfinite(raw)):
        raise ResearchError("relative-strength scoring lacks its required causal history")
    scores = cross_sectional_zscore(raw)
    return StrategyScores(
        matrix_to_long_frame(
            times=panel.times[time_slice],
            symbols=panel.symbols,
            value_name="score",
            values=scores,
        )
    )


@dataclass(frozen=True, slots=True)
class FittedRelativeStrengthStrategy:
    parameters: RelativeStrengthParameters

    def score(self, features: pl.DataFrame, *, context: FoldContext) -> StrategyScores:
        panel = projected_panel(
            features,
            required_features=RELATIVE_STRENGTH_FEATURES,
            start_ms=context.test_start_ms,
            end_ms=context.test_end_ms,
            role="relative-strength scoring",
        )
        return _score_panel(panel, parameters=self.parameters, context=context)

    def score_panel(self, features: PanelData, *, context: FoldContext) -> StrategyScores:
        return _score_panel(features, parameters=self.parameters, context=context)


@dataclass(frozen=True, slots=True)
class RelativeStrengthStrategy:
    parameters: RelativeStrengthParameters
    strategy_id: str = field(default="relative_strength", init=False)
    strategy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return RELATIVE_STRENGTH_FEATURES

    def target_column(self) -> str | None:
        return None

    def fit(
        self,
        train: TrainingDataset,
        *,
        context: FoldContext,
    ) -> FittedRelativeStrengthStrategy:
        if train.target is not None:
            raise ResearchError("relative strength is non-trainable and does not accept a target")
        projected_panel(
            train.features,
            required_features=self.required_features(),
            start_ms=context.train_start_ms,
            end_ms=context.train_end_ms,
            role="relative-strength training",
        )
        return FittedRelativeStrengthStrategy(self.parameters)

    def fit_panel(
        self,
        train: PanelData,
        *,
        target: pl.DataFrame | None,
        context: FoldContext,
    ) -> FittedRelativeStrengthStrategy:
        validate_fixed_panel_fit(
            train,
            required_features=self.required_features(),
            target=target,
            context=context,
            role="relative strength",
        )
        return FittedRelativeStrengthStrategy(self.parameters)


__all__ = [
    "RELATIVE_STRENGTH_FEATURES",
    "FittedRelativeStrengthStrategy",
    "RelativeStrengthParameters",
    "RelativeStrengthStrategy",
    "causal_relative_strength",
]
