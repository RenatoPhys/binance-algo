"""Causal stateful Donchian breakout for absolute time-series trend."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext, StrategyScores, TrainingDataset
from binance_algo.research.panel import PanelData, matrix_to_long_frame
from binance_algo.research.strategies.fixed import projected_panel, validate_fixed_panel_fit

DONCHIAN_BREAKOUT_FEATURES = ("log_return_1h",)
HOUR_MS = 3_600_000


@dataclass(frozen=True, slots=True)
class DonchianBreakoutParameters:
    entry_window_hours: int
    exit_window_hours: int

    def __post_init__(self) -> None:
        if not 24 <= self.entry_window_hours <= 24 * 90:
            raise ResearchError("Donchian entry window must be between one and 90 days")
        if not 4 <= self.exit_window_hours <= 24 * 30:
            raise ResearchError("Donchian exit window must be between four hours and 30 days")
        if self.exit_window_hours >= self.entry_window_hours:
            raise ResearchError("Donchian exit window must be shorter than entry window")


def _prior_extrema(
    values: np.ndarray[Any, np.dtype[np.float64]],
    window: int,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    maximum = np.full(len(values), np.nan, dtype=np.float64)
    minimum = np.full(len(values), np.nan, dtype=np.float64)
    maximum_indices: deque[int] = deque()
    minimum_indices: deque[int] = deque()
    for index in range(len(values)):
        previous = index - 1
        if previous >= 0:
            while maximum_indices and values[maximum_indices[-1]] <= values[previous]:
                maximum_indices.pop()
            maximum_indices.append(previous)
            while minimum_indices and values[minimum_indices[-1]] >= values[previous]:
                minimum_indices.pop()
            minimum_indices.append(previous)
        first_valid = index - window
        while maximum_indices and maximum_indices[0] < first_valid:
            maximum_indices.popleft()
        while minimum_indices and minimum_indices[0] < first_valid:
            minimum_indices.popleft()
        if index >= window:
            maximum[index] = values[maximum_indices[0]]
            minimum[index] = values[minimum_indices[0]]
    return maximum, minimum


def _segment_signal(
    hourly_returns: np.ndarray[Any, np.dtype[np.float64]],
    *,
    parameters: DonchianBreakoutParameters,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    relative_log_price = np.cumsum(hourly_returns)
    price = np.exp(relative_log_price - relative_log_price[0])
    entry_high, entry_low = _prior_extrema(price, parameters.entry_window_hours)
    exit_high, exit_low = _prior_extrema(price, parameters.exit_window_hours)
    signal = np.zeros(len(price), dtype=np.float64)
    position = 0.0
    for index in range(parameters.entry_window_hours, len(price)):
        if price[index] > entry_high[index]:
            position = 1.0
        elif price[index] < entry_low[index]:
            position = -1.0
        elif (position > 0 and price[index] < exit_low[index]) or (
            position < 0 and price[index] > exit_high[index]
        ):
            position = 0.0
        signal[index] = position
    return signal


def causal_donchian_breakout(
    panel: PanelData,
    *,
    parameters: DonchianBreakoutParameters,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    hourly_returns = np.asarray(panel.matrix("log_return_1h"), dtype=np.float64)
    signal = np.zeros_like(hourly_returns)
    breaks = np.flatnonzero(np.diff(panel.times) != HOUR_MS) + 1
    starts = np.concatenate((np.asarray([0]), breaks))
    ends = np.concatenate((breaks, np.asarray([len(panel.times)])))
    for start, end in zip(starts, ends, strict=True):
        segment = hourly_returns[start:end]
        if len(segment) <= parameters.entry_window_hours:
            continue
        for symbol in range(len(panel.symbols)):
            signal[start:end, symbol] = _segment_signal(
                segment[:, symbol],
                parameters=parameters,
            )
    return signal


def _score_panel(
    panel: PanelData,
    *,
    parameters: DonchianBreakoutParameters,
    context: FoldContext,
) -> StrategyScores:
    panel.require_complete_range(
        context.test_start_ms,
        context.test_end_ms,
        role="Donchian breakout scoring",
    )
    time_slice = panel.time_slice(context.test_start_ms, context.test_end_ms)
    scores = causal_donchian_breakout(panel, parameters=parameters)[time_slice]
    return StrategyScores(
        matrix_to_long_frame(
            times=panel.times[time_slice],
            symbols=panel.symbols,
            value_name="score",
            values=scores,
        )
    )


@dataclass(frozen=True, slots=True)
class FittedDonchianBreakoutStrategy:
    parameters: DonchianBreakoutParameters

    def score(self, features: pl.DataFrame, *, context: FoldContext) -> StrategyScores:
        panel = projected_panel(
            features,
            required_features=DONCHIAN_BREAKOUT_FEATURES,
            start_ms=context.test_start_ms,
            end_ms=context.test_end_ms,
            role="Donchian breakout scoring",
        )
        return _score_panel(panel, parameters=self.parameters, context=context)

    def score_panel(self, features: PanelData, *, context: FoldContext) -> StrategyScores:
        return _score_panel(features, parameters=self.parameters, context=context)


@dataclass(frozen=True, slots=True)
class DonchianBreakoutStrategy:
    parameters: DonchianBreakoutParameters
    strategy_id: str = field(default="donchian_breakout", init=False)
    strategy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return DONCHIAN_BREAKOUT_FEATURES

    def target_column(self) -> str | None:
        return None

    def fit(
        self,
        train: TrainingDataset,
        *,
        context: FoldContext,
    ) -> FittedDonchianBreakoutStrategy:
        if train.target is not None:
            raise ResearchError("Donchian breakout is non-trainable and does not accept a target")
        projected_panel(
            train.features,
            required_features=self.required_features(),
            start_ms=context.train_start_ms,
            end_ms=context.train_end_ms,
            role="Donchian breakout training",
        )
        return FittedDonchianBreakoutStrategy(self.parameters)

    def fit_panel(
        self,
        train: PanelData,
        *,
        target: pl.DataFrame | None,
        context: FoldContext,
    ) -> FittedDonchianBreakoutStrategy:
        validate_fixed_panel_fit(
            train,
            required_features=self.required_features(),
            target=target,
            context=context,
            role="Donchian breakout",
        )
        return FittedDonchianBreakoutStrategy(self.parameters)


__all__ = [
    "DONCHIAN_BREAKOUT_FEATURES",
    "DonchianBreakoutParameters",
    "DonchianBreakoutStrategy",
    "FittedDonchianBreakoutStrategy",
    "causal_donchian_breakout",
]
