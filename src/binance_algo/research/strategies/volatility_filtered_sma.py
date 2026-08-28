"""Causal slow SMA trend enabled below a training-only volatility quantile."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext, StrategyScores, TrainingDataset
from binance_algo.research.panel import PanelData, matrix_to_long_frame
from binance_algo.research.strategies.fixed import projected_panel
from binance_algo.research.strategies.sma_crossover import (
    SmaCrossoverParameters,
    causal_sma_crossover,
)

VOLATILITY_FILTERED_SMA_FEATURES = ("log_return_1h", "market_volatility_regime")


@dataclass(frozen=True, slots=True)
class VolatilityFilteredSmaParameters:
    fast_window_hours: int
    slow_window_hours: int
    maximum_volatility_quantile: float

    def __post_init__(self) -> None:
        SmaCrossoverParameters(
            fast_window_hours=self.fast_window_hours,
            slow_window_hours=self.slow_window_hours,
        )
        if (
            not math.isfinite(self.maximum_volatility_quantile)
            or not 0.25 <= self.maximum_volatility_quantile <= 0.90
        ):
            raise ResearchError("maximum volatility quantile must be finite and in [0.25, 0.90]")

    def sma_parameters(self) -> SmaCrossoverParameters:
        return SmaCrossoverParameters(
            fast_window_hours=self.fast_window_hours,
            slow_window_hours=self.slow_window_hours,
        )


def _training_volatility_threshold(
    panel: PanelData,
    *,
    parameters: VolatilityFilteredSmaParameters,
    context: FoldContext,
) -> float:
    regime = np.asarray(
        panel.matrix(
            "market_volatility_regime",
            start_ms=context.train_start_ms,
            end_ms=context.train_end_ms,
        ),
        dtype=np.float64,
    )
    if np.any(np.max(regime, axis=1) - np.min(regime, axis=1) > 1e-12):
        raise ResearchError("market volatility regime must be identical across symbols")
    threshold = float(np.quantile(regime[:, 0], parameters.maximum_volatility_quantile))
    if not math.isfinite(threshold) or threshold <= 0:
        raise ResearchError("training volatility threshold must be finite and positive")
    return threshold


@dataclass(frozen=True, slots=True)
class FittedVolatilityFilteredSmaStrategy:
    parameters: VolatilityFilteredSmaParameters
    volatility_threshold: float

    def _score_panel(self, panel: PanelData, *, context: FoldContext) -> StrategyScores:
        panel.require_complete_range(
            context.test_start_ms,
            context.test_end_ms,
            role="volatility-filtered SMA scoring",
        )
        time_slice = panel.time_slice(context.test_start_ms, context.test_end_ms)
        scores = causal_sma_crossover(
            panel,
            parameters=self.parameters.sma_parameters(),
        )[time_slice]
        regime = np.asarray(
            panel.matrix(
                "market_volatility_regime",
                start_ms=context.test_start_ms,
                end_ms=context.test_end_ms,
            ),
            dtype=np.float64,
        )
        if np.any(np.max(regime, axis=1) - np.min(regime, axis=1) > 1e-12):
            raise ResearchError("market volatility regime must be identical across symbols")
        scores[regime[:, 0] > self.volatility_threshold] = 0.0
        if np.any(~np.isfinite(scores)):
            raise ResearchError(
                "volatility-filtered SMA lacks the causal history required by its slow window"
            )
        return StrategyScores(
            matrix_to_long_frame(
                times=panel.times[time_slice],
                symbols=panel.symbols,
                value_name="score",
                values=scores,
            )
        )

    def score(self, features: pl.DataFrame, *, context: FoldContext) -> StrategyScores:
        panel = projected_panel(
            features,
            required_features=VOLATILITY_FILTERED_SMA_FEATURES,
            start_ms=context.test_start_ms,
            end_ms=context.test_end_ms,
            role="volatility-filtered SMA scoring",
        )
        return self._score_panel(panel, context=context)

    def score_panel(self, features: PanelData, *, context: FoldContext) -> StrategyScores:
        return self._score_panel(features, context=context)


@dataclass(frozen=True, slots=True)
class VolatilityFilteredSmaStrategy:
    parameters: VolatilityFilteredSmaParameters
    strategy_id: str = field(default="volatility_filtered_sma", init=False)
    strategy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return VOLATILITY_FILTERED_SMA_FEATURES

    def target_column(self) -> str | None:
        return None

    def fit(
        self,
        train: TrainingDataset,
        *,
        context: FoldContext,
    ) -> FittedVolatilityFilteredSmaStrategy:
        if train.target is not None:
            raise ResearchError("volatility-filtered SMA does not accept a target")
        panel = projected_panel(
            train.features,
            required_features=self.required_features(),
            start_ms=context.train_start_ms,
            end_ms=context.train_end_ms,
            role="volatility-filtered SMA training",
        )
        threshold = _training_volatility_threshold(
            panel,
            parameters=self.parameters,
            context=context,
        )
        return FittedVolatilityFilteredSmaStrategy(self.parameters, threshold)

    def fit_panel(
        self,
        train: PanelData,
        *,
        target: pl.DataFrame | None,
        context: FoldContext,
    ) -> FittedVolatilityFilteredSmaStrategy:
        if target is not None:
            raise ResearchError("volatility-filtered SMA does not accept a target")
        train.require_complete_range(
            context.train_start_ms,
            context.train_end_ms,
            role="volatility-filtered SMA training",
        )
        threshold = _training_volatility_threshold(
            train,
            parameters=self.parameters,
            context=context,
        )
        return FittedVolatilityFilteredSmaStrategy(self.parameters, threshold)


__all__ = [
    "VOLATILITY_FILTERED_SMA_FEATURES",
    "FittedVolatilityFilteredSmaStrategy",
    "VolatilityFilteredSmaParameters",
    "VolatilityFilteredSmaStrategy",
]
