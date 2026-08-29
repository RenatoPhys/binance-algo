"""Breakout after fold-frozen relative volatility compression."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext, StrategyScores, TrainingDataset
from binance_algo.research.features.rolling import frozen_symbol_quantiles
from binance_algo.research.panel import PanelData, matrix_to_long_frame
from binance_algo.research.state_machine import causal_sparse_state
from binance_algo.research.strategies.fixed import projected_panel

VOLATILITY_COMPRESSION_BREAKOUT_FEATURES = (
    "volatility_compression_4h_168h",
    "log_close_level",
    "prior_high_72h",
    "prior_low_72h",
    "quote_volume_zscore_24h",
    "efficiency_ratio_24h",
)


@dataclass(frozen=True, slots=True)
class VolatilityCompressionBreakoutParameters:
    compression_quantile: float
    hold_hours: int

    def __post_init__(self) -> None:
        if self.compression_quantile not in {0.2, 0.3}:
            raise ResearchError("compression quantile must be 0.20 or 0.30")
        if self.hold_hours not in {12, 24}:
            raise ResearchError("compression-breakout holding period must be 12 or 24 hours")


def _training_thresholds(
    panel: PanelData,
    *,
    context: FoldContext,
    quantile: float,
) -> tuple[float, ...]:
    values = np.asarray(
        panel.matrix(
            "volatility_compression_4h_168h",
            start_ms=context.train_start_ms,
            end_ms=context.train_end_ms,
        ),
        dtype=np.float64,
    )
    thresholds = frozen_symbol_quantiles(values, quantile)
    if not np.all(np.isfinite(thresholds)):
        raise ResearchError("compression threshold is unavailable for one or more symbols")
    return tuple(float(value) for value in thresholds)


@dataclass(frozen=True, slots=True)
class FittedVolatilityCompressionBreakoutStrategy:
    parameters: VolatilityCompressionBreakoutParameters
    symbols: tuple[str, ...]
    thresholds: tuple[float, ...]

    def _score_panel(self, panel: PanelData, *, context: FoldContext) -> StrategyScores:
        if panel.symbols != self.symbols:
            raise ResearchError("compression-breakout symbols differ from fitted fold")
        panel.require_complete_range(
            context.test_start_ms,
            context.test_end_ms,
            role="compression-breakout scoring",
        )
        time_slice = panel.time_slice(context.test_start_ms, context.test_end_ms)
        indices = np.arange(len(panel.times))[time_slice]
        compression = np.asarray(panel.matrix("volatility_compression_4h_168h"), dtype=np.float64)
        prior_compression = np.full((len(indices), len(panel.symbols)), np.nan)
        valid_previous = indices > 0
        prior_compression[valid_previous] = compression[indices[valid_previous] - 1]
        close = np.exp(np.asarray(panel.matrix("log_close_level"), dtype=np.float64)[time_slice])
        prior_high = np.asarray(panel.matrix("prior_high_72h"), dtype=np.float64)[time_slice]
        prior_low = np.asarray(panel.matrix("prior_low_72h"), dtype=np.float64)[time_slice]
        volume = np.asarray(panel.matrix("quote_volume_zscore_24h"), dtype=np.float64)[time_slice]
        efficiency = np.asarray(panel.matrix("efficiency_ratio_24h"), dtype=np.float64)[time_slice]
        compressed = prior_compression <= np.asarray(self.thresholds)[None, :]
        filters = compressed & (volume >= 0.0) & (efficiency >= 0.25)
        entry = np.zeros_like(close)
        entry[filters & (close > prior_high)] = 1.0
        entry[filters & (close < prior_low)] = -1.0
        scores = causal_sparse_state(entry, hold_hours=self.parameters.hold_hours).values
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
            required_features=VOLATILITY_COMPRESSION_BREAKOUT_FEATURES,
            start_ms=context.test_start_ms,
            end_ms=context.test_end_ms,
            role="compression-breakout scoring",
        )
        return self._score_panel(panel, context=context)

    def score_panel(self, features: PanelData, *, context: FoldContext) -> StrategyScores:
        return self._score_panel(features, context=context)


@dataclass(frozen=True, slots=True)
class VolatilityCompressionBreakoutStrategy:
    parameters: VolatilityCompressionBreakoutParameters
    strategy_id: str = field(default="volatility_compression_breakout", init=False)
    strategy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return VOLATILITY_COMPRESSION_BREAKOUT_FEATURES

    def target_column(self) -> str | None:
        return None

    def fit(
        self,
        train: TrainingDataset,
        *,
        context: FoldContext,
    ) -> FittedVolatilityCompressionBreakoutStrategy:
        if train.target is not None:
            raise ResearchError("compression breakout does not accept a training target")
        panel = projected_panel(
            train.features,
            required_features=self.required_features(),
            start_ms=context.train_start_ms,
            end_ms=context.train_end_ms,
            role="compression-breakout training",
        )
        return FittedVolatilityCompressionBreakoutStrategy(
            parameters=self.parameters,
            symbols=panel.symbols,
            thresholds=_training_thresholds(
                panel,
                context=context,
                quantile=self.parameters.compression_quantile,
            ),
        )

    def fit_panel(
        self,
        train: PanelData,
        *,
        target: pl.DataFrame | None,
        context: FoldContext,
    ) -> FittedVolatilityCompressionBreakoutStrategy:
        if target is not None:
            raise ResearchError("compression breakout does not accept a training target")
        train.require_complete_range(
            context.train_start_ms,
            context.train_end_ms,
            role="compression-breakout training",
        )
        for feature in self.required_features():
            train.matrix(feature, start_ms=context.train_start_ms, end_ms=context.train_end_ms)
        return FittedVolatilityCompressionBreakoutStrategy(
            parameters=self.parameters,
            symbols=train.symbols,
            thresholds=_training_thresholds(
                train,
                context=context,
                quantile=self.parameters.compression_quantile,
            ),
        )


__all__ = [
    "VOLATILITY_COMPRESSION_BREAKOUT_FEATURES",
    "FittedVolatilityCompressionBreakoutStrategy",
    "VolatilityCompressionBreakoutParameters",
    "VolatilityCompressionBreakoutStrategy",
]
