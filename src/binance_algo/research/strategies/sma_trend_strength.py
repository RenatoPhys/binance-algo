"""Raw causal SMA trend strength for exposure gating and slow rebalancing."""

from __future__ import annotations

from dataclasses import dataclass, field

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
from binance_algo.research.strategies.fixed import projected_panel, validate_fixed_panel_fit
from binance_algo.research.strategies.sma_crossover import (
    SMA_CROSSOVER_FEATURES,
    SmaCrossoverParameters,
    causal_sma_crossover,
)


def _score_panel(
    panel: PanelData,
    *,
    parameters: SmaCrossoverParameters,
    context: FoldContext,
) -> StrategyScores:
    panel.require_complete_range(
        context.test_start_ms,
        context.test_end_ms,
        role="SMA trend strength scoring",
    )
    time_slice = panel.time_slice(context.test_start_ms, context.test_end_ms)
    scores = causal_sma_crossover(panel, parameters=parameters)[time_slice]
    if np.any(~np.isfinite(scores)):
        raise ResearchError(
            "SMA trend strength scoring lacks the causal history required by its slow window"
        )
    return StrategyScores(
        matrix_to_long_frame(
            times=panel.times[time_slice],
            symbols=panel.symbols,
            value_name="score",
            values=scores,
        )
    )


@dataclass(frozen=True, slots=True)
class FittedSmaTrendStrengthStrategy:
    parameters: SmaCrossoverParameters

    def score(self, features: pl.DataFrame, *, context: FoldContext) -> StrategyScores:
        projected = select_feature_view(features, required_features=SMA_CROSSOVER_FEATURES)
        panel = PanelData.from_frame(projected, feature_columns=SMA_CROSSOVER_FEATURES)
        return _score_panel(panel, parameters=self.parameters, context=context)

    def score_panel(self, features: PanelData, *, context: FoldContext) -> StrategyScores:
        return _score_panel(features, parameters=self.parameters, context=context)


@dataclass(frozen=True, slots=True)
class SmaTrendStrengthStrategy:
    parameters: SmaCrossoverParameters
    strategy_id: str = field(default="sma_trend_strength", init=False)
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
    ) -> FittedSmaTrendStrengthStrategy:
        if train.target is not None:
            raise ResearchError("SMA trend strength is non-trainable and does not accept a target")
        projected_panel(
            train.features,
            required_features=self.required_features(),
            start_ms=context.train_start_ms,
            end_ms=context.train_end_ms,
            role="SMA trend strength training",
        )
        return FittedSmaTrendStrengthStrategy(self.parameters)

    def fit_panel(
        self,
        train: PanelData,
        *,
        target: pl.DataFrame | None,
        context: FoldContext,
    ) -> FittedSmaTrendStrengthStrategy:
        validate_fixed_panel_fit(
            train,
            required_features=self.required_features(),
            target=target,
            context=context,
            role="SMA trend strength",
        )
        return FittedSmaTrendStrengthStrategy(self.parameters)


__all__ = [
    "FittedSmaTrendStrengthStrategy",
    "SmaTrendStrengthStrategy",
]
