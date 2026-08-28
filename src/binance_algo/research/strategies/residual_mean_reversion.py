"""Fixed short-horizon residual mean-reversion score."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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

RESIDUAL_MEAN_REVERSION_FEATURES = (
    "residual_momentum_1h",
    "residual_momentum_4h",
    "realized_volatility_24h",
)


@dataclass(frozen=True, slots=True)
class ResidualMeanReversionParameters:
    momentum_weight_1h: float
    momentum_weight_4h: float
    volatility_adjustment: float

    def __post_init__(self) -> None:
        weights = (self.momentum_weight_1h, self.momentum_weight_4h)
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in weights):
            raise ResearchError("mean-reversion momentum weights must be finite in [0, 1]")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-12):
            raise ResearchError("mean-reversion momentum weights must sum to one")
        if (
            not math.isfinite(self.volatility_adjustment)
            or not 0 <= self.volatility_adjustment <= 2
        ):
            raise ResearchError("volatility adjustment must be finite and between zero and two")


def _score_panel(
    panel: PanelData,
    *,
    parameters: ResidualMeanReversionParameters,
    context: FoldContext,
) -> StrategyScores:
    panel.require_complete_range(
        context.test_start_ms,
        context.test_end_ms,
        role="residual mean reversion scoring",
    )
    arrays = {
        name: np.asarray(
            panel.matrix(name, start_ms=context.test_start_ms, end_ms=context.test_end_ms),
            dtype=np.float64,
        )
        for name in RESIDUAL_MEAN_REVERSION_FEATURES
    }
    score = -(
        parameters.momentum_weight_1h * cross_sectional_zscore(arrays["residual_momentum_1h"])
        + parameters.momentum_weight_4h * cross_sectional_zscore(arrays["residual_momentum_4h"])
    )
    if parameters.volatility_adjustment:
        volatility = arrays["realized_volatility_24h"]
        median = np.median(volatility, axis=1, keepdims=True)
        relative = np.divide(
            volatility,
            median,
            out=np.ones_like(volatility),
            where=median > 1e-15,
        )
        score = score / np.power(np.maximum(relative, 1e-12), parameters.volatility_adjustment)
    time_slice = panel.time_slice(context.test_start_ms, context.test_end_ms)
    return StrategyScores(
        matrix_to_long_frame(
            times=panel.times[time_slice],
            symbols=panel.symbols,
            value_name="score",
            values=score,
        )
    )


@dataclass(frozen=True, slots=True)
class FittedResidualMeanReversionStrategy:
    parameters: ResidualMeanReversionParameters

    def score(self, features: pl.DataFrame, *, context: FoldContext) -> StrategyScores:
        panel = projected_panel(
            features,
            required_features=RESIDUAL_MEAN_REVERSION_FEATURES,
            start_ms=context.test_start_ms,
            end_ms=context.test_end_ms,
            role="residual mean reversion scoring",
        )
        return _score_panel(panel, parameters=self.parameters, context=context)

    def score_panel(self, features: PanelData, *, context: FoldContext) -> StrategyScores:
        return _score_panel(features, parameters=self.parameters, context=context)


@dataclass(frozen=True, slots=True)
class ResidualMeanReversionStrategy:
    parameters: ResidualMeanReversionParameters
    strategy_id: str = field(default="residual_mean_reversion", init=False)
    strategy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return RESIDUAL_MEAN_REVERSION_FEATURES

    def target_column(self) -> str | None:
        return None

    def fit(
        self,
        train: TrainingDataset,
        *,
        context: FoldContext,
    ) -> FittedResidualMeanReversionStrategy:
        if train.target is not None:
            raise ResearchError(
                "residual mean reversion is non-trainable and does not accept a target"
            )
        projected_panel(
            train.features,
            required_features=self.required_features(),
            start_ms=context.train_start_ms,
            end_ms=context.train_end_ms,
            role="residual mean reversion training",
        )
        return FittedResidualMeanReversionStrategy(self.parameters)

    def fit_panel(
        self,
        train: PanelData,
        *,
        target: pl.DataFrame | None,
        context: FoldContext,
    ) -> FittedResidualMeanReversionStrategy:
        validate_fixed_panel_fit(
            train,
            required_features=self.required_features(),
            target=target,
            context=context,
            role="residual mean reversion",
        )
        return FittedResidualMeanReversionStrategy(self.parameters)


__all__ = [
    "RESIDUAL_MEAN_REVERSION_FEATURES",
    "FittedResidualMeanReversionStrategy",
    "ResidualMeanReversionParameters",
    "ResidualMeanReversionStrategy",
]
