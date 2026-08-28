"""Fixed funding-carry score with optional residual-momentum confirmation."""

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

FUNDING_CARRY_FEATURES = (
    "funding_rate_current",
    "funding_rate_change",
    "residual_momentum_4h",
)


@dataclass(frozen=True, slots=True)
class FundingCarryParameters:
    funding_rate_weight: float
    funding_change_weight: float
    momentum_confirmation_weight: float

    def __post_init__(self) -> None:
        values = (
            self.funding_rate_weight,
            self.funding_change_weight,
            self.momentum_confirmation_weight,
        )
        if any(not math.isfinite(value) or not 0 <= value <= 5 for value in values):
            raise ResearchError("funding carry weights must be finite and between zero and five")
        if self.funding_rate_weight + self.funding_change_weight <= 0:
            raise ResearchError("funding carry requires a positive funding weight")


def _score_panel(
    panel: PanelData,
    *,
    parameters: FundingCarryParameters,
    context: FoldContext,
) -> StrategyScores:
    panel.require_complete_range(
        context.test_start_ms,
        context.test_end_ms,
        role="funding carry scoring",
    )
    arrays = {
        name: np.asarray(
            panel.matrix(name, start_ms=context.test_start_ms, end_ms=context.test_end_ms),
            dtype=np.float64,
        )
        for name in FUNDING_CARRY_FEATURES
    }
    score = (
        -parameters.funding_rate_weight * cross_sectional_zscore(arrays["funding_rate_current"])
        - parameters.funding_change_weight * cross_sectional_zscore(arrays["funding_rate_change"])
        + parameters.momentum_confirmation_weight
        * cross_sectional_zscore(arrays["residual_momentum_4h"])
    )
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
class FittedFundingCarryStrategy:
    parameters: FundingCarryParameters

    def score(self, features: pl.DataFrame, *, context: FoldContext) -> StrategyScores:
        panel = projected_panel(
            features,
            required_features=FUNDING_CARRY_FEATURES,
            start_ms=context.test_start_ms,
            end_ms=context.test_end_ms,
            role="funding carry scoring",
        )
        return _score_panel(panel, parameters=self.parameters, context=context)

    def score_panel(self, features: PanelData, *, context: FoldContext) -> StrategyScores:
        return _score_panel(features, parameters=self.parameters, context=context)


@dataclass(frozen=True, slots=True)
class FundingCarryStrategy:
    parameters: FundingCarryParameters
    strategy_id: str = field(default="funding_carry", init=False)
    strategy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return FUNDING_CARRY_FEATURES

    def target_column(self) -> str | None:
        return None

    def fit(
        self,
        train: TrainingDataset,
        *,
        context: FoldContext,
    ) -> FittedFundingCarryStrategy:
        if train.target is not None:
            raise ResearchError("funding carry is non-trainable and does not accept a target")
        projected_panel(
            train.features,
            required_features=self.required_features(),
            start_ms=context.train_start_ms,
            end_ms=context.train_end_ms,
            role="funding carry training",
        )
        return FittedFundingCarryStrategy(self.parameters)

    def fit_panel(
        self,
        train: PanelData,
        *,
        target: pl.DataFrame | None,
        context: FoldContext,
    ) -> FittedFundingCarryStrategy:
        validate_fixed_panel_fit(
            train,
            required_features=self.required_features(),
            target=target,
            context=context,
            role="funding carry",
        )
        return FittedFundingCarryStrategy(self.parameters)


__all__ = [
    "FUNDING_CARRY_FEATURES",
    "FittedFundingCarryStrategy",
    "FundingCarryParameters",
    "FundingCarryStrategy",
]
