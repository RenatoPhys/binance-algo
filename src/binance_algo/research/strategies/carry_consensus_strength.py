"""Funding carry with a fast/slow relative-strength consensus sleeve."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext, StrategyScores, TrainingDataset
from binance_algo.research.panel import PanelData
from binance_algo.research.strategies.carry_multi_horizon import (
    CARRY_MULTI_HORIZON_FEATURES,
    CarryMultiHorizonParameters,
)
from binance_algo.research.strategies.carry_multi_horizon import (
    _score_panel as score_carry_multi_horizon,
)
from binance_algo.research.strategies.fixed import projected_panel, validate_fixed_panel_fit

CARRY_CONSENSUS_STRENGTH_FEATURES = CARRY_MULTI_HORIZON_FEATURES


@dataclass(frozen=True, slots=True)
class CarryConsensusStrengthParameters:
    funding_rate_weight: float
    funding_change_weight: float
    momentum_confirmation_weight: float
    fast_lookback_hours: int
    slow_lookback_hours: int

    def __post_init__(self) -> None:
        self.core_parameters()

    def core_parameters(self) -> CarryMultiHorizonParameters:
        return CarryMultiHorizonParameters(
            funding_rate_weight=self.funding_rate_weight,
            funding_change_weight=self.funding_change_weight,
            momentum_confirmation_weight=self.momentum_confirmation_weight,
            fast_lookback_hours=self.fast_lookback_hours,
            slow_lookback_hours=self.slow_lookback_hours,
        )


def _score_panel(
    panel: PanelData,
    *,
    parameters: CarryConsensusStrengthParameters,
    context: FoldContext,
) -> StrategyScores:
    core = score_carry_multi_horizon(
        panel,
        parameters=parameters.core_parameters(),
        context=context,
    ).frame
    fast = core["fast_strength_score"].to_numpy()
    slow = core["slow_strength_score"].to_numpy()
    agreement = np.sign(fast) == np.sign(slow)
    consensus = np.where(agreement, (fast + slow) / 2, 0.0)
    diagnostic = (core["carry_score"].to_numpy() + consensus) / 2
    return StrategyScores(
        core.select("decision_time_ms", "symbol").with_columns(
            pl.Series("score", diagnostic),
            core["carry_score"],
            pl.Series("relative_strength_score", consensus),
        )
    )


@dataclass(frozen=True, slots=True)
class FittedCarryConsensusStrengthStrategy:
    parameters: CarryConsensusStrengthParameters

    def score(self, features: pl.DataFrame, *, context: FoldContext) -> StrategyScores:
        panel = projected_panel(
            features,
            required_features=CARRY_CONSENSUS_STRENGTH_FEATURES,
            start_ms=context.test_start_ms,
            end_ms=context.test_end_ms,
            role="carry consensus-strength scoring",
        )
        return _score_panel(panel, parameters=self.parameters, context=context)

    def score_panel(self, features: PanelData, *, context: FoldContext) -> StrategyScores:
        return _score_panel(features, parameters=self.parameters, context=context)


@dataclass(frozen=True, slots=True)
class CarryConsensusStrengthStrategy:
    parameters: CarryConsensusStrengthParameters
    strategy_id: str = field(default="carry_consensus_strength", init=False)
    strategy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return CARRY_CONSENSUS_STRENGTH_FEATURES

    def target_column(self) -> str | None:
        return None

    def fit(
        self,
        train: TrainingDataset,
        *,
        context: FoldContext,
    ) -> FittedCarryConsensusStrengthStrategy:
        if train.target is not None:
            raise ResearchError("carry consensus-strength is non-trainable and rejects a target")
        projected_panel(
            train.features,
            required_features=self.required_features(),
            start_ms=context.train_start_ms,
            end_ms=context.train_end_ms,
            role="carry consensus-strength training",
        )
        return FittedCarryConsensusStrengthStrategy(self.parameters)

    def fit_panel(
        self,
        train: PanelData,
        *,
        target: pl.DataFrame | None,
        context: FoldContext,
    ) -> FittedCarryConsensusStrengthStrategy:
        validate_fixed_panel_fit(
            train,
            required_features=self.required_features(),
            target=target,
            context=context,
            role="carry consensus-strength",
        )
        return FittedCarryConsensusStrengthStrategy(self.parameters)


__all__ = [
    "CARRY_CONSENSUS_STRENGTH_FEATURES",
    "CarryConsensusStrengthParameters",
    "CarryConsensusStrengthStrategy",
    "FittedCarryConsensusStrengthStrategy",
]
