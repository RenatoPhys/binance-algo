"""Carry multi-horizon scores with a causal market-regime trend sleeve."""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import (
    FEATURE_KEY_COLUMNS,
    FoldContext,
    StrategyScores,
    TrainingDataset,
)
from binance_algo.research.panel import PanelData
from binance_algo.research.strategies.carry_multi_horizon import (
    CARRY_MULTI_HORIZON_FEATURES,
    CarryMultiHorizonParameters,
)
from binance_algo.research.strategies.carry_multi_horizon import (
    _score_panel as score_carry_multi_horizon,
)
from binance_algo.research.strategies.fixed import projected_panel, validate_fixed_panel_fit
from binance_algo.research.strategies.market_regime_trend import (
    MarketRegimeTrendParameters,
)
from binance_algo.research.strategies.market_regime_trend import (
    _score_panel as score_market_regime_trend,
)

CARRY_MULTI_REGIME_FEATURES = CARRY_MULTI_HORIZON_FEATURES


@dataclass(frozen=True, slots=True)
class CarryMultiRegimeParameters:
    funding_rate_weight: float
    funding_change_weight: float
    momentum_confirmation_weight: float
    fast_lookback_hours: int
    slow_lookback_hours: int
    regime_fast_lookback_hours: int
    regime_slow_lookback_hours: int

    def __post_init__(self) -> None:
        self.carry_parameters()
        self.regime_parameters()

    def carry_parameters(self) -> CarryMultiHorizonParameters:
        return CarryMultiHorizonParameters(
            funding_rate_weight=self.funding_rate_weight,
            funding_change_weight=self.funding_change_weight,
            momentum_confirmation_weight=self.momentum_confirmation_weight,
            fast_lookback_hours=self.fast_lookback_hours,
            slow_lookback_hours=self.slow_lookback_hours,
        )

    def regime_parameters(self) -> MarketRegimeTrendParameters:
        return MarketRegimeTrendParameters(
            fast_lookback_hours=self.regime_fast_lookback_hours,
            slow_lookback_hours=self.regime_slow_lookback_hours,
        )


def _score_panel(
    panel: PanelData,
    *,
    parameters: CarryMultiRegimeParameters,
    context: FoldContext,
) -> StrategyScores:
    carry = score_carry_multi_horizon(
        panel,
        parameters=parameters.carry_parameters(),
        context=context,
    ).frame
    regime = score_market_regime_trend(
        panel,
        parameters=parameters.regime_parameters(),
        context=context,
    ).frame.select(
        *FEATURE_KEY_COLUMNS,
        pl.col("score").alias("regime_trend_score"),
    )
    return StrategyScores(carry.join(regime, on=list(FEATURE_KEY_COLUMNS), how="inner"))


@dataclass(frozen=True, slots=True)
class FittedCarryMultiRegimeStrategy:
    parameters: CarryMultiRegimeParameters

    def score(self, features: pl.DataFrame, *, context: FoldContext) -> StrategyScores:
        panel = projected_panel(
            features,
            required_features=CARRY_MULTI_REGIME_FEATURES,
            start_ms=context.test_start_ms,
            end_ms=context.test_end_ms,
            role="carry multi-regime scoring",
        )
        return _score_panel(panel, parameters=self.parameters, context=context)

    def score_panel(self, features: PanelData, *, context: FoldContext) -> StrategyScores:
        return _score_panel(features, parameters=self.parameters, context=context)


@dataclass(frozen=True, slots=True)
class CarryMultiRegimeStrategy:
    parameters: CarryMultiRegimeParameters
    strategy_id: str = field(default="carry_multi_regime", init=False)
    strategy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return CARRY_MULTI_REGIME_FEATURES

    def target_column(self) -> str | None:
        return None

    def fit(
        self,
        train: TrainingDataset,
        *,
        context: FoldContext,
    ) -> FittedCarryMultiRegimeStrategy:
        if train.target is not None:
            raise ResearchError("carry multi-regime is non-trainable and rejects a target")
        projected_panel(
            train.features,
            required_features=self.required_features(),
            start_ms=context.train_start_ms,
            end_ms=context.train_end_ms,
            role="carry multi-regime training",
        )
        return FittedCarryMultiRegimeStrategy(self.parameters)

    def fit_panel(
        self,
        train: PanelData,
        *,
        target: pl.DataFrame | None,
        context: FoldContext,
    ) -> FittedCarryMultiRegimeStrategy:
        validate_fixed_panel_fit(
            train,
            required_features=self.required_features(),
            target=target,
            context=context,
            role="carry multi-regime",
        )
        return FittedCarryMultiRegimeStrategy(self.parameters)


__all__ = [
    "CARRY_MULTI_REGIME_FEATURES",
    "CarryMultiRegimeParameters",
    "CarryMultiRegimeStrategy",
    "FittedCarryMultiRegimeStrategy",
]
