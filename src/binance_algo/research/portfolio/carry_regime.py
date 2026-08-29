"""Convex blend of a neutral three-sleeve carry core and a long/flat regime sleeve."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FEATURE_KEY_COLUMNS, FoldContext, StrategyScores
from binance_algo.research.panel import PanelData
from binance_algo.research.portfolio.long_flat import (
    BufferedLongFlatParameters,
    BufferedLongFlatPolicy,
)
from binance_algo.research.portfolio.neutral_long_short import NEUTRAL_LONG_SHORT_FEATURES
from binance_algo.research.portfolio.three_sleeve_neutral import (
    BufferedThreeSleeveNeutralParameters,
    BufferedThreeSleeveNeutralPolicy,
)


@dataclass(frozen=True, slots=True)
class BufferedCarryRegimeParameters:
    core_carry_weight: float
    core_fast_strength_weight: float
    core_slow_strength_weight: float
    regime_trend_weight: float
    no_trade_score_band: float
    signal_threshold: float
    rebalance_interval_hours: int
    gross_exposure: float
    annual_volatility_target: float
    max_symbol_weight: float

    def __post_init__(self) -> None:
        core_weights = (
            self.core_carry_weight,
            self.core_fast_strength_weight,
            self.core_slow_strength_weight,
        )
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in core_weights):
            raise ResearchError("carry-regime core weights must be finite and in [0, 1]")
        if not math.isclose(sum(core_weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ResearchError("carry-regime core weights must sum to one")
        if not math.isfinite(self.regime_trend_weight) or not 0 <= self.regime_trend_weight <= 0.5:
            raise ResearchError("carry-regime trend weight must be finite and in [0, 0.5]")
        self.core_parameters()
        self.trend_parameters()

    def core_parameters(self) -> BufferedThreeSleeveNeutralParameters:
        return BufferedThreeSleeveNeutralParameters(
            carry_weight=self.core_carry_weight,
            fast_strength_weight=self.core_fast_strength_weight,
            slow_strength_weight=self.core_slow_strength_weight,
            no_trade_score_band=self.no_trade_score_band,
            rebalance_interval_hours=self.rebalance_interval_hours,
            gross_exposure=self.gross_exposure,
            annual_volatility_target=self.annual_volatility_target,
            max_symbol_weight=self.max_symbol_weight,
        )

    def trend_parameters(self) -> BufferedLongFlatParameters:
        return BufferedLongFlatParameters(
            signal_threshold=self.signal_threshold,
            rebalance_interval_hours=self.rebalance_interval_hours,
            gross_exposure=self.gross_exposure,
            annual_volatility_target=self.annual_volatility_target,
            max_symbol_weight=self.max_symbol_weight,
        )


def _trend_scores(scores: StrategyScores) -> StrategyScores:
    if "regime_trend_score" not in scores.frame.columns:
        raise ResearchError("carry-regime scores are missing column: regime_trend_score")
    return StrategyScores(
        scores.frame.select(
            *FEATURE_KEY_COLUMNS,
            pl.col("regime_trend_score").alias("score"),
        )
    )


def _blend_targets(
    core: pl.DataFrame,
    trend: pl.DataFrame,
    *,
    regime_trend_weight: float,
) -> pl.DataFrame:
    return (
        core.rename({"target_weight": "core_target_weight"})
        .join(
            trend.rename({"target_weight": "trend_target_weight"}),
            on=list(FEATURE_KEY_COLUMNS),
            how="inner",
        )
        .select(
            *FEATURE_KEY_COLUMNS,
            (
                (1 - regime_trend_weight) * pl.col("core_target_weight")
                + regime_trend_weight * pl.col("trend_target_weight")
            ).alias("target_weight"),
        )
    )


@dataclass(frozen=True, slots=True)
class BufferedCarryRegimePolicy:
    parameters: BufferedCarryRegimeParameters
    policy_id: str = field(default="buffered_carry_regime", init=False)
    policy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return NEUTRAL_LONG_SHORT_FEATURES

    def target_weights(
        self,
        scores: pl.DataFrame,
        market_state: pl.DataFrame,
        *,
        context: FoldContext,
    ) -> pl.DataFrame:
        validated = StrategyScores(scores)
        core = BufferedThreeSleeveNeutralPolicy(self.parameters.core_parameters()).target_weights(
            validated.frame,
            market_state,
            context=context,
        )
        trend = BufferedLongFlatPolicy(self.parameters.trend_parameters()).target_weights(
            _trend_scores(validated).frame,
            market_state,
            context=context,
        )
        return _blend_targets(
            core,
            trend,
            regime_trend_weight=self.parameters.regime_trend_weight,
        )

    def target_weights_panel(
        self,
        scores: StrategyScores,
        market_state: PanelData,
        *,
        context: FoldContext,
    ) -> pl.DataFrame:
        core = BufferedThreeSleeveNeutralPolicy(
            self.parameters.core_parameters()
        ).target_weights_panel(scores, market_state, context=context)
        trend = BufferedLongFlatPolicy(self.parameters.trend_parameters()).target_weights_panel(
            _trend_scores(scores), market_state, context=context
        )
        return _blend_targets(
            core,
            trend,
            regime_trend_weight=self.parameters.regime_trend_weight,
        )


__all__ = [
    "BufferedCarryRegimeParameters",
    "BufferedCarryRegimePolicy",
]
