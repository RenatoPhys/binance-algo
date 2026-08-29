"""Funding carry diversified by relative-strength and SMA-trend sleeves."""

from __future__ import annotations

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
from binance_algo.research.strategies.funding_carry import FundingCarryParameters
from binance_algo.research.strategies.relative_strength import (
    RelativeStrengthParameters,
    causal_relative_strength,
)
from binance_algo.research.strategies.sma_crossover import (
    SmaCrossoverParameters,
    causal_sma_crossover,
)

CARRY_DUAL_TREND_FEATURES = (
    "funding_rate_current",
    "funding_rate_change",
    "residual_momentum_4h",
    "log_return_1h",
)


@dataclass(frozen=True, slots=True)
class CarryDualTrendParameters:
    funding_rate_weight: float
    funding_change_weight: float
    momentum_confirmation_weight: float
    relative_strength_lookback_hours: int
    sma_fast_window_hours: int
    sma_slow_window_hours: int

    def __post_init__(self) -> None:
        FundingCarryParameters(
            funding_rate_weight=self.funding_rate_weight,
            funding_change_weight=self.funding_change_weight,
            momentum_confirmation_weight=self.momentum_confirmation_weight,
        )
        RelativeStrengthParameters(lookback_hours=self.relative_strength_lookback_hours)
        SmaCrossoverParameters(
            fast_window_hours=self.sma_fast_window_hours,
            slow_window_hours=self.sma_slow_window_hours,
        )

    def funding_parameters(self) -> FundingCarryParameters:
        return FundingCarryParameters(
            funding_rate_weight=self.funding_rate_weight,
            funding_change_weight=self.funding_change_weight,
            momentum_confirmation_weight=self.momentum_confirmation_weight,
        )


def _score_panel(
    panel: PanelData,
    *,
    parameters: CarryDualTrendParameters,
    context: FoldContext,
) -> StrategyScores:
    panel.require_complete_range(
        context.test_start_ms,
        context.test_end_ms,
        role="carry dual-trend scoring",
    )
    time_slice = panel.time_slice(context.test_start_ms, context.test_end_ms)
    funding = parameters.funding_parameters()
    current = np.asarray(
        panel.matrix(
            "funding_rate_current",
            start_ms=context.test_start_ms,
            end_ms=context.test_end_ms,
        ),
        dtype=np.float64,
    )
    change = np.asarray(
        panel.matrix(
            "funding_rate_change",
            start_ms=context.test_start_ms,
            end_ms=context.test_end_ms,
        ),
        dtype=np.float64,
    )
    momentum = np.asarray(
        panel.matrix(
            "residual_momentum_4h",
            start_ms=context.test_start_ms,
            end_ms=context.test_end_ms,
        ),
        dtype=np.float64,
    )
    carry_score = (
        -funding.funding_rate_weight * cross_sectional_zscore(current)
        - funding.funding_change_weight * cross_sectional_zscore(change)
        + funding.momentum_confirmation_weight * cross_sectional_zscore(momentum)
    )
    relative_raw = causal_relative_strength(
        panel,
        parameters=RelativeStrengthParameters(
            lookback_hours=parameters.relative_strength_lookback_hours
        ),
    )[time_slice]
    sma_raw = causal_sma_crossover(
        panel,
        parameters=SmaCrossoverParameters(
            fast_window_hours=parameters.sma_fast_window_hours,
            slow_window_hours=parameters.sma_slow_window_hours,
        ),
    )[time_slice]
    if np.any(~np.isfinite(relative_raw)) or np.any(~np.isfinite(sma_raw)):
        raise ResearchError("carry dual-trend scoring lacks causal return history")
    relative_score = cross_sectional_zscore(relative_raw)
    sma_score = cross_sectional_zscore(sma_raw)
    diagnostic_score = (carry_score + relative_score + sma_score) / 3
    frame = matrix_to_long_frame(
        times=panel.times[time_slice],
        symbols=panel.symbols,
        value_name="score",
        values=diagnostic_score,
    ).with_columns(
        pl.Series("carry_score", carry_score.reshape(-1)),
        pl.Series("fast_strength_score", relative_score.reshape(-1)),
        pl.Series("slow_strength_score", sma_score.reshape(-1)),
    )
    return StrategyScores(frame)


@dataclass(frozen=True, slots=True)
class FittedCarryDualTrendStrategy:
    parameters: CarryDualTrendParameters

    def score(self, features: pl.DataFrame, *, context: FoldContext) -> StrategyScores:
        panel = projected_panel(
            features,
            required_features=CARRY_DUAL_TREND_FEATURES,
            start_ms=context.test_start_ms,
            end_ms=context.test_end_ms,
            role="carry dual-trend scoring",
        )
        return _score_panel(panel, parameters=self.parameters, context=context)

    def score_panel(self, features: PanelData, *, context: FoldContext) -> StrategyScores:
        return _score_panel(features, parameters=self.parameters, context=context)


@dataclass(frozen=True, slots=True)
class CarryDualTrendStrategy:
    parameters: CarryDualTrendParameters
    strategy_id: str = field(default="carry_dual_trend", init=False)
    strategy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return CARRY_DUAL_TREND_FEATURES

    def target_column(self) -> str | None:
        return None

    def fit(
        self,
        train: TrainingDataset,
        *,
        context: FoldContext,
    ) -> FittedCarryDualTrendStrategy:
        if train.target is not None:
            raise ResearchError("carry dual-trend is non-trainable and rejects a target")
        projected_panel(
            train.features,
            required_features=self.required_features(),
            start_ms=context.train_start_ms,
            end_ms=context.train_end_ms,
            role="carry dual-trend training",
        )
        return FittedCarryDualTrendStrategy(self.parameters)

    def fit_panel(
        self,
        train: PanelData,
        *,
        target: pl.DataFrame | None,
        context: FoldContext,
    ) -> FittedCarryDualTrendStrategy:
        validate_fixed_panel_fit(
            train,
            required_features=self.required_features(),
            target=target,
            context=context,
            role="carry dual-trend",
        )
        return FittedCarryDualTrendStrategy(self.parameters)


__all__ = [
    "CARRY_DUAL_TREND_FEATURES",
    "CarryDualTrendParameters",
    "CarryDualTrendStrategy",
    "FittedCarryDualTrendStrategy",
]
