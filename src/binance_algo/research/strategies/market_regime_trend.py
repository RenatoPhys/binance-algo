"""Causal long/flat trend requiring an upward equal-weight market regime."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext, StrategyScores, TrainingDataset
from binance_algo.research.panel import PanelData, matrix_to_long_frame
from binance_algo.research.strategies.fixed import projected_panel, validate_fixed_panel_fit
from binance_algo.research.strategies.multi_horizon_trend import causal_rolling_return

MARKET_REGIME_TREND_FEATURES = ("log_return_1h",)


@dataclass(frozen=True, slots=True)
class MarketRegimeTrendParameters:
    fast_lookback_hours: int
    slow_lookback_hours: int

    def __post_init__(self) -> None:
        if not 24 <= self.fast_lookback_hours < self.slow_lookback_hours <= 24 * 90:
            raise ResearchError(
                "market-regime trend lookbacks must be increasing between one and 90 days"
            )


def _score_panel(
    panel: PanelData,
    *,
    parameters: MarketRegimeTrendParameters,
    context: FoldContext,
) -> StrategyScores:
    panel.require_complete_range(
        context.test_start_ms,
        context.test_end_ms,
        role="market-regime trend scoring",
    )
    time_slice = panel.time_slice(context.test_start_ms, context.test_end_ms)
    fast = causal_rolling_return(
        panel,
        lookback_hours=parameters.fast_lookback_hours,
    )[time_slice]
    slow = causal_rolling_return(
        panel,
        lookback_hours=parameters.slow_lookback_hours,
    )[time_slice]
    if np.any(~np.isfinite(fast)) or np.any(~np.isfinite(slow)):
        raise ResearchError("market-regime trend scoring lacks causal return history")

    market_up = np.mean(slow, axis=1) > 0
    individual_up = (fast > 0) & (slow > 0)
    score = np.where(market_up[:, None] & individual_up, 1.0, 0.0)
    return StrategyScores(
        matrix_to_long_frame(
            times=panel.times[time_slice],
            symbols=panel.symbols,
            value_name="score",
            values=score,
        )
    )


@dataclass(frozen=True, slots=True)
class FittedMarketRegimeTrendStrategy:
    parameters: MarketRegimeTrendParameters

    def score(self, features: pl.DataFrame, *, context: FoldContext) -> StrategyScores:
        panel = projected_panel(
            features,
            required_features=MARKET_REGIME_TREND_FEATURES,
            start_ms=context.test_start_ms,
            end_ms=context.test_end_ms,
            role="market-regime trend scoring",
        )
        return _score_panel(panel, parameters=self.parameters, context=context)

    def score_panel(self, features: PanelData, *, context: FoldContext) -> StrategyScores:
        return _score_panel(features, parameters=self.parameters, context=context)


@dataclass(frozen=True, slots=True)
class MarketRegimeTrendStrategy:
    parameters: MarketRegimeTrendParameters
    strategy_id: str = field(default="market_regime_trend", init=False)
    strategy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return MARKET_REGIME_TREND_FEATURES

    def target_column(self) -> str | None:
        return None

    def fit(
        self,
        train: TrainingDataset,
        *,
        context: FoldContext,
    ) -> FittedMarketRegimeTrendStrategy:
        if train.target is not None:
            raise ResearchError("market-regime trend is non-trainable and rejects a target")
        projected_panel(
            train.features,
            required_features=self.required_features(),
            start_ms=context.train_start_ms,
            end_ms=context.train_end_ms,
            role="market-regime trend training",
        )
        return FittedMarketRegimeTrendStrategy(self.parameters)

    def fit_panel(
        self,
        train: PanelData,
        *,
        target: pl.DataFrame | None,
        context: FoldContext,
    ) -> FittedMarketRegimeTrendStrategy:
        validate_fixed_panel_fit(
            train,
            required_features=self.required_features(),
            target=target,
            context=context,
            role="market-regime trend",
        )
        return FittedMarketRegimeTrendStrategy(self.parameters)


__all__ = [
    "MARKET_REGIME_TREND_FEATURES",
    "FittedMarketRegimeTrendStrategy",
    "MarketRegimeTrendParameters",
    "MarketRegimeTrendStrategy",
]
