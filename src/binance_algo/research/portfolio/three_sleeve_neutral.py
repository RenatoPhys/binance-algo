"""Buffered convex combination of carry and two strength sleeves."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FEATURE_KEY_COLUMNS, FoldContext, StrategyScores
from binance_algo.research.datasets.views import build_feature_view
from binance_algo.research.panel import PanelData
from binance_algo.research.portfolio.neutral_long_short import (
    NEUTRAL_LONG_SHORT_FEATURES,
    BufferedNeutralLongShortParameters,
    _buffered_target_weight_frame,
)

THREE_SLEEVE_SCORE_COLUMNS = (
    "carry_score",
    "fast_strength_score",
    "slow_strength_score",
)


@dataclass(frozen=True, slots=True)
class BufferedThreeSleeveNeutralParameters:
    carry_weight: float
    fast_strength_weight: float
    slow_strength_weight: float
    no_trade_score_band: float
    rebalance_interval_hours: int
    gross_exposure: float
    annual_volatility_target: float
    max_symbol_weight: float

    def __post_init__(self) -> None:
        weights = (
            self.carry_weight,
            self.fast_strength_weight,
            self.slow_strength_weight,
        )
        if any(not math.isfinite(value) or value < 0 or value > 1 for value in weights):
            raise ResearchError("three-sleeve weights must be finite and in [0, 1]")
        if not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ResearchError("three-sleeve weights must sum to one")
        BufferedNeutralLongShortParameters(
            no_trade_score_band=self.no_trade_score_band,
            rebalance_interval_hours=self.rebalance_interval_hours,
            gross_exposure=self.gross_exposure,
            annual_volatility_target=self.annual_volatility_target,
            max_symbol_weight=self.max_symbol_weight,
            minimum_score_spread=0.0,
        )

    def sleeve_parameters(self) -> BufferedNeutralLongShortParameters:
        return BufferedNeutralLongShortParameters(
            no_trade_score_band=self.no_trade_score_band,
            rebalance_interval_hours=self.rebalance_interval_hours,
            gross_exposure=self.gross_exposure,
            annual_volatility_target=self.annual_volatility_target,
            max_symbol_weight=self.max_symbol_weight,
            minimum_score_spread=0.0,
        )


def _frame_arrays(
    scores: pl.DataFrame,
    market_state: pl.DataFrame,
) -> tuple[
    tuple[str, ...],
    np.ndarray[Any, np.dtype[np.int64]],
    dict[str, np.ndarray[Any, np.dtype[np.float64]]],
]:
    validated = StrategyScores(scores).frame
    required_scores = (*FEATURE_KEY_COLUMNS, *THREE_SLEEVE_SCORE_COLUMNS)
    missing = sorted(set(required_scores).difference(validated.columns))
    if missing:
        raise ResearchError(f"three-sleeve scores are missing columns: {missing}")
    score_frame = validated.select(required_scores).sort(FEATURE_KEY_COLUMNS)
    market_frame = build_feature_view(
        market_state,
        required_features=NEUTRAL_LONG_SHORT_FEATURES,
    ).sort(FEATURE_KEY_COLUMNS)
    if not score_frame.select(FEATURE_KEY_COLUMNS).equals(
        market_frame.select(FEATURE_KEY_COLUMNS), null_equal=True
    ):
        raise ResearchError("three-sleeve scores and market state keys must align exactly")
    fields = (*THREE_SLEEVE_SCORE_COLUMNS, *NEUTRAL_LONG_SHORT_FEATURES)
    panel = PanelData.from_frame(
        score_frame.join(market_frame, on=list(FEATURE_KEY_COLUMNS), how="inner"),
        feature_columns=fields,
    )
    panel.require_complete(role="three-sleeve portfolio input")
    return panel.symbols, panel.times, dict(panel.features)


def _panel_arrays(
    scores: StrategyScores,
    market_state: PanelData,
    *,
    context: FoldContext,
) -> tuple[
    tuple[str, ...],
    np.ndarray[Any, np.dtype[np.int64]],
    dict[str, np.ndarray[Any, np.dtype[np.float64]]],
]:
    missing = sorted(set(THREE_SLEEVE_SCORE_COLUMNS).difference(scores.frame.columns))
    if missing:
        raise ResearchError(f"three-sleeve scores are missing columns: {missing}")
    score_panel = PanelData.from_frame(
        scores.frame,
        feature_columns=THREE_SLEEVE_SCORE_COLUMNS,
    )
    time_slice = market_state.time_slice(context.test_start_ms, context.test_end_ms)
    times = market_state.times[time_slice]
    market_state.require_complete_range(
        context.test_start_ms,
        context.test_end_ms,
        role="three-sleeve portfolio market state",
    )
    if score_panel.symbols != market_state.symbols or not np.array_equal(score_panel.times, times):
        raise ResearchError("three-sleeve scores and market state keys must align exactly")
    arrays = {
        name: np.asarray(score_panel.features[name], dtype=np.float64)
        for name in THREE_SLEEVE_SCORE_COLUMNS
    }
    arrays.update(
        {
            feature: np.asarray(
                market_state.matrix(
                    feature,
                    start_ms=context.test_start_ms,
                    end_ms=context.test_end_ms,
                ),
                dtype=np.float64,
            )
            for feature in NEUTRAL_LONG_SHORT_FEATURES
        }
    )
    return market_state.symbols, times, arrays


def _sleeve_targets(
    *,
    score_column: str,
    symbols: tuple[str, ...],
    times: np.ndarray[Any, np.dtype[np.int64]],
    arrays: dict[str, np.ndarray[Any, np.dtype[np.float64]]],
    parameters: BufferedNeutralLongShortParameters,
    context: FoldContext,
) -> pl.DataFrame:
    common = {feature: arrays[feature] for feature in NEUTRAL_LONG_SHORT_FEATURES}
    return _buffered_target_weight_frame(
        symbols=symbols,
        times=times,
        arrays={**common, "score": arrays[score_column]},
        parameters=parameters,
        context=context,
    )


def _target_weights(
    *,
    symbols: tuple[str, ...],
    times: np.ndarray[Any, np.dtype[np.int64]],
    arrays: dict[str, np.ndarray[Any, np.dtype[np.float64]]],
    parameters: BufferedThreeSleeveNeutralParameters,
    context: FoldContext,
) -> pl.DataFrame:
    sleeve_parameters = parameters.sleeve_parameters()
    carry = _sleeve_targets(
        score_column="carry_score",
        symbols=symbols,
        times=times,
        arrays=arrays,
        parameters=sleeve_parameters,
        context=context,
    ).rename({"target_weight": "carry_target_weight"})
    fast = _sleeve_targets(
        score_column="fast_strength_score",
        symbols=symbols,
        times=times,
        arrays=arrays,
        parameters=sleeve_parameters,
        context=context,
    ).rename({"target_weight": "fast_target_weight"})
    slow = _sleeve_targets(
        score_column="slow_strength_score",
        symbols=symbols,
        times=times,
        arrays=arrays,
        parameters=sleeve_parameters,
        context=context,
    ).rename({"target_weight": "slow_target_weight"})
    return (
        carry.join(fast, on=list(FEATURE_KEY_COLUMNS), how="inner")
        .join(
            slow,
            on=list(FEATURE_KEY_COLUMNS),
            how="inner",
        )
        .select(
            *FEATURE_KEY_COLUMNS,
            (
                parameters.carry_weight * pl.col("carry_target_weight")
                + parameters.fast_strength_weight * pl.col("fast_target_weight")
                + parameters.slow_strength_weight * pl.col("slow_target_weight")
            ).alias("target_weight"),
        )
    )


@dataclass(frozen=True, slots=True)
class BufferedThreeSleeveNeutralPolicy:
    parameters: BufferedThreeSleeveNeutralParameters
    policy_id: str = field(default="buffered_three_sleeve_neutral", init=False)
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
        symbols, times, arrays = _frame_arrays(scores, market_state)
        return _target_weights(
            symbols=symbols,
            times=times,
            arrays=arrays,
            parameters=self.parameters,
            context=context,
        )

    def target_weights_panel(
        self,
        scores: StrategyScores,
        market_state: PanelData,
        *,
        context: FoldContext,
    ) -> pl.DataFrame:
        symbols, times, arrays = _panel_arrays(scores, market_state, context=context)
        return _target_weights(
            symbols=symbols,
            times=times,
            arrays=arrays,
            parameters=self.parameters,
            context=context,
        )


__all__ = [
    "THREE_SLEEVE_SCORE_COLUMNS",
    "BufferedThreeSleeveNeutralParameters",
    "BufferedThreeSleeveNeutralPolicy",
]
