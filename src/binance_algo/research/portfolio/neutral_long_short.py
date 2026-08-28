"""Neutral long/short portfolio policy extracted from the Phase 3 baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import (
    FEATURE_KEY_COLUMNS,
    FoldContext,
    StrategyScores,
)
from binance_algo.research.datasets.views import build_feature_view
from binance_algo.research.panel import PanelData, matrix_to_long_frame

NEUTRAL_LONG_SHORT_FEATURES = ("rolling_beta", "realized_volatility_24h")
HOUR_MS = 3_600_000


@dataclass(frozen=True, slots=True)
class NeutralLongShortParameters:
    """Immutable constraints for the baseline neutral long/short policy."""

    no_trade_score_band: float
    gross_exposure: float
    annual_volatility_target: float
    max_symbol_weight: float

    def __post_init__(self) -> None:
        values = (
            self.no_trade_score_band,
            self.gross_exposure,
            self.annual_volatility_target,
            self.max_symbol_weight,
        )
        if any(not math.isfinite(value) for value in values):
            raise ResearchError("portfolio parameters must be finite")
        if self.no_trade_score_band < 0:
            raise ResearchError("no-trade score band cannot be negative")
        if not 0 < self.gross_exposure <= 1:
            raise ResearchError("gross exposure must be in (0, 1]")
        if not 0 < self.annual_volatility_target <= 1:
            raise ResearchError("annual volatility target must be in (0, 1]")
        if not 0 < self.max_symbol_weight <= 1:
            raise ResearchError("maximum symbol weight must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class BufferedNeutralLongShortParameters:
    """Risk constraints plus a causal minimum interval between rebalances."""

    no_trade_score_band: float
    rebalance_interval_hours: int
    gross_exposure: float
    annual_volatility_target: float
    max_symbol_weight: float
    minimum_score_spread: float = 0.0

    def __post_init__(self) -> None:
        NeutralLongShortParameters(
            no_trade_score_band=self.no_trade_score_band,
            gross_exposure=self.gross_exposure,
            annual_volatility_target=self.annual_volatility_target,
            max_symbol_weight=self.max_symbol_weight,
        )
        if not 1 <= self.rebalance_interval_hours <= 24 * 30:
            raise ResearchError("rebalance interval must be between one hour and 30 days")
        if not math.isfinite(self.minimum_score_spread) or self.minimum_score_spread < 0:
            raise ResearchError("minimum score spread must be finite and non-negative")

    def base_parameters(self) -> NeutralLongShortParameters:
        return NeutralLongShortParameters(
            no_trade_score_band=self.no_trade_score_band,
            gross_exposure=self.gross_exposure,
            annual_volatility_target=self.annual_volatility_target,
            max_symbol_weight=self.max_symbol_weight,
        )


def _neutral_weights(
    *,
    top_index: int,
    bottom_index: int,
    betas: np.ndarray[Any, np.dtype[np.float64]],
    realized_volatility: np.ndarray[Any, np.dtype[np.float64]],
    parameters: NeutralLongShortParameters,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    raw = np.zeros(len(betas), dtype=np.float64)
    raw[top_index] = 1
    raw[bottom_index] = -1
    constraints = np.vstack((np.ones(len(betas), dtype=np.float64), betas))
    projected = raw - constraints.T @ np.linalg.pinv(constraints @ constraints.T) @ (
        constraints @ raw
    )
    if (
        float(np.sum(np.abs(projected))) <= 1e-12
        or projected[top_index] <= 0
        or projected[bottom_index] >= 0
    ):
        projected = raw
    projected /= float(np.sum(np.abs(projected)))
    annualized_asset_volatility = realized_volatility * math.sqrt(365)
    volatility_proxy = float(math.sqrt(np.sum(np.square(projected * annualized_asset_volatility))))
    target_gross = parameters.gross_exposure
    if volatility_proxy > 0:
        target_gross = min(target_gross, parameters.annual_volatility_target / volatility_proxy)
    weights = projected * target_gross
    maximum_weight = float(np.max(np.abs(weights)))
    if maximum_weight > parameters.max_symbol_weight:
        weights *= parameters.max_symbol_weight / maximum_weight
    if float(np.sum(np.abs(weights))) > 1 + 1e-12:
        raise ResearchError("portfolio policy attempted economic leverage")
    return weights


def _aligned_panel(
    scores: pl.DataFrame,
    market_state: pl.DataFrame,
) -> tuple[
    tuple[str, ...],
    np.ndarray[Any, np.dtype[np.int64]],
    dict[str, np.ndarray[Any, np.dtype[np.float64]]],
]:
    score_frame = (
        StrategyScores(scores).frame.select(*FEATURE_KEY_COLUMNS, "score").sort(FEATURE_KEY_COLUMNS)
    )
    market_frame = build_feature_view(
        market_state,
        required_features=NEUTRAL_LONG_SHORT_FEATURES,
    ).sort(FEATURE_KEY_COLUMNS)
    if not score_frame.select(FEATURE_KEY_COLUMNS).equals(
        market_frame.select(FEATURE_KEY_COLUMNS), null_equal=True
    ):
        raise ResearchError("strategy scores and portfolio market state keys must align exactly")
    panel = score_frame.join(market_frame, on=list(FEATURE_KEY_COLUMNS), how="inner")
    fields = ("score", *NEUTRAL_LONG_SHORT_FEATURES)
    panel_data = PanelData.from_frame(panel, feature_columns=fields)
    panel_data.require_complete(role="portfolio input")
    return panel_data.symbols, panel_data.times, dict(panel_data.features)


def _aligned_panel_data(
    scores: StrategyScores,
    market_state: PanelData,
    *,
    context: FoldContext,
) -> tuple[
    tuple[str, ...],
    np.ndarray[Any, np.dtype[np.int64]],
    dict[str, np.ndarray[Any, np.dtype[np.float64]]],
]:
    score_panel = PanelData.from_frame(scores.frame, feature_columns=("score",))
    time_slice = market_state.time_slice(context.test_start_ms, context.test_end_ms)
    times = market_state.times[time_slice]
    market_state.require_complete_range(
        context.test_start_ms,
        context.test_end_ms,
        role="portfolio market state",
    )
    if score_panel.symbols != market_state.symbols or not np.array_equal(score_panel.times, times):
        raise ResearchError("strategy scores and portfolio market state keys must align exactly")
    arrays = {"score": np.asarray(score_panel.features["score"], dtype=np.float64)}
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


def _target_weight_frame(
    *,
    symbols: tuple[str, ...],
    times: np.ndarray[Any, np.dtype[np.int64]],
    arrays: dict[str, np.ndarray[Any, np.dtype[np.float64]]],
    parameters: NeutralLongShortParameters,
    context: FoldContext,
) -> pl.DataFrame:
    if int(times[0]) < context.test_start_ms or int(times[-1]) > context.test_end_ms:
        raise ResearchError("portfolio input contains decisions outside its fold context")
    targets = np.empty_like(arrays["score"])
    previous_top: int | None = None
    previous_bottom: int | None = None
    band = parameters.no_trade_score_band
    for period in range(len(times)):
        top = int(np.argmax(arrays["score"][period]))
        bottom = int(np.argmin(arrays["score"][period]))
        if (
            previous_top is not None
            and arrays["score"][period, top] - arrays["score"][period, previous_top] <= band
        ):
            top = previous_top
        if (
            previous_bottom is not None
            and arrays["score"][period, previous_bottom] - arrays["score"][period, bottom] <= band
        ):
            bottom = previous_bottom
        if top == bottom:
            raise ResearchError("cross-sectional score did not produce distinct tails")
        targets[period] = _neutral_weights(
            top_index=top,
            bottom_index=bottom,
            betas=arrays["rolling_beta"][period],
            realized_volatility=arrays["realized_volatility_24h"][period],
            parameters=parameters,
        )
        previous_top, previous_bottom = top, bottom
    return matrix_to_long_frame(
        times=times,
        symbols=symbols,
        value_name="target_weight",
        values=targets,
    )


def _buffered_target_weight_frame(
    *,
    symbols: tuple[str, ...],
    times: np.ndarray[Any, np.dtype[np.int64]],
    arrays: dict[str, np.ndarray[Any, np.dtype[np.float64]]],
    parameters: BufferedNeutralLongShortParameters,
    context: FoldContext,
) -> pl.DataFrame:
    if int(times[0]) < context.test_start_ms or int(times[-1]) > context.test_end_ms:
        raise ResearchError("portfolio input contains decisions outside its fold context")
    targets = np.zeros_like(arrays["score"])
    previous_weights = np.zeros(len(symbols), dtype=np.float64)
    previous_top: int | None = None
    previous_bottom: int | None = None
    last_rebalance_ms: int | None = None
    interval_ms = parameters.rebalance_interval_hours * HOUR_MS
    base_parameters = parameters.base_parameters()
    for period, decision_time in enumerate(times):
        if last_rebalance_ms is not None and int(decision_time) - last_rebalance_ms < interval_ms:
            targets[period] = previous_weights
            continue
        score_spread = float(np.max(arrays["score"][period]) - np.min(arrays["score"][period]))
        if score_spread < parameters.minimum_score_spread:
            previous_weights = np.zeros(len(symbols), dtype=np.float64)
            previous_top = None
            previous_bottom = None
            last_rebalance_ms = int(decision_time)
            targets[period] = previous_weights
            continue
        top = int(np.argmax(arrays["score"][period]))
        bottom = int(np.argmin(arrays["score"][period]))
        band = parameters.no_trade_score_band
        if (
            previous_top is not None
            and arrays["score"][period, top] - arrays["score"][period, previous_top] <= band
        ):
            top = previous_top
        if (
            previous_bottom is not None
            and arrays["score"][period, previous_bottom] - arrays["score"][period, bottom] <= band
        ):
            bottom = previous_bottom
        if top == bottom:
            raise ResearchError("cross-sectional score did not produce distinct tails")
        previous_weights = _neutral_weights(
            top_index=top,
            bottom_index=bottom,
            betas=arrays["rolling_beta"][period],
            realized_volatility=arrays["realized_volatility_24h"][period],
            parameters=base_parameters,
        )
        targets[period] = previous_weights
        previous_top, previous_bottom = top, bottom
        last_rebalance_ms = int(decision_time)
    return matrix_to_long_frame(
        times=times,
        symbols=symbols,
        value_name="target_weight",
        values=targets,
    )


@dataclass(frozen=True, slots=True)
class NeutralLongShortPolicy:
    """Select distinct score tails and impose the Phase 3 portfolio constraints."""

    parameters: NeutralLongShortParameters
    policy_id: str = field(default="neutral_long_short", init=False)
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
        symbols, times, arrays = _aligned_panel(scores, market_state)
        return _target_weight_frame(
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
        symbols, times, arrays = _aligned_panel_data(scores, market_state, context=context)
        return _target_weight_frame(
            symbols=symbols,
            times=times,
            arrays=arrays,
            parameters=self.parameters,
            context=context,
        )


@dataclass(frozen=True, slots=True)
class BufferedNeutralLongShortPolicy:
    """Hold risk-controlled weights between explicit causal rebalance times."""

    parameters: BufferedNeutralLongShortParameters
    policy_id: str = field(default="buffered_neutral_long_short", init=False)
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
        symbols, times, arrays = _aligned_panel(scores, market_state)
        return _buffered_target_weight_frame(
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
        symbols, times, arrays = _aligned_panel_data(scores, market_state, context=context)
        return _buffered_target_weight_frame(
            symbols=symbols,
            times=times,
            arrays=arrays,
            parameters=self.parameters,
            context=context,
        )


__all__ = [
    "NEUTRAL_LONG_SHORT_FEATURES",
    "BufferedNeutralLongShortParameters",
    "BufferedNeutralLongShortPolicy",
    "NeutralLongShortParameters",
    "NeutralLongShortPolicy",
]
