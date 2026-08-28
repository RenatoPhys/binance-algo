"""Buffered volatility-scaled portfolio policy for absolute directional scores."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FEATURE_KEY_COLUMNS, FoldContext, StrategyScores
from binance_algo.research.datasets.views import build_feature_view
from binance_algo.research.panel import PanelData, matrix_to_long_frame

DIRECTIONAL_FEATURES = ("realized_volatility_24h",)
HOUR_MS = 3_600_000


@dataclass(frozen=True, slots=True)
class BufferedDirectionalParameters:
    """Risk controls and a causal interval for absolute-score portfolios."""

    signal_threshold: float
    rebalance_interval_hours: int
    gross_exposure: float
    annual_volatility_target: float
    max_symbol_weight: float

    def __post_init__(self) -> None:
        values = (
            self.signal_threshold,
            self.gross_exposure,
            self.annual_volatility_target,
            self.max_symbol_weight,
        )
        if any(not math.isfinite(value) for value in values):
            raise ResearchError("directional portfolio parameters must be finite")
        if not 0 <= self.signal_threshold <= 1:
            raise ResearchError("directional signal threshold must be in [0, 1]")
        if not 1 <= self.rebalance_interval_hours <= 24 * 30:
            raise ResearchError(
                "directional rebalance interval must be between one hour and 30 days"
            )
        if not 0 < self.gross_exposure <= 1:
            raise ResearchError("directional gross exposure must be in (0, 1]")
        if not 0 < self.annual_volatility_target <= 1:
            raise ResearchError("directional annual volatility target must be in (0, 1]")
        if not 0 < self.max_symbol_weight <= 1:
            raise ResearchError("directional maximum symbol weight must be in (0, 1]")


def _aligned_frame_data(
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
        required_features=DIRECTIONAL_FEATURES,
    ).sort(FEATURE_KEY_COLUMNS)
    if not score_frame.select(FEATURE_KEY_COLUMNS).equals(
        market_frame.select(FEATURE_KEY_COLUMNS), null_equal=True
    ):
        raise ResearchError("directional scores and market state keys must align exactly")
    joined = score_frame.join(market_frame, on=list(FEATURE_KEY_COLUMNS), how="inner")
    fields = ("score", *DIRECTIONAL_FEATURES)
    panel = PanelData.from_frame(joined, feature_columns=fields)
    panel.require_complete(role="directional portfolio input")
    return panel.symbols, panel.times, dict(panel.features)


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
        role="directional portfolio market state",
    )
    if score_panel.symbols != market_state.symbols or not np.array_equal(score_panel.times, times):
        raise ResearchError("directional scores and market state keys must align exactly")
    return (
        market_state.symbols,
        times,
        {
            "score": np.asarray(score_panel.features["score"], dtype=np.float64),
            "realized_volatility_24h": np.asarray(
                market_state.matrix(
                    "realized_volatility_24h",
                    start_ms=context.test_start_ms,
                    end_ms=context.test_end_ms,
                ),
                dtype=np.float64,
            ),
        },
    )


def _directional_weights(
    scores: np.ndarray[Any, np.dtype[np.float64]],
    realized_volatility: np.ndarray[Any, np.dtype[np.float64]],
    *,
    parameters: BufferedDirectionalParameters,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    active = np.abs(scores) > parameters.signal_threshold
    if not np.any(active):
        return np.zeros_like(scores)
    inverse_volatility = np.divide(
        1.0,
        realized_volatility,
        out=np.zeros_like(realized_volatility),
        where=realized_volatility > 1e-15,
    )
    raw = np.sign(scores) * inverse_volatility * active
    raw_gross = float(np.sum(np.abs(raw)))
    if raw_gross <= 1e-15:
        return np.zeros_like(scores)
    normalized = raw / raw_gross
    annualized_asset_volatility = realized_volatility * math.sqrt(365)
    volatility_proxy = float(math.sqrt(np.sum(np.square(normalized * annualized_asset_volatility))))
    target_gross = parameters.gross_exposure
    if volatility_proxy > 0:
        target_gross = min(target_gross, parameters.annual_volatility_target / volatility_proxy)
    weights = normalized * target_gross
    maximum_weight = float(np.max(np.abs(weights)))
    if maximum_weight > parameters.max_symbol_weight:
        weights *= parameters.max_symbol_weight / maximum_weight
    if float(np.sum(np.abs(weights))) > 1 + 1e-12:
        raise ResearchError("directional portfolio attempted economic leverage")
    return np.asarray(weights, dtype=np.float64)


def _target_weight_frame(
    *,
    symbols: tuple[str, ...],
    times: np.ndarray[Any, np.dtype[np.int64]],
    arrays: dict[str, np.ndarray[Any, np.dtype[np.float64]]],
    parameters: BufferedDirectionalParameters,
    context: FoldContext,
) -> pl.DataFrame:
    if int(times[0]) < context.test_start_ms or int(times[-1]) > context.test_end_ms:
        raise ResearchError(
            "directional portfolio input contains decisions outside its fold context"
        )
    targets = np.zeros_like(arrays["score"])
    previous_weights = np.zeros(len(symbols), dtype=np.float64)
    last_rebalance_ms: int | None = None
    interval_ms = parameters.rebalance_interval_hours * HOUR_MS
    for period, decision_time in enumerate(times):
        if last_rebalance_ms is not None and int(decision_time) - last_rebalance_ms < interval_ms:
            targets[period] = previous_weights
            continue
        previous_weights = _directional_weights(
            arrays["score"][period],
            arrays["realized_volatility_24h"][period],
            parameters=parameters,
        )
        last_rebalance_ms = int(decision_time)
        targets[period] = previous_weights
    return matrix_to_long_frame(
        times=times,
        symbols=symbols,
        value_name="target_weight",
        values=targets,
    )


@dataclass(frozen=True, slots=True)
class BufferedDirectionalPolicy:
    """Map each absolute score sign to a held, inverse-volatility-scaled position."""

    parameters: BufferedDirectionalParameters
    policy_id: str = field(default="buffered_directional", init=False)
    policy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return DIRECTIONAL_FEATURES

    def target_weights(
        self,
        scores: pl.DataFrame,
        market_state: pl.DataFrame,
        *,
        context: FoldContext,
    ) -> pl.DataFrame:
        symbols, times, arrays = _aligned_frame_data(scores, market_state)
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


__all__ = [
    "DIRECTIONAL_FEATURES",
    "BufferedDirectionalParameters",
    "BufferedDirectionalPolicy",
]
