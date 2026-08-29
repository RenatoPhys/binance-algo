"""Buffered volatility-scaled long/flat policy for directional trend scores."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext, StrategyScores
from binance_algo.research.panel import PanelData, matrix_to_long_frame
from binance_algo.research.portfolio.directional import (
    DIRECTIONAL_FEATURES,
    HOUR_MS,
    _aligned_frame_data,
    _aligned_panel_data,
)


@dataclass(frozen=True, slots=True)
class BufferedLongFlatParameters:
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
            raise ResearchError("long-flat portfolio parameters must be finite")
        if not 0 <= self.signal_threshold <= 1:
            raise ResearchError("long-flat signal threshold must be in [0, 1]")
        if not 1 <= self.rebalance_interval_hours <= 24 * 30:
            raise ResearchError("long-flat rebalance interval must be between one hour and 30 days")
        if not 0 < self.gross_exposure <= 1:
            raise ResearchError("long-flat gross exposure must be in (0, 1]")
        if not 0 < self.annual_volatility_target <= 1:
            raise ResearchError("long-flat annual volatility target must be in (0, 1]")
        if not 0 < self.max_symbol_weight <= 1:
            raise ResearchError("long-flat maximum symbol weight must be in (0, 1]")


def _long_flat_weights(
    scores: np.ndarray[Any, np.dtype[np.float64]],
    realized_volatility: np.ndarray[Any, np.dtype[np.float64]],
    *,
    parameters: BufferedLongFlatParameters,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    active = scores > parameters.signal_threshold
    if not np.any(active):
        return np.zeros_like(scores)
    inverse_volatility = np.divide(
        1.0,
        realized_volatility,
        out=np.zeros_like(realized_volatility),
        where=realized_volatility > 1e-15,
    )
    raw = inverse_volatility * active
    raw_gross = float(np.sum(raw))
    if raw_gross <= 1e-15:
        return np.zeros_like(scores)
    normalized = raw / raw_gross
    annualized_asset_volatility = realized_volatility * math.sqrt(365)
    volatility_proxy = float(math.sqrt(np.sum(np.square(normalized * annualized_asset_volatility))))
    target_gross = parameters.gross_exposure
    if volatility_proxy > 0:
        target_gross = min(target_gross, parameters.annual_volatility_target / volatility_proxy)
    weights = normalized * target_gross
    maximum_weight = float(np.max(weights))
    if maximum_weight > parameters.max_symbol_weight:
        weights *= parameters.max_symbol_weight / maximum_weight
    if float(np.sum(weights)) > 1 + 1e-12:
        raise ResearchError("long-flat portfolio attempted economic leverage")
    return np.asarray(weights, dtype=np.float64)


def _target_weight_frame(
    *,
    symbols: tuple[str, ...],
    times: np.ndarray[Any, np.dtype[np.int64]],
    arrays: dict[str, np.ndarray[Any, np.dtype[np.float64]]],
    parameters: BufferedLongFlatParameters,
    context: FoldContext,
) -> pl.DataFrame:
    if int(times[0]) < context.test_start_ms or int(times[-1]) > context.test_end_ms:
        raise ResearchError("long-flat portfolio input contains decisions outside its fold context")
    targets = np.zeros_like(arrays["score"])
    previous_weights = np.zeros(len(symbols), dtype=np.float64)
    last_rebalance_ms: int | None = None
    interval_ms = parameters.rebalance_interval_hours * HOUR_MS
    for period, decision_time in enumerate(times):
        if last_rebalance_ms is not None and int(decision_time) - last_rebalance_ms < interval_ms:
            targets[period] = previous_weights
            continue
        previous_weights = _long_flat_weights(
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
class BufferedLongFlatPolicy:
    parameters: BufferedLongFlatParameters
    policy_id: str = field(default="buffered_long_flat", init=False)
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
    "BufferedLongFlatParameters",
    "BufferedLongFlatPolicy",
]
