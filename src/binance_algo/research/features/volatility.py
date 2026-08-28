"""Causal realized-volatility, range, and regime feature implementations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from binance_algo.research.features.base import FeatureDefinition


def compute_volatility_features(
    *,
    minute_log_returns: np.ndarray[Any, np.dtype[np.float64]],
    highs: np.ndarray[Any, np.dtype[np.float64]],
    lows: np.ndarray[Any, np.dtype[np.float64]],
    decision_indices: np.ndarray[Any, np.dtype[np.int64]],
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    realized = np.empty((len(decision_indices), highs.shape[1]), dtype=np.float64)
    intraday_range = np.empty_like(realized)
    for row_index, minute_index in enumerate(decision_indices):
        day_slice = slice(minute_index - 1_440 + 1, minute_index + 1)
        range_slice = slice(minute_index - 240 + 1, minute_index + 1)
        realized[row_index] = np.sqrt(np.nansum(np.square(minute_log_returns[day_slice]), axis=0))
        intraday_range[row_index] = (
            np.max(highs[range_slice], axis=0) / np.min(lows[range_slice], axis=0) - 1
        )
    return realized, intraday_range


def compute_market_volatility_regime(
    market_returns: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    output = np.full(len(market_returns), np.nan, dtype=np.float64)
    for row_index in range(23, len(market_returns)):
        output[row_index] = float(
            np.std(market_returns[row_index - 23 : row_index + 1]) * math.sqrt(24 * 365)
        )
    return output


def _definition(name: str, *, lookback: str, description: str) -> FeatureDefinition:
    return FeatureDefinition(
        feature_id=f"{name}:v1",
        name=name,
        version="v1",
        description=description,
        dtype="Float64",
        lookback=lookback,
        timestamp_semantics="closed one-minute bars at or before decision_time_ms",
        required_datasets=("klines",),
        required_columns=("open_time_ms", "high", "low", "close"),
        implementation_path="binance_algo.research.features.volatility",
        parameters={},
    )


VOLATILITY_FEATURES = (
    _definition(
        "realized_volatility_24h",
        lookback="24h",
        description="Square root of summed squared one-minute log returns.",
    ),
    _definition(
        "intraday_range_4h",
        lookback="4h",
        description="Trailing high divided by trailing low, minus one.",
    ),
    _definition(
        "market_volatility_regime",
        lookback="24 hourly observations",
        description="Annualized trailing volatility of the BTC/ETH benchmark.",
    ),
)


__all__ = [
    "VOLATILITY_FEATURES",
    "compute_market_volatility_regime",
    "compute_volatility_features",
]
