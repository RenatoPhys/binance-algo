"""Causal bar-level market microstructure feature implementations."""

from __future__ import annotations

from typing import Any

import numpy as np

from binance_algo.research.features.base import FeatureDefinition


def compute_taker_imbalance(
    *,
    taker_quote_volume: np.ndarray[Any, np.dtype[np.float64]],
    hourly_quote_volume: np.ndarray[Any, np.dtype[np.float64]],
    decision_indices: np.ndarray[Any, np.dtype[np.int64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    output = np.empty_like(hourly_quote_volume)
    symbol_count = hourly_quote_volume.shape[1]
    for row_index, minute_index in enumerate(decision_indices):
        hour_slice = slice(minute_index - 60 + 1, minute_index + 1)
        total_taker = np.sum(taker_quote_volume[hour_slice], axis=0)
        output[row_index] = (
            np.divide(
                2 * total_taker,
                hourly_quote_volume[row_index],
                out=np.full(symbol_count, np.nan),
                where=hourly_quote_volume[row_index] > 0,
            )
            - 1
        )
    return output


MICROSTRUCTURE_FEATURES = (
    FeatureDefinition(
        feature_id="taker_buy_imbalance_1h:v1",
        name="taker_buy_imbalance_1h",
        version="v1",
        description="Two times taker-buy quote volume divided by quote volume, minus one.",
        dtype="Float64",
        lookback="1h",
        timestamp_semantics="closed one-minute bars at or before decision_time_ms",
        required_datasets=("klines",),
        required_columns=("open_time_ms", "quote_volume", "taker_buy_quote_volume"),
        implementation_path="binance_algo.research.features.microstructure",
        parameters={},
    ),
)


__all__ = ["MICROSTRUCTURE_FEATURES", "compute_taker_imbalance"]
