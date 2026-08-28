"""Causal quote-volume feature implementations."""

from __future__ import annotations

from typing import Any

import numpy as np

from binance_algo.research.features.base import FeatureDefinition


def _zscore(values: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values))
    if standard_deviation <= 1e-15:
        return np.zeros_like(values)
    return (values - mean) / standard_deviation


def compute_volume_features(
    quote_volume: np.ndarray[Any, np.dtype[np.float64]],
    decision_indices: np.ndarray[Any, np.dtype[np.int64]],
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    hourly = np.empty((len(decision_indices), quote_volume.shape[1]), dtype=np.float64)
    for row_index, minute_index in enumerate(decision_indices):
        hour_slice = slice(minute_index - 60 + 1, minute_index + 1)
        hourly[row_index] = np.sum(quote_volume[hour_slice], axis=0)
    zscore = np.full_like(hourly, np.nan)
    for row_index in range(23, len(decision_indices)):
        for symbol_index in range(quote_volume.shape[1]):
            zscore[row_index, symbol_index] = _zscore(
                hourly[row_index - 23 : row_index + 1, symbol_index]
            )[-1]
    return hourly, zscore


VOLUME_FEATURES = (
    FeatureDefinition(
        feature_id="quote_volume_1h:v1",
        name="quote_volume_1h",
        version="v1",
        description="Sum of quote volume over the trailing hour.",
        dtype="Float64",
        lookback="1h",
        timestamp_semantics="closed one-minute bars at or before decision_time_ms",
        required_datasets=("klines",),
        required_columns=("open_time_ms", "quote_volume"),
        implementation_path="binance_algo.research.features.volume",
        parameters={},
    ),
    FeatureDefinition(
        feature_id="quote_volume_zscore_24h:v1",
        name="quote_volume_zscore_24h",
        version="v1",
        description="Z-score of hourly quote volume over 24 hourly observations.",
        dtype="Float64",
        lookback="24 hourly observations",
        timestamp_semantics="closed one-minute bars at or before decision_time_ms",
        required_datasets=("klines",),
        required_columns=("open_time_ms", "quote_volume"),
        implementation_path="binance_algo.research.features.volume",
        parameters={},
    ),
)


__all__ = ["VOLUME_FEATURES", "compute_volume_features"]
