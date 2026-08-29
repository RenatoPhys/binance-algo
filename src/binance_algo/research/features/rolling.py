"""Shared causal rolling primitives for two-dimensional research panels."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from binance_algo.common.errors import ResearchError

FloatMatrix = np.ndarray[Any, np.dtype[np.float64]]


def _matrix(values: FloatMatrix, *, role: str) -> FloatMatrix:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ResearchError(f"{role} must be a two-dimensional matrix")
    return array


def trailing_sum(values: FloatMatrix, window: int, *, shift: int = 0) -> FloatMatrix:
    """Return a full-window trailing sum, optionally ending ``shift`` rows earlier."""

    array = _matrix(values, role="trailing sum input")
    if window < 1 or shift < 0:
        raise ResearchError("rolling window must be positive and shift cannot be negative")
    finite = np.isfinite(array)
    sums = np.vstack(
        (
            np.zeros((1, array.shape[1]), dtype=np.float64),
            np.cumsum(np.where(finite, array, 0.0), axis=0),
        )
    )
    counts = np.vstack(
        (
            np.zeros((1, array.shape[1]), dtype=np.int64),
            np.cumsum(finite, axis=0),
        )
    )
    output = np.full_like(array, np.nan)
    for row in range(window - 1 + shift, len(array)):
        end = row - shift + 1
        start = end - window
        valid = counts[end] - counts[start] == window
        output[row, valid] = (sums[end] - sums[start])[valid]
    return output


def rolling_mean_std(
    values: FloatMatrix,
    window: int,
    *,
    shift: int = 0,
) -> tuple[FloatMatrix, FloatMatrix]:
    """Return population mean/std using only the declared trailing window."""

    array = _matrix(values, role="rolling statistics input")
    sums = trailing_sum(array, window, shift=shift)
    squared_sums = trailing_sum(np.square(array), window, shift=shift)
    mean = sums / window
    variance = np.maximum(squared_sums / window - np.square(mean), 0.0)
    return mean, np.sqrt(variance)


def rolling_zscore(values: FloatMatrix, window: int, *, shift: int = 0) -> FloatMatrix:
    """Standardize the current observation with causal trailing population moments."""

    array = _matrix(values, role="rolling z-score input")
    mean, standard_deviation = rolling_mean_std(array, window, shift=shift)
    output = np.full_like(array, np.nan)
    valid = np.isfinite(mean) & (standard_deviation > 1e-15)
    output[valid] = (array[valid] - mean[valid]) / standard_deviation[valid]
    constant = np.isfinite(mean) & (standard_deviation <= 1e-15) & np.isfinite(array)
    output[constant] = 0.0
    return output


def trailing_realized_volatility(values: FloatMatrix, window: int) -> FloatMatrix:
    """Square root of the causal trailing sum of squared returns."""

    squared = trailing_sum(np.square(_matrix(values, role="return input")), window)
    return np.sqrt(squared)


def prior_rolling_extrema(
    values: FloatMatrix,
    window: int,
) -> tuple[FloatMatrix, FloatMatrix]:
    """Rolling high/low over the preceding rows, excluding the current row."""

    array = _matrix(values, role="rolling extrema input")
    if window < 1:
        raise ResearchError("rolling extrema window must be positive")
    maximum = np.full_like(array, np.nan)
    minimum = np.full_like(array, np.nan)
    for column in range(array.shape[1]):
        maximum_indices: deque[int] = deque()
        minimum_indices: deque[int] = deque()
        series = array[:, column]
        for row in range(len(series)):
            previous = row - 1
            if previous >= 0 and np.isfinite(series[previous]):
                while maximum_indices and series[maximum_indices[-1]] <= series[previous]:
                    maximum_indices.pop()
                maximum_indices.append(previous)
                while minimum_indices and series[minimum_indices[-1]] >= series[previous]:
                    minimum_indices.pop()
                minimum_indices.append(previous)
            first_valid = row - window
            while maximum_indices and maximum_indices[0] < first_valid:
                maximum_indices.popleft()
            while minimum_indices and minimum_indices[0] < first_valid:
                minimum_indices.popleft()
            if row >= window and maximum_indices and minimum_indices:
                maximum[row, column] = series[maximum_indices[0]]
                minimum[row, column] = series[minimum_indices[0]]
    return maximum, minimum


def kaufman_efficiency_ratio(log_price: FloatMatrix, horizon: int) -> FloatMatrix:
    """Causal Kaufman path efficiency over ``horizon`` return intervals."""

    values = _matrix(log_price, role="Kaufman log-price input")
    if horizon < 1:
        raise ResearchError("Kaufman horizon must be positive")
    output = np.full_like(values, np.nan)
    absolute_changes = np.abs(np.diff(values, axis=0))
    denominator = trailing_sum(absolute_changes, horizon)
    for row in range(horizon, len(values)):
        path = denominator[row - 1]
        displacement = np.abs(values[row] - values[row - horizon])
        valid = np.isfinite(path) & (path > 1e-15)
        output[row, valid] = displacement[valid] / path[valid]
        output[row, np.isfinite(path) & (path <= 1e-15)] = 0.0
    return output


def frozen_symbol_quantiles(values: FloatMatrix, quantile: float) -> FloatMatrix:
    """Fit one finite quantile per symbol; callers freeze the returned vector per fold."""

    array = _matrix(values, role="training quantile input")
    if not 0 < quantile < 1:
        raise ResearchError("training quantile must be strictly between zero and one")
    thresholds = np.full(array.shape[1], np.nan, dtype=np.float64)
    for column in range(array.shape[1]):
        finite = array[np.isfinite(array[:, column]), column]
        if len(finite):
            thresholds[column] = float(np.quantile(finite, quantile))
    thresholds.setflags(write=False)
    return thresholds


__all__ = [
    "frozen_symbol_quantiles",
    "kaufman_efficiency_ratio",
    "prior_rolling_extrema",
    "rolling_mean_std",
    "rolling_zscore",
    "trailing_realized_volatility",
    "trailing_sum",
]
