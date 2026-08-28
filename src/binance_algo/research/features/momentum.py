"""Causal return, beta, and residual-momentum feature implementations."""

from __future__ import annotations

from typing import Any

import numpy as np

from binance_algo.research.features.base import FeatureDefinition

MINUTES_PER_HOUR = 60


def compute_log_returns(
    log_close: np.ndarray[Any, np.dtype[np.float64]],
    decision_indices: np.ndarray[Any, np.dtype[np.int64]],
    *,
    horizons: tuple[int, ...],
) -> dict[int, np.ndarray[Any, np.dtype[np.float64]]]:
    return {
        window: log_close[decision_indices] - log_close[decision_indices - window]
        for window in horizons
    }


def compute_residual_momentum(
    hourly_returns: np.ndarray[Any, np.dtype[np.float64]],
    *,
    symbols: tuple[str, ...],
    beta_window_hours: int,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    btc_index = symbols.index("BTCUSDT")
    eth_index = symbols.index("ETHUSDT")
    market_returns = (hourly_returns[:, btc_index] + hourly_returns[:, eth_index]) / 2
    benchmark_returns = np.tile(market_returns[:, None], (1, len(symbols)))
    benchmark_returns[:, btc_index] = hourly_returns[:, eth_index]
    benchmark_returns[:, eth_index] = hourly_returns[:, btc_index]
    beta = np.full_like(hourly_returns, np.nan)
    residual_returns = np.full_like(hourly_returns, np.nan)
    for row_index in range(beta_window_hours - 1, len(hourly_returns)):
        start = row_index - beta_window_hours + 1
        for symbol_index in range(len(symbols)):
            x = benchmark_returns[start : row_index + 1, symbol_index]
            y = hourly_returns[start : row_index + 1, symbol_index]
            variance = float(np.var(x))
            if variance <= 1e-18:
                continue
            beta[row_index, symbol_index] = float(
                np.mean((x - np.mean(x)) * (y - np.mean(y))) / variance
            )
            residual_returns[row_index, symbol_index] = (
                hourly_returns[row_index, symbol_index]
                - beta[row_index, symbol_index] * benchmark_returns[row_index, symbol_index]
            )

    residual_momentum_4h = np.full_like(residual_returns, np.nan)
    residual_momentum_24h = np.full_like(residual_returns, np.nan)
    for row_index in range(len(hourly_returns)):
        if row_index >= 3:
            residual_momentum_4h[row_index] = np.sum(
                residual_returns[row_index - 3 : row_index + 1], axis=0
            )
        if row_index >= 23:
            residual_momentum_24h[row_index] = np.sum(
                residual_returns[row_index - 23 : row_index + 1], axis=0
            )
    return (
        benchmark_returns,
        beta,
        residual_returns,
        residual_momentum_4h,
        residual_momentum_24h,
    )


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
        required_columns=("symbol", "open_time_ms", "close"),
        implementation_path="binance_algo.research.features.momentum",
        parameters={},
    )


MOMENTUM_FEATURES = (
    *(
        _definition(
            f"log_return_{label}",
            lookback=label,
            description=f"Log close return over the trailing {label} window.",
        )
        for label in ("5m", "15m", "1h", "4h", "24h")
    ),
    _definition(
        "benchmark_return_1h",
        lookback="1h",
        description="Leave-one-anchor-out BTC/ETH benchmark return.",
    ),
    _definition(
        "rolling_beta",
        lookback="configured beta_window_hours",
        description="Causal OLS beta to the leave-one-anchor-out benchmark.",
    ),
    *(
        _definition(
            f"residual_momentum_{label}",
            lookback=label,
            description=f"Causal benchmark-residual momentum over {label}.",
        )
        for label in ("1h", "4h", "24h")
    ),
)


__all__ = ["MOMENTUM_FEATURES", "compute_log_returns", "compute_residual_momentum"]
