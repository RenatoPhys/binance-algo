from __future__ import annotations

from types import MappingProxyType

import numpy as np
import polars as pl

from binance_algo.research.features.base import FeatureComputeContext
from binance_algo.research.features.clock_phase import compute_clock_phase_features
from binance_algo.research.features.rolling import (
    frozen_symbol_quantiles,
    kaufman_efficiency_ratio,
    prior_rolling_extrema,
    rolling_zscore,
    trailing_realized_volatility,
)


def _context(minutes: int, decision_index: int) -> FeatureComputeContext:
    open_times = np.arange(minutes, dtype=np.int64) * 60_000
    quote = np.full((minutes, 1), 10.0)
    taker = np.full((minutes, 1), 5.0)
    quarter = np.isin(np.arange(minutes) % 60, [0, 15, 30, 45])
    taker[quarter] = 10.0
    log_open = np.zeros((minutes, 1), dtype=np.float64)
    log_close = log_open.copy()
    log_close[quarter] = 0.01
    return FeatureComputeContext(
        symbols=("BTCUSDT",),
        decision_indices=np.asarray([decision_index], dtype=np.int64),
        decision_times=np.asarray([decision_index * 60_000 + 59_999], dtype=np.int64),
        open_times=open_times,
        log_open=log_open,
        log_close=log_close,
        minute_log_returns=np.zeros_like(log_close),
        highs=np.ones_like(log_close),
        lows=np.ones_like(log_close),
        quote_volume=quote,
        taker_quote_volume=taker,
        funding=pl.DataFrame(),
        prior_outputs=MappingProxyType({}),
    )


def test_quarter_hour_bundle_uses_exact_utc_opening_minutes() -> None:
    outputs = compute_clock_phase_features(_context(60, 59))

    assert outputs["quarter_open_signed_flow_1h"][0, 0] == 40.0
    assert outputs["quarter_open_quote_volume_1h"][0, 0] == 40.0
    assert outputs["quarter_open_flow_share_1h"][0, 0] == 1.0
    assert outputs["non_open_flow_share_1h"][0, 0] == 0.0
    assert outputs["quarter_flow_excess_1h"][0, 0] == 1.0
    assert outputs["quarter_open_return_1h"][0, 0] == 0.04


def test_clock_phase_rejects_an_incomplete_hour() -> None:
    outputs = compute_clock_phase_features(_context(59, 58))

    assert all(np.isnan(values[0, 0]) for values in outputs.values())


def test_rolling_operations_are_trailing_and_prior_extrema_exclude_current() -> None:
    values = np.arange(200, dtype=np.float64)[:, None]
    original = rolling_zscore(values, 168)
    changed = values.copy()
    changed[190] = -1_000_000.0
    mutated = rolling_zscore(changed, 168)

    assert original[180, 0] == mutated[180, 0]
    assert original[190, 0] != mutated[190, 0]
    high, low = prior_rolling_extrema(np.asarray([[1.0], [2.0], [100.0]]), 2)
    assert high[2, 0] == 2.0
    assert low[2, 0] == 1.0


def test_training_quantiles_realized_volatility_and_efficiency_are_deterministic() -> None:
    values = np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    thresholds = frozen_symbol_quantiles(values, 0.5)
    volatility = trailing_realized_volatility(np.full((4, 2), 0.5), 4)
    efficiency = kaufman_efficiency_ratio(np.log(values), 2)

    assert thresholds.tolist() == [2.5, 5.0]
    assert volatility[3].tolist() == [1.0, 1.0]
    assert np.allclose(efficiency[2:], 1.0)
    assert rolling_zscore(values, 3).tobytes() == rolling_zscore(values, 3).tobytes()
