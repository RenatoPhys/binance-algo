from __future__ import annotations

import pytest

import binance_algo.clock as clock_module
from binance_algo.clock import ExchangeClock


class FixedServerTime:
    async def server_time(self) -> int:
        return 2_000


async def test_clock_uses_epoch_midpoint_and_monotonic_duration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    epoch_values = iter([1_000_000_000, 3_000_000_000])
    monotonic_values = iter([10_000_000_000, 10_100_000_000])
    monkeypatch.setattr(clock_module.time, "time_ns", lambda: next(epoch_values))
    monkeypatch.setattr(clock_module.time, "monotonic_ns", lambda: next(monotonic_values))

    sync = await ExchangeClock().synchronize(FixedServerTime())

    assert sync.local_midpoint_ms == 2_000
    assert sync.offset_ms == 0
    assert sync.round_trip_ms == 100.0
