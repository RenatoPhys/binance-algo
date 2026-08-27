"""Exchange clock synchronization without mixing epoch and monotonic time."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol


class ServerTimeSource(Protocol):
    async def server_time(self) -> int: ...


@dataclass(frozen=True, slots=True)
class ClockSync:
    server_time_ms: int
    local_midpoint_ms: int
    offset_ms: int
    round_trip_ms: float
    synchronized_at_ns: int


class ExchangeClock:
    """Tracks the observed exchange/local epoch offset."""

    def __init__(self) -> None:
        self._sync: ClockSync | None = None

    @property
    def sync(self) -> ClockSync | None:
        return self._sync

    @property
    def offset_ms(self) -> int:
        return self._sync.offset_ms if self._sync else 0

    async def synchronize(self, source: ServerTimeSource) -> ClockSync:
        epoch_before_ns = time.time_ns()
        monotonic_before_ns = time.monotonic_ns()
        server_time_ms = await source.server_time()
        monotonic_after_ns = time.monotonic_ns()
        epoch_after_ns = time.time_ns()

        local_midpoint_ns = (epoch_before_ns + epoch_after_ns) // 2
        offset_ms = round(server_time_ms - (local_midpoint_ns / 1_000_000))
        sync = ClockSync(
            server_time_ms=server_time_ms,
            local_midpoint_ms=local_midpoint_ns // 1_000_000,
            offset_ms=offset_ms,
            round_trip_ms=(monotonic_after_ns - monotonic_before_ns) / 1_000_000,
            synchronized_at_ns=epoch_after_ns,
        )
        self._sync = sync
        return sync

    def exchange_time_ms(self) -> int:
        return (time.time_ns() // 1_000_000) + self.offset_ms
