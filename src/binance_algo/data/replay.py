"""Network-free deterministic temporal replay over manifested stream Parquet files."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import duckdb

from binance_algo.common.errors import ReplayError
from binance_algo.data.manifest import DataFileRecord, DataFileStatus
from binance_algo.data.state_store import StateStore
from binance_algo.exchange.binance_usdm.public_streams import (
    STREAM_EVENT_TYPES,
    StreamEventType,
)


@dataclass(frozen=True, slots=True)
class ReplayEvent:
    event_type: StreamEventType
    event_id: str
    symbol: str
    event_time_ms: int
    received_time_ns: int
    connection_id: str
    ingestion_run_id: str
    payload_json: str
    data: dict[str, object]


@dataclass(frozen=True, slots=True)
class ReplayResult:
    event_count: int
    digest: str
    first_event_time_ms: int | None
    last_event_time_ms: int | None
    speed: float


class ReplayClock(Protocol):
    async def sleep(self, seconds: float) -> None: ...


class RealReplayClock:
    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


@dataclass(slots=True)
class VirtualReplayClock:
    elapsed_seconds: float = 0.0
    sleep_calls: int = 0

    async def sleep(self, seconds: float) -> None:
        if seconds < 0:
            raise ReplayError("virtual replay clock cannot move backward")
        self.elapsed_seconds += seconds
        self.sleep_calls += 1


ReplayHandler = Callable[[ReplayEvent], Awaitable[None]]


def select_replay_files(
    state_store: StateStore,
    *,
    datasets: tuple[StreamEventType, ...],
    symbols: tuple[str, ...] | None,
    start_time_ms: int,
    end_time_ms: int,
) -> list[DataFileRecord]:
    records = state_store.list_data_files(
        layer="raw",
        statuses={DataFileStatus.VALIDATED},
        symbols=symbols,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
    )
    selected = [
        record
        for record in records
        if record.source == "binance_market_stream" and record.logical_dataset in datasets
    ]
    if not selected:
        raise ReplayError("no manifested recorder files match the replay request")
    missing = sorted(set(datasets).difference(record.logical_dataset for record in selected))
    if missing:
        raise ReplayError(f"no recorder files match requested datasets: {missing}")
    return selected


class ReplayEngine:
    def __init__(
        self,
        *,
        records: list[DataFileRecord],
        start_time_ms: int,
        end_time_ms: int,
    ) -> None:
        if end_time_ms < start_time_ms:
            raise ReplayError("replay end must be on or after start")
        self._records = records
        self._start_time_ms = start_time_ms
        self._end_time_ms = end_time_ms

    def events(self) -> Iterator[ReplayEvent]:
        query = self._query()
        try:
            with duckdb.connect() as connection:
                reader = connection.execute(query).to_arrow_reader(batch_size=10_000)
                for batch in reader:
                    columns = batch.to_pydict()
                    for index in range(batch.num_rows):
                        row = {name: values[index] for name, values in columns.items()}
                        yield _replay_event(row)
        except duckdb.Error as exc:
            raise ReplayError(f"cannot read replay dataset: {exc}") from exc

    async def stream(
        self,
        *,
        speed: float,
        clock: ReplayClock,
    ) -> AsyncIterator[ReplayEvent]:
        if speed <= 0:
            raise ReplayError("replay speed must be positive")
        previous_time_ms: int | None = None
        for event in self.events():
            if previous_time_ms is not None:
                delay = max(0.0, (event.event_time_ms - previous_time_ms) / 1_000 / speed)
                if delay:
                    await clock.sleep(delay)
            previous_time_ms = event.event_time_ms
            yield event

    async def run(
        self,
        *,
        speed: float,
        clock: ReplayClock,
        handler: ReplayHandler | None = None,
    ) -> ReplayResult:
        digest = hashlib.sha256()
        count = 0
        first: int | None = None
        last: int | None = None
        async for event in self.stream(speed=speed, clock=clock):
            if handler is not None:
                await handler(event)
            _update_digest(digest, event)
            count += 1
            first = event.event_time_ms if first is None else first
            last = event.event_time_ms
        return ReplayResult(
            event_count=count,
            digest=digest.hexdigest(),
            first_event_time_ms=first,
            last_event_time_ms=last,
            speed=speed,
        )

    def digest(self) -> ReplayResult:
        digest = hashlib.sha256()
        count = 0
        first: int | None = None
        last: int | None = None
        for event in self.events():
            _update_digest(digest, event)
            count += 1
            first = event.event_time_ms if first is None else first
            last = event.event_time_ms
        return ReplayResult(
            event_count=count,
            digest=digest.hexdigest(),
            first_event_time_ms=first,
            last_event_time_ms=last,
            speed=0.0,
        )

    def verify_determinism(self) -> ReplayResult:
        first = self.digest()
        second = self.digest()
        if first.event_count != second.event_count or first.digest != second.digest:
            raise ReplayError("replay is not deterministic across two executions")
        return first

    def _query(self) -> str:
        grouped: dict[str, list[DataFileRecord]] = {}
        for record in self._records:
            if record.logical_dataset not in STREAM_EVENT_TYPES:
                raise ReplayError(f"unsupported replay dataset: {record.logical_dataset}")
            grouped.setdefault(record.logical_dataset, []).append(record)
        selects: list[str] = []
        for event_type, records in sorted(grouped.items()):
            paths = ", ".join(
                _sql_string(str(Path(record.path).resolve()).replace("\\", "/"))
                for record in records
            )
            selects.append(
                f"SELECT {_sql_string(event_type)} AS replay_event_type, * "
                f"FROM read_parquet([{paths}], union_by_name = true) "
                f"WHERE event_time_ms BETWEEN {self._start_time_ms} AND {self._end_time_ms}"
            )
        if not selects:
            raise ReplayError("replay has no input files")
        union = " UNION ALL BY NAME ".join(selects)
        return (
            f"SELECT * FROM ({union}) "
            "ORDER BY event_time_ms, received_time_ns, event_id, replay_event_type"
        )


def parse_replay_datasets(value: str) -> tuple[StreamEventType, ...]:
    if value.strip().lower() == "all":
        return STREAM_EVENT_TYPES
    raw = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    invalid = sorted(set(raw).difference(STREAM_EVENT_TYPES))
    if invalid:
        raise ReplayError(f"unsupported replay datasets: {invalid}")
    if not raw:
        raise ReplayError("at least one replay dataset is required")
    if len(raw) != len(set(raw)):
        raise ReplayError("replay datasets must not contain duplicates")
    return cast(tuple[StreamEventType, ...], raw)


def _replay_event(row: dict[str, Any]) -> ReplayEvent:
    event_type = str(row.pop("replay_event_type"))
    if event_type not in STREAM_EVENT_TYPES:
        raise ReplayError(f"unexpected replay event type: {event_type}")
    required = {
        "event_id",
        "symbol",
        "event_time_ms",
        "received_time_ns",
        "connection_id",
        "ingestion_run_id",
        "payload_json",
    }
    if any(row.get(column) is None for column in required):
        raise ReplayError(f"replay row is missing common fields for {event_type}")
    data = {key: value for key, value in row.items() if value is not None}
    return ReplayEvent(
        event_type=event_type,
        event_id=str(row["event_id"]),
        symbol=str(row["symbol"]),
        event_time_ms=int(row["event_time_ms"]),
        received_time_ns=int(row["received_time_ns"]),
        connection_id=str(row["connection_id"]),
        ingestion_run_id=str(row["ingestion_run_id"]),
        payload_json=str(row["payload_json"]),
        data=data,
    )


def _update_digest(digest: Any, event: ReplayEvent) -> None:
    digest.update(
        (
            f"{event.event_type}\x1f{event.event_time_ms}\x1f{event.received_time_ns}"
            f"\x1f{event.event_id}\n"
        ).encode()
    )


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
