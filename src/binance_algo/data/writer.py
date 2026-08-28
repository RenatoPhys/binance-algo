"""Atomic manifest-backed Parquet micro-batches for real-time market events."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import orjson
import polars as pl

from binance_algo.common.errors import BinanceAlgoError, DataContractError, RecorderError
from binance_algo.data.archive_client import sha256_file
from binance_algo.data.manifest import (
    DataFileRecord,
    DataFileStatus,
    deterministic_file_id,
    now_ms,
)
from binance_algo.data.state_store import StateStore
from binance_algo.data.storage import LocalFilesystemStorage
from binance_algo.exchange.binance_usdm.public_streams import (
    STREAM_EVENT_TYPES,
    STREAM_SCHEMA_VERSION,
    STREAM_SCHEMAS,
    CanonicalStreamEvent,
    StreamEventType,
    stream_frame,
)
from binance_algo.observability.metrics import RecorderHealthState, RecorderMetrics

PartitionKey = tuple[StreamEventType, str, int, str]


@dataclass(frozen=True, slots=True)
class StreamFileResult:
    file_id: str
    event_type: StreamEventType
    symbol: str
    path: str
    row_count: int
    start_time_ms: int
    end_time_ms: int
    checksum: str
    size_bytes: int


@dataclass(slots=True)
class StreamWriterStats:
    files: list[StreamFileResult] = field(default_factory=list)
    rows_by_stream: Counter[str] = field(default_factory=Counter)
    bytes_written: int = 0
    temporary_files_quarantined: int = 0
    orphan_files_recovered: int = 0
    inflight_files_recovered: int = 0
    invalid_files_quarantined: int = 0


class StreamMicroBatchWriter:
    def __init__(
        self,
        *,
        storage: LocalFilesystemStorage,
        state_store: StateStore,
        ingestion_run_id: str,
        compression: str,
        max_rows: int,
        max_seconds: float,
        metrics: RecorderMetrics,
        health: RecorderHealthState,
        stats: StreamWriterStats,
    ) -> None:
        self._storage = storage
        self._state_store = state_store
        self._ingestion_run_id = ingestion_run_id
        self._compression = compression
        self._max_rows = max_rows
        self._max_seconds = max_seconds
        self._metrics = metrics
        self._health = health
        self._stats = stats
        self._buffers: dict[PartitionKey, list[dict[str, object]]] = defaultdict(list)
        self._sequences: Counter[PartitionKey] = Counter()
        self._buffered_rows = 0

    async def run(
        self,
        queue: asyncio.Queue[CanonicalStreamEvent],
        producers_done: asyncio.Event,
    ) -> None:
        last_flush = time.monotonic()
        try:
            while not producers_done.is_set() or not queue.empty():
                remaining = max(0.01, self._max_seconds - (time.monotonic() - last_flush))
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=min(remaining, 0.5))
                except TimeoutError:
                    flush_due = time.monotonic() - last_flush >= self._max_seconds
                    if self._buffered_rows and (flush_due or producers_done.is_set()):
                        await asyncio.to_thread(self.flush_all)
                        last_flush = time.monotonic()
                    continue
                self._append(event)
                queue.task_done()
                self._metrics.queue_size.set(queue.qsize())
                if self._buffered_rows >= self._max_rows:
                    await asyncio.to_thread(self.flush_all)
                    last_flush = time.monotonic()
            if self._buffered_rows:
                await asyncio.to_thread(self.flush_all)
        except (BinanceAlgoError, OSError, pl.exceptions.PolarsError) as exc:
            self._health.writer_healthy = False
            self._health.last_error = f"{type(exc).__name__}: {exc}"
            raise RecorderError(f"stream writer failed: {exc}") from exc

    def _append(self, event: CanonicalStreamEvent) -> None:
        observed = datetime.fromtimestamp(event.event_time_ms / 1_000, tz=UTC)
        key: PartitionKey = (
            event.event_type,
            observed.date().isoformat(),
            observed.hour,
            event.symbol,
        )
        self._buffers[key].append(event.row)
        self._buffered_rows += 1

    def flush_all(self) -> None:
        pending = self._buffers
        self._buffers = defaultdict(list)
        self._buffered_rows = 0
        for key in sorted(pending):
            rows = pending[key]
            if rows:
                self._flush_partition(key, rows)

    def _flush_partition(self, key: PartitionKey, rows: list[dict[str, object]]) -> None:
        event_type, day, hour, symbol = key
        started_ns = time.monotonic_ns()
        self._sequences[key] += 1
        sequence = self._sequences[key]
        frame = stream_frame(event_type, rows).sort(
            ["event_time_ms", "received_time_ns", "event_id"]
        )
        start_time_ms = cast(int, frame["event_time_ms"].min())
        end_time_ms = cast(int, frame["event_time_ms"].max())
        target = self._storage.path(
            "raw",
            "binance",
            "usdm",
            event_type,
            f"date={day}",
            f"hour={hour:02d}",
            f"symbol={symbol}",
            f"events_{self._ingestion_run_id}_{sequence:08d}_{start_time_ms}_{end_time_ms}.parquet",
        )
        relative = target.relative_to(self._storage.root).as_posix()
        file_id = deterministic_file_id("binance_market_stream", relative)
        timestamp = now_ms()
        interval = (
            "1m" if event_type == "kline_1m" else "1s" if event_type == "mark_price" else "realtime"
        )
        record = self._state_store.register_data_file(
            DataFileRecord(
                file_id=file_id,
                logical_dataset=event_type,
                layer="raw",
                source="binance_market_stream",
                symbol=symbol,
                interval=interval,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                row_count=None,
                schema_version=STREAM_SCHEMA_VERSION,
                checksum=None,
                path=str(target),
                status=DataFileStatus.DOWNLOADING,
                created_at_ms=timestamp,
                updated_at_ms=timestamp,
                ingestion_run_id=self._ingestion_run_id,
            )
        )
        try:
            self._storage.write_parquet_atomic(target, frame, compression=self._compression)
            checksum = sha256_file(target)
            if record.status is DataFileStatus.DOWNLOADING:
                record = self._state_store.transition_data_file(
                    file_id,
                    DataFileStatus.DOWNLOADED,
                    checksum=checksum,
                    row_count=frame.height,
                )
            if record.status is DataFileStatus.DOWNLOADED:
                stream_key, checkpoint_json = _checkpoint(
                    frame,
                    event_type=event_type,
                    symbol=symbol,
                    run_id=self._ingestion_run_id,
                    file_id=file_id,
                )
                self._state_store.validate_stream_file_with_checkpoint(
                    file_id,
                    checksum=checksum,
                    row_count=frame.height,
                    stream_key=stream_key,
                    checkpoint_json=checkpoint_json,
                )
        except (BinanceAlgoError, OSError, pl.exceptions.PolarsError) as exc:
            current = self._state_store.get_data_file(file_id)
            if current is not None and current.status not in {
                DataFileStatus.FAILED,
                DataFileStatus.QUARANTINED,
                DataFileStatus.VALIDATED,
            }:
                self._state_store.transition_data_file(
                    file_id, DataFileStatus.FAILED, last_error=str(exc)
                )
            raise

        size_bytes = target.stat().st_size
        result = StreamFileResult(
            file_id=file_id,
            event_type=event_type,
            symbol=symbol,
            path=str(target),
            row_count=frame.height,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            checksum=checksum,
            size_bytes=size_bytes,
        )
        self._stats.files.append(result)
        self._stats.rows_by_stream[event_type] += frame.height
        self._stats.bytes_written += size_bytes
        self._metrics.parquet_flushes.labels(event_type=event_type).inc()
        self._metrics.parquet_flush_rows.labels(event_type=event_type).inc(frame.height)
        self._metrics.parquet_flush_latency.labels(event_type=event_type).observe(
            (time.monotonic_ns() - started_ns) / 1_000_000_000
        )


def recover_recorder_artifacts(
    *,
    storage: LocalFilesystemStorage,
    state_store: StateStore,
    stats: StreamWriterStats,
) -> None:
    stream_root = storage.path("raw", "binance", "usdm")
    if not stream_root.exists():
        return
    quarantine_root = storage.path(
        "quarantine",
        "recorder_recovery",
        datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S%fZ"),
    )
    for temporary in sorted(stream_root.rglob("*.tmp")):
        quarantine_root.mkdir(parents=True, exist_ok=True)
        destination = quarantine_root / f"temporary_{uuid.uuid4().hex}.tmp"
        try:
            os.replace(_os_path(temporary), _os_path(destination))
        except OSError as exc:
            raise RecorderError(f"cannot quarantine recorder temporary {temporary}: {exc}") from exc
        stats.temporary_files_quarantined += 1

    records = state_store.list_data_files()
    for record in records:
        if record.source != "binance_market_stream" or record.status not in {
            DataFileStatus.DOWNLOADING,
            DataFileStatus.DOWNLOADED,
        }:
            continue
        parquet = Path(record.path)
        if not parquet.exists():
            continue
        try:
            frame, event_type, symbol, run_id = _inspect_recorder_parquet(
                parquet,
                event_type=record.logical_dataset,
                symbol=record.symbol,
                run_id=record.ingestion_run_id,
            )
        except (DataContractError, RecorderError) as exc:
            state_store.transition_data_file(
                record.file_id, DataFileStatus.QUARANTINED, last_error=str(exc)
            )
            stats.invalid_files_quarantined += 1
            continue
        checksum = sha256_file(parquet)
        current = record
        if current.status is DataFileStatus.DOWNLOADING:
            current = state_store.transition_data_file(
                record.file_id,
                DataFileStatus.DOWNLOADED,
                checksum=checksum,
                row_count=frame.height,
            )
        if current.status is DataFileStatus.DOWNLOADED:
            stream_key, checkpoint_json = _checkpoint(
                frame,
                event_type=event_type,
                symbol=symbol,
                run_id=run_id,
                file_id=record.file_id,
            )
            state_store.validate_stream_file_with_checkpoint(
                record.file_id,
                checksum=checksum,
                row_count=frame.height,
                stream_key=stream_key,
                checkpoint_json=checkpoint_json,
            )
        stats.inflight_files_recovered += 1

    manifested_paths = {Path(record.path).resolve() for record in records}
    for parquet in sorted(stream_root.rglob("*.parquet")):
        if parquet.resolve() in manifested_paths:
            continue
        try:
            _recover_orphan(parquet, storage=storage, state_store=state_store)
        except (DataContractError, RecorderError):
            quarantine_root.mkdir(parents=True, exist_ok=True)
            destination = quarantine_root / f"invalid_{uuid.uuid4().hex}.parquet"
            os.replace(_os_path(parquet), _os_path(destination))
            stats.invalid_files_quarantined += 1
        else:
            stats.orphan_files_recovered += 1


def _os_path(path: Path) -> str:
    resolved = str(path.resolve())
    return f"\\\\?\\{resolved}" if os.name == "nt" else resolved


def _recover_orphan(
    parquet: Path,
    *,
    storage: LocalFilesystemStorage,
    state_store: StateStore,
) -> None:
    relative = parquet.resolve().relative_to(storage.root).as_posix()
    frame, typed_event_type, symbol, run_id = _inspect_recorder_parquet(parquet)
    start_time_ms = cast(int, frame["event_time_ms"].min())
    end_time_ms = cast(int, frame["event_time_ms"].max())
    file_id = deterministic_file_id("binance_market_stream", relative)
    timestamp = now_ms()
    interval = (
        "1m"
        if typed_event_type == "kline_1m"
        else "1s"
        if typed_event_type == "mark_price"
        else "realtime"
    )
    state_store.register_data_file(
        DataFileRecord(
            file_id=file_id,
            logical_dataset=typed_event_type,
            layer="raw",
            source="binance_market_stream",
            symbol=symbol,
            interval=interval,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            row_count=None,
            schema_version=STREAM_SCHEMA_VERSION,
            checksum=None,
            path=str(parquet.resolve()),
            status=DataFileStatus.DOWNLOADING,
            created_at_ms=timestamp,
            updated_at_ms=timestamp,
            ingestion_run_id=run_id,
        )
    )
    checksum = sha256_file(parquet)
    state_store.transition_data_file(
        file_id, DataFileStatus.DOWNLOADED, checksum=checksum, row_count=frame.height
    )
    stream_key, checkpoint_json = _checkpoint(
        frame,
        event_type=typed_event_type,
        symbol=symbol,
        run_id=run_id,
        file_id=file_id,
    )
    state_store.validate_stream_file_with_checkpoint(
        file_id,
        checksum=checksum,
        row_count=frame.height,
        stream_key=stream_key,
        checkpoint_json=checkpoint_json,
    )


def _inspect_recorder_parquet(
    parquet: Path,
    *,
    event_type: str | None = None,
    symbol: str | None = None,
    run_id: str | None = None,
) -> tuple[pl.DataFrame, StreamEventType, str, str]:
    candidate_event_type = event_type
    if candidate_event_type is None:
        parts = parquet.parts
        try:
            usdm_index = parts.index("usdm")
            candidate_event_type = parts[usdm_index + 1]
        except (ValueError, IndexError) as exc:
            raise RecorderError(f"cannot identify orphan recorder dataset: {parquet}") from exc
    if candidate_event_type not in STREAM_EVENT_TYPES:
        raise RecorderError(f"unsupported recorder dataset {candidate_event_type}: {parquet}")
    typed_event_type = candidate_event_type
    try:
        frame = pl.read_parquet(parquet)
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise RecorderError(f"cannot read recorder Parquet {parquet}: {exc}") from exc
    expected_schema = STREAM_SCHEMAS[typed_event_type]
    if (
        frame.is_empty()
        or list(frame.columns) != list(expected_schema)
        or dict(frame.schema) != expected_schema
    ):
        raise DataContractError(f"recorder Parquet failed schema validation: {parquet}")
    symbols = frame["symbol"].unique().to_list()
    run_ids = frame["ingestion_run_id"].unique().to_list()
    if len(symbols) != 1 or len(run_ids) != 1:
        raise DataContractError(f"recorder Parquet mixes symbols or runs: {parquet}")
    observed_symbol = str(symbols[0])
    observed_run_id = str(run_ids[0])
    if symbol is not None and symbol != observed_symbol:
        raise DataContractError(f"recorder Parquet symbol differs from manifest: {parquet}")
    if run_id is not None and run_id != observed_run_id:
        raise DataContractError(f"recorder Parquet run differs from manifest: {parquet}")
    return frame, typed_event_type, observed_symbol, observed_run_id


def _checkpoint(
    frame: pl.DataFrame,
    *,
    event_type: StreamEventType,
    symbol: str,
    run_id: str,
    file_id: str,
) -> tuple[str, str]:
    last_row = frame.sort("received_time_ns").row(-1, named=True)
    checkpoint = {
        "event_id": last_row["event_id"],
        "event_time_ms": last_row["event_time_ms"],
        "received_time_ns": last_row["received_time_ns"],
        "file_id": file_id,
        "ingestion_run_id": run_id,
    }
    return (
        f"{event_type}:{symbol}",
        orjson.dumps(checkpoint, option=orjson.OPT_SORT_KEYS).decode("utf-8"),
    )
