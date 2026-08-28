from __future__ import annotations

from pathlib import Path

import orjson
import polars as pl

from binance_algo.data.manifest import (
    DataFileRecord,
    DataFileStatus,
    deterministic_file_id,
    now_ms,
)
from binance_algo.data.state_store import StateStore
from binance_algo.data.storage import LocalFilesystemStorage
from binance_algo.data.writer import StreamWriterStats, recover_recorder_artifacts
from binance_algo.exchange.binance_usdm.public_streams import (
    STREAM_SCHEMA_VERSION,
    parse_combined_message,
    stream_frame,
)


def _book_frame(*, run_id: str, received_time_ns: int) -> pl.DataFrame:
    payload = {
        "e": "bookTicker",
        "E": 1_800_000_000_000,
        "T": 1_799_999_999_999,
        "s": "BTCUSDT",
        "u": received_time_ns,
        "b": "100.0",
        "B": "2.0",
        "a": "100.1",
        "A": "3.0",
    }
    event = parse_combined_message(
        orjson.dumps({"stream": "btcusdt@bookTicker", "data": payload}),
        received_time_ns=received_time_ns,
        processed_time_ns=received_time_ns + 1,
        connection_id="connection-1",
        ingestion_run_id=run_id,
    )
    assert event is not None
    return stream_frame("book_ticker", [event.row])


def _target(storage: LocalFilesystemStorage, name: str) -> Path:
    return storage.path(
        "raw",
        "binance",
        "usdm",
        "book_ticker",
        "date=2027-01-15",
        "hour=08",
        "symbol=BTCUSDT",
        name,
    )


def test_recovers_inflight_and_orphan_files_and_quarantines_artifacts(
    tmp_path: Path,
) -> None:
    storage = LocalFilesystemStorage(tmp_path / "data")
    state_store = StateStore(tmp_path / "state.sqlite3")
    state_store.initialize()

    temporary = _target(storage, ".interrupted.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_bytes(b"partial parquet")

    inflight = _target(storage, "events_inflight.parquet")
    inflight_frame = _book_frame(run_id="run-inflight", received_time_ns=10)
    storage.write_parquet_atomic(inflight, inflight_frame, compression="zstd")
    inflight_relative = inflight.relative_to(storage.root).as_posix()
    inflight_id = deterministic_file_id("binance_market_stream", inflight_relative)
    timestamp = now_ms()
    state_store.register_data_file(
        DataFileRecord(
            file_id=inflight_id,
            logical_dataset="book_ticker",
            layer="raw",
            source="binance_market_stream",
            symbol="BTCUSDT",
            interval="realtime",
            start_time_ms=1_800_000_000_000,
            end_time_ms=1_800_000_000_000,
            row_count=None,
            schema_version=STREAM_SCHEMA_VERSION,
            checksum=None,
            path=str(inflight),
            status=DataFileStatus.DOWNLOADING,
            created_at_ms=timestamp,
            updated_at_ms=timestamp,
            ingestion_run_id="run-inflight",
        )
    )

    orphan = _target(storage, "events_orphan.parquet")
    storage.write_parquet_atomic(
        orphan,
        _book_frame(run_id="run-orphan", received_time_ns=20),
        compression="zstd",
    )
    invalid = _target(storage, "events_invalid.parquet")
    invalid.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"invalid": [1]}).write_parquet(invalid)

    stats = StreamWriterStats()
    recover_recorder_artifacts(storage=storage, state_store=state_store, stats=stats)

    assert stats.temporary_files_quarantined == 1
    assert stats.inflight_files_recovered == 1
    assert stats.orphan_files_recovered == 1
    assert stats.invalid_files_quarantined == 1
    assert not temporary.exists()
    assert not invalid.exists()
    assert len(list((storage.root / "quarantine" / "recorder_recovery").rglob("*.tmp"))) == 1
    assert len(list((storage.root / "quarantine" / "recorder_recovery").rglob("*.parquet"))) == 1
    validated = state_store.list_data_files(statuses={DataFileStatus.VALIDATED})
    assert {Path(record.path) for record in validated} == {inflight, orphan}
    recovered_inflight = state_store.get_data_file(inflight_id)
    assert recovered_inflight is not None
    assert recovered_inflight.status is DataFileStatus.VALIDATED
    checkpoint = state_store.get_stream_checkpoint("book_ticker:BTCUSDT")
    assert checkpoint is not None
    assert orjson.loads(checkpoint)["file_id"] in {record.file_id for record in validated}
