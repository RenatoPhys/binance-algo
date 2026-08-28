from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from binance_algo.common.errors import DataQualityError
from binance_algo.data.catalog import rebuild_kline_catalog
from binance_algo.data.manifest import DataFileRecord, DataFileStatus
from binance_algo.data.normalize import KlineNormalizer, NormalizeOutcome, normalize_kline_csv
from binance_algo.data.quality import audit_kline_files, audit_kline_frame
from binance_algo.data.state_store import StateStore
from binance_algo.data.storage import LocalFilesystemStorage

START_MS = 1_787_616_000_000


def _raw_record(path: Path, checksum: str) -> DataFileRecord:
    return DataFileRecord(
        file_id="raw-file-1",
        logical_dataset="klines",
        layer="raw_archives",
        source="binance_public_data",
        symbol="BTCUSDT",
        interval="1m",
        start_time_ms=START_MS,
        end_time_ms=START_MS + 119_999,
        row_count=3,
        schema_version=1,
        checksum=checksum,
        path=str(path),
        status=DataFileStatus.VALIDATED,
        created_at_ms=100,
        updated_at_ms=100,
        ingestion_run_id="download-1",
    )


def _csv() -> bytes:
    return (
        "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
        "taker_buy_volume,taker_buy_quote_volume,ignore\n"
        f"{START_MS + 60_000},11,13,10,12,2,{START_MS + 119_999},24,4,1,12,0\n"
        f"{START_MS},10,12,9,11,1,{START_MS + 59_999},11,3,0.5,5.5,0\n"
        f"{START_MS},10,12,9,11,1,{START_MS + 59_999},11,3,0.5,5.5,0\n"
    ).encode()


def _normalized_fixture(
    tmp_path: Path,
) -> tuple[StateStore, KlineNormalizer, DataFileRecord]:
    storage = LocalFilesystemStorage(tmp_path / "data")
    archive = storage.path(
        "raw_archives",
        "binance",
        "usdm",
        "klines",
        "BTCUSDT",
        "1m",
        "BTCUSDT-1m-2026-08-25.zip",
    )
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"fixture archive")
    csv_path = archive.parent / "extracted" / "BTCUSDT-1m-2026-08-25.csv"
    csv_path.parent.mkdir()
    csv_path.write_bytes(_csv())
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    record = _raw_record(archive, checksum)
    state_store = StateStore(tmp_path / "state.sqlite3")
    state_store.initialize()
    state_store.register_data_file(record)
    normalizer = KlineNormalizer(
        storage=storage,
        state_store=state_store,
        compression="zstd",
        max_uncompressed_bytes=1_000_000,
        chunk_bytes=65_536,
    )
    return state_store, normalizer, record


def test_normalization_is_typed_deduplicated_sorted_and_idempotent(tmp_path: Path) -> None:
    state_store, normalizer, record = _normalized_fixture(tmp_path)

    first = normalizer.normalize_many([record], ingestion_run_id="normalize-1")[0]
    assert first.outcome is NormalizeOutcome.NORMALIZED
    assert first.source_rows == 3
    assert first.output_rows == 2
    assert first.duplicate_rows_removed == 1
    assert first.source_out_of_order_count == 1
    assert first.parquet_path is not None
    frame = pl.read_parquet(first.parquet_path)
    assert frame["open_time_ms"].to_list() == [START_MS, START_MS + 60_000]
    assert frame["schema_version"].dtype == pl.Int32
    assert frame["open"].dtype == pl.Float64

    refreshed_raw = state_store.get_data_file(record.file_id)
    assert refreshed_raw is not None
    second = normalizer.normalize_many([refreshed_raw], ingestion_run_id="normalize-2")[0]
    assert second.outcome is NormalizeOutcome.SKIPPED
    assert second.checksum == first.checksum
    bronze = state_store.list_data_files(
        logical_dataset="klines",
        layer="bronze",
        statuses={DataFileStatus.NORMALIZED},
    )
    assert len(bronze) == 1


def test_normalization_accepts_legacy_headerless_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "legacy.csv"
    csv_path.write_bytes(_csv().split(b"\n", 1)[1])

    frame, source_rows, duplicate_rows, out_of_order_rows = normalize_kline_csv(
        csv_path,
        symbol="BTCUSDT",
        interval="1m",
        ingested_at_ns=123,
    )

    assert source_rows == 3
    assert frame.height == 2
    assert duplicate_rows == 1
    assert out_of_order_rows == 1
    assert frame["open_time_ms"].to_list() == [START_MS, START_MS + 60_000]


def test_quality_gate_and_duckdb_catalog(tmp_path: Path) -> None:
    state_store, normalizer, record = _normalized_fixture(tmp_path)
    result = normalizer.normalize_many([record], ingestion_run_id="normalize-1")[0]
    assert result.outcome is NormalizeOutcome.NORMALIZED
    bronze = state_store.list_data_files(layer="bronze")

    report = audit_kline_files(
        bronze,
        state_store=state_store,
        start_time_ms=START_MS,
        end_time_ms=START_MS + 119_999,
    )
    assert report.passed
    assert report.aggregate[0].gap_count == 0
    with state_store.transaction() as connection:
        assert connection.execute("SELECT COUNT(*) FROM schema_versions").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM quality_results").fetchone()[0] == 1
    catalog = rebuild_kline_catalog(database_path=tmp_path / "market_data.duckdb", records=bronze)
    assert catalog.row_count == 2
    assert catalog.min_open_time_ms == START_MS
    assert catalog.max_open_time_ms == START_MS + 60_000


def test_quality_gate_rejects_entirely_missing_requested_symbol(tmp_path: Path) -> None:
    state_store, normalizer, record = _normalized_fixture(tmp_path)
    normalizer.normalize_many([record], ingestion_run_id="normalize-1")
    bronze = state_store.list_data_files(layer="bronze")

    with pytest.raises(DataQualityError, match="ETHUSDT"):
        audit_kline_files(
            bronze,
            state_store=state_store,
            start_time_ms=START_MS,
            end_time_ms=START_MS + 119_999,
            expected_symbols=("BTCUSDT", "ETHUSDT"),
        )


def test_quality_gate_detects_gap_and_invalid_ohlc(tmp_path: Path) -> None:
    _, normalizer, record = _normalized_fixture(tmp_path)
    result = normalizer.normalize_many([record], ingestion_run_id="normalize-1")[0]
    assert result.parquet_path is not None
    frame = pl.read_parquet(result.parquet_path).head(1).with_columns(pl.lit(20.0).alias("close"))

    audit = audit_kline_frame(
        frame,
        scope="test",
        symbol="BTCUSDT",
        interval="1m",
        start_time_ms=START_MS,
        end_time_ms=START_MS + 119_999,
        checksum_status="PASSED",
    )

    assert not audit.passed
    assert audit.gap_count == 1
    assert audit.gap_event_count == 1
    assert audit.invalid_ohlc_count == 1


def test_list_data_files_filters_ranges(tmp_path: Path) -> None:
    state_store, _, record = _normalized_fixture(tmp_path)
    other = replace(
        record,
        file_id="raw-file-2",
        symbol="ETHUSDT",
        path=str(tmp_path / "other.zip"),
        start_time_ms=START_MS + 86_400_000,
        end_time_ms=START_MS + 86_519_999,
    )
    state_store.register_data_file(other)

    selected = state_store.list_data_files(
        logical_dataset="klines",
        symbols=("BTCUSDT",),
        start_time_ms=START_MS,
        end_time_ms=START_MS + 119_999,
    )

    assert [item.file_id for item in selected] == ["raw-file-1"]
