from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from binance_algo.common.errors import InvalidStateTransition, StateStoreError
from binance_algo.data.manifest import (
    BackfillJobRecord,
    BackfillJobStatus,
    DataFileRecord,
    DataFileStatus,
)
from binance_algo.data.state_store import StateStore


def data_file_record(path: Path) -> DataFileRecord:
    return DataFileRecord(
        file_id="file-1",
        logical_dataset="klines",
        layer="raw_archives",
        source="binance_public_data",
        symbol="BTCUSDT",
        interval="1m",
        start_time_ms=1_700_000_000_000,
        end_time_ms=1_700_086_399_999,
        row_count=None,
        schema_version=1,
        checksum=None,
        path=str(path),
        status=DataFileStatus.DOWNLOADING,
        created_at_ms=100,
        updated_at_ms=100,
        ingestion_run_id="run-1",
    )


def test_initializes_wal_schema_and_persists_manifest(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "state" / "ingestion.sqlite3")
    store.initialize()

    assert store.journal_mode().lower() == "wal"
    registered = store.register_data_file(data_file_record(tmp_path / "archive.zip"))
    assert registered.status is DataFileStatus.DOWNLOADING

    downloaded = store.transition_data_file("file-1", DataFileStatus.DOWNLOADED, checksum="a" * 64)
    assert downloaded.checksum == "a" * 64
    validated = store.transition_data_file("file-1", DataFileStatus.VALIDATED, row_count=1_440)
    assert validated.row_count == 1_440
    assert StateStore(store.path).get_data_file("file-1") == validated
    assert store.data_file_counts() == {"VALIDATED": 1}


def test_rejects_invalid_file_transition(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "ingestion.sqlite3")
    store.initialize()
    store.register_data_file(data_file_record(tmp_path / "archive.zip"))
    store.transition_data_file("file-1", DataFileStatus.DOWNLOADED)
    store.transition_data_file("file-1", DataFileStatus.VALIDATED)

    with pytest.raises(InvalidStateTransition, match="VALIDATED -> COMPACTED"):
        store.transition_data_file("file-1", DataFileStatus.COMPACTED)


def test_transaction_rolls_back_on_failure(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "ingestion.sqlite3")
    store.initialize()

    with pytest.raises(RuntimeError, match="force rollback"), store.transaction() as connection:
        connection.execute(
            """
            INSERT INTO schema_versions(
                logical_dataset, schema_version, schema_json, created_at_ms
            ) VALUES ('test', 1, '{}', 1)
            """
        )
        raise RuntimeError("force rollback")

    with store.transaction() as connection:
        count = connection.execute("SELECT COUNT(*) FROM schema_versions").fetchone()[0]
    assert count == 0


def test_write_lock_times_out_explicitly_and_recovers(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "ingestion.sqlite3", busy_timeout_ms=25)
    store.initialize()
    store.register_data_file(data_file_record(tmp_path / "archive.zip"))

    blocker = sqlite3.connect(store.path, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(StateStoreError, match="locked"):
            store.transition_data_file("file-1", DataFileStatus.DOWNLOADED)
    finally:
        blocker.rollback()
        blocker.close()

    recovered = store.transition_data_file("file-1", DataFileStatus.DOWNLOADED)
    assert recovered.status is DataFileStatus.DOWNLOADED


def test_backfill_job_state_machine(tmp_path: Path) -> None:
    store = StateStore(tmp_path / "ingestion.sqlite3")
    store.initialize()
    job = BackfillJobRecord(
        job_id="job-1",
        dataset="klines",
        symbols_json='["BTCUSDT"]',
        interval="1m",
        start_date="2026-08-25",
        end_date="2026-08-25",
        status=BackfillJobStatus.PENDING,
        total_files=1,
        completed_files=0,
        failed_files=0,
        created_at_ms=1,
        updated_at_ms=1,
    )
    store.create_backfill_job(job)
    store.transition_backfill_job("job-1", BackfillJobStatus.RUNNING)
    store.transition_backfill_job(
        "job-1", BackfillJobStatus.COMPLETED, completed_files=1, failed_files=0
    )

    with pytest.raises(InvalidStateTransition, match="COMPLETED -> FAILED"):
        store.transition_backfill_job("job-1", BackfillJobStatus.FAILED)
