"""Small transactional SQLite state store in WAL mode."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any

from binance_algo.common.errors import InvalidStateTransition, StateStoreError
from binance_algo.data.manifest import (
    ALLOWED_FILE_TRANSITIONS,
    ALLOWED_JOB_TRANSITIONS,
    BackfillJobRecord,
    BackfillJobStatus,
    DataFileRecord,
    DataFileStatus,
    now_ms,
)

MIGRATION_1 = (
    """
    CREATE TABLE IF NOT EXISTS data_files (
        file_id TEXT PRIMARY KEY,
        logical_dataset TEXT NOT NULL,
        layer TEXT NOT NULL,
        source TEXT NOT NULL,
        symbol TEXT NOT NULL,
        interval TEXT NOT NULL,
        start_time_ms INTEGER NOT NULL,
        end_time_ms INTEGER NOT NULL,
        row_count INTEGER,
        schema_version INTEGER NOT NULL,
        checksum TEXT,
        path TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL CHECK (status IN (
            'DOWNLOADING', 'DOWNLOADED', 'VALIDATED', 'NORMALIZED',
            'COMPACTED', 'QUARANTINED', 'FAILED'
        )),
        created_at_ms INTEGER NOT NULL,
        updated_at_ms INTEGER NOT NULL,
        ingestion_run_id TEXT NOT NULL,
        parent_file_ids_json TEXT NOT NULL DEFAULT '[]',
        last_error TEXT
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_data_files_lookup
    ON data_files(logical_dataset, symbol, interval, start_time_ms, status)
    """,
    """
    CREATE TABLE IF NOT EXISTS backfill_jobs (
        job_id TEXT PRIMARY KEY,
        dataset TEXT NOT NULL,
        symbols_json TEXT NOT NULL,
        interval TEXT NOT NULL,
        start_date TEXT NOT NULL,
        end_date TEXT NOT NULL,
        status TEXT NOT NULL CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')),
        total_files INTEGER NOT NULL,
        completed_files INTEGER NOT NULL DEFAULT 0,
        failed_files INTEGER NOT NULL DEFAULT 0,
        created_at_ms INTEGER NOT NULL,
        updated_at_ms INTEGER NOT NULL,
        last_error TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stream_checkpoints (
        stream_key TEXT PRIMARY KEY,
        checkpoint_json TEXT NOT NULL,
        updated_at_ms INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS quality_results (
        result_id TEXT PRIMARY KEY,
        file_id TEXT NOT NULL,
        check_name TEXT NOT NULL,
        passed INTEGER NOT NULL CHECK (passed IN (0, 1)),
        details_json TEXT NOT NULL,
        created_at_ms INTEGER NOT NULL,
        FOREIGN KEY(file_id) REFERENCES data_files(file_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS schema_versions (
        logical_dataset TEXT NOT NULL,
        schema_version INTEGER NOT NULL,
        schema_json TEXT NOT NULL,
        created_at_ms INTEGER NOT NULL,
        PRIMARY KEY(logical_dataset, schema_version)
    )
    """,
)


class StateStore:
    def __init__(self, path: Path, *, busy_timeout_ms: int = 5_000) -> None:
        self.path = path.resolve()
        self._busy_timeout_ms = busy_timeout_ms

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version INTEGER PRIMARY KEY,
                        applied_at_ms INTEGER NOT NULL
                    )
                    """
                )
                applied = {
                    int(row[0])
                    for row in connection.execute("SELECT version FROM schema_migrations")
                }
                if 1 not in applied:
                    for statement in MIGRATION_1:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT INTO schema_migrations(version, applied_at_ms) VALUES (?, ?)",
                        (1, now_ms()),
                    )
        except sqlite3.Error as exc:
            raise StateStoreError(f"cannot initialize state store {self.path}: {exc}") from exc

    def journal_mode(self) -> str:
        try:
            with closing(self._connect()) as connection:
                row = connection.execute("PRAGMA journal_mode").fetchone()
        except sqlite3.Error as exc:
            raise StateStoreError(f"cannot read SQLite journal mode: {exc}") from exc
        return str(row[0]) if row is not None else "unknown"

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def register_data_file(self, record: DataFileRecord) -> DataFileRecord:
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO data_files(
                        file_id, logical_dataset, layer, source, symbol, interval,
                        start_time_ms, end_time_ms, row_count, schema_version, checksum,
                        path, status, created_at_ms, updated_at_ms, ingestion_run_id,
                        parent_file_ids_json, last_error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.file_id,
                        record.logical_dataset,
                        record.layer,
                        record.source,
                        record.symbol,
                        record.interval,
                        record.start_time_ms,
                        record.end_time_ms,
                        record.row_count,
                        record.schema_version,
                        record.checksum,
                        record.path,
                        record.status.value,
                        record.created_at_ms,
                        record.updated_at_ms,
                        record.ingestion_run_id,
                        record.parent_file_ids_json,
                        record.last_error,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM data_files WHERE file_id = ?", (record.file_id,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise StateStoreError(f"cannot register data file {record.file_id}: {exc}") from exc
        if row is None:
            raise StateStoreError(f"registered data file disappeared: {record.file_id}")
        existing = self._data_file_from_row(row)
        if (
            existing.path != record.path
            or existing.logical_dataset != record.logical_dataset
            or existing.symbol != record.symbol
            or existing.interval != record.interval
        ):
            raise StateStoreError(f"file id collision for {record.file_id}")
        return existing

    def get_data_file(self, file_id: str) -> DataFileRecord | None:
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT * FROM data_files WHERE file_id = ?", (file_id,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise StateStoreError(f"cannot read data file {file_id}: {exc}") from exc
        return self._data_file_from_row(row) if row is not None else None

    def list_data_files(
        self,
        *,
        logical_dataset: str | None = None,
        layer: str | None = None,
        statuses: set[DataFileStatus] | None = None,
        symbols: tuple[str, ...] | None = None,
        interval: str | None = None,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        ingestion_run_id: str | None = None,
    ) -> list[DataFileRecord]:
        clauses: list[str] = []
        parameters: list[object] = []
        for column, value in (
            ("logical_dataset", logical_dataset),
            ("layer", layer),
            ("interval", interval),
            ("ingestion_run_id", ingestion_run_id),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        if statuses:
            placeholders = ", ".join("?" for _ in statuses)
            clauses.append(f"status IN ({placeholders})")
            parameters.extend(sorted(status.value for status in statuses))
        if symbols:
            placeholders = ", ".join("?" for _ in symbols)
            clauses.append(f"symbol IN ({placeholders})")
            parameters.extend(symbols)
        if start_time_ms is not None:
            clauses.append("end_time_ms >= ?")
            parameters.append(start_time_ms)
        if end_time_ms is not None:
            clauses.append("start_time_ms <= ?")
            parameters.append(end_time_ms)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    "SELECT * FROM data_files"
                    f"{where} ORDER BY symbol, interval, start_time_ms, file_id",
                    parameters,
                ).fetchall()
        except sqlite3.Error as exc:
            raise StateStoreError(f"cannot list data files: {exc}") from exc
        return [self._data_file_from_row(row) for row in rows]

    def get_stream_checkpoint(self, stream_key: str) -> str | None:
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT checkpoint_json FROM stream_checkpoints WHERE stream_key = ?",
                    (stream_key,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise StateStoreError(f"cannot read stream checkpoint {stream_key}: {exc}") from exc
        return str(row["checkpoint_json"]) if row is not None else None

    def upsert_stream_checkpoint(self, stream_key: str, checkpoint_json: str) -> None:
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO stream_checkpoints(stream_key, checkpoint_json, updated_at_ms)
                    VALUES (?, ?, ?)
                    ON CONFLICT(stream_key) DO UPDATE SET
                        checkpoint_json = excluded.checkpoint_json,
                        updated_at_ms = excluded.updated_at_ms
                    """,
                    (stream_key, checkpoint_json, now_ms()),
                )
        except sqlite3.Error as exc:
            raise StateStoreError(f"cannot write stream checkpoint {stream_key}: {exc}") from exc

    def list_stream_checkpoints(self) -> dict[str, str]:
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    """
                    SELECT stream_key, checkpoint_json
                    FROM stream_checkpoints ORDER BY stream_key
                    """
                ).fetchall()
        except sqlite3.Error as exc:
            raise StateStoreError(f"cannot list stream checkpoints: {exc}") from exc
        return {str(row["stream_key"]): str(row["checkpoint_json"]) for row in rows}

    def validate_stream_file_with_checkpoint(
        self,
        file_id: str,
        *,
        checksum: str,
        row_count: int,
        stream_key: str,
        checkpoint_json: str,
    ) -> DataFileRecord:
        """Atomically make a downloaded stream file visible and advance its checkpoint."""

        try:
            with self.transaction() as connection:
                row = connection.execute(
                    "SELECT * FROM data_files WHERE file_id = ?", (file_id,)
                ).fetchone()
                if row is None:
                    raise StateStoreError(f"unknown data file: {file_id}")
                current = DataFileStatus(str(row["status"]))
                if current not in {DataFileStatus.DOWNLOADED, DataFileStatus.VALIDATED}:
                    raise InvalidStateTransition(
                        f"data file {file_id}: {current.value} -> VALIDATED with checkpoint"
                    )
                connection.execute(
                    """
                    UPDATE data_files
                    SET status = ?, checksum = ?, row_count = ?, updated_at_ms = ?,
                        last_error = NULL
                    WHERE file_id = ?
                    """,
                    (DataFileStatus.VALIDATED.value, checksum, row_count, now_ms(), file_id),
                )
                connection.execute(
                    """
                    INSERT INTO stream_checkpoints(stream_key, checkpoint_json, updated_at_ms)
                    VALUES (?, ?, ?)
                    ON CONFLICT(stream_key) DO UPDATE SET
                        checkpoint_json = excluded.checkpoint_json,
                        updated_at_ms = excluded.updated_at_ms
                    """,
                    (stream_key, checkpoint_json, now_ms()),
                )
                updated = connection.execute(
                    "SELECT * FROM data_files WHERE file_id = ?", (file_id,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise StateStoreError(
                f"cannot validate stream file and checkpoint {file_id}: {exc}"
            ) from exc
        if updated is None:
            raise StateStoreError(f"validated stream file disappeared: {file_id}")
        return self._data_file_from_row(updated)

    def register_schema_version(
        self, logical_dataset: str, schema_version: int, schema_json: str
    ) -> None:
        try:
            with self.transaction() as connection:
                row = connection.execute(
                    """
                    SELECT schema_json FROM schema_versions
                    WHERE logical_dataset = ? AND schema_version = ?
                    """,
                    (logical_dataset, schema_version),
                ).fetchone()
                if row is not None and str(row["schema_json"]) != schema_json:
                    raise StateStoreError(
                        f"schema content mismatch for {logical_dataset} v{schema_version}"
                    )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO schema_versions(
                        logical_dataset, schema_version, schema_json, created_at_ms
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (logical_dataset, schema_version, schema_json, now_ms()),
                )
        except StateStoreError:
            raise
        except sqlite3.Error as exc:
            raise StateStoreError(
                f"cannot register schema {logical_dataset} v{schema_version}: {exc}"
            ) from exc

    def register_quality_result(
        self, *, file_id: str, check_name: str, passed: bool, details_json: str
    ) -> str:
        result_id = hashlib.sha256(
            f"{file_id}\x1f{check_name}\x1f{details_json}".encode()
        ).hexdigest()
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO quality_results(
                        result_id, file_id, check_name, passed, details_json, created_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (result_id, file_id, check_name, int(passed), details_json, now_ms()),
                )
        except sqlite3.Error as exc:
            raise StateStoreError(f"cannot register quality result {result_id}: {exc}") from exc
        return result_id

    def transition_data_file(
        self,
        file_id: str,
        new_status: DataFileStatus,
        *,
        checksum: str | None = None,
        row_count: int | None = None,
        last_error: str | None = None,
    ) -> DataFileRecord:
        try:
            with self.transaction() as connection:
                row = connection.execute(
                    "SELECT * FROM data_files WHERE file_id = ?", (file_id,)
                ).fetchone()
                if row is None:
                    raise StateStoreError(f"unknown data file: {file_id}")
                current = DataFileStatus(str(row["status"]))
                if new_status != current and new_status not in ALLOWED_FILE_TRANSITIONS[current]:
                    raise InvalidStateTransition(
                        f"data file {file_id}: {current.value} -> {new_status.value}"
                    )
                connection.execute(
                    """
                    UPDATE data_files
                    SET status = ?, checksum = COALESCE(?, checksum),
                        row_count = COALESCE(?, row_count), updated_at_ms = ?, last_error = ?
                    WHERE file_id = ?
                    """,
                    (
                        new_status.value,
                        checksum,
                        row_count,
                        now_ms(),
                        last_error,
                        file_id,
                    ),
                )
                updated = connection.execute(
                    "SELECT * FROM data_files WHERE file_id = ?", (file_id,)
                ).fetchone()
        except sqlite3.Error as exc:
            raise StateStoreError(f"cannot transition data file {file_id}: {exc}") from exc
        if updated is None:
            raise StateStoreError(f"transitioned data file disappeared: {file_id}")
        return self._data_file_from_row(updated)

    def create_backfill_job(self, record: BackfillJobRecord) -> None:
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT INTO backfill_jobs(
                        job_id, dataset, symbols_json, interval, start_date, end_date,
                        status, total_files, completed_files, failed_files,
                        created_at_ms, updated_at_ms, last_error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.job_id,
                        record.dataset,
                        record.symbols_json,
                        record.interval,
                        record.start_date,
                        record.end_date,
                        record.status.value,
                        record.total_files,
                        record.completed_files,
                        record.failed_files,
                        record.created_at_ms,
                        record.updated_at_ms,
                        record.last_error,
                    ),
                )
        except sqlite3.Error as exc:
            raise StateStoreError(f"cannot create backfill job {record.job_id}: {exc}") from exc

    def transition_backfill_job(
        self,
        job_id: str,
        new_status: BackfillJobStatus,
        *,
        completed_files: int | None = None,
        failed_files: int | None = None,
        last_error: str | None = None,
    ) -> None:
        try:
            with self.transaction() as connection:
                row = connection.execute(
                    "SELECT status FROM backfill_jobs WHERE job_id = ?", (job_id,)
                ).fetchone()
                if row is None:
                    raise StateStoreError(f"unknown backfill job: {job_id}")
                current = BackfillJobStatus(str(row["status"]))
                if new_status != current and new_status not in ALLOWED_JOB_TRANSITIONS[current]:
                    raise InvalidStateTransition(
                        f"backfill job {job_id}: {current.value} -> {new_status.value}"
                    )
                connection.execute(
                    """
                    UPDATE backfill_jobs
                    SET status = ?, completed_files = COALESCE(?, completed_files),
                        failed_files = COALESCE(?, failed_files), updated_at_ms = ?,
                        last_error = ?
                    WHERE job_id = ?
                    """,
                    (
                        new_status.value,
                        completed_files,
                        failed_files,
                        now_ms(),
                        last_error,
                        job_id,
                    ),
                )
        except sqlite3.Error as exc:
            raise StateStoreError(f"cannot transition backfill job {job_id}: {exc}") from exc

    def data_file_counts(self) -> dict[str, int]:
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    "SELECT status, COUNT(*) AS count FROM data_files GROUP BY status"
                ).fetchall()
        except sqlite3.Error as exc:
            raise StateStoreError(f"cannot summarize data files: {exc}") from exc
        return {str(row["status"]): int(row["count"]) for row in rows}

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path, timeout=self._busy_timeout_ms / 1_000, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @staticmethod
    def _data_file_from_row(row: sqlite3.Row) -> DataFileRecord:
        values: dict[str, Any] = dict(row)
        return DataFileRecord(
            file_id=str(values["file_id"]),
            logical_dataset=str(values["logical_dataset"]),
            layer=str(values["layer"]),
            source=str(values["source"]),
            symbol=str(values["symbol"]),
            interval=str(values["interval"]),
            start_time_ms=int(values["start_time_ms"]),
            end_time_ms=int(values["end_time_ms"]),
            row_count=int(values["row_count"]) if values["row_count"] is not None else None,
            schema_version=int(values["schema_version"]),
            checksum=str(values["checksum"]) if values["checksum"] is not None else None,
            path=str(values["path"]),
            status=DataFileStatus(str(values["status"])),
            created_at_ms=int(values["created_at_ms"]),
            updated_at_ms=int(values["updated_at_ms"]),
            ingestion_run_id=str(values["ingestion_run_id"]),
            parent_file_ids_json=str(values["parent_file_ids_json"]),
            last_error=str(values["last_error"]) if values["last_error"] else None,
        )
