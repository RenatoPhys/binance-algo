"""Canonical operational manifest records and explicit state machines."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from enum import StrEnum


class DataFileStatus(StrEnum):
    DOWNLOADING = "DOWNLOADING"
    DOWNLOADED = "DOWNLOADED"
    VALIDATED = "VALIDATED"
    NORMALIZED = "NORMALIZED"
    COMPACTED = "COMPACTED"
    QUARANTINED = "QUARANTINED"
    FAILED = "FAILED"


class BackfillJobStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


ALLOWED_FILE_TRANSITIONS: dict[DataFileStatus, frozenset[DataFileStatus]] = {
    DataFileStatus.DOWNLOADING: frozenset(
        {DataFileStatus.DOWNLOADED, DataFileStatus.FAILED, DataFileStatus.QUARANTINED}
    ),
    DataFileStatus.DOWNLOADED: frozenset(
        {DataFileStatus.VALIDATED, DataFileStatus.FAILED, DataFileStatus.QUARANTINED}
    ),
    DataFileStatus.VALIDATED: frozenset(
        {
            DataFileStatus.NORMALIZED,
            DataFileStatus.FAILED,
            DataFileStatus.QUARANTINED,
        }
    ),
    DataFileStatus.NORMALIZED: frozenset(
        {DataFileStatus.COMPACTED, DataFileStatus.FAILED, DataFileStatus.QUARANTINED}
    ),
    DataFileStatus.COMPACTED: frozenset({DataFileStatus.QUARANTINED}),
    DataFileStatus.QUARANTINED: frozenset({DataFileStatus.DOWNLOADING}),
    DataFileStatus.FAILED: frozenset({DataFileStatus.DOWNLOADING}),
}

ALLOWED_JOB_TRANSITIONS: dict[BackfillJobStatus, frozenset[BackfillJobStatus]] = {
    BackfillJobStatus.PENDING: frozenset({BackfillJobStatus.RUNNING, BackfillJobStatus.FAILED}),
    BackfillJobStatus.RUNNING: frozenset({BackfillJobStatus.COMPLETED, BackfillJobStatus.FAILED}),
    BackfillJobStatus.COMPLETED: frozenset(),
    BackfillJobStatus.FAILED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class DataFileRecord:
    file_id: str
    logical_dataset: str
    layer: str
    source: str
    symbol: str
    interval: str
    start_time_ms: int
    end_time_ms: int
    row_count: int | None
    schema_version: int
    checksum: str | None
    path: str
    status: DataFileStatus
    created_at_ms: int
    updated_at_ms: int
    ingestion_run_id: str
    parent_file_ids_json: str = "[]"
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class BackfillJobRecord:
    job_id: str
    dataset: str
    symbols_json: str
    interval: str
    start_date: str
    end_date: str
    status: BackfillJobStatus
    total_files: int
    completed_files: int
    failed_files: int
    created_at_ms: int
    updated_at_ms: int
    last_error: str | None = None


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def deterministic_file_id(*identity: str) -> str:
    canonical = "\x1f".join(identity).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
