"""Streaming content checksums and path-independent dataset lineage fingerprints."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.data.manifest import DataFileRecord, DataFileStatus
from binance_algo.data.state_store import StateStore
from binance_algo.research.features.registry import FeatureSetSpec
from binance_algo.research.labels.base import LabelDefinition

FINGERPRINT_METHOD = "lineage_v2"
DATASET_BUILDER_VERSION = "phase3.5-pr3-v1"


@dataclass(frozen=True, slots=True)
class InputFileFingerprint:
    file_id: str
    logical_dataset: str
    symbol: str
    interval: str
    start_time_ms: int
    end_time_ms: int
    row_count: int
    schema_version: int
    checksum: str

    @classmethod
    def from_record(cls, record: DataFileRecord) -> InputFileFingerprint:
        if record.status is not DataFileStatus.NORMALIZED:
            raise ResearchError(f"research lineage input is not normalized: {record.file_id}")
        if record.checksum is None or record.row_count is None:
            raise ResearchError(
                f"research lineage input lacks checksum or row count: {record.file_id}"
            )
        if not Path(record.path).is_file():
            raise ResearchError(f"research lineage input is missing: {record.path}")
        return cls(
            file_id=record.file_id,
            logical_dataset=record.logical_dataset,
            symbol=record.symbol,
            interval=record.interval,
            start_time_ms=record.start_time_ms,
            end_time_ms=record.end_time_ms,
            row_count=record.row_count,
            schema_version=record.schema_version,
            checksum=record.checksum,
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "file_id": self.file_id,
            "logical_dataset": self.logical_dataset,
            "symbol": self.symbol,
            "interval": self.interval,
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "row_count": self.row_count,
            "schema_version": self.schema_version,
            "checksum": self.checksum,
        }


def collect_research_lineage(
    *,
    state_db_path: Path,
    symbols: tuple[str, ...],
    start_time_ms: int,
    end_time_ms: int,
    funding_required: bool,
) -> tuple[InputFileFingerprint, ...]:
    """Select the manifested canonical files that can affect the bounded queries."""

    state_store = StateStore(state_db_path)
    kline_records = state_store.list_data_files(
        logical_dataset="klines",
        layer="bronze",
        statuses={DataFileStatus.NORMALIZED},
        symbols=symbols,
        interval="1m",
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
    )
    if not kline_records:
        raise ResearchError("no normalized kline manifests match the research range")
    funding_records = state_store.list_data_files(
        logical_dataset="funding_rates",
        layer="bronze",
        statuses={DataFileStatus.NORMALIZED},
        symbols=symbols,
        end_time_ms=end_time_ms,
    )
    if funding_required and not funding_records:
        raise ResearchError("no normalized funding manifests match the research range")
    records = (*kline_records, *funding_records)
    return tuple(
        sorted(
            (InputFileFingerprint.from_record(record) for record in records),
            key=lambda item: (
                item.logical_dataset,
                item.symbol,
                item.interval,
                item.start_time_ms,
                item.file_id,
            ),
        )
    )


def logical_content_checksum(frame: pl.DataFrame) -> str:
    """Hash a frame incrementally without materializing it as a list of row dictionaries."""

    digest = hashlib.sha256()
    schema = [(column, str(frame.schema[column])) for column in frame.columns]
    digest.update(orjson.dumps(schema))
    digest.update(b"\n")
    for row in frame.iter_rows(named=False, buffer_size=1_024):
        digest.update(orjson.dumps(row))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_path(path: Path, *, chunk_bytes: int = 1_048_576) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def build_lineage_payload(
    *,
    input_files: Iterable[InputFileFingerprint],
    dataset_schema_version: int,
    universe_version: str,
    symbols: tuple[str, ...],
    start_time_ms: int,
    end_time_ms: int,
    feature_set: FeatureSetSpec,
    label: LabelDefinition,
    builder_parameters: Mapping[str, Any],
    builder_version: str = DATASET_BUILDER_VERSION,
) -> dict[str, object]:
    files = tuple(
        sorted(
            input_files,
            key=lambda item: (
                item.logical_dataset,
                item.symbol,
                item.interval,
                item.start_time_ms,
                item.file_id,
            ),
        )
    )
    if not files:
        raise ResearchError("lineage_v2 requires at least one manifested input file")
    return {
        "fingerprint_method": FINGERPRINT_METHOD,
        "input_files": [item.identity_payload() for item in files],
        "input_schema_versions": sorted(
            {f"{item.logical_dataset}:{item.schema_version}" for item in files}
        ),
        "dataset_schema_version": dataset_schema_version,
        "universe_version": universe_version,
        "symbols": sorted(symbols),
        "start_time_ms": start_time_ms,
        "end_time_ms": end_time_ms,
        "feature_set_id": feature_set.feature_set_id,
        "feature_set_checksum": feature_set.canonical_checksum,
        "label": label.to_manifest(),
        "builder_parameters": dict(builder_parameters),
        "builder_version": builder_version,
    }


def lineage_fingerprint(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)).hexdigest()


__all__ = [
    "DATASET_BUILDER_VERSION",
    "FINGERPRINT_METHOD",
    "InputFileFingerprint",
    "build_lineage_payload",
    "collect_research_lineage",
    "lineage_fingerprint",
    "logical_content_checksum",
    "sha256_path",
]
