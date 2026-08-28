"""Deterministic Binance kline CSV to canonical immutable Parquet normalization."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import orjson
import polars as pl

from binance_algo.common.errors import BinanceAlgoError, DataContractError, StorageError
from binance_algo.data.archive_client import (
    KLINE_HEADER,
    kline_csv_has_header,
    sha256_file,
    validate_and_extract_kline_archive,
)
from binance_algo.data.manifest import (
    DataFileRecord,
    DataFileStatus,
    deterministic_file_id,
    now_ms,
)
from binance_algo.data.state_store import StateStore
from binance_algo.data.storage import LocalFilesystemStorage

KLINE_SCHEMA_VERSION = 1
KLINE_KEY = ("symbol", "interval", "open_time_ms")
KLINE_COLUMNS = (
    "symbol",
    "interval",
    "open_time_ms",
    "close_time_ms",
    "open",
    "high",
    "low",
    "close",
    "base_volume",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "is_closed",
    "ingested_at_ns",
    "source",
    "schema_version",
)
KLINE_SCHEMA_CONTRACT = {
    "dataset": "klines",
    "key": list(KLINE_KEY),
    "schema_version": KLINE_SCHEMA_VERSION,
    "columns": {
        "symbol": "String",
        "interval": "String",
        "open_time_ms": "Int64",
        "close_time_ms": "Int64",
        "open": "Float64",
        "high": "Float64",
        "low": "Float64",
        "close": "Float64",
        "base_volume": "Float64",
        "quote_volume": "Float64",
        "trade_count": "Int64",
        "taker_buy_base_volume": "Float64",
        "taker_buy_quote_volume": "Float64",
        "is_closed": "Boolean",
        "ingested_at_ns": "Int64",
        "source": "String",
        "schema_version": "Int32",
    },
}


class NormalizeOutcome(StrEnum):
    NORMALIZED = "normalized"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    source_file_id: str
    output_file_id: str | None
    symbol: str
    interval: str
    start_time_ms: int
    outcome: NormalizeOutcome
    source_rows: int | None
    output_rows: int | None
    duplicate_rows_removed: int | None
    source_out_of_order_count: int | None
    parquet_path: str | None
    checksum: str | None
    duration_ms: float
    error: str | None = None


def schema_contract_json() -> str:
    return orjson.dumps(KLINE_SCHEMA_CONTRACT, option=orjson.OPT_SORT_KEYS).decode("utf-8")


def normalize_kline_csv(
    csv_path: Path,
    *,
    symbol: str,
    interval: str,
    ingested_at_ns: int,
) -> tuple[pl.DataFrame, int, int, int]:
    """Return canonical rows and source quality counters before deterministic deduplication."""

    try:
        has_header = kline_csv_has_header(csv_path)
        if has_header:
            raw = pl.read_csv(
                csv_path,
                has_header=True,
                schema_overrides={column: pl.String for column in KLINE_HEADER},
            )
        else:
            raw = pl.read_csv(
                csv_path,
                has_header=False,
                new_columns=list(KLINE_HEADER),
                schema_overrides={column: pl.String for column in KLINE_HEADER},
            )
    except (OSError, pl.exceptions.PolarsError) as exc:
        raise DataContractError(f"cannot parse kline CSV {csv_path}: {exc}") from exc
    if tuple(raw.columns) != KLINE_HEADER:
        raise DataContractError(
            f"unexpected kline CSV columns in {csv_path}: {tuple(raw.columns)!r}"
        )
    try:
        frame = raw.select(
            pl.lit(symbol, dtype=pl.String).alias("symbol"),
            pl.lit(interval, dtype=pl.String).alias("interval"),
            pl.col("open_time").cast(pl.Int64, strict=True).alias("open_time_ms"),
            pl.col("close_time").cast(pl.Int64, strict=True).alias("close_time_ms"),
            pl.col("open").cast(pl.Float64, strict=True),
            pl.col("high").cast(pl.Float64, strict=True),
            pl.col("low").cast(pl.Float64, strict=True),
            pl.col("close").cast(pl.Float64, strict=True),
            pl.col("volume").cast(pl.Float64, strict=True).alias("base_volume"),
            pl.col("quote_volume").cast(pl.Float64, strict=True),
            pl.col("count").cast(pl.Int64, strict=True).alias("trade_count"),
            pl.col("taker_buy_volume").cast(pl.Float64, strict=True).alias("taker_buy_base_volume"),
            pl.col("taker_buy_quote_volume").cast(pl.Float64, strict=True),
            pl.lit(True, dtype=pl.Boolean).alias("is_closed"),
            pl.lit(ingested_at_ns, dtype=pl.Int64).alias("ingested_at_ns"),
            pl.lit("binance_public_data", dtype=pl.String).alias("source"),
            pl.lit(KLINE_SCHEMA_VERSION, dtype=pl.Int32).alias("schema_version"),
        )
    except pl.exceptions.PolarsError as exc:
        raise DataContractError(f"kline type conversion failed for {csv_path}: {exc}") from exc

    source_rows = frame.height
    duplicate_rows = source_rows - frame.unique(subset=list(KLINE_KEY)).height
    if source_rows < 2:
        out_of_order_count = 0
    else:
        out_of_order_count = int(
            frame.select((pl.col("open_time_ms").diff() < 0).sum().fill_null(0)).item()
        )
    canonical = (
        frame.unique(subset=list(KLINE_KEY), keep="first", maintain_order=True)
        .sort(list(KLINE_KEY))
        .select(list(KLINE_COLUMNS))
    )
    return canonical, source_rows, duplicate_rows, out_of_order_count


class KlineNormalizer:
    def __init__(
        self,
        *,
        storage: LocalFilesystemStorage,
        state_store: StateStore,
        compression: str,
        max_uncompressed_bytes: int,
        chunk_bytes: int,
    ) -> None:
        self._storage = storage
        self._state_store = state_store
        self._compression = compression
        self._max_uncompressed_bytes = max_uncompressed_bytes
        self._chunk_bytes = chunk_bytes

    def normalize_many(
        self, records: list[DataFileRecord], *, ingestion_run_id: str | None = None
    ) -> list[NormalizationResult]:
        run_id = ingestion_run_id or uuid.uuid4().hex
        self._state_store.register_schema_version(
            "klines", KLINE_SCHEMA_VERSION, schema_contract_json()
        )
        return [self.normalize_one(record, ingestion_run_id=run_id) for record in records]

    def normalize_one(
        self, raw_record: DataFileRecord, *, ingestion_run_id: str
    ) -> NormalizationResult:
        started_ns = time.monotonic_ns()
        output_id: str | None = None
        try:
            if raw_record.logical_dataset != "klines" or raw_record.layer != "raw_archives":
                raise DataContractError(f"not a raw kline archive: {raw_record.file_id}")
            if raw_record.status not in {DataFileStatus.VALIDATED, DataFileStatus.NORMALIZED}:
                raise DataContractError(
                    f"raw archive {raw_record.file_id} is not validated: {raw_record.status.value}"
                )
            if raw_record.checksum is None:
                raise DataContractError(f"raw archive has no checksum: {raw_record.file_id}")

            raw_path = Path(raw_record.path)
            day = datetime.fromtimestamp(raw_record.start_time_ms / 1_000, tz=UTC).date()
            csv_filename = f"{raw_record.symbol}-{raw_record.interval}-{day.isoformat()}.csv"
            csv_path = raw_path.parent / "extracted" / csv_filename
            if not csv_path.exists():
                validate_and_extract_kline_archive(
                    raw_path,
                    csv_path,
                    expected_csv_filename=csv_filename,
                    max_uncompressed_bytes=self._max_uncompressed_bytes,
                    chunk_bytes=self._chunk_bytes,
                )
            frame, source_rows, duplicates, out_of_order = normalize_kline_csv(
                csv_path,
                symbol=raw_record.symbol,
                interval=raw_record.interval,
                ingested_at_ns=raw_record.created_at_ms * 1_000_000,
            )
            checksum_prefix = raw_record.checksum[:12]
            output_path = self._storage.path(
                "bronze",
                "binance",
                "usdm",
                "klines",
                f"date={day.isoformat()}",
                f"symbol={raw_record.symbol}",
                f"klines_{raw_record.interval}_{day.isoformat()}_{checksum_prefix}.parquet",
            )
            output_id = deterministic_file_id(
                "canonical_parquet", raw_record.file_id, str(KLINE_SCHEMA_VERSION)
            )
            timestamp = now_ms()
            output_record = self._state_store.register_data_file(
                DataFileRecord(
                    file_id=output_id,
                    logical_dataset="klines",
                    layer="bronze",
                    source="binance_public_data",
                    symbol=raw_record.symbol,
                    interval=raw_record.interval,
                    start_time_ms=raw_record.start_time_ms,
                    end_time_ms=raw_record.end_time_ms,
                    row_count=None,
                    schema_version=KLINE_SCHEMA_VERSION,
                    checksum=None,
                    path=str(output_path),
                    status=DataFileStatus.DOWNLOADING,
                    created_at_ms=timestamp,
                    updated_at_ms=timestamp,
                    ingestion_run_id=ingestion_run_id,
                    parent_file_ids_json=orjson.dumps([raw_record.file_id]).decode("utf-8"),
                )
            )
            if output_record.status is DataFileStatus.NORMALIZED:
                if not output_path.exists():
                    raise StorageError(f"manifested Parquet is missing: {output_path}")
                checksum = sha256_file(output_path, chunk_bytes=self._chunk_bytes)
                if checksum != output_record.checksum or frame.height != output_record.row_count:
                    raise StorageError(f"manifested Parquet changed: {output_path}")
                if raw_record.status is DataFileStatus.VALIDATED:
                    self._state_store.transition_data_file(
                        raw_record.file_id, DataFileStatus.NORMALIZED
                    )
                return self._result(
                    raw_record,
                    output_id=output_id,
                    outcome=NormalizeOutcome.SKIPPED,
                    source_rows=source_rows,
                    output_rows=frame.height,
                    duplicates=duplicates,
                    out_of_order=out_of_order,
                    output_path=output_path,
                    checksum=checksum,
                    started_ns=started_ns,
                )
            if output_record.status in {DataFileStatus.FAILED, DataFileStatus.QUARANTINED}:
                output_record = self._state_store.transition_data_file(
                    output_id, DataFileStatus.DOWNLOADING
                )

            self._storage.write_parquet_atomic(output_path, frame, compression=self._compression)
            checksum = sha256_file(output_path, chunk_bytes=self._chunk_bytes)
            if output_record.status is DataFileStatus.DOWNLOADING:
                output_record = self._state_store.transition_data_file(
                    output_id,
                    DataFileStatus.DOWNLOADED,
                    checksum=checksum,
                    row_count=frame.height,
                )
            if output_record.status is DataFileStatus.DOWNLOADED:
                output_record = self._state_store.transition_data_file(
                    output_id,
                    DataFileStatus.VALIDATED,
                    checksum=checksum,
                    row_count=frame.height,
                )
            if output_record.status is DataFileStatus.VALIDATED:
                self._state_store.transition_data_file(
                    output_id,
                    DataFileStatus.NORMALIZED,
                    checksum=checksum,
                    row_count=frame.height,
                )
            if raw_record.status is DataFileStatus.VALIDATED:
                self._state_store.transition_data_file(
                    raw_record.file_id, DataFileStatus.NORMALIZED
                )
            return self._result(
                raw_record,
                output_id=output_id,
                outcome=NormalizeOutcome.NORMALIZED,
                source_rows=source_rows,
                output_rows=frame.height,
                duplicates=duplicates,
                out_of_order=out_of_order,
                output_path=output_path,
                checksum=checksum,
                started_ns=started_ns,
            )
        except (BinanceAlgoError, OSError) as exc:
            if output_id is not None:
                existing = self._state_store.get_data_file(output_id)
                if existing is not None and existing.status not in {
                    DataFileStatus.FAILED,
                    DataFileStatus.QUARANTINED,
                    DataFileStatus.NORMALIZED,
                }:
                    self._state_store.transition_data_file(
                        output_id, DataFileStatus.FAILED, last_error=str(exc)
                    )
            return self._result(
                raw_record,
                output_id=output_id,
                outcome=NormalizeOutcome.FAILED,
                source_rows=None,
                output_rows=None,
                duplicates=None,
                out_of_order=None,
                output_path=None,
                checksum=None,
                started_ns=started_ns,
                error=str(exc),
            )

    @staticmethod
    def _result(
        raw_record: DataFileRecord,
        *,
        output_id: str | None,
        outcome: NormalizeOutcome,
        source_rows: int | None,
        output_rows: int | None,
        duplicates: int | None,
        out_of_order: int | None,
        output_path: Path | None,
        checksum: str | None,
        started_ns: int,
        error: str | None = None,
    ) -> NormalizationResult:
        return NormalizationResult(
            source_file_id=raw_record.file_id,
            output_file_id=output_id,
            symbol=raw_record.symbol,
            interval=raw_record.interval,
            start_time_ms=raw_record.start_time_ms,
            outcome=outcome,
            source_rows=source_rows,
            output_rows=output_rows,
            duplicate_rows_removed=duplicates,
            source_out_of_order_count=out_of_order,
            parquet_path=str(output_path) if output_path is not None else None,
            checksum=checksum,
            duration_ms=(time.monotonic_ns() - started_ns) / 1_000_000,
            error=error,
        )


def write_normalization_report(
    *, results: list[NormalizationResult], reports_root: Path
) -> tuple[Path, Path]:
    storage = LocalFilesystemStorage(reports_root)
    timestamp = now_ms()
    base = f"normalization_{timestamp}_{uuid.uuid4().hex[:8]}"
    json_path = storage.path(f"{base}.json")
    markdown_path = storage.path(f"{base}.md")
    summary = {
        outcome.value: sum(result.outcome is outcome for result in results)
        for outcome in NormalizeOutcome
    }
    summary["source_rows"] = sum(result.source_rows or 0 for result in results)
    summary["output_rows"] = sum(result.output_rows or 0 for result in results)
    summary["duplicate_rows_removed"] = sum(
        result.duplicate_rows_removed or 0 for result in results
    )
    storage.write_json_atomic(
        json_path,
        {
            "schema": KLINE_SCHEMA_CONTRACT,
            "summary": summary,
            "files": [asdict(r) for r in results],
        },
    )
    lines = [
        "# Normalization report",
        "",
        f"- Files: {len(results)}",
        f"- Normalized: {summary['normalized']}",
        f"- Skipped: {summary['skipped']}",
        f"- Failed: {summary['failed']}",
        f"- Source rows: {summary['source_rows']}",
        f"- Output rows: {summary['output_rows']}",
        f"- Duplicate rows removed: {summary['duplicate_rows_removed']}",
        "",
        "| Symbol | Start ms | Outcome | Source rows | Output rows | Duplicates | Error |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        error = (result.error or "").replace("|", "\\|")
        lines.append(
            f"| {result.symbol} | {result.start_time_ms} | {result.outcome.value} | "
            f"{result.source_rows or ''} | {result.output_rows or ''} | "
            f"{result.duplicate_rows_removed or 0} | {error} |"
        )
    storage.write_bytes_atomic(markdown_path, ("\n".join(lines) + "\n").encode("utf-8"))
    return json_path, markdown_path
