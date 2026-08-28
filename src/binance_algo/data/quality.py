"""Canonical kline invariants, continuity checks, and durable audit reports."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import orjson
import polars as pl

from binance_algo.common.errors import DataQualityError
from binance_algo.data.archive_client import sha256_file
from binance_algo.data.manifest import DataFileRecord, now_ms
from binance_algo.data.normalize import (
    KLINE_COLUMNS,
    KLINE_KEY,
    KLINE_SCHEMA_CONTRACT,
    KLINE_SCHEMA_VERSION,
)
from binance_algo.data.state_store import StateStore
from binance_algo.data.storage import LocalFilesystemStorage

INTERVAL_MS = {"1m": 60_000}


@dataclass(frozen=True, slots=True)
class KlineAudit:
    scope: str
    symbol: str
    interval: str
    start_time_ms: int
    end_time_ms: int
    row_count: int
    min_timestamp_ms: int | None
    max_timestamp_ms: int | None
    duplicate_key_count: int
    null_count_by_column: dict[str, int]
    out_of_order_count: int
    gap_count: int
    gap_event_count: int
    gap_duration_seconds: int
    negative_price_count: int
    negative_quantity_count: int
    invalid_ohlc_count: int
    invalid_close_time_count: int
    unaligned_timestamp_count: int
    out_of_range_count: int
    non_finite_numeric_count: int
    contract_value_mismatch_count: int
    open_bar_count: int
    checksum_status: str
    schema_status: str
    source_comparison_status: str
    passed: bool


@dataclass(frozen=True, slots=True)
class QualityReport:
    created_at_ms: int
    schema_version: int
    passed: bool
    files_audited: int
    aggregate: tuple[KlineAudit, ...]
    partitions: tuple[KlineAudit, ...]


def _gap_metrics(
    timestamps: list[int], *, start_time_ms: int, end_time_ms: int, interval_ms: int
) -> tuple[int, int]:
    expected_last = end_time_ms // interval_ms * interval_ms
    unique = sorted(set(timestamps))
    missing = 0
    events = 0
    previous = start_time_ms - interval_ms
    for current in unique:
        if current < start_time_ms or current > expected_last:
            continue
        difference = current - previous
        if difference > interval_ms:
            missing += difference // interval_ms - 1
            events += 1
        previous = current
    if expected_last - previous >= interval_ms:
        missing += (expected_last - previous) // interval_ms
        events += 1
    return missing, events


def _schema_status(frame: pl.DataFrame) -> str:
    expected = KLINE_SCHEMA_CONTRACT["columns"]
    actual = {name: str(dtype) for name, dtype in frame.schema.items()}
    return (
        "PASSED" if list(frame.columns) == list(KLINE_COLUMNS) and actual == expected else "FAILED"
    )


def audit_kline_frame(
    frame: pl.DataFrame,
    *,
    scope: str,
    symbol: str,
    interval: str,
    start_time_ms: int,
    end_time_ms: int,
    checksum_status: str,
) -> KlineAudit:
    if interval not in INTERVAL_MS:
        raise DataQualityError(f"unsupported audit interval: {interval}")
    schema_status = _schema_status(frame)
    missing_columns = set(KLINE_COLUMNS).difference(frame.columns)
    if missing_columns:
        raise DataQualityError(f"cannot audit missing canonical columns: {sorted(missing_columns)}")

    row_count = frame.height
    min_timestamp = cast(int, frame["open_time_ms"].min()) if row_count else None
    max_timestamp = cast(int, frame["open_time_ms"].max()) if row_count else None
    duplicate_count = row_count - frame.unique(subset=list(KLINE_KEY)).height
    out_of_order = (
        int(frame.select((pl.col("open_time_ms").diff() < 0).sum().fill_null(0)).item())
        if row_count > 1
        else 0
    )
    null_counts = {column: int(frame[column].null_count()) for column in frame.columns}
    interval_ms = INTERVAL_MS[interval]
    gaps, gap_events = _gap_metrics(
        frame["open_time_ms"].drop_nulls().to_list(),
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        interval_ms=interval_ms,
    )
    counts = frame.select(
        pl.any_horizontal(
            pl.col("open") < 0,
            pl.col("high") < 0,
            pl.col("low") < 0,
            pl.col("close") < 0,
        )
        .sum()
        .alias("negative_price_count"),
        pl.any_horizontal(
            pl.col("base_volume") < 0,
            pl.col("quote_volume") < 0,
            pl.col("trade_count") < 0,
            pl.col("taker_buy_base_volume") < 0,
            pl.col("taker_buy_quote_volume") < 0,
        )
        .sum()
        .alias("negative_quantity_count"),
        (
            (pl.col("low") > pl.min_horizontal("open", "close"))
            | (pl.col("high") < pl.max_horizontal("open", "close"))
            | (pl.col("low") > pl.col("high"))
        )
        .sum()
        .alias("invalid_ohlc_count"),
        (pl.col("close_time_ms") <= pl.col("open_time_ms")).sum().alias("invalid_close_time_count"),
        (pl.col("open_time_ms") % interval_ms != 0).sum().alias("unaligned_timestamp_count"),
        (
            (pl.col("open_time_ms") < start_time_ms)
            | (pl.col("open_time_ms") > end_time_ms // interval_ms * interval_ms)
        )
        .sum()
        .alias("out_of_range_count"),
        pl.any_horizontal(
            ~pl.col("open").is_finite(),
            ~pl.col("high").is_finite(),
            ~pl.col("low").is_finite(),
            ~pl.col("close").is_finite(),
            ~pl.col("base_volume").is_finite(),
            ~pl.col("quote_volume").is_finite(),
            ~pl.col("taker_buy_base_volume").is_finite(),
            ~pl.col("taker_buy_quote_volume").is_finite(),
        )
        .sum()
        .alias("non_finite_numeric_count"),
        pl.any_horizontal(
            pl.col("symbol") != symbol,
            pl.col("interval") != interval,
            pl.col("source") != "binance_public_data",
            pl.col("schema_version") != KLINE_SCHEMA_VERSION,
        )
        .sum()
        .alias("contract_value_mismatch_count"),
        (~pl.col("is_closed")).sum().alias("open_bar_count"),
    ).row(0, named=True)
    numeric_counts = {key: int(value or 0) for key, value in counts.items()}
    passed = all(
        (
            duplicate_count == 0,
            sum(null_counts.values()) == 0,
            out_of_order == 0,
            gaps == 0,
            numeric_counts["negative_price_count"] == 0,
            numeric_counts["negative_quantity_count"] == 0,
            numeric_counts["invalid_ohlc_count"] == 0,
            numeric_counts["invalid_close_time_count"] == 0,
            numeric_counts["unaligned_timestamp_count"] == 0,
            numeric_counts["out_of_range_count"] == 0,
            numeric_counts["non_finite_numeric_count"] == 0,
            numeric_counts["contract_value_mismatch_count"] == 0,
            numeric_counts["open_bar_count"] == 0,
            checksum_status == "PASSED",
            schema_status == "PASSED",
        )
    )
    return KlineAudit(
        scope=scope,
        symbol=symbol,
        interval=interval,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        row_count=row_count,
        min_timestamp_ms=min_timestamp,
        max_timestamp_ms=max_timestamp,
        duplicate_key_count=duplicate_count,
        null_count_by_column=null_counts,
        out_of_order_count=out_of_order,
        gap_count=gaps,
        gap_event_count=gap_events,
        gap_duration_seconds=gaps * interval_ms // 1_000,
        negative_price_count=numeric_counts["negative_price_count"],
        negative_quantity_count=numeric_counts["negative_quantity_count"],
        invalid_ohlc_count=numeric_counts["invalid_ohlc_count"],
        invalid_close_time_count=numeric_counts["invalid_close_time_count"],
        unaligned_timestamp_count=numeric_counts["unaligned_timestamp_count"],
        out_of_range_count=numeric_counts["out_of_range_count"],
        non_finite_numeric_count=numeric_counts["non_finite_numeric_count"],
        contract_value_mismatch_count=numeric_counts["contract_value_mismatch_count"],
        open_bar_count=numeric_counts["open_bar_count"],
        checksum_status=checksum_status,
        schema_status=schema_status,
        source_comparison_status="NOT_RUN",
        passed=passed,
    )


def audit_kline_files(
    records: list[DataFileRecord],
    *,
    state_store: StateStore,
    start_time_ms: int,
    end_time_ms: int,
    expected_symbols: tuple[str, ...] | None = None,
) -> QualityReport:
    if not records:
        raise DataQualityError("no normalized kline files match the requested audit")
    if expected_symbols is not None:
        present_symbols = {record.symbol for record in records}
        missing_symbols = sorted(set(expected_symbols).difference(present_symbols))
        if missing_symbols:
            raise DataQualityError(
                f"no normalized kline files for requested symbols: {missing_symbols}"
            )
    partitions: list[KlineAudit] = []
    frames_by_series: dict[tuple[str, str], list[pl.DataFrame]] = {}
    checksums_by_series: dict[tuple[str, str], list[bool]] = {}
    for record in records:
        path = Path(record.path)
        if not path.exists():
            raise DataQualityError(f"manifested Parquet is missing: {path}")
        try:
            frame = pl.read_parquet(path)
        except (OSError, pl.exceptions.PolarsError) as exc:
            raise DataQualityError(f"cannot read Parquet {path}: {exc}") from exc
        checksum_ok = record.checksum is not None and sha256_file(path) == record.checksum
        partition = audit_kline_frame(
            frame,
            scope=record.file_id,
            symbol=record.symbol,
            interval=record.interval,
            start_time_ms=record.start_time_ms,
            end_time_ms=record.end_time_ms,
            checksum_status="PASSED" if checksum_ok else "FAILED",
        )
        partitions.append(partition)
        details_json = orjson.dumps(asdict(partition), option=orjson.OPT_SORT_KEYS).decode("utf-8")
        state_store.register_quality_result(
            file_id=record.file_id,
            check_name=f"canonical_kline_audit_v{KLINE_SCHEMA_VERSION}",
            passed=partition.passed,
            details_json=details_json,
        )
        key = (record.symbol, record.interval)
        frames_by_series.setdefault(key, []).append(frame)
        checksums_by_series.setdefault(key, []).append(checksum_ok)

    aggregate: list[KlineAudit] = []
    for (symbol, interval), frames in sorted(frames_by_series.items()):
        combined = pl.concat(frames, how="vertical_relaxed", rechunk=True)
        aggregate.append(
            audit_kline_frame(
                combined,
                scope="requested_range",
                symbol=symbol,
                interval=interval,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
                checksum_status=(
                    "PASSED" if all(checksums_by_series[(symbol, interval)]) else "FAILED"
                ),
            )
        )
    return QualityReport(
        created_at_ms=now_ms(),
        schema_version=KLINE_SCHEMA_VERSION,
        passed=all(item.passed for item in (*partitions, *aggregate)),
        files_audited=len(records),
        aggregate=tuple(aggregate),
        partitions=tuple(partitions),
    )


def write_quality_report(*, report: QualityReport, reports_root: Path) -> tuple[Path, Path]:
    storage = LocalFilesystemStorage(reports_root)
    base = f"data_quality_{report.created_at_ms}_{uuid.uuid4().hex[:8]}"
    json_path = storage.path(f"{base}.json")
    markdown_path = storage.path(f"{base}.md")
    storage.write_json_atomic(json_path, asdict(report))
    lines = [
        "# Data quality report",
        "",
        f"- Gate: {'PASS' if report.passed else 'FAIL'}",
        f"- Schema version: {report.schema_version}",
        f"- Files audited: {report.files_audited}",
        "",
        "| Symbol | Rows | Range | Duplicates | Gaps | Gap events | Invalid OHLC | "
        "Checksum | Schema | Gate |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report.aggregate:
        lines.append(
            f"| {item.symbol} | {item.row_count} | {item.start_time_ms}..{item.end_time_ms} | "
            f"{item.duplicate_key_count} | {item.gap_count} | {item.gap_event_count} | "
            f"{item.invalid_ohlc_count} | {item.checksum_status} | {item.schema_status} | "
            f"{'PASS' if item.passed else 'FAIL'} |"
        )
    storage.write_bytes_atomic(markdown_path, ("\n".join(lines) + "\n").encode("utf-8"))
    return json_path, markdown_path
