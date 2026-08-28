"""DuckDB catalog and objective quality report for recorded market streams."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import polars as pl

from binance_algo.common.errors import DataQualityError
from binance_algo.data.archive_client import sha256_file
from binance_algo.data.manifest import DataFileRecord, now_ms
from binance_algo.data.storage import LocalFilesystemStorage
from binance_algo.exchange.binance_usdm.public_streams import (
    STREAM_EVENT_TYPES,
    STREAM_SCHEMAS,
    StreamEventType,
)
from binance_algo.exchange.binance_usdm.websocket import WebSocketRunStats

VIEW_NAMES: dict[StreamEventType, str] = {
    "book_ticker": "realtime_book_ticker",
    "aggregate_trades": "realtime_aggregate_trades",
    "mark_price": "realtime_mark_price",
    "kline_1m": "realtime_kline_1m",
}


@dataclass(frozen=True, slots=True)
class StreamDatasetAudit:
    event_type: StreamEventType
    messages_received: int
    rows_persisted: int
    file_count: int
    file_size_bytes: int
    min_event_time_ms: int | None
    max_event_time_ms: int | None
    duplicate_event_count: int
    sequence_gap_count: int
    sequence_regression_count: int
    invalid_market_value_count: int
    checksum_status: str
    schema_status: str
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    latency_p99_ms: float | None
    staleness_p95_seconds: float | None
    staleness_max_seconds: float | None
    passed: bool


@dataclass(frozen=True, slots=True)
class RecorderQualityReport:
    run_id: str
    started_at_ns: int
    ended_at_ns: int
    duration_seconds: float
    clock_offset_ms: int
    passed: bool
    queue_capacity: int
    queue_peak: int
    dropped_events_total: int
    connections_total: int
    reconnects_total: int
    messages_received_total: int
    rows_persisted_total: int
    files_total: int
    file_size_bytes: int
    temporary_files_quarantined: int
    orphan_files_recovered: int
    inflight_files_recovered: int
    invalid_files_quarantined: int
    messages_by_symbol: dict[str, int]
    disconnects: tuple[dict[str, object], ...]
    datasets: tuple[StreamDatasetAudit, ...]


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def rebuild_stream_catalog(*, database_path: Path, records: list[DataFileRecord]) -> None:
    grouped: dict[str, list[DataFileRecord]] = {}
    for record in records:
        if record.logical_dataset in STREAM_EVENT_TYPES:
            grouped.setdefault(record.logical_dataset, []).append(record)
    if not grouped:
        raise DataQualityError("cannot build stream catalog without validated recorder files")
    database_path = database_path.resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with duckdb.connect(str(database_path)) as connection:
            for event_type in STREAM_EVENT_TYPES:
                selected = grouped.get(event_type)
                view_name = VIEW_NAMES[event_type]
                if not selected:
                    connection.execute(f"DROP VIEW IF EXISTS {view_name}")
                    continue
                paths = ", ".join(
                    _sql_string(str(Path(record.path).resolve()).replace("\\", "/"))
                    for record in selected
                )
                connection.execute(
                    f"CREATE OR REPLACE VIEW {view_name} AS "
                    f"SELECT * FROM read_parquet([{paths}], union_by_name = true)"
                )
    except duckdb.Error as exc:
        raise DataQualityError(f"cannot rebuild DuckDB stream catalog: {exc}") from exc


def audit_recorder_run(
    *,
    database_path: Path,
    records: list[DataFileRecord],
    run_id: str,
    started_at_ns: int,
    ended_at_ns: int,
    clock_offset_ms: int,
    queue_capacity: int,
    websocket_stats: WebSocketRunStats,
    temporary_files_quarantined: int,
    orphan_files_recovered: int,
    inflight_files_recovered: int,
    invalid_files_quarantined: int,
    expected_event_types: tuple[StreamEventType, ...],
) -> RecorderQualityReport:
    run_records = [record for record in records if record.ingestion_run_id == run_id]
    if not run_records:
        raise DataQualityError(f"recorder run {run_id} produced no manifested files")
    present_event_types = {record.logical_dataset for record in run_records}
    missing_event_types = sorted(set(expected_event_types).difference(present_event_types))
    if missing_event_types:
        raise DataQualityError(
            f"recorder run {run_id} produced no files for streams: {missing_event_types}"
        )
    audits: list[StreamDatasetAudit] = []
    try:
        with duckdb.connect(str(database_path), read_only=True) as connection:
            for event_type in STREAM_EVENT_TYPES:
                selected = [r for r in run_records if r.logical_dataset == event_type]
                if not selected:
                    continue
                audits.append(
                    _audit_dataset(
                        connection,
                        event_type=event_type,
                        records=selected,
                        run_id=run_id,
                        clock_offset_ms=clock_offset_ms,
                        messages_received=int(websocket_stats.messages_by_stream[event_type]),
                    )
                )
    except duckdb.Error as exc:
        raise DataQualityError(f"cannot audit recorder run {run_id}: {exc}") from exc
    rows_persisted = sum(item.rows_persisted for item in audits)
    messages_received = sum(websocket_stats.messages_by_stream.values())
    passed = all(
        (
            websocket_stats.dropped_events == 0,
            messages_received == rows_persisted,
            all(item.passed for item in audits),
        )
    )
    return RecorderQualityReport(
        run_id=run_id,
        started_at_ns=started_at_ns,
        ended_at_ns=ended_at_ns,
        duration_seconds=(ended_at_ns - started_at_ns) / 1_000_000_000,
        clock_offset_ms=clock_offset_ms,
        passed=passed,
        queue_capacity=queue_capacity,
        queue_peak=websocket_stats.queue_peak,
        dropped_events_total=websocket_stats.dropped_events,
        connections_total=sum(websocket_stats.connections_by_route.values()),
        reconnects_total=sum(websocket_stats.reconnects_by_route.values()),
        messages_received_total=messages_received,
        rows_persisted_total=rows_persisted,
        files_total=len(run_records),
        file_size_bytes=sum(Path(record.path).stat().st_size for record in run_records),
        temporary_files_quarantined=temporary_files_quarantined,
        orphan_files_recovered=orphan_files_recovered,
        inflight_files_recovered=inflight_files_recovered,
        invalid_files_quarantined=invalid_files_quarantined,
        messages_by_symbol=dict(sorted(websocket_stats.messages_by_symbol.items())),
        disconnects=tuple(websocket_stats.disconnect_payloads()),
        datasets=tuple(audits),
    )


def _audit_dataset(
    connection: duckdb.DuckDBPyConnection,
    *,
    event_type: StreamEventType,
    records: list[DataFileRecord],
    run_id: str,
    clock_offset_ms: int,
    messages_received: int,
) -> StreamDatasetAudit:
    view = VIEW_NAMES[event_type]
    basic = connection.execute(
        f"""
        SELECT COUNT(*)::BIGINT,
               MIN(event_time_ms)::BIGINT,
               MAX(event_time_ms)::BIGINT,
               (COUNT(*) - COUNT(DISTINCT event_id))::BIGINT,
               quantile_cont(received_time_ns / 1000000.0 + ? - event_time_ms, [0.5, 0.95, 0.99])
        FROM {view}
        WHERE ingestion_run_id = ?
        """,
        (clock_offset_ms, run_id),
    ).fetchone()
    if basic is None:
        raise DataQualityError(f"empty DuckDB audit result for {event_type}")
    staleness = connection.execute(
        f"""
        WITH ordered AS (
            SELECT (received_time_ns - LAG(received_time_ns) OVER (
                       PARTITION BY symbol ORDER BY received_time_ns, event_id
                   )) / 1000000000.0 AS seconds
            FROM {view}
            WHERE ingestion_run_id = ?
        )
        SELECT quantile_cont(seconds, 0.95), MAX(seconds)
        FROM ordered WHERE seconds IS NOT NULL
        """,
        (run_id,),
    ).fetchone()
    sequence_gaps, sequence_regressions = _sequence_counts(
        connection, event_type=event_type, view=view, run_id=run_id
    )
    invalid = _invalid_count(connection, event_type=event_type, view=view, run_id=run_id)
    checksum_ok = all(
        record.checksum is not None and sha256_file(Path(record.path)) == record.checksum
        for record in records
    )
    expected_schema = STREAM_SCHEMAS[event_type]
    schema_ok = True
    for record in records:
        try:
            actual = dict(pl.read_parquet_schema(record.path))
        except (OSError, pl.exceptions.PolarsError):
            schema_ok = False
            break
        if actual != expected_schema:
            schema_ok = False
            break
    row_count = int(basic[0])
    duplicate_count = int(basic[3])
    quantiles = basic[4]
    latency = [float(value) for value in quantiles] if quantiles is not None else [None] * 3
    staleness_p95 = float(staleness[0]) if staleness and staleness[0] is not None else None
    staleness_max = float(staleness[1]) if staleness and staleness[1] is not None else None
    passed = all(
        (
            row_count == messages_received,
            duplicate_count == 0,
            sequence_gaps == 0,
            sequence_regressions == 0,
            invalid == 0,
            checksum_ok,
            schema_ok,
        )
    )
    return StreamDatasetAudit(
        event_type=event_type,
        messages_received=messages_received,
        rows_persisted=row_count,
        file_count=len(records),
        file_size_bytes=sum(Path(record.path).stat().st_size for record in records),
        min_event_time_ms=int(basic[1]) if basic[1] is not None else None,
        max_event_time_ms=int(basic[2]) if basic[2] is not None else None,
        duplicate_event_count=duplicate_count,
        sequence_gap_count=sequence_gaps,
        sequence_regression_count=sequence_regressions,
        invalid_market_value_count=invalid,
        checksum_status="PASSED" if checksum_ok else "FAILED",
        schema_status="PASSED" if schema_ok else "FAILED",
        latency_p50_ms=latency[0],
        latency_p95_ms=latency[1],
        latency_p99_ms=latency[2],
        staleness_p95_seconds=staleness_p95,
        staleness_max_seconds=staleness_max,
        passed=passed,
    )


def _sequence_counts(
    connection: duckdb.DuckDBPyConnection,
    *,
    event_type: StreamEventType,
    view: str,
    run_id: str,
) -> tuple[int, int]:
    if event_type == "aggregate_trades":
        gap_row = connection.execute(
            f"""
            WITH ids AS (
                SELECT DISTINCT symbol, aggregate_trade_id FROM {view}
                WHERE ingestion_run_id = ?
            ), ordered AS (
                SELECT aggregate_trade_id,
                       LAG(aggregate_trade_id) OVER (
                           PARTITION BY symbol ORDER BY aggregate_trade_id
                       ) AS previous_id
                FROM ids
            )
            SELECT COALESCE(SUM(GREATEST(aggregate_trade_id - previous_id - 1, 0)), 0)
            FROM ordered
            """,
            (run_id,),
        ).fetchone()
        regression_row = connection.execute(
            f"""
            WITH ordered AS (
                SELECT aggregate_trade_id,
                       LAG(aggregate_trade_id) OVER (
                           PARTITION BY symbol, connection_id
                           ORDER BY received_time_ns, event_id
                       ) AS previous_id
                FROM {view} WHERE ingestion_run_id = ?
            )
            SELECT COUNT(*) FROM ordered
            WHERE previous_id IS NOT NULL AND aggregate_trade_id <= previous_id
            """,
            (run_id,),
        ).fetchone()
        return int(gap_row[0] if gap_row else 0), int(regression_row[0] if regression_row else 0)
    if event_type == "book_ticker":
        row = connection.execute(
            f"""
            WITH ordered AS (
                SELECT update_id,
                       LAG(update_id) OVER (
                           PARTITION BY symbol, connection_id
                           ORDER BY received_time_ns, event_id
                       ) AS previous_id
                FROM {view} WHERE ingestion_run_id = ?
            )
            SELECT COUNT(*) FROM ordered
            WHERE previous_id IS NOT NULL AND update_id <= previous_id
            """,
            (run_id,),
        ).fetchone()
        return 0, int(row[0] if row else 0)
    return 0, 0


def _invalid_count(
    connection: duckdb.DuckDBPyConnection,
    *,
    event_type: StreamEventType,
    view: str,
    run_id: str,
) -> int:
    predicate = {
        "book_ticker": (
            "bid_price <= 0 OR ask_price <= 0 OR bid_qty < 0 OR ask_qty < 0 "
            "OR bid_price > ask_price"
        ),
        "aggregate_trades": "price <= 0 OR quantity <= 0 OR first_trade_id > last_trade_id",
        "mark_price": "mark_price <= 0 OR index_price <= 0",
        "kline_1m": (
            "open <= 0 OR high <= 0 OR low <= 0 OR close <= 0 OR base_volume < 0 "
            "OR quote_volume < 0 OR low > LEAST(open, close) OR high < GREATEST(open, close) "
            "OR close_time_ms <= open_time_ms OR interval <> '1m'"
        ),
    }[event_type]
    row = connection.execute(
        f"SELECT COUNT(*) FROM {view} WHERE ingestion_run_id = ? AND ({predicate})",
        (run_id,),
    ).fetchone()
    return int(row[0] if row else 0)


def write_recorder_report(
    *, report: RecorderQualityReport, reports_root: Path
) -> tuple[Path, Path]:
    storage = LocalFilesystemStorage(reports_root)
    base = f"recorder_{now_ms()}_{report.run_id[:8]}_{uuid.uuid4().hex[:6]}"
    json_path = storage.path(f"{base}.json")
    markdown_path = storage.path(f"{base}.md")
    storage.write_json_atomic(json_path, asdict(report))
    lines = [
        "# Recorder quality report",
        "",
        f"- Run: `{report.run_id}`",
        f"- Gate: {'PASS' if report.passed else 'FAIL'}",
        f"- Duration: {report.duration_seconds:.3f}s",
        f"- Clock offset: {report.clock_offset_ms}ms",
        f"- Messages received: {report.messages_received_total}",
        f"- Rows persisted: {report.rows_persisted_total}",
        f"- Dropped events: {report.dropped_events_total}",
        f"- Connections / reconnects: {report.connections_total} / {report.reconnects_total}",
        f"- Queue peak / capacity: {report.queue_peak} / {report.queue_capacity}",
        f"- Files / bytes: {report.files_total} / {report.file_size_bytes}",
        f"- Recovery (temporary/orphan/in-flight/invalid): "
        f"{report.temporary_files_quarantined}/{report.orphan_files_recovered}/"
        f"{report.inflight_files_recovered}/{report.invalid_files_quarantined}",
        "",
        "| Stream | Messages | Rows | Duplicates | Gaps | Regressions | Invalid | "
        "p50 ms | p95 ms | p99 ms | Max stale s | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report.datasets:
        lines.append(
            f"| {item.event_type} | {item.messages_received} | {item.rows_persisted} | "
            f"{item.duplicate_event_count} | {item.sequence_gap_count} | "
            f"{item.sequence_regression_count} | {item.invalid_market_value_count} | "
            f"{_number(item.latency_p50_ms)} | {_number(item.latency_p95_ms)} | "
            f"{_number(item.latency_p99_ms)} | {_number(item.staleness_max_seconds)} | "
            f"{'PASS' if item.passed else 'FAIL'} |"
        )
    storage.write_bytes_atomic(markdown_path, ("\n".join(lines) + "\n").encode("utf-8"))
    return json_path, markdown_path


def _number(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"
