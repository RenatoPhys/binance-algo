"""End-to-end public market recorder orchestration and graceful shutdown."""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from binance_algo.common.errors import (
    DataQualityError,
    QueueSaturatedError,
    RecorderError,
    WebSocketStreamError,
)
from binance_algo.config import Settings
from binance_algo.data.manifest import DataFileStatus
from binance_algo.data.state_store import StateStore
from binance_algo.data.storage import LocalFilesystemStorage
from binance_algo.data.stream_quality import (
    RecorderQualityReport,
    audit_recorder_run,
    rebuild_stream_catalog,
    write_recorder_report,
)
from binance_algo.data.writer import (
    StreamMicroBatchWriter,
    StreamWriterStats,
    recover_recorder_artifacts,
)
from binance_algo.exchange.binance_usdm.public_streams import (
    STREAM_SCHEMA_VERSION,
    CanonicalStreamEvent,
    StreamEventType,
    build_stream_subscriptions,
    stream_schema_contract_json,
)
from binance_algo.exchange.binance_usdm.websocket import (
    RoutedPublicWebSocket,
    WebSocketRunStats,
)
from binance_algo.logging import get_logger
from binance_algo.observability.metrics import (
    HealthServer,
    RecorderHealthState,
    RecorderMetrics,
)


@dataclass(frozen=True, slots=True)
class RecorderRunResult:
    report: RecorderQualityReport
    json_report: Path
    markdown_report: Path
    health_port: int | None


def enabled_event_types(settings: Settings) -> tuple[StreamEventType, ...]:
    enabled: list[StreamEventType] = []
    if settings.streams.book_ticker:
        enabled.append("book_ticker")
    if settings.streams.aggregate_trades:
        enabled.append("aggregate_trades")
    if settings.streams.mark_price:
        enabled.append("mark_price")
    if settings.streams.kline_1m:
        enabled.append("kline_1m")
    return tuple(enabled)


async def run_recorder(
    settings: Settings,
    *,
    symbols: tuple[str, ...],
    duration_seconds: float,
    clock_offset_ms: int,
    metrics_port: int | None = None,
    metrics: RecorderMetrics | None = None,
) -> RecorderRunResult:
    if duration_seconds <= 0:
        raise RecorderError("recorder duration must be positive")
    run_id = uuid.uuid4().hex
    started_at_ns = time.time_ns()
    storage = LocalFilesystemStorage(settings.data_root)
    _verify_storage_writable(storage)
    state_store = StateStore(settings.state_db_path)
    state_store.initialize()
    event_types = enabled_event_types(settings)
    for event_type in event_types:
        state_store.register_schema_version(
            event_type, STREAM_SCHEMA_VERSION, stream_schema_contract_json(event_type)
        )
    writer_stats = StreamWriterStats()
    recover_recorder_artifacts(storage=storage, state_store=state_store, stats=writer_stats)
    subscriptions = build_stream_subscriptions(
        symbols,
        book_ticker=settings.streams.book_ticker,
        aggregate_trades=settings.streams.aggregate_trades,
        mark_price=settings.streams.mark_price,
        kline_1m=settings.streams.kline_1m,
    )
    required_routes = frozenset(subscription.route for subscription in subscriptions)
    queue: asyncio.Queue[CanonicalStreamEvent] = asyncio.Queue(
        maxsize=settings.recorder.queue_capacity
    )
    metrics = metrics or RecorderMetrics(queue_capacity=settings.recorder.queue_capacity)
    metrics.clock_offset.set(clock_offset_ms)
    health = RecorderHealthState(
        required_routes=required_routes,
        stale_after_seconds=settings.recorder.stale_after_seconds,
    )
    health_server = HealthServer(
        host=settings.recorder.metrics_host,
        port=settings.recorder.metrics_port if metrics_port is None else metrics_port,
        health=health,
        metrics=metrics,
    )
    websocket_stats = WebSocketRunStats()
    producer_done = asyncio.Event()
    writer = StreamMicroBatchWriter(
        storage=storage,
        state_store=state_store,
        ingestion_run_id=run_id,
        compression=settings.storage.parquet_compression,
        max_rows=settings.storage.micro_batch_max_rows,
        max_seconds=settings.storage.micro_batch_max_seconds,
        metrics=metrics,
        health=health,
        stats=writer_stats,
    )
    stop_event = asyncio.Event()
    connections = [
        RoutedPublicWebSocket(
            base_url=settings.binance.market_ws_base_url,
            subscription=subscription,
            queue=queue,
            ingestion_run_id=run_id,
            stats=websocket_stats,
            metrics=metrics,
            health=health,
            queue_put_timeout_seconds=settings.recorder.queue_put_timeout_seconds,
            stale_after_seconds=settings.recorder.stale_after_seconds,
            reconnect_base_seconds=settings.recorder.reconnect_base_seconds,
            reconnect_max_seconds=settings.binance.reconnect_max_seconds,
            reconnect_stable_after_seconds=settings.recorder.reconnect_stable_after_seconds,
            connection_max_seconds=settings.recorder.connection_max_seconds,
        )
        for subscription in subscriptions
    ]
    logger = get_logger(operation="recorder", run_id=run_id)
    await health_server.start()
    logger.info(
        "recorder_started",
        symbols=list(symbols),
        duration_seconds=duration_seconds,
        routes=sorted(required_routes),
        health_port=health_server.bound_port,
    )
    producer_tasks = [
        asyncio.create_task(connection.run(stop_event), name=f"ws-{connection.subscription.route}")
        for connection in connections
    ]
    writer_task = asyncio.create_task(writer.run(queue, producer_done), name="stream-writer")
    monitor_task = asyncio.create_task(
        _monitor_health(stop_event, health=health, metrics=metrics), name="recorder-health"
    )
    failure: RecorderError | WebSocketStreamError | QueueSaturatedError | None = None
    interrupted = False
    try:
        done, _ = await asyncio.wait(
            [*producer_tasks, writer_task],
            timeout=duration_seconds,
            return_when=asyncio.FIRST_EXCEPTION,
        )
        for task in done:
            try:
                task.result()
            except (RecorderError, WebSocketStreamError, QueueSaturatedError) as exc:
                failure = exc
                break
            if task is writer_task:
                failure = RecorderError("stream writer stopped before recorder duration elapsed")
                break
    except asyncio.CancelledError:
        interrupted = True
        failure = RecorderError("recorder interrupted; graceful shutdown requested")
        current_task = asyncio.current_task()
        if current_task is not None:
            current_task.uncancel()
    finally:
        stop_event.set()
        if interrupted:
            for task in producer_tasks:
                task.cancel()
            await asyncio.gather(*producer_tasks, return_exceptions=True)
        else:
            await _finish_producers(
                producer_tasks, timeout_seconds=settings.recorder.shutdown_timeout_seconds
            )
        producer_done.set()
        if not writer_task.done():
            try:
                await asyncio.wait_for(
                    writer_task, timeout=settings.recorder.shutdown_timeout_seconds
                )
            except TimeoutError:
                writer_task.cancel()
                await asyncio.gather(writer_task, return_exceptions=True)
                if failure is None:
                    failure = RecorderError("writer shutdown timed out")
            except RecorderError as exc:
                if failure is None:
                    failure = exc
        monitor_task.cancel()
        await asyncio.gather(monitor_task, return_exceptions=True)
        health.stopping = True
        await health_server.close()

    ended_at_ns = time.time_ns()
    all_stream_records = [
        record
        for record in state_store.list_data_files(layer="raw", statuses={DataFileStatus.VALIDATED})
        if record.source == "binance_market_stream"
    ]
    run_records = [record for record in all_stream_records if record.ingestion_run_id == run_id]
    if not run_records:
        metrics.data_quality_failures.inc()
        detail = f": {failure}" if failure is not None else ""
        raise RecorderError(
            f"recorder run {run_id} produced no validated files{detail}"
        ) from failure
    try:
        rebuild_stream_catalog(database_path=settings.duckdb_path, records=all_stream_records)
        report = audit_recorder_run(
            database_path=settings.duckdb_path,
            records=all_stream_records,
            run_id=run_id,
            started_at_ns=started_at_ns,
            ended_at_ns=ended_at_ns,
            clock_offset_ms=clock_offset_ms,
            queue_capacity=settings.recorder.queue_capacity,
            websocket_stats=websocket_stats,
            temporary_files_quarantined=writer_stats.temporary_files_quarantined,
            orphan_files_recovered=writer_stats.orphan_files_recovered,
            inflight_files_recovered=writer_stats.inflight_files_recovered,
            invalid_files_quarantined=writer_stats.invalid_files_quarantined,
            expected_event_types=event_types,
        )
    except DataQualityError:
        metrics.data_quality_failures.inc()
        raise
    json_report, markdown_report = write_recorder_report(
        report=report, reports_root=settings.reports_root
    )
    logger.info(
        "recorder_complete",
        passed=report.passed,
        messages=report.messages_received_total,
        rows=report.rows_persisted_total,
        dropped=report.dropped_events_total,
        reconnects=report.reconnects_total,
        json_report=str(json_report),
    )
    if failure is not None:
        metrics.data_quality_failures.inc()
        raise RecorderError(
            f"recorder stopped after preserving its report at {json_report}: {failure}"
        ) from failure
    if not report.passed:
        metrics.data_quality_failures.inc()
        raise RecorderError(f"recorder quality gate failed; inspect {json_report}")
    return RecorderRunResult(
        report=report,
        json_report=json_report,
        markdown_report=markdown_report,
        health_port=health_server.bound_port,
    )


async def _finish_producers(tasks: list[asyncio.Task[None]], *, timeout_seconds: float) -> None:
    if not tasks:
        return
    _, pending = await asyncio.wait(tasks, timeout=timeout_seconds)
    for task in pending:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def _verify_storage_writable(storage: LocalFilesystemStorage) -> None:
    probe = storage.path(".health", f"recorder_{uuid.uuid4().hex}.probe")
    try:
        storage.write_bytes_atomic(probe, b"recorder storage probe\n")
    except OSError as exc:
        raise RecorderError(f"recorder storage is not writable: {exc}") from exc
    finally:
        with suppress(OSError):
            probe.unlink()


async def _monitor_health(
    stop_event: asyncio.Event,
    *,
    health: RecorderHealthState,
    metrics: RecorderMetrics,
) -> None:
    while not stop_event.is_set():
        now = time.monotonic_ns()
        for route in health.required_routes:
            last = health.last_event_monotonic_ns.get(route)
            age = (now - last) / 1_000_000_000 if last is not None else float("inf")
            metrics.ws_last_event_age.labels(route=route).set(age)
        with suppress(TimeoutError):
            await asyncio.wait_for(stop_event.wait(), timeout=1.0)
