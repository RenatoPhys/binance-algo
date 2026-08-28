from __future__ import annotations

import asyncio

import aiohttp
import orjson
import pytest

from binance_algo.common.errors import QueueSaturatedError
from binance_algo.exchange.binance_usdm.public_streams import (
    CanonicalStreamEvent,
    StreamSubscription,
    parse_combined_message,
)
from binance_algo.exchange.binance_usdm.websocket import (
    RoutedPublicWebSocket,
    WebSocketRunStats,
)
from binance_algo.observability.metrics import (
    HealthServer,
    RecorderHealthState,
    RecorderMetrics,
)


def _event() -> CanonicalStreamEvent:
    event = parse_combined_message(
        orjson.dumps(
            {
                "stream": "btcusdt@bookTicker",
                "data": {
                    "e": "bookTicker",
                    "E": 1_000,
                    "T": 999,
                    "s": "BTCUSDT",
                    "u": 1,
                    "b": "100",
                    "B": "1",
                    "a": "101",
                    "A": "1",
                },
            }
        ),
        received_time_ns=1_000_000_000,
        connection_id="connection-1",
        ingestion_run_id="run-1",
    )
    assert event is not None
    return event


def _connection(
    queue: asyncio.Queue[CanonicalStreamEvent],
    *,
    stats: WebSocketRunStats,
    metrics: RecorderMetrics,
    health: RecorderHealthState,
) -> RoutedPublicWebSocket:
    return RoutedPublicWebSocket(
        base_url="wss://example.test",
        subscription=StreamSubscription(route="public", names=("btcusdt@bookTicker",)),
        queue=queue,
        ingestion_run_id="run-1",
        stats=stats,
        metrics=metrics,
        health=health,
        queue_put_timeout_seconds=0.01,
        stale_after_seconds=5,
        reconnect_base_seconds=0,
        reconnect_max_seconds=1,
        reconnect_stable_after_seconds=1,
        connection_max_seconds=60,
    )


@pytest.mark.asyncio
async def test_queue_saturation_fails_explicitly_and_counts_the_event() -> None:
    queue: asyncio.Queue[CanonicalStreamEvent] = asyncio.Queue(maxsize=1)
    queue.put_nowait(_event())
    stats = WebSocketRunStats()
    metrics = RecorderMetrics(queue_capacity=1)
    health = RecorderHealthState(required_routes=frozenset({"public"}), stale_after_seconds=5)
    connection = _connection(
        queue,
        stats=stats,
        metrics=metrics,
        health=health,
    )

    with pytest.raises(QueueSaturatedError, match="remained full"):
        await connection._enqueue(_event())

    assert stats.dropped_events == 1
    assert queue.qsize() == 1


@pytest.mark.asyncio
async def test_subscribe_and_unsubscribe_use_monotonic_request_ids() -> None:
    class FakeSocket:
        closed = False

        def __init__(self) -> None:
            self.messages: list[str] = []

        async def send_str(self, message: str) -> None:
            self.messages.append(message)

    queue: asyncio.Queue[CanonicalStreamEvent] = asyncio.Queue(maxsize=1)
    stats = WebSocketRunStats()
    metrics = RecorderMetrics(queue_capacity=1)
    health = RecorderHealthState(required_routes=frozenset({"public"}), stale_after_seconds=5)
    connection = _connection(queue, stats=stats, metrics=metrics, health=health)
    socket = FakeSocket()
    connection._websocket = socket  # type: ignore[assignment]

    assert await connection.subscribe(("ethusdt@bookTicker",)) == 1
    assert await connection.unsubscribe(("btcusdt@bookTicker",)) == 2
    assert [orjson.loads(message) for message in socket.messages] == [
        {"method": "SUBSCRIBE", "params": ["ethusdt@bookTicker"], "id": 1},
        {"method": "UNSUBSCRIBE", "params": ["btcusdt@bookTicker"], "id": 2},
    ]


@pytest.mark.asyncio
async def test_health_and_metrics_endpoints_reflect_readiness() -> None:
    metrics = RecorderMetrics(queue_capacity=123)
    metrics.clock_offset.set(4)
    health = RecorderHealthState(
        required_routes=frozenset({"public", "market"}), stale_after_seconds=5
    )
    server = HealthServer(host="127.0.0.1", port=0, health=health, metrics=metrics)
    await server.start()
    assert server.bound_port is not None
    base_url = f"http://127.0.0.1:{server.bound_port}"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{base_url}/health/live") as response:
                assert response.status == 200
                assert await response.json() == {"live": True}
            async with session.get(f"{base_url}/health/ready") as response:
                assert response.status == 503

            health.mark_event("public")
            health.mark_event("market")
            async with session.get(f"{base_url}/health/ready") as response:
                assert response.status == 200
                assert await response.json() == {"ready": True, "error": None}
            async with session.get(f"{base_url}/metrics") as response:
                payload = await response.text()
                assert "recorder_queue_capacity 123.0" in payload
                assert "clock_offset_ms 4.0" in payload
                for metric_name in (
                    "binance_rest_requests_total",
                    "binance_rest_errors_total",
                    "binance_rest_latency_seconds",
                    "binance_rate_limit_used",
                    "binance_ws_connections_total",
                    "binance_ws_reconnects_total",
                    "binance_ws_messages_total",
                    "binance_ws_last_event_age_seconds",
                    "binance_ws_sequence_gaps_total",
                    "recorder_queue_size",
                    "recorder_queue_capacity",
                    "recorder_dropped_events_total",
                    "parquet_flush_total",
                    "parquet_flush_rows",
                    "parquet_flush_latency_seconds",
                    "data_quality_failures_total",
                    "clock_offset_ms",
                ):
                    assert metric_name in payload

            health.queue_saturated = True
            async with session.get(f"{base_url}/health/ready") as response:
                assert response.status == 503
    finally:
        await server.close()
