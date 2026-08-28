"""Per-recorder Prometheus registry and local liveness/readiness endpoints."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, cast

from aiohttp import web
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class RecorderMetrics:
    def __init__(self, *, queue_capacity: int) -> None:
        self.registry = CollectorRegistry()
        self.rest_requests = Counter(
            "binance_rest_requests_total",
            "Binance REST request attempts.",
            ("path", "status"),
            registry=self.registry,
        )
        self.rest_errors = Counter(
            "binance_rest_errors_total",
            "Binance REST request errors.",
            ("path", "error_type"),
            registry=self.registry,
        )
        self.rest_latency = Histogram(
            "binance_rest_latency_seconds",
            "Binance REST attempt latency.",
            ("path",),
            registry=self.registry,
        )
        self.rate_limit_used = Gauge(
            "binance_rate_limit_used",
            "Latest Binance rate-limit header value.",
            ("header",),
            registry=self.registry,
        )
        self.ws_connections = Counter(
            "binance_ws_connections_total",
            "WebSocket connections opened.",
            ("route",),
            registry=self.registry,
        )
        self.ws_reconnects = Counter(
            "binance_ws_reconnects_total",
            "WebSocket reconnection attempts.",
            ("route",),
            registry=self.registry,
        )
        self.ws_messages = Counter(
            "binance_ws_messages_total",
            "Canonical WebSocket messages received.",
            ("route", "event_type", "symbol"),
            registry=self.registry,
        )
        self.ws_last_event_age = Gauge(
            "binance_ws_last_event_age_seconds",
            "Age of the last event on a routed connection.",
            ("route",),
            registry=self.registry,
        )
        self.ws_sequence_gaps = Counter(
            "binance_ws_sequence_gaps_total",
            "Observed aggregate-trade sequence gaps.",
            ("symbol",),
            registry=self.registry,
        )
        self.queue_size = Gauge(
            "recorder_queue_size",
            "Current recorder queue size.",
            registry=self.registry,
        )
        self.queue_capacity = Gauge(
            "recorder_queue_capacity",
            "Configured recorder queue capacity.",
            registry=self.registry,
        )
        self.queue_capacity.set(queue_capacity)
        self.dropped_events = Counter(
            "recorder_dropped_events_total",
            "Events not enqueued because the lossless contract failed.",
            registry=self.registry,
        )
        self.parquet_flushes = Counter(
            "parquet_flush_total",
            "Atomic Parquet micro-batches promoted.",
            ("event_type",),
            registry=self.registry,
        )
        self.parquet_flush_rows = Gauge(
            "parquet_flush_rows",
            "Cumulative rows promoted in Parquet micro-batches.",
            ("event_type",),
            registry=self.registry,
        )
        self.parquet_flush_latency = Histogram(
            "parquet_flush_latency_seconds",
            "Parquet micro-batch flush latency.",
            ("event_type",),
            registry=self.registry,
        )
        self.data_quality_failures = Counter(
            "data_quality_failures_total",
            "Recorder quality-gate failures.",
            registry=self.registry,
        )
        self.clock_offset = Gauge(
            "clock_offset_ms",
            "Observed exchange minus local clock offset in milliseconds.",
            registry=self.registry,
        )

    def observe_rest_response(
        self,
        *,
        path: str,
        status: int,
        latency_seconds: float,
        rate_limits: dict[str, str],
    ) -> None:
        self.rest_requests.labels(path=path, status=str(status)).inc()
        self.rest_latency.labels(path=path).observe(latency_seconds)
        if status >= 400:
            self.rest_errors.labels(path=path, error_type=f"http_{status}").inc()
        for header, raw_value in rate_limits.items():
            try:
                value = float(raw_value)
            except ValueError:
                continue
            self.rate_limit_used.labels(header=header).set(value)

    def observe_rest_error(self, *, path: str, error_type: str, latency_seconds: float) -> None:
        self.rest_requests.labels(path=path, status="network_error").inc()
        self.rest_errors.labels(path=path, error_type=error_type).inc()
        self.rest_latency.labels(path=path).observe(latency_seconds)


@dataclass(slots=True)
class RecorderHealthState:
    required_routes: frozenset[str]
    stale_after_seconds: float
    started_monotonic_ns: int = field(default_factory=time.monotonic_ns)
    last_event_monotonic_ns: dict[str, int] = field(default_factory=dict)
    writer_healthy: bool = True
    queue_saturated: bool = False
    stopping: bool = False
    last_error: str | None = None

    def mark_event(self, route: str) -> None:
        self.last_event_monotonic_ns[route] = time.monotonic_ns()

    def live(self) -> bool:
        return not self.stopping

    def ready(self) -> bool:
        if self.stopping or not self.writer_healthy or self.queue_saturated:
            return False
        now = time.monotonic_ns()
        for route in self.required_routes:
            last = self.last_event_monotonic_ns.get(route)
            if last is None or (now - last) / 1_000_000_000 > self.stale_after_seconds:
                return False
        return True


class HealthServer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        health: RecorderHealthState,
        metrics: RecorderMetrics,
    ) -> None:
        self._host = host
        self._port = port
        self._health = health
        self._metrics = metrics
        self._runner: web.AppRunner | None = None
        self.bound_port: int | None = None

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/health/live", self._live)
        app.router.add_get("/health/ready", self._ready)
        app.router.add_get("/metrics", self._prometheus)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        server = cast(Any, site._server)
        if server is not None and server.sockets:
            self.bound_port = int(server.sockets[0].getsockname()[1])

    async def close(self) -> None:
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _live(self, _request: web.Request) -> web.Response:
        live = self._health.live()
        return web.json_response({"live": live}, status=200 if live else 503)

    async def _ready(self, _request: web.Request) -> web.Response:
        ready = self._health.ready()
        return web.json_response(
            {"ready": ready, "error": self._health.last_error},
            status=200 if ready else 503,
        )

    async def _prometheus(self, _request: web.Request) -> web.Response:
        return web.Response(
            body=generate_latest(self._metrics.registry),
            content_type="text/plain",
            charset="utf-8",
        )
