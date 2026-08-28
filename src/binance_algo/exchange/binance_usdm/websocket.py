"""Resilient routed Binance USD-M public WebSocket connections."""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from collections import Counter
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from typing import Literal, cast

import aiohttp

from binance_algo import __version__
from binance_algo.common.errors import (
    DataContractError,
    QueueSaturatedError,
    WebSocketStreamError,
)
from binance_algo.exchange.binance_usdm.public_streams import (
    CanonicalStreamEvent,
    StreamSubscription,
    combined_stream_url,
    parse_combined_message,
    subscription_command,
)
from binance_algo.logging import get_logger
from binance_algo.observability.metrics import RecorderHealthState, RecorderMetrics

Jitter = Callable[[float], float]


@dataclass(frozen=True, slots=True)
class DisconnectRecord:
    route: str
    connection_id: str
    opened_at_ns: int
    closed_at_ns: int
    reason: str
    stable: bool


@dataclass(slots=True)
class WebSocketRunStats:
    connections_by_route: Counter[str] = field(default_factory=Counter)
    reconnects_by_route: Counter[str] = field(default_factory=Counter)
    messages_by_stream: Counter[str] = field(default_factory=Counter)
    messages_by_symbol: Counter[str] = field(default_factory=Counter)
    sequence_gaps_by_symbol: Counter[str] = field(default_factory=Counter)
    sequence_regressions_by_symbol: Counter[str] = field(default_factory=Counter)
    disconnects: list[DisconnectRecord] = field(default_factory=list)
    dropped_events: int = 0
    queue_peak: int = 0

    def disconnect_payloads(self) -> list[dict[str, object]]:
        return [asdict(record) for record in self.disconnects]


class RoutedPublicWebSocket:
    def __init__(
        self,
        *,
        base_url: str,
        subscription: StreamSubscription,
        queue: asyncio.Queue[CanonicalStreamEvent],
        ingestion_run_id: str,
        stats: WebSocketRunStats,
        metrics: RecorderMetrics,
        health: RecorderHealthState,
        queue_put_timeout_seconds: float,
        stale_after_seconds: float,
        reconnect_base_seconds: float,
        reconnect_max_seconds: float,
        reconnect_stable_after_seconds: float,
        connection_max_seconds: float,
        jitter: Jitter | None = None,
    ) -> None:
        self.url = combined_stream_url(base_url, subscription)
        self.subscription = subscription
        self._queue = queue
        self._ingestion_run_id = ingestion_run_id
        self._stats = stats
        self._metrics = metrics
        self._health = health
        self._queue_put_timeout_seconds = queue_put_timeout_seconds
        self._stale_after_seconds = stale_after_seconds
        self._reconnect_base_seconds = reconnect_base_seconds
        self._reconnect_max_seconds = reconnect_max_seconds
        self._reconnect_stable_after_seconds = reconnect_stable_after_seconds
        self._connection_max_seconds = connection_max_seconds
        self._jitter = jitter or (lambda ceiling: random.uniform(0.0, ceiling))
        self._websocket: aiohttp.ClientWebSocketResponse | None = None
        self._request_id = 0
        self._last_aggregate_trade_id: dict[str, int] = {}

    async def subscribe(self, names: tuple[str, ...]) -> int:
        return await self._send_command("SUBSCRIBE", names)

    async def unsubscribe(self, names: tuple[str, ...]) -> int:
        return await self._send_command("UNSUBSCRIBE", names)

    async def _send_command(
        self, method: Literal["SUBSCRIBE", "UNSUBSCRIBE"], names: tuple[str, ...]
    ) -> int:
        websocket = self._websocket
        if websocket is None or websocket.closed:
            raise WebSocketStreamError("cannot change subscriptions without an open connection")
        self._request_id += 1
        await websocket.send_str(
            subscription_command(method, names, self._request_id).decode("utf-8")
        )
        return self._request_id

    async def run(self, stop_event: asyncio.Event) -> None:
        attempt = 0
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=15, sock_read=None)
        async with aiohttp.ClientSession(
            timeout=timeout,
            headers={"User-Agent": f"binance-algo/{__version__} public-stream"},
        ) as session:
            while not stop_event.is_set():
                connection_id = uuid.uuid4().hex
                opened_epoch_ns = time.time_ns()
                opened_monotonic_ns = time.monotonic_ns()
                reason = "normal shutdown"
                stable = False
                logger = get_logger(
                    operation="websocket",
                    connection_id=connection_id,
                    route=self.subscription.route,
                )
                try:
                    async with session.ws_connect(
                        self.url,
                        autoping=True,
                        autoclose=True,
                        heartbeat=None,
                        max_msg_size=2 * 1024 * 1024,
                    ) as websocket:
                        self._websocket = websocket
                        route = self.subscription.route
                        self._stats.connections_by_route[route] += 1
                        self._metrics.ws_connections.labels(route=route).inc()
                        logger.info(
                            "websocket_connected", stream_count=len(self.subscription.names)
                        )
                        await self._receive(websocket, connection_id, stop_event)
                except QueueSaturatedError:
                    reason = "bounded recorder queue saturated"
                    self._health.queue_saturated = True
                    self._health.last_error = reason
                    raise
                except asyncio.CancelledError:
                    reason = "task cancelled"
                    raise
                except (
                    aiohttp.ClientError,
                    TimeoutError,
                    DataContractError,
                    WebSocketStreamError,
                ) as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    self._health.last_error = reason
                finally:
                    self._websocket = None
                    elapsed = (time.monotonic_ns() - opened_monotonic_ns) / 1_000_000_000
                    stable = elapsed >= self._reconnect_stable_after_seconds
                    self._stats.disconnects.append(
                        DisconnectRecord(
                            route=self.subscription.route,
                            connection_id=connection_id,
                            opened_at_ns=opened_epoch_ns,
                            closed_at_ns=time.time_ns(),
                            reason=reason,
                            stable=stable,
                        )
                    )
                    log = (
                        logger.info
                        if stop_event.is_set() and reason == "normal shutdown"
                        else logger.warning
                    )
                    log(
                        "websocket_disconnected",
                        reason=reason,
                        stable=stable,
                        reconnecting=not stop_event.is_set(),
                    )
                if stop_event.is_set():
                    break
                attempt = 0 if stable else attempt + 1
                self._stats.reconnects_by_route[self.subscription.route] += 1
                self._metrics.ws_reconnects.labels(route=self.subscription.route).inc()
                ceiling = min(
                    self._reconnect_max_seconds,
                    self._reconnect_base_seconds * (2 ** max(0, attempt - 1)),
                )
                delay = self._jitter(ceiling) if ceiling else 0.0
                with suppress(TimeoutError):
                    await asyncio.wait_for(stop_event.wait(), timeout=delay)

    async def _receive(
        self,
        websocket: aiohttp.ClientWebSocketResponse,
        connection_id: str,
        stop_event: asyncio.Event,
    ) -> None:
        connected_ns = time.monotonic_ns()
        while not stop_event.is_set():
            age_seconds = (time.monotonic_ns() - connected_ns) / 1_000_000_000
            remaining = self._connection_max_seconds - age_seconds
            if remaining <= 0:
                raise WebSocketStreamError("proactive connection rotation")
            receive_timeout = min(self._stale_after_seconds, remaining)
            try:
                message = await websocket.receive(timeout=receive_timeout)
            except TimeoutError as exc:
                raise WebSocketStreamError(
                    f"stream stale for {self._stale_after_seconds:.1f}s"
                ) from exc
            if message.type is aiohttp.WSMsgType.TEXT:
                received_ns = time.time_ns()
                event = parse_combined_message(
                    message.data,
                    received_time_ns=received_ns,
                    connection_id=connection_id,
                    ingestion_run_id=self._ingestion_run_id,
                )
                if event is None:
                    continue
                await self._enqueue(event)
                self._observe(event)
                continue
            if message.type in {aiohttp.WSMsgType.PING, aiohttp.WSMsgType.PONG}:
                continue
            if message.type in {
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSING,
            }:
                raise WebSocketStreamError(
                    f"remote close code={websocket.close_code} data={message.data!r}"
                )
            if message.type is aiohttp.WSMsgType.ERROR:
                raise WebSocketStreamError(f"WebSocket receive error: {websocket.exception()}")
            raise WebSocketStreamError(f"unsupported WebSocket frame type: {message.type.name}")

    async def _enqueue(self, event: CanonicalStreamEvent) -> None:
        try:
            await asyncio.wait_for(self._queue.put(event), timeout=self._queue_put_timeout_seconds)
        except TimeoutError as exc:
            self._stats.dropped_events += 1
            self._metrics.dropped_events.inc()
            raise QueueSaturatedError(
                f"queue remained full for {self._queue_put_timeout_seconds:.3f}s"
            ) from exc
        queue_size = self._queue.qsize()
        self._stats.queue_peak = max(self._stats.queue_peak, queue_size)
        self._metrics.queue_size.set(queue_size)

    def _observe(self, event: CanonicalStreamEvent) -> None:
        route = self.subscription.route
        self._stats.messages_by_stream[event.event_type] += 1
        self._stats.messages_by_symbol[event.symbol] += 1
        self._metrics.ws_messages.labels(
            route=route, event_type=event.event_type, symbol=event.symbol
        ).inc()
        self._health.mark_event(route)
        self._metrics.ws_last_event_age.labels(route=route).set(0)
        if event.event_type != "aggregate_trades":
            return
        current = cast(int, event.row["aggregate_trade_id"])
        previous = self._last_aggregate_trade_id.get(event.symbol)
        if previous is not None:
            if current > previous + 1:
                gap = current - previous - 1
                self._stats.sequence_gaps_by_symbol[event.symbol] += gap
                self._metrics.ws_sequence_gaps.labels(symbol=event.symbol).inc(gap)
            elif current <= previous:
                self._stats.sequence_regressions_by_symbol[event.symbol] += 1
        self._last_aggregate_trade_id[event.symbol] = current
