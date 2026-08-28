from __future__ import annotations

import asyncio
from collections import Counter
from pathlib import Path
from typing import Any

import orjson
import pytest
from aiohttp import web

from binance_algo.common.errors import RecorderError, ReplayError
from binance_algo.config import load_settings
from binance_algo.data.manifest import DataFileStatus
from binance_algo.data.recorder import run_recorder
from binance_algo.data.replay import (
    ReplayEngine,
    VirtualReplayClock,
    parse_replay_datasets,
    select_replay_files,
)
from binance_algo.data.state_store import StateStore
from binance_algo.exchange.binance_usdm.public_streams import STREAM_EVENT_TYPES


def test_replay_dataset_selection_is_strict() -> None:
    assert parse_replay_datasets("all") == STREAM_EVENT_TYPES
    assert parse_replay_datasets("book_ticker,mark_price") == (
        "book_ticker",
        "mark_price",
    )
    with pytest.raises(ReplayError, match="unsupported"):
        parse_replay_datasets("depth")
    with pytest.raises(ReplayError, match="duplicates"):
        parse_replay_datasets("book_ticker,book_ticker")


class FakeMarketStreams:
    def __init__(self) -> None:
        self.connections: Counter[str] = Counter()
        self.agg_ids: Counter[str] = Counter()
        self.update_ids: Counter[str] = Counter()
        self.event_time_ms = 1_800_000_000_000
        self.force_market_close = True
        self.connection_events = {
            "public": asyncio.Event(),
            "market": asyncio.Event(),
        }

    async def handler(self, request: web.Request) -> web.WebSocketResponse:
        route = request.match_info["route"]
        websocket = web.WebSocketResponse(autoping=True)
        await websocket.prepare(request)
        self.connections[route] += 1
        self.connection_events[route].set()
        streams = tuple(request.query.get("streams", "").split("/"))
        cycles = 0
        try:
            while not websocket.closed:
                for stream in streams:
                    self.event_time_ms += 1
                    payload = self._payload(stream)
                    await websocket.send_str(
                        orjson.dumps({"stream": stream, "data": payload}).decode("utf-8")
                    )
                cycles += 1
                if route == "market" and self.force_market_close and cycles == 1:
                    self.force_market_close = False
                    await websocket.close(code=1001, message=b"forced reconnect")
                    break
                await asyncio.sleep(0.02)
        except ConnectionResetError:
            pass
        return websocket

    def _payload(self, stream: str) -> dict[str, Any]:
        symbol = stream.split("@")[0].upper()
        if "bookTicker" in stream:
            self.update_ids[symbol] += 1
            return {
                "e": "bookTicker",
                "E": self.event_time_ms,
                "T": self.event_time_ms - 1,
                "s": symbol,
                "u": self.update_ids[symbol],
                "b": "100.0",
                "B": "2.0",
                "a": "100.1",
                "A": "3.0",
            }
        if "aggTrade" in stream:
            self.agg_ids[symbol] += 1
            aggregate_id = self.agg_ids[symbol]
            return {
                "e": "aggTrade",
                "E": self.event_time_ms,
                "T": self.event_time_ms - 1,
                "s": symbol,
                "a": aggregate_id,
                "p": "100.05",
                "q": "0.5",
                "f": aggregate_id,
                "l": aggregate_id,
                "m": False,
            }
        if "markPrice" in stream:
            return {
                "e": "markPriceUpdate",
                "E": self.event_time_ms,
                "s": symbol,
                "p": "100.02",
                "i": "100.01",
                "P": "100.03",
                "r": "0.0001",
                "T": self.event_time_ms + 1_000,
            }
        open_time = self.event_time_ms // 60_000 * 60_000
        return {
            "e": "kline",
            "E": self.event_time_ms,
            "s": symbol,
            "k": {
                "t": open_time,
                "T": open_time + 59_999,
                "i": "1m",
                "o": "100",
                "h": "101",
                "l": "99",
                "c": "100.5",
                "v": "5",
                "q": "502.5",
                "n": 10,
                "V": "2",
                "Q": "201",
                "x": False,
            },
        }


@pytest.fixture
async def fake_market_streams() -> Any:
    source = FakeMarketStreams()
    app = web.Application()
    app.router.add_get("/{route}/stream", source.handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield source, f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def _recorder_settings(tmp_path: Path, base_url: str) -> Any:
    settings = load_settings()
    return settings.model_copy(
        update={
            "binance": settings.binance.model_copy(update={"market_ws_base_url": base_url}),
            "storage": settings.storage.model_copy(
                update={
                    "root": tmp_path / "data",
                    "state_db": tmp_path / "state.sqlite3",
                    "duckdb": tmp_path / "market_data.duckdb",
                    "reports_root": tmp_path / "reports",
                    "micro_batch_max_rows": 1_000,
                    "micro_batch_max_seconds": 5.0,
                }
            ),
            "recorder": settings.recorder.model_copy(
                update={
                    "queue_capacity": 1_000,
                    "stale_after_seconds": 2.0,
                    "reconnect_base_seconds": 0.0,
                    "reconnect_stable_after_seconds": 0.1,
                    "connection_max_seconds": 60.0,
                    "shutdown_timeout_seconds": 10.0,
                    "metrics_port": 0,
                }
            ),
        }
    )


@pytest.mark.asyncio
async def test_recorder_reconnects_flushes_and_replays_deterministically(
    tmp_path: Path, fake_market_streams: tuple[FakeMarketStreams, str]
) -> None:
    source, base_url = fake_market_streams
    settings = _recorder_settings(tmp_path, base_url)
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")

    result = await run_recorder(
        settings,
        symbols=symbols,
        duration_seconds=0.4,
        clock_offset_ms=0,
        metrics_port=0,
    )

    assert result.report.passed
    assert result.report.dropped_events_total == 0
    assert result.report.messages_received_total == result.report.rows_persisted_total
    assert {item.event_type for item in result.report.datasets} == set(STREAM_EVENT_TYPES)
    assert all(item.duplicate_event_count == 0 for item in result.report.datasets)
    assert all(item.sequence_gap_count == 0 for item in result.report.datasets)
    assert source.connections["market"] >= 2
    assert result.report.reconnects_total >= 1

    state_store = StateStore(settings.state_db_path)
    state_store.initialize()
    records = select_replay_files(
        state_store,
        datasets=STREAM_EVENT_TYPES,
        symbols=symbols,
        start_time_ms=min(item.min_event_time_ms or 0 for item in result.report.datasets),
        end_time_ms=max(item.max_event_time_ms or 0 for item in result.report.datasets),
    )
    engine = ReplayEngine(
        records=records,
        start_time_ms=min(item.min_event_time_ms or 0 for item in result.report.datasets),
        end_time_ms=max(item.max_event_time_ms or 0 for item in result.report.datasets),
    )
    deterministic = engine.verify_determinism()
    one_x_clock = VirtualReplayClock()
    fast_clock = VirtualReplayClock()
    one_x = await engine.run(speed=1.0, clock=one_x_clock)
    accelerated = await engine.run(speed=100.0, clock=fast_clock)
    assert deterministic.digest == one_x.digest == accelerated.digest
    assert deterministic.event_count == result.report.rows_persisted_total
    assert one_x_clock.elapsed_seconds == pytest.approx(fast_clock.elapsed_seconds * 100)

    manifested = state_store.list_data_files(
        layer="raw",
        statuses={DataFileStatus.VALIDATED},
        ingestion_run_id=result.report.run_id,
    )
    assert len(manifested) == result.report.files_total
    assert len(state_store.list_stream_checkpoints()) == len(STREAM_EVENT_TYPES) * len(symbols)
    with state_store.transaction() as connection:
        registered_schemas = connection.execute(
            "SELECT logical_dataset FROM schema_versions WHERE schema_version = 1"
        ).fetchall()
    assert {str(row[0]) for row in registered_schemas} == set(STREAM_EVENT_TYPES)


@pytest.mark.asyncio
async def test_cancellation_drains_and_preserves_a_quality_report(
    tmp_path: Path, fake_market_streams: tuple[FakeMarketStreams, str]
) -> None:
    source, base_url = fake_market_streams
    settings = _recorder_settings(tmp_path, base_url)
    settings = settings.model_copy(
        update={
            "streams": settings.streams.model_copy(
                update={
                    "aggregate_trades": False,
                    "mark_price": False,
                    "kline_1m": False,
                }
            )
        }
    )
    task = asyncio.create_task(
        run_recorder(
            settings,
            symbols=("BTCUSDT",),
            duration_seconds=60,
            clock_offset_ms=0,
            metrics_port=0,
        )
    )
    await asyncio.wait_for(source.connection_events["public"].wait(), timeout=2)
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(RecorderError, match="preserving its report"):
        await task

    reports = list(settings.reports_root.glob("recorder_*.json"))
    assert len(reports) == 1
    report = orjson.loads(reports[0].read_bytes())
    assert report["passed"] is True
    assert report["duration_seconds"] < 2
    assert report["messages_received_total"] == report["rows_persisted_total"]
    state_store = StateStore(settings.state_db_path)
    state_store.initialize()
    assert state_store.list_data_files(layer="raw", statuses={DataFileStatus.VALIDATED})
