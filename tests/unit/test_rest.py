from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from aiohttp import web
from prometheus_client import generate_latest

from binance_algo.exchange.binance_usdm.errors import BinanceAPIError, BinanceResponseError
from binance_algo.exchange.binance_usdm.rest import BinanceUSDMRestClient
from binance_algo.observability.metrics import RecorderMetrics


@asynccontextmanager
async def fake_binance(handler: Any) -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_get("/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    server = site._server
    assert server is not None
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


async def test_public_endpoints_and_rate_limit_headers() -> None:
    async def handler(request: web.Request) -> web.Response:
        if request.path == "/fapi/v1/ping":
            return web.json_response({}, headers={"X-MBX-USED-WEIGHT-1M": "1"})
        if request.path == "/fapi/v1/time":
            return web.json_response({"serverTime": 1_787_873_855_926})
        if request.path == "/fapi/v1/exchangeInfo":
            return web.json_response({"serverTime": 1_787_873_855_926, "symbols": []})
        raise web.HTTPNotFound()

    metrics = RecorderMetrics(queue_capacity=1)
    async with (
        fake_binance(handler) as base_url,
        BinanceUSDMRestClient(base_url=base_url, metrics=metrics) as client,
    ):
        await client.ping()
        assert client.last_rate_limits == {"x-mbx-used-weight-1m": "1"}
        assert await client.server_time() == 1_787_873_855_926
        assert (await client.exchange_info())["symbols"] == []

    samples = generate_latest(metrics.registry).decode("utf-8")
    assert 'binance_rest_requests_total{path="/fapi/v1/ping",status="200"} 1.0' in samples
    assert 'binance_rate_limit_used{header="x-mbx-used-weight-1m"} 1.0' in samples


async def test_retries_safe_get_on_server_error() -> None:
    delays: list[float] = []
    calls = 0

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    async def handler(_request: web.Request) -> web.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return web.json_response({"code": -1000, "msg": "Internal error"}, status=500)
        return web.json_response({"serverTime": 123})

    async with (
        fake_binance(handler) as base_url,
        BinanceUSDMRestClient(
            base_url=base_url,
            max_attempts=2,
            retry_base_seconds=0.25,
            sleep=record_sleep,
            jitter=lambda _ceiling: 0,
        ) as client,
    ):
        assert await client.server_time() == 123

    assert calls == 2
    assert delays == [0.25]


async def test_does_not_retry_non_rate_limit_client_error() -> None:
    calls = 0

    async def handler(_request: web.Request) -> web.Response:
        nonlocal calls
        calls += 1
        return web.json_response({"code": -1121, "msg": "Invalid symbol."}, status=400)

    async with (
        fake_binance(handler) as base_url,
        BinanceUSDMRestClient(base_url=base_url) as client,
    ):
        with pytest.raises(BinanceAPIError, match="Invalid symbol"):
            await client.server_time()
    assert calls == 1


async def test_rejects_invalid_success_payload() -> None:
    async def handler(_request: web.Request) -> web.Response:
        return web.json_response({"serverTime": "not-an-int"})

    async with (
        fake_binance(handler) as base_url,
        BinanceUSDMRestClient(base_url=base_url) as client,
    ):
        with pytest.raises(BinanceResponseError, match="serverTime"):
            await client.server_time()
