import pytest

from binance_algo.config import load_settings
from binance_algo.exchange.binance_usdm.rest import BinanceUSDMRestClient


@pytest.mark.network
async def test_demo_public_rest_contract() -> None:
    settings = load_settings()
    async with BinanceUSDMRestClient(
        base_url=settings.binance.rest_base_url,
        timeout_seconds=settings.binance.request_timeout_seconds,
    ) as client:
        await client.ping()
        assert await client.server_time() > 0
        exchange_info = await client.exchange_info()
        symbols = {item["symbol"] for item in exchange_info["symbols"]}
        assert {"BTCUSDT", "ETHUSDT", "SOLUSDT"} <= symbols
