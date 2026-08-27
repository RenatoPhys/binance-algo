"""Minimal, lifecycle-explicit public REST client for USD-M Futures."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import aiohttp
import orjson

from binance_algo.exchange.binance_usdm import endpoints
from binance_algo.exchange.binance_usdm.errors import (
    BinanceAPIError,
    BinanceNetworkError,
    BinanceResponseError,
)

Sleep = Callable[[float], Awaitable[None]]
Jitter = Callable[[float], float]


class BinanceUSDMRestClient:
    """Public-only client with bounded retries for idempotent GET requests."""

    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: float = 10.0,
        max_attempts: int = 3,
        retry_base_seconds: float = 0.25,
        session: aiohttp.ClientSession | None = None,
        sleep: Sleep = asyncio.sleep,
        jitter: Jitter | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._max_attempts = max_attempts
        self._retry_base_seconds: float = retry_base_seconds
        self._session = session
        self._owns_session = session is None
        self._sleep = sleep
        self._jitter: Jitter = jitter or (lambda ceiling: random.uniform(0.0, ceiling))
        self._last_rate_limits: dict[str, str] = {}

    @property
    def last_rate_limits(self) -> Mapping[str, str]:
        return dict(self._last_rate_limits)

    async def __aenter__(self) -> BinanceUSDMRestClient:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": "binance-algo/0.1 public-data"}
            )
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()

    async def ping(self) -> None:
        payload = await self._get_json(endpoints.PING)
        if payload != {}:
            raise BinanceResponseError(f"unexpected ping payload: {payload!r}")

    async def server_time(self) -> int:
        payload = await self._get_json(endpoints.SERVER_TIME)
        server_time = payload.get("serverTime")
        if not isinstance(server_time, int):
            raise BinanceResponseError("server time response is missing integer serverTime")
        return server_time

    async def exchange_info(self) -> dict[str, Any]:
        payload = await self._get_json(endpoints.EXCHANGE_INFO)
        if not isinstance(payload.get("symbols"), list) or not isinstance(
            payload.get("serverTime"), int
        ):
            raise BinanceResponseError(
                "exchangeInfo response must contain integer serverTime and symbols list"
            )
        return payload

    async def _get_json(self, path: str) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("REST client must be used as an async context manager")

        last_network_error: BaseException | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                async with self._session.get(
                    f"{self._base_url}{path}", timeout=self._timeout
                ) as response:
                    self._capture_rate_limits(response.headers)
                    body = await response.read()
                    payload = self._decode_payload(body, path)
                    if (
                        response.status == 429 or 500 <= response.status < 600
                    ) and attempt < self._max_attempts:
                        await self._sleep(self._retry_delay(attempt, response.headers))
                        continue
                    if response.status >= 400:
                        code = payload.get("code")
                        message = payload.get("msg", "unknown Binance error")
                        raise BinanceAPIError(
                            status=response.status,
                            code=code if isinstance(code, int) else None,
                            message=str(message),
                            path=path,
                        )
                    return payload
            except (aiohttp.ClientError, TimeoutError) as exc:
                last_network_error = exc
                if attempt < self._max_attempts:
                    await self._sleep(self._retry_delay(attempt, {}))
                    continue
                break

        detail = f": {last_network_error}" if last_network_error else ""
        raise BinanceNetworkError(
            f"public GET {path} failed after {self._max_attempts} attempts{detail}"
        ) from last_network_error

    @staticmethod
    def _decode_payload(body: bytes, path: str) -> dict[str, Any]:
        try:
            value = orjson.loads(body) if body else {}
        except orjson.JSONDecodeError as exc:
            raise BinanceResponseError(f"invalid JSON returned by {path}") from exc
        if not isinstance(value, dict):
            raise BinanceResponseError(f"JSON object expected from {path}")
        return {str(key): item for key, item in value.items()}

    def _retry_delay(self, attempt: int, headers: Mapping[str, str]) -> float:
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after is not None:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
        exponential: float = self._retry_base_seconds * (2 ** (attempt - 1))
        jitter = float(self._jitter(self._retry_base_seconds))
        delay: float = exponential + jitter
        return delay if delay < 60.0 else 60.0

    def _capture_rate_limits(self, headers: Mapping[str, str]) -> None:
        self._last_rate_limits = {
            name.lower(): value
            for name, value in headers.items()
            if name.lower().startswith("x-mbx-used-weight")
            or name.lower().startswith("x-mbx-order-count")
        }
