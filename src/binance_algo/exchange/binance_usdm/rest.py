"""Minimal, lifecycle-explicit public REST client for USD-M Futures."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Protocol

import aiohttp
import orjson

from binance_algo import __version__
from binance_algo.exchange.binance_usdm import endpoints
from binance_algo.exchange.binance_usdm.errors import (
    BinanceAPIError,
    BinanceNetworkError,
    BinanceResponseError,
)
from binance_algo.exchange.binance_usdm.models import FundingRatePayload

Sleep = Callable[[float], Awaitable[None]]
Jitter = Callable[[float], float]


class RestMetrics(Protocol):
    def observe_rest_response(
        self,
        *,
        path: str,
        status: int,
        latency_seconds: float,
        rate_limits: dict[str, str],
    ) -> None: ...

    def observe_rest_error(self, *, path: str, error_type: str, latency_seconds: float) -> None: ...


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
        metrics: RestMetrics | None = None,
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
        self._metrics = metrics

    @property
    def last_rate_limits(self) -> Mapping[str, str]:
        return dict(self._last_rate_limits)

    async def __aenter__(self) -> BinanceUSDMRestClient:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": f"binance-algo/{__version__} public-data"}
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

    async def funding_rate_history(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1_000,
    ) -> list[FundingRatePayload]:
        """Return public funding events in chronological order without credentials."""

        if not symbol or symbol != symbol.upper():
            raise ValueError("symbol must be a non-empty uppercase Binance symbol")
        if end_time_ms < start_time_ms:
            raise ValueError("end_time_ms must not precede start_time_ms")
        if not 1 <= limit <= 1_000:
            raise ValueError("funding history limit must be between 1 and 1000")
        payload = await self._get_json_value(
            endpoints.FUNDING_RATE_HISTORY,
            params={
                "symbol": symbol,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
                "limit": limit,
            },
        )
        if not isinstance(payload, list):
            raise BinanceResponseError("fundingRate response must be a JSON array")
        try:
            return [FundingRatePayload.model_validate(item) for item in payload]
        except (TypeError, ValueError) as exc:
            raise BinanceResponseError(f"incompatible fundingRate response: {exc}") from exc

    async def _get_json(self, path: str) -> dict[str, Any]:
        payload = await self._get_json_value(path)
        if not isinstance(payload, dict):
            raise BinanceResponseError(f"JSON object expected from {path}")
        return {str(key): item for key, item in payload.items()}

    async def _get_json_value(
        self, path: str, *, params: Mapping[str, str | int] | None = None
    ) -> Any:
        if self._session is None:
            raise RuntimeError("REST client must be used as an async context manager")

        last_network_error: BaseException | None = None
        for attempt in range(1, self._max_attempts + 1):
            attempt_started_ns = time.monotonic_ns()
            try:
                async with self._session.get(
                    f"{self._base_url}{path}", params=params, timeout=self._timeout
                ) as response:
                    self._capture_rate_limits(response.headers)
                    body = await response.read()
                    if self._metrics is not None:
                        self._metrics.observe_rest_response(
                            path=path,
                            status=response.status,
                            latency_seconds=(time.monotonic_ns() - attempt_started_ns)
                            / 1_000_000_000,
                            rate_limits=self._last_rate_limits,
                        )
                    payload = self._decode_payload(body, path)
                    if (
                        response.status == 429 or 500 <= response.status < 600
                    ) and attempt < self._max_attempts:
                        await self._sleep(self._retry_delay(attempt, response.headers))
                        continue
                    if response.status >= 400:
                        error_payload = payload if isinstance(payload, dict) else {}
                        code = error_payload.get("code")
                        message = error_payload.get("msg", "unknown Binance error")
                        raise BinanceAPIError(
                            status=response.status,
                            code=code if isinstance(code, int) else None,
                            message=str(message),
                            path=path,
                        )
                    return payload
            except (aiohttp.ClientError, TimeoutError) as exc:
                if self._metrics is not None:
                    self._metrics.observe_rest_error(
                        path=path,
                        error_type=type(exc).__name__,
                        latency_seconds=(time.monotonic_ns() - attempt_started_ns) / 1_000_000_000,
                    )
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
    def _decode_payload(body: bytes, path: str) -> Any:
        try:
            value = orjson.loads(body) if body else {}
        except orjson.JSONDecodeError as exc:
            raise BinanceResponseError(f"invalid JSON returned by {path}") from exc
        if not isinstance(value, (dict, list)):
            raise BinanceResponseError(f"JSON object or array expected from {path}")
        return value

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
