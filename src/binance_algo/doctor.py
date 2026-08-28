"""Environment diagnostics for the public-data milestone."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from dataclasses import dataclass
from urllib.parse import urlparse

from binance_algo.clock import ExchangeClock
from binance_algo.common.errors import StateStoreError
from binance_algo.config import Settings
from binance_algo.data.state_store import StateStore
from binance_algo.exchange.binance_usdm.errors import BinanceRestError
from binance_algo.exchange.binance_usdm.rest import BinanceUSDMRestClient


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


async def run_doctor(settings: Settings) -> list[CheckResult]:
    results = [
        CheckResult(
            "python",
            sys.version_info >= (3, 12),
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        ),
        CheckResult(
            "safety",
            not settings.safety.live_trading
            and not settings.safety.allow_order_submission
            and settings.safety.max_order_notional_usdt == 0,
            "live trading and order submission disabled; max order notional is zero",
        ),
        CheckResult(
            "credentials",
            True,
            "configured (unused by public commands)"
            if settings.credentials.configured
            else "not configured (not required for public commands)",
        ),
    ]

    try:
        settings.data_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=settings.data_root, prefix="doctor-", delete=True
        ) as tmp:
            tmp.write(b"ok")
            tmp.flush()
        results.append(CheckResult("storage", True, f"writable: {settings.data_root}"))
    except OSError as exc:
        results.append(CheckResult("storage", False, f"not writable: {exc}"))

    try:
        state_store = StateStore(settings.state_db_path)
        state_store.initialize()
        journal_mode = state_store.journal_mode()
        results.append(
            CheckResult(
                "state_store",
                journal_mode.lower() == "wal",
                f"SQLite journal_mode={journal_mode}: {settings.state_db_path}",
            )
        )
    except StateStoreError as exc:
        results.append(CheckResult("state_store", False, str(exc)))

    host = urlparse(settings.binance.rest_base_url).hostname
    if host is None:
        results.append(CheckResult("dns", False, "REST base URL has no hostname"))
    else:
        try:
            await asyncio.get_running_loop().getaddrinfo(host, 443)
            results.append(CheckResult("dns", True, f"resolved {host}"))
        except OSError as exc:
            results.append(CheckResult("dns", False, f"cannot resolve {host}: {exc}"))

    archive_host = urlparse(settings.archives.base_url).hostname
    if archive_host is None:
        results.append(CheckResult("archive_dns", False, "archive base URL has no hostname"))
    else:
        try:
            await asyncio.get_running_loop().getaddrinfo(archive_host, 443)
            results.append(CheckResult("archive_dns", True, f"resolved {archive_host}"))
        except OSError as exc:
            results.append(
                CheckResult("archive_dns", False, f"cannot resolve {archive_host}: {exc}")
            )

    clock = ExchangeClock()
    try:
        async with BinanceUSDMRestClient(
            base_url=settings.binance.rest_base_url,
            timeout_seconds=settings.binance.request_timeout_seconds,
            max_attempts=settings.binance.retry_max_attempts,
            retry_base_seconds=settings.binance.retry_base_seconds,
        ) as client:
            await client.ping()
            results.append(CheckResult("rest_ping", True, settings.binance.rest_base_url))
            sync = await clock.synchronize(client)
            offset_ok = abs(sync.offset_ms) <= settings.binance.clock_max_offset_ms
            results.append(
                CheckResult(
                    "clock",
                    offset_ok,
                    f"offset={sync.offset_ms}ms round_trip={sync.round_trip_ms:.1f}ms "
                    f"limit={settings.binance.clock_max_offset_ms}ms",
                )
            )
            rate_limits = client.last_rate_limits
            detail = ", ".join(f"{key}={value}" for key, value in rate_limits.items())
            results.append(CheckResult("rate_limit_headers", True, detail or "none returned"))
    except BinanceRestError as exc:
        results.append(CheckResult("rest_ping", False, str(exc)))

    return results
