"""Explicit REST failure models."""

from __future__ import annotations

from binance_algo.common.errors import BinanceAlgoError


class BinanceRestError(BinanceAlgoError):
    """Base class for public REST failures."""


class BinanceAPIError(BinanceRestError):
    def __init__(self, *, status: int, code: int | None, message: str, path: str) -> None:
        super().__init__(f"Binance HTTP {status} code={code} path={path}: {message}")
        self.status = status
        self.code = code
        self.message = message
        self.path = path


class BinanceNetworkError(BinanceRestError):
    """A safe public request exhausted its bounded retry policy."""


class BinanceResponseError(BinanceRestError):
    """A successful HTTP response did not satisfy the expected data contract."""
