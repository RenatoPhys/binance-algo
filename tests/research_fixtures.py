"""Deterministic synthetic research panel shared by Phase 3.5 tests."""

from __future__ import annotations

import math

import polars as pl

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
START_MS = 1_767_225_600_000


def research_frame(days: int = 11) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for hour in range(days * 24):
        decision = START_MS + hour * 3_600_000 + 3_599_999
        common_volatility = 0.012 + 0.002 * math.sin(hour / 24)
        for symbol_index, symbol in enumerate(SYMBOLS):
            phase = math.sin(hour / 9 + symbol_index)
            residual_1h = 0.002 * phase
            rows.append(
                {
                    "decision_time_ms": decision,
                    "execution_time_ms": decision + 1,
                    "label_end_time_ms": decision + 3_600_001,
                    "symbol": symbol,
                    "residual_momentum_1h": residual_1h,
                    "residual_momentum_4h": residual_1h * 2 + symbol_index * 0.0001,
                    "residual_momentum_24h": residual_1h * 4 - symbol_index * 0.0001,
                    "realized_volatility_24h": common_volatility * (1 + symbol_index * 0.1),
                    "rolling_beta": 0.8 + symbol_index * 0.25,
                    "future_return_1h": 0.0015 * phase - 0.0002 * symbol_index,
                    "future_residual_return_1h": 0.0012 * phase,
                    "outcome_funding_rate_1h": (
                        0.0001 * (symbol_index + 1) if hour % 8 == 7 else 0.0
                    ),
                    "outcome_quote_volume_1h": 100_000_000.0,
                    "market_volatility_regime": common_volatility * math.sqrt(365),
                }
            )
    return pl.DataFrame(rows)


__all__ = ["START_MS", "SYMBOLS", "research_frame"]
