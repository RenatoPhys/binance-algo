from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl

from binance_algo.config import load_settings
from binance_algo.research.dataset import MINUTE_MS, build_point_in_time_frame
from binance_algo.research.datasets.fingerprints import logical_content_checksum
from binance_algo.research.features.registry import phase3_feature_plan

PROJECT_ROOT = Path(__file__).parents[2]
BASE_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
START_MS = 1_767_225_600_000  # 2026-01-01T00:00:00Z


def _inputs(days: int = 4) -> tuple[pl.DataFrame, pl.DataFrame]:
    periods = days * 1_440
    rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(SYMBOLS):
        base = (50_000.0, 3_000.0, 150.0)[symbol_index]
        for index in range(periods):
            common = 0.00002 * index + 0.003 * math.sin(index / 73)
            residual = (symbol_index - 1) * 0.000004 * index
            close = base * math.exp(common + residual)
            open_price = close * (1 - 0.00005 * math.sin(index / 11))
            rows.append(
                {
                    "symbol": symbol,
                    "open_time_ms": START_MS + index * MINUTE_MS,
                    "close_time_ms": START_MS + (index + 1) * MINUTE_MS - 1,
                    "open": open_price,
                    "high": max(open_price, close) * 1.0005,
                    "low": min(open_price, close) * 0.9995,
                    "close": close,
                    "quote_volume": 1_000_000.0 + symbol_index * 100_000 + index * 10,
                    "taker_buy_quote_volume": 510_000.0 + symbol_index * 50_000 + index * 5,
                    "is_closed": True,
                }
            )
    funding_rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(SYMBOLS):
        for hour in range(32, days * 24, 8):
            funding_rows.append(
                {
                    "symbol": symbol,
                    "funding_time_ms": START_MS + hour * 3_600_000,
                    "funding_rate": 0.0001 * (symbol_index + 1),
                    "rate_type": "Regular",
                    "mark_price": 1.0,
                }
            )
    return pl.DataFrame(rows), pl.DataFrame(funding_rows)


def test_point_in_time_features_ignore_future_shock_and_labels_start_next_open() -> None:
    settings = load_settings(BASE_CONFIG)
    config = settings.research.model_copy(update={"beta_window_hours": 24})
    klines, funding = _inputs()
    baseline, audit, _, _ = build_point_in_time_frame(
        klines=klines, funding=funding, symbols=SYMBOLS, config=config
    )

    shock_start = START_MS + (4 * 1_440 - 180) * MINUTE_MS
    shocked = klines.with_columns(
        pl.when(pl.col("open_time_ms") >= shock_start)
        .then(pl.col("close") * 3)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    changed, changed_audit, _, _ = build_point_in_time_frame(
        klines=shocked, funding=funding, symbols=SYMBOLS, config=config
    )
    feature_columns = [
        "decision_time_ms",
        "symbol",
        "log_return_5m",
        "log_return_15m",
        "log_return_1h",
        "log_return_4h",
        "log_return_24h",
        "rolling_beta",
        "residual_momentum_24h",
        "funding_rate_current",
    ]
    before_shock = pl.col("decision_time_ms") < shock_start
    assert (
        baseline.filter(before_shock)
        .select(feature_columns)
        .equals(changed.filter(before_shock).select(feature_columns), null_equal=True)
    )
    assert audit.passed and changed_audit.passed
    assert baseline["execution_time_ms"].min() > baseline["decision_time_ms"].min()
    assert baseline["label_end_time_ms"].min() > baseline["execution_time_ms"].min()

    row = baseline.row(0, named=True)
    source = klines.filter(pl.col("symbol") == row["symbol"])
    entry = source.filter(pl.col("open_time_ms") == row["execution_time_ms"])["open"].item()
    exit_price = source.filter(pl.col("open_time_ms") == row["label_end_time_ms"])["open"].item()
    assert math.isclose(row["future_return_1h"], exit_price / entry - 1, abs_tol=1e-14)
    assert baseline["decision_time_ms"].min() >= START_MS + 40 * 3_600_000
    assert baseline.shape == (72, 33)
    assert (
        logical_content_checksum(baseline)
        == "6dab21ac8fc1b0a4fb5e46a1554d97d1057f61a1986b8f09ed14877980760839"
    )
    plan = phase3_feature_plan(config)
    assert tuple(bundle.bundle.bundle_id for bundle in plan.bundles) == (
        "returns_momentum",
        "volatility",
        "volume",
        "microstructure",
        "funding",
    )
    assert plan.feature_set.canonical_checksum.startswith("62e0aa63b7e61f92")


def test_funding_event_at_entry_is_charged_to_previous_position() -> None:
    settings = load_settings(BASE_CONFIG)
    config = settings.research.model_copy(update={"beta_window_hours": 24})
    klines, funding = _inputs()
    frame, _, _, _ = build_point_in_time_frame(
        klines=klines, funding=funding, symbols=SYMBOLS, config=config
    )
    event_time = int(funding["funding_time_ms"][3])
    at_entry = frame.filter(pl.col("execution_time_ms") == event_time)
    before_exit = frame.filter(pl.col("label_end_time_ms") == event_time)
    if not at_entry.is_empty():
        assert np.allclose(at_entry["outcome_funding_rate_1h"].to_numpy(), 0.0)
    if not before_exit.is_empty():
        assert np.any(before_exit["outcome_funding_rate_1h"].to_numpy() != 0.0)
