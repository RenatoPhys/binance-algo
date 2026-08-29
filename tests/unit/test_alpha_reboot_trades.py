from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from binance_algo.research.backtest import funding_pnl_contributions
from binance_algo.research.trades import build_trade_metrics, reconstruct_trade_events


def test_trade_reconstruction_counts_entry_hold_and_fold_close_once() -> None:
    positions = pl.DataFrame(
        {
            "fold": [1, 1, 1],
            "symbol": ["BTCUSDT"] * 3,
            "execution_time_ms": [1_000, 3_601_000, 7_201_000],
            "label_end_time_ms": [3_601_000, 7_201_000, 10_801_000],
            "previous_weight": [0.0, 0.1, 0.1],
            "target_weight": [0.1, 0.1, 0.1],
            "trade_weight": [0.1, 0.0, 0.1],
            "price_pnl": [0.001, -0.0005, 0.002],
            "funding_pnl": [0.0, 0.0001, 0.0],
            "allocated_fee": [0.0004, 0.0, 0.0004],
            "allocated_spread_cost": [0.0001, 0.0, 0.0001],
            "allocated_slippage_cost": [0.0001, 0.0, 0.0001],
        }
    )
    events = reconstruct_trade_events(positions, strategy_id="fixture")
    metrics = build_trade_metrics(events, positions)
    overall = metrics.filter(pl.col("scope") == "overall").row(0, named=True)

    assert events.height == 1
    assert events["exit_reason"][0] == "fold_close"
    assert events["holding_hours"][0] == 3.0
    assert events["explicit_cost"][0] == pytest.approx(0.0012)
    assert overall["completed_trades"] == 1
    assert overall["entries"] == 1
    assert overall["exits"] == 1
    assert overall["turnover"] == 0.2


def test_funding_sign_is_correct_for_long_and_short_legs() -> None:
    contributions = funding_pnl_contributions(np.asarray([0.20, -0.30]), np.asarray([0.01, 0.01]))

    assert contributions.tolist() == pytest.approx([-0.002, 0.003])


def test_trade_reconstruction_allocates_flip_and_fold_close_cost_once() -> None:
    positions = pl.DataFrame(
        {
            "fold": [1, 1],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "execution_time_ms": [1_000, 3_601_000],
            "label_end_time_ms": [3_601_000, 7_201_000],
            "previous_weight": [0.0, 0.1],
            "target_weight": [0.1, -0.2],
            "trade_weight": [0.1, 0.5],
            "price_pnl": [0.001, 0.002],
            "funding_pnl": [-0.0001, 0.0002],
            "allocated_fee": [0.0001, 0.0005],
            "allocated_spread_cost": [0.0, 0.0],
            "allocated_slippage_cost": [0.0, 0.0],
        }
    )
    events = reconstruct_trade_events(positions, strategy_id="fixture").sort("entry_time_ms")
    metrics = build_trade_metrics(events, positions)
    overall = metrics.filter(pl.col("scope") == "overall").row(0, named=True)

    assert events.height == 2
    assert events["exit_reason"].to_list() == ["direction_flip", "fold_close"]
    assert events["explicit_cost"].to_list() == pytest.approx([0.0002, 0.0004])
    assert overall["direction_flips"] == 1
    assert overall["turnover"] == pytest.approx(0.6)
    assert overall["explicit_cost"] == pytest.approx(0.0006)
