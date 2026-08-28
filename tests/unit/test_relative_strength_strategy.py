from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext
from binance_algo.research.panel import PanelData
from binance_algo.research.strategies.registry import build_strategy
from binance_algo.research.strategies.relative_strength import RELATIVE_STRENGTH_FEATURES

HOUR_MS = 3_600_000


def test_relative_strength_is_causal_fixed_and_cross_sectional() -> None:
    periods = 100
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    times = 1_700_000_000_000 + np.arange(periods, dtype=np.int64) * HOUR_MS
    returns = np.tile(np.asarray([0.002, 0.0, -0.002]), (periods, 1))
    frame = pl.DataFrame(
        {
            "decision_time_ms": np.repeat(times, len(symbols)),
            "symbol": np.tile(np.asarray(symbols), periods),
            "log_return_1h": returns.reshape(-1),
        }
    )
    panel = PanelData.from_frame(frame, feature_columns=RELATIVE_STRENGTH_FEATURES)
    context = FoldContext(
        fold=1,
        train_start_ms=int(times[0]),
        train_end_ms=int(times[71]),
        test_start_ms=int(times[73]),
        test_end_ms=int(times[-1]),
        embargo_bars=1,
        random_seed=42,
    )
    strategy = build_strategy("relative_strength", "v1", {"lookback_hours": 48})

    scores = (
        strategy.fit_panel(panel, target=None, context=context)
        .score_panel(panel, context=context)
        .frame.filter(pl.col("decision_time_ms") == context.test_start_ms)
        .sort("symbol")
    )

    assert strategy.required_features() == RELATIVE_STRENGTH_FEATURES
    assert strategy.target_column() is None
    assert scores.filter(pl.col("symbol") == "BTCUSDT")["score"].item() > 0
    assert scores.filter(pl.col("symbol") == "ETHUSDT")["score"].item() == 0
    assert scores.filter(pl.col("symbol") == "SOLUSDT")["score"].item() < 0


def test_relative_strength_factory_rejects_invalid_or_extra_parameters() -> None:
    with pytest.raises(ResearchError, match="greater than or equal to 24"):
        build_strategy("relative_strength", "1", {"lookback_hours": 12})
    with pytest.raises(ResearchError, match="extra_forbidden"):
        build_strategy(
            "relative_strength",
            "1",
            {"lookback_hours": 168, "future_return": True},
        )
