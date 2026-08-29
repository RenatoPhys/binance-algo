from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext
from binance_algo.research.panel import PanelData
from binance_algo.research.strategies.market_regime_trend import (
    MARKET_REGIME_TREND_FEATURES,
)
from binance_algo.research.strategies.registry import build_strategy

HOUR_MS = 3_600_000


def _score(first_two_returns: tuple[float, float]) -> list[float]:
    periods = 500
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    times = 1_700_000_000_000 + np.arange(periods, dtype=np.int64) * HOUR_MS
    returns = np.tile(np.asarray([*first_two_returns, -0.004]), periods)
    panel = PanelData.from_frame(
        pl.DataFrame(
            {
                "decision_time_ms": np.repeat(times, len(symbols)),
                "symbol": np.tile(np.asarray(symbols), periods),
                "log_return_1h": returns,
            }
        ),
        feature_columns=MARKET_REGIME_TREND_FEATURES,
    )
    context = FoldContext(
        fold=1,
        train_start_ms=int(times[0]),
        train_end_ms=int(times[399]),
        test_start_ms=int(times[401]),
        test_end_ms=int(times[-1]),
        embargo_bars=1,
        random_seed=42,
    )
    strategy = build_strategy(
        "market_regime_trend",
        "v1",
        {"fast_lookback_hours": 72, "slow_lookback_hours": 336},
    )
    scores = strategy.fit_panel(panel, target=None, context=context).score_panel(
        panel, context=context
    )
    return (
        scores.frame.filter(pl.col("decision_time_ms") == context.test_start_ms)
        .sort("symbol")["score"]
        .to_list()
    )


def test_market_regime_trend_requires_individual_and_market_uptrends() -> None:
    assert _score((0.003, 0.002)) == [1.0, 1.0, 0.0]
    assert _score((0.001, 0.001)) == [0.0, 0.0, 0.0]


def test_market_regime_trend_rejects_reversed_or_extra_parameters() -> None:
    with pytest.raises(ResearchError, match="lookbacks must be increasing"):
        build_strategy(
            "market_regime_trend",
            "1",
            {"fast_lookback_hours": 336, "slow_lookback_hours": 72},
        )
    with pytest.raises(ResearchError, match="extra_forbidden"):
        build_strategy(
            "market_regime_trend",
            "1",
            {
                "fast_lookback_hours": 72,
                "slow_lookback_hours": 336,
                "future_filter": True,
            },
        )
