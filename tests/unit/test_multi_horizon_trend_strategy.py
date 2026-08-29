from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext
from binance_algo.research.panel import PanelData
from binance_algo.research.portfolio.registry import build_portfolio_policy
from binance_algo.research.strategies.multi_horizon_trend import (
    MULTI_HORIZON_TREND_FEATURES,
)
from binance_algo.research.strategies.registry import build_strategy

HOUR_MS = 3_600_000


def _panel() -> tuple[PanelData, FoldContext]:
    periods = 900
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    times = 1_700_000_000_000 + np.arange(periods, dtype=np.int64) * HOUR_MS
    returns = np.tile(np.asarray([0.001, 0.0, -0.001]), periods)
    frame = pl.DataFrame(
        {
            "decision_time_ms": np.repeat(times, len(symbols)),
            "symbol": np.tile(np.asarray(symbols), periods),
            "log_return_1h": returns,
            "realized_volatility_24h": np.tile(np.asarray([0.01, 0.012, 0.014]), periods),
        }
    )
    context = FoldContext(
        fold=1,
        train_start_ms=int(times[0]),
        train_end_ms=int(times[749]),
        test_start_ms=int(times[751]),
        test_end_ms=int(times[-1]),
        embargo_bars=1,
        random_seed=42,
    )
    return (
        PanelData.from_frame(
            frame,
            feature_columns=(*MULTI_HORIZON_TREND_FEATURES, "realized_volatility_24h"),
        ),
        context,
    )


def test_multi_horizon_trend_votes_causally_and_long_flat_ignores_bearish_assets() -> None:
    panel, context = _panel()
    strategy = build_strategy(
        "multi_horizon_trend",
        "v1",
        {
            "short_lookback_hours": 72,
            "medium_lookback_hours": 336,
            "long_lookback_hours": 720,
            "short_weight": 0.2,
            "medium_weight": 0.3,
            "long_weight": 0.5,
        },
    )
    scores = strategy.fit_panel(panel, target=None, context=context).score_panel(
        panel, context=context
    )
    policy = build_portfolio_policy(
        "buffered_long_flat",
        "1",
        {
            "signal_threshold": 0.0,
            "rebalance_interval_hours": 72,
            "gross_exposure": 0.5,
            "annual_volatility_target": 0.15,
            "max_symbol_weight": 0.25,
        },
    )
    first_scores = scores.frame.filter(pl.col("decision_time_ms") == context.test_start_ms).sort(
        "symbol"
    )
    weights = policy.target_weights_panel(scores, panel, context=context).sort(
        "decision_time_ms", "symbol"
    )
    first_weights = weights.filter(pl.col("decision_time_ms") == context.test_start_ms).sort(
        "symbol"
    )

    assert strategy.required_features() == MULTI_HORIZON_TREND_FEATURES
    assert first_scores["score"].to_list() == [1.0, 0.0, -1.0]
    assert first_weights["target_weight"].to_list() == [0.25, 0.0, 0.0]
    assert weights["target_weight"].min() >= 0
    assert weights.filter(pl.col("symbol") != "BTCUSDT")["target_weight"].max() == 0


def test_multi_horizon_trend_factories_reject_non_convex_or_extra_parameters() -> None:
    with pytest.raises(ResearchError, match="sum to one"):
        build_strategy(
            "multi_horizon_trend",
            "1",
            {
                "short_lookback_hours": 72,
                "medium_lookback_hours": 336,
                "long_lookback_hours": 720,
                "short_weight": 0.4,
                "medium_weight": 0.4,
                "long_weight": 0.4,
            },
        )
    with pytest.raises(ResearchError, match="extra_forbidden"):
        build_portfolio_policy(
            "buffered_long_flat",
            "1",
            {
                "signal_threshold": 0.0,
                "rebalance_interval_hours": 72,
                "gross_exposure": 0.5,
                "annual_volatility_target": 0.15,
                "max_symbol_weight": 0.25,
                "allow_short": True,
            },
        )
