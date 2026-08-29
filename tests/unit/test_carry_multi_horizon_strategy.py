from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext, StrategyScores
from binance_algo.research.panel import PanelData
from binance_algo.research.portfolio.neutral_long_short import (
    BufferedNeutralLongShortParameters,
    BufferedNeutralLongShortPolicy,
)
from binance_algo.research.portfolio.registry import build_portfolio_policy
from binance_algo.research.strategies.carry_dual_trend import CARRY_DUAL_TREND_FEATURES
from binance_algo.research.strategies.carry_multi_horizon import (
    CARRY_MULTI_HORIZON_FEATURES,
)
from binance_algo.research.strategies.registry import build_strategy

HOUR_MS = 3_600_000


def _panel() -> tuple[PanelData, FoldContext]:
    periods = 260
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    times = 1_700_000_000_000 + np.arange(periods, dtype=np.int64) * HOUR_MS
    symbol_pattern = np.tile(np.asarray([-1.0, 0.0, 1.0]), periods)
    returns = np.tile(np.asarray([0.002, 0.0, -0.002]), periods)
    frame = pl.DataFrame(
        {
            "decision_time_ms": np.repeat(times, len(symbols)),
            "symbol": np.tile(np.asarray(symbols), periods),
            "funding_rate_current": 0.0001 * symbol_pattern,
            "funding_rate_change": 0.00001 * symbol_pattern,
            "residual_momentum_4h": -0.001 * symbol_pattern,
            "log_return_1h": returns,
            "rolling_beta": np.tile(np.asarray([0.8, 1.0, 1.2]), periods),
            "realized_volatility_24h": np.tile(np.asarray([0.01, 0.012, 0.014]), periods),
        }
    )
    context = FoldContext(
        fold=1,
        train_start_ms=int(times[0]),
        train_end_ms=int(times[199]),
        test_start_ms=int(times[201]),
        test_end_ms=int(times[-1]),
        embargo_bars=1,
        random_seed=42,
    )
    return (
        PanelData.from_frame(
            frame,
            feature_columns=(
                *CARRY_MULTI_HORIZON_FEATURES,
                "rolling_beta",
                "realized_volatility_24h",
            ),
        ),
        context,
    )


def _sleeve_policy() -> BufferedNeutralLongShortPolicy:
    return BufferedNeutralLongShortPolicy(
        BufferedNeutralLongShortParameters(
            no_trade_score_band=1.0,
            rebalance_interval_hours=24,
            gross_exposure=0.5,
            annual_volatility_target=0.15,
            max_symbol_weight=0.25,
        )
    )


def test_three_sleeve_policy_combines_causal_target_weights() -> None:
    panel, context = _panel()
    strategy = build_strategy(
        "carry_multi_horizon",
        "v1",
        {
            "funding_rate_weight": 1.0,
            "funding_change_weight": 0.0,
            "momentum_confirmation_weight": 0.25,
            "fast_lookback_hours": 48,
            "slow_lookback_hours": 96,
        },
    )
    scores = strategy.fit_panel(panel, target=None, context=context).score_panel(
        panel, context=context
    )
    policy = build_portfolio_policy(
        "buffered_three_sleeve_neutral",
        "1",
        {
            "carry_weight": 0.6,
            "fast_strength_weight": 0.25,
            "slow_strength_weight": 0.15,
            "no_trade_score_band": 1.0,
            "rebalance_interval_hours": 24,
            "gross_exposure": 0.5,
            "annual_volatility_target": 0.15,
            "max_symbol_weight": 0.25,
        },
    )

    combined = policy.target_weights_panel(scores, panel, context=context).sort(
        "decision_time_ms", "symbol"
    )
    expected = np.zeros(combined.height, dtype=np.float64)
    for column, weight in (
        ("carry_score", 0.6),
        ("fast_strength_score", 0.25),
        ("slow_strength_score", 0.15),
    ):
        sleeve_scores = StrategyScores(
            scores.frame.select(
                "decision_time_ms",
                "symbol",
                pl.col(column).alias("score"),
            )
        )
        sleeve = (
            _sleeve_policy()
            .target_weights_panel(sleeve_scores, panel, context=context)
            .sort("decision_time_ms", "symbol")
        )
        expected += weight * sleeve["target_weight"].to_numpy()

    assert strategy.required_features() == CARRY_MULTI_HORIZON_FEATURES
    assert strategy.target_column() is None
    assert {
        "carry_score",
        "fast_strength_score",
        "slow_strength_score",
    }.issubset(scores.frame.columns)
    assert np.allclose(combined["target_weight"].to_numpy(), expected)


def test_multi_horizon_factories_reject_invalid_parameters() -> None:
    with pytest.raises(ResearchError, match="shorter than slow"):
        build_strategy(
            "carry_multi_horizon",
            "1",
            {
                "funding_rate_weight": 1.0,
                "funding_change_weight": 0.0,
                "momentum_confirmation_weight": 0.25,
                "fast_lookback_hours": 336,
                "slow_lookback_hours": 168,
            },
        )
    with pytest.raises(ResearchError, match="sum to one"):
        build_portfolio_policy(
            "buffered_three_sleeve_neutral",
            "1",
            {
                "carry_weight": 0.7,
                "fast_strength_weight": 0.2,
                "slow_strength_weight": 0.2,
                "no_trade_score_band": 1.0,
                "rebalance_interval_hours": 48,
                "gross_exposure": 0.5,
                "annual_volatility_target": 0.15,
                "max_symbol_weight": 0.25,
            },
        )


def test_carry_dual_trend_scores_three_causal_sleeves() -> None:
    panel, context = _panel()
    strategy = build_strategy(
        "carry_dual_trend",
        "v1",
        {
            "funding_rate_weight": 1.0,
            "funding_change_weight": 0.0,
            "momentum_confirmation_weight": 0.25,
            "relative_strength_lookback_hours": 168,
            "sma_fast_window_hours": 12,
            "sma_slow_window_hours": 168,
        },
    )

    scores = strategy.fit_panel(panel, target=None, context=context).score_panel(
        panel, context=context
    )
    first = scores.frame.filter(pl.col("decision_time_ms") == context.test_start_ms).sort("symbol")

    assert strategy.required_features() == CARRY_DUAL_TREND_FEATURES
    assert {
        "carry_score",
        "fast_strength_score",
        "slow_strength_score",
    }.issubset(scores.frame.columns)
    assert first.filter(pl.col("symbol") == "BTCUSDT")["fast_strength_score"].item() > 0
    assert first.filter(pl.col("symbol") == "BTCUSDT")["slow_strength_score"].item() > 0
