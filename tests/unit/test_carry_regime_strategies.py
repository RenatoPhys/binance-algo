from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext, StrategyScores
from binance_algo.research.panel import PanelData
from binance_algo.research.portfolio.registry import build_portfolio_policy
from binance_algo.research.strategies.carry_multi_regime import CARRY_MULTI_REGIME_FEATURES
from binance_algo.research.strategies.registry import build_strategy

HOUR_MS = 3_600_000


def _panel() -> tuple[PanelData, FoldContext]:
    periods = 260
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    times = 1_700_000_000_000 + np.arange(periods, dtype=np.int64) * HOUR_MS
    symbol_pattern = np.tile(np.asarray([-1.0, 0.0, 1.0]), periods)
    returns = np.tile(np.asarray([0.003, 0.002, -0.001]), periods)
    panel = PanelData.from_frame(
        pl.DataFrame(
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
        ),
        feature_columns=(
            *CARRY_MULTI_REGIME_FEATURES,
            "rolling_beta",
            "realized_volatility_24h",
        ),
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
    return panel, context


def _carry_parameters() -> dict[str, float | int]:
    return {
        "funding_rate_weight": 1.0,
        "funding_change_weight": 0.0,
        "momentum_confirmation_weight": 0.25,
        "fast_lookback_hours": 48,
        "slow_lookback_hours": 96,
    }


def test_carry_regime_policy_is_exact_convex_blend() -> None:
    panel, context = _panel()
    strategy = build_strategy(
        "carry_multi_regime",
        "v1",
        {
            **_carry_parameters(),
            "regime_fast_lookback_hours": 48,
            "regime_slow_lookback_hours": 96,
        },
    )
    scores = strategy.fit_panel(panel, target=None, context=context).score_panel(
        panel, context=context
    )
    common = {
        "rebalance_interval_hours": 24,
        "gross_exposure": 0.5,
        "annual_volatility_target": 0.15,
        "max_symbol_weight": 0.25,
    }
    combined = build_portfolio_policy(
        "buffered_carry_regime",
        "1",
        {
            "core_carry_weight": 0.6,
            "core_fast_strength_weight": 0.3,
            "core_slow_strength_weight": 0.1,
            "regime_trend_weight": 0.2,
            "no_trade_score_band": 1.0,
            "signal_threshold": 0.0,
            **common,
        },
    ).target_weights_panel(scores, panel, context=context)
    core = build_portfolio_policy(
        "buffered_three_sleeve_neutral",
        "1",
        {
            "carry_weight": 0.6,
            "fast_strength_weight": 0.3,
            "slow_strength_weight": 0.1,
            "no_trade_score_band": 1.0,
            **common,
        },
    ).target_weights_panel(scores, panel, context=context)
    trend_scores = StrategyScores(
        scores.frame.select(
            "decision_time_ms",
            "symbol",
            pl.col("regime_trend_score").alias("score"),
        )
    )
    trend = build_portfolio_policy(
        "buffered_long_flat",
        "1",
        {"signal_threshold": 0.0, **common},
    ).target_weights_panel(trend_scores, panel, context=context)

    actual = combined.sort("decision_time_ms", "symbol")["target_weight"].to_numpy()
    expected = (
        0.8 * core.sort("decision_time_ms", "symbol")["target_weight"].to_numpy()
        + 0.2 * trend.sort("decision_time_ms", "symbol")["target_weight"].to_numpy()
    )
    assert "regime_trend_score" in scores.frame.columns
    assert np.allclose(actual, expected)
    assert np.max(np.abs(actual)) <= 0.25 + 1e-12


def test_consensus_strategy_exposes_only_agreeing_fast_and_slow_strength() -> None:
    panel, context = _panel()
    scores = (
        build_strategy("carry_consensus_strength", "1", _carry_parameters())
        .fit_panel(panel, target=None, context=context)
        .score_panel(panel, context=context)
        .frame
    )

    assert {"carry_score", "relative_strength_score"}.issubset(scores.columns)
    assert scores["relative_strength_score"].abs().max() > 0
    assert np.all(np.isfinite(scores["score"].to_numpy()))

    flat_consensus = StrategyScores(
        scores.with_columns(pl.lit(0.0).alias("relative_strength_score"))
    )
    targets = build_portfolio_policy(
        "buffered_two_sleeve_neutral",
        "1",
        {
            "carry_weight": 0.7,
            "no_trade_score_band": 1.0,
            "rebalance_interval_hours": 48,
            "gross_exposure": 0.5,
            "annual_volatility_target": 0.15,
            "max_symbol_weight": 0.25,
        },
    ).target_weights_panel(flat_consensus, panel, context=context)
    assert np.all(np.isfinite(targets["target_weight"].to_numpy()))


def test_carry_regime_factories_reject_unsafe_weights_and_extra_parameters() -> None:
    common = {
        "core_carry_weight": 0.6,
        "core_fast_strength_weight": 0.3,
        "core_slow_strength_weight": 0.1,
        "no_trade_score_band": 1.0,
        "signal_threshold": 0.0,
        "rebalance_interval_hours": 48,
        "gross_exposure": 0.5,
        "annual_volatility_target": 0.15,
        "max_symbol_weight": 0.25,
    }
    with pytest.raises(ResearchError, match="regime_trend_weight"):
        build_portfolio_policy(
            "buffered_carry_regime",
            "1",
            {**common, "regime_trend_weight": 0.6},
        )
    with pytest.raises(ResearchError, match="extra_forbidden"):
        build_strategy(
            "carry_consensus_strength",
            "1",
            {**_carry_parameters(), "future_filter": True},
        )
