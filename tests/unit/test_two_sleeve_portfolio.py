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
from binance_algo.research.strategies.carry_relative_strength import (
    CARRY_RELATIVE_STRENGTH_FEATURES,
)
from binance_algo.research.strategies.registry import build_strategy

HOUR_MS = 3_600_000


def _panel() -> tuple[PanelData, FoldContext]:
    periods = 100
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    times = 1_700_000_000_000 + np.arange(periods, dtype=np.int64) * HOUR_MS
    symbol_pattern = np.tile(np.asarray([-1.0, 0.0, 1.0]), periods)
    return_pattern = np.tile(np.asarray([0.002, 0.0, -0.002]), periods)
    frame = pl.DataFrame(
        {
            "decision_time_ms": np.repeat(times, len(symbols)),
            "symbol": np.tile(np.asarray(symbols), periods),
            "funding_rate_current": 0.0001 * symbol_pattern,
            "funding_rate_change": 0.00001 * symbol_pattern,
            "residual_momentum_4h": -0.001 * symbol_pattern,
            "log_return_1h": return_pattern,
            "rolling_beta": np.tile(np.asarray([0.8, 1.0, 1.2]), periods),
            "realized_volatility_24h": np.tile(np.asarray([0.01, 0.012, 0.014]), periods),
        }
    )
    context = FoldContext(
        fold=1,
        train_start_ms=int(times[0]),
        train_end_ms=int(times[71]),
        test_start_ms=int(times[73]),
        test_end_ms=int(times[-1]),
        embargo_bars=1,
        random_seed=42,
    )
    return (
        PanelData.from_frame(
            frame,
            feature_columns=(
                *CARRY_RELATIVE_STRENGTH_FEATURES,
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


def test_two_sleeve_policy_combines_actual_target_weights() -> None:
    panel, context = _panel()
    strategy = build_strategy(
        "carry_relative_strength",
        "v1",
        {
            "funding_rate_weight": 1.0,
            "funding_change_weight": 0.0,
            "momentum_confirmation_weight": 0.25,
            "relative_strength_lookback_hours": 48,
        },
    )
    scores = strategy.fit_panel(panel, target=None, context=context).score_panel(
        panel, context=context
    )
    policy = build_portfolio_policy(
        "buffered_two_sleeve_neutral",
        "1",
        {
            "carry_weight": 0.7,
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
    carry_scores = StrategyScores(
        scores.frame.select(
            "decision_time_ms",
            "symbol",
            pl.col("carry_score").alias("score"),
        )
    )
    relative_scores = StrategyScores(
        scores.frame.select(
            "decision_time_ms",
            "symbol",
            pl.col("relative_strength_score").alias("score"),
        )
    )
    carry = (
        _sleeve_policy()
        .target_weights_panel(carry_scores, panel, context=context)
        .sort("decision_time_ms", "symbol")
    )
    relative = (
        _sleeve_policy()
        .target_weights_panel(relative_scores, panel, context=context)
        .sort("decision_time_ms", "symbol")
    )
    expected = 0.7 * carry["target_weight"] + 0.3 * relative["target_weight"]

    assert strategy.target_column() is None
    assert {"carry_score", "relative_strength_score"}.issubset(scores.frame.columns)
    assert np.allclose(combined["target_weight"].to_numpy(), expected.to_numpy())


def test_two_sleeve_factory_rejects_unknown_parameters() -> None:
    with pytest.raises(ResearchError, match="extra_forbidden"):
        build_portfolio_policy(
            "buffered_two_sleeve_neutral",
            "1",
            {
                "carry_weight": 0.7,
                "no_trade_score_band": 1.0,
                "rebalance_interval_hours": 48,
                "gross_exposure": 0.5,
                "annual_volatility_target": 0.15,
                "max_symbol_weight": 0.25,
                "dynamic_weight": True,
            },
        )
