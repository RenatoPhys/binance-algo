from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from binance_algo.common.errors import ResearchError
from binance_algo.config import load_settings
from binance_algo.research.backtest import run_walk_forward
from binance_algo.research.contracts import FoldContext, StrategyScores
from binance_algo.research.panel import PanelData
from binance_algo.research.portfolio.directional import (
    BufferedDirectionalParameters,
    BufferedDirectionalPolicy,
)
from binance_algo.research.portfolio.registry import build_portfolio_policy
from binance_algo.research.strategies.registry import build_strategy

from ..research_fixtures import research_frame

HOUR_MS = 3_600_000
BASE_CONFIG = Path(__file__).parents[2] / "configs" / "base.yaml"


def _context() -> FoldContext:
    return FoldContext(
        fold=1,
        train_start_ms=HOUR_MS,
        train_end_ms=2 * HOUR_MS,
        test_start_ms=4 * HOUR_MS,
        test_end_ms=6 * HOUR_MS,
        embargo_bars=1,
        random_seed=42,
    )


def _inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    times = np.arange(4, 7, dtype=np.int64) * HOUR_MS
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    scores = pl.DataFrame(
        {
            "decision_time_ms": np.repeat(times, len(symbols)),
            "symbol": np.tile(np.asarray(symbols), len(times)),
            "score": [0.02, 0.01, -0.03, -0.02, -0.01, 0.03, -0.02, -0.01, 0.03],
        }
    )
    market_state = scores.select("decision_time_ms", "symbol").with_columns(
        pl.Series("realized_volatility_24h", [0.01, 0.02, 0.01] * len(times))
    )
    return scores, market_state


def _policy(*, threshold: float = 0.0) -> BufferedDirectionalPolicy:
    return BufferedDirectionalPolicy(
        BufferedDirectionalParameters(
            signal_threshold=threshold,
            rebalance_interval_hours=2,
            gross_exposure=0.5,
            annual_volatility_target=0.15,
            max_symbol_weight=0.25,
        )
    )


def test_directional_policy_uses_absolute_signs_and_holds_between_rebalances() -> None:
    scores, market_state = _inputs()

    targets = _policy().target_weights(scores, market_state, context=_context())
    first = targets.filter(pl.col("decision_time_ms") == 4 * HOUR_MS).sort("symbol")
    second = targets.filter(pl.col("decision_time_ms") == 5 * HOUR_MS).sort("symbol")
    third = targets.filter(pl.col("decision_time_ms") == 6 * HOUR_MS).sort("symbol")

    assert first["target_weight"].to_list() == second["target_weight"].to_list()
    assert first.filter(pl.col("symbol") == "BTCUSDT")["target_weight"].item() > 0
    assert first.filter(pl.col("symbol") == "SOLUSDT")["target_weight"].item() < 0
    assert third.filter(pl.col("symbol") == "BTCUSDT")["target_weight"].item() < 0
    assert third.filter(pl.col("symbol") == "SOLUSDT")["target_weight"].item() > 0
    for period in (first, third):
        weights = period["target_weight"].to_list()
        assert sum(abs(value) for value in weights) <= 0.5 + 1e-12
        assert max(abs(value) for value in weights) <= 0.25 + 1e-12


def test_directional_panel_path_matches_frame_path_and_threshold_can_flatten() -> None:
    scores, market_state = _inputs()
    context = _context()
    frame_targets = _policy().target_weights(scores, market_state, context=context)
    panel = PanelData.from_frame(
        market_state,
        feature_columns=("realized_volatility_24h",),
    )
    panel_targets = _policy().target_weights_panel(
        StrategyScores(scores),
        panel,
        context=context,
    )

    assert frame_targets.equals(panel_targets, null_equal=True)
    flat = _policy(threshold=0.1).target_weights(scores, market_state, context=context)
    assert flat["target_weight"].to_list() == [0.0] * flat.height


def test_directional_factory_is_strict_and_parameters_are_finite() -> None:
    parameters = {
        "signal_threshold": 0.002,
        "rebalance_interval_hours": 24,
        "gross_exposure": 0.5,
        "annual_volatility_target": 0.15,
        "max_symbol_weight": 0.25,
    }
    policy = build_portfolio_policy("buffered_directional", "v1", parameters)

    assert policy.policy_id == "buffered_directional"
    assert math.isclose(policy.parameters.signal_threshold, 0.002)
    with pytest.raises(ResearchError, match="extra_forbidden"):
        build_portfolio_policy(
            "buffered_directional",
            "1",
            {**parameters, "leverage": 2},
        )
    with pytest.raises(ResearchError, match="must be finite"):
        BufferedDirectionalParameters(
            signal_threshold=float("nan"),
            rebalance_interval_hours=24,
            gross_exposure=0.5,
            annual_volatility_target=0.15,
            max_symbol_weight=0.25,
        )


def test_directional_walk_forward_loads_accounting_beta_independently() -> None:
    config = load_settings(BASE_CONFIG).research.model_copy(
        update={"walk_forward_train_days": 7, "walk_forward_test_days": 1}
    )
    strategy = build_strategy(
        "sma_trend_strength",
        "1",
        {"fast_window_hours": 4, "slow_window_hours": 24},
    )
    policy = build_portfolio_policy(
        "buffered_directional",
        "1",
        {
            "signal_threshold": 0.0,
            "rebalance_interval_hours": 24,
            "gross_exposure": 0.5,
            "annual_volatility_target": 0.15,
            "max_symbol_weight": 0.25,
        },
    )

    result = run_walk_forward(
        research_frame(),
        config=config,
        strategy=strategy,
        portfolio_policy=policy,
    )

    assert result.metrics.periods > 0
    assert math.isfinite(result.metrics.maximum_absolute_beta_exposure)
