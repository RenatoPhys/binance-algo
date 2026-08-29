from __future__ import annotations

import numpy as np
import polars as pl

from binance_algo.research.contracts import FoldContext, StrategyScores
from binance_algo.research.panel import PanelData
from binance_algo.research.portfolio.pair_spread import (
    BufferedPairSpreadParameters,
    BufferedPairSpreadPolicy,
)
from binance_algo.research.state_machine import causal_sparse_state
from binance_algo.research.strategies.pair_spread_reversion import (
    PairSpreadReversionParameters,
    PairSpreadReversionStrategy,
)

HOUR_MS = 3_600_000


def test_sparse_state_machine_holds_ignores_retriggers_and_exits_causally() -> None:
    entry = np.asarray([[2.0], [-3.0], [0.0], [0.0], [-1.0], [0.0]])
    result = causal_sparse_state(entry, hold_hours=3)

    assert result.values[:, 0].tolist() == [2.0, 2.0, 2.0, 0.0, -1.0, -1.0]
    assert result.entries[:, 0].tolist() == [True, False, False, False, True, False]
    assert result.forced_time_exits[:, 0].tolist() == [False, False, False, True, False, False]

    explicit = causal_sparse_state(
        np.asarray([[1.0], [0.0], [-1.0], [0.0]]),
        hold_hours=10,
        explicit_exit=np.asarray([[False], [False], [True], [False]]),
    )
    assert explicit.values[:, 0].tolist() == [1.0, 1.0, 0.0, 0.0]


def _pair_panel(*, test_shock: float = 0.0) -> PanelData:
    generator = np.random.default_rng(7)
    rows = 800
    btc = 8.0 + np.cumsum(generator.normal(0.0, 0.002, rows))
    residual_eth = np.zeros(rows)
    residual_sol = np.zeros(rows)
    for index in range(1, rows):
        residual_eth[index] = 0.90 * residual_eth[index - 1] + generator.normal(0.0, 0.003)
        residual_sol[index] = 0.85 * residual_sol[index - 1] + generator.normal(0.0, 0.004)
    eth = 0.2 + 1.1 * btc + residual_eth
    sol = -0.1 + 0.8 * btc + residual_sol
    eth[730:] += test_shock
    times = np.arange(rows, dtype=np.int64) * HOUR_MS
    return PanelData.from_frame(
        pl.DataFrame(
            {
                "decision_time_ms": np.repeat(times, 3),
                "symbol": np.tile(np.asarray(["BTCUSDT", "ETHUSDT", "SOLUSDT"]), rows),
                "log_close_level": np.column_stack((btc, eth, sol)).reshape(-1),
            }
        ),
        feature_columns=("log_close_level",),
    )


def test_pair_fit_is_frozen_to_training_rows() -> None:
    context = FoldContext(
        fold=1,
        train_start_ms=0,
        train_end_ms=719 * HOUR_MS,
        test_start_ms=721 * HOUR_MS,
        test_end_ms=799 * HOUR_MS,
        embargo_bars=1,
        random_seed=1,
    )
    strategy = PairSpreadReversionStrategy(PairSpreadReversionParameters(168, 1.5))
    fitted = strategy.fit_panel(_pair_panel(), target=None, context=context)
    shocked = strategy.fit_panel(_pair_panel(test_shock=5.0), target=None, context=context)

    assert fitted.models == shocked.models
    assert all(model.eligible for model in fitted.models)


def test_pair_policy_preserves_hedge_ratio_before_global_scaling() -> None:
    time = HOUR_MS
    symbols = np.asarray(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    score_frame = pl.DataFrame(
        {
            "decision_time_ms": np.repeat(np.asarray([time]), 3),
            "symbol": symbols,
            "score": [-0.5, 1.0, 0.0],
            "pair_eth_btc": [-0.5, 1.0, 0.0],
            "pair_sol_btc": [0.0, 0.0, 0.0],
        }
    )
    market = PanelData.from_frame(
        pl.DataFrame(
            {
                "decision_time_ms": np.repeat(np.asarray([time]), 3),
                "symbol": symbols,
                "realized_volatility_24h": [0.1, 0.1, 0.1],
            }
        ),
        feature_columns=("realized_volatility_24h",),
    )
    context = FoldContext(1, 0, 1, time, time, 1, 1)
    policy = BufferedPairSpreadPolicy(BufferedPairSpreadParameters(0.35, 0.10, 0.20, 2))
    result = policy.target_weights_panel(StrategyScores(score_frame), market, context=context)
    weights = result.sort("symbol")["target_weight"].to_list()

    assert weights[0] / weights[1] == -0.5
    assert abs(weights[0]) <= 0.20
    assert abs(weights[1]) <= 0.20
