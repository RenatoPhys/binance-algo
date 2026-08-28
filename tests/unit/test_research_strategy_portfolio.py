from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext, TrainingDataset, select_feature_view
from binance_algo.research.panel import PanelData
from binance_algo.research.portfolio.neutral_long_short import (
    BufferedNeutralLongShortParameters,
    BufferedNeutralLongShortPolicy,
    NeutralLongShortParameters,
    NeutralLongShortPolicy,
)
from binance_algo.research.portfolio.registry import build_portfolio_policy
from binance_algo.research.strategies.funding_carry import FUNDING_CARRY_FEATURES
from binance_algo.research.strategies.registry import build_strategy
from binance_algo.research.strategies.residual_mean_reversion import (
    RESIDUAL_MEAN_REVERSION_FEATURES,
)
from binance_algo.research.strategies.residual_momentum import (
    RESIDUAL_MOMENTUM_FEATURES,
    ResidualMomentumParameters,
    ResidualMomentumStrategy,
)


def _context() -> FoldContext:
    return FoldContext(
        fold=1,
        train_start_ms=1_000,
        train_end_ms=2_000,
        test_start_ms=3_000,
        test_end_ms=4_000,
        embargo_bars=1,
        random_seed=42,
    )


def _strategy() -> ResidualMomentumStrategy:
    return ResidualMomentumStrategy(
        ResidualMomentumParameters(
            momentum_weight_1h=0.2,
            momentum_weight_4h=0.3,
            momentum_weight_24h=0.5,
        )
    )


def _feature_frame(decision_time_ms: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "decision_time_ms": [decision_time_ms] * 3,
            "symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "residual_momentum_1h": [1.0, 0.0, -1.0],
            "residual_momentum_4h": [2.0, 0.0, -2.0],
            "residual_momentum_24h": [4.0, 0.0, -4.0],
        }
    )


def _policy() -> NeutralLongShortPolicy:
    return NeutralLongShortPolicy(
        NeutralLongShortParameters(
            no_trade_score_band=0.25,
            gross_exposure=0.5,
            annual_volatility_target=0.15,
            max_symbol_weight=0.25,
        )
    )


def _portfolio_inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    scores = pl.DataFrame(
        {
            "decision_time_ms": [3_000] * 3 + [4_000] * 3,
            "symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"] * 2,
            "score": [2.0, -2.0, 0.0, 1.9, -2.0, 2.0],
        }
    )
    market_state = pl.DataFrame(
        {
            "decision_time_ms": [3_000] * 3 + [4_000] * 3,
            "symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"] * 2,
            "rolling_beta": [0.8, 1.0, 1.2] * 2,
            "realized_volatility_24h": [0.001, 0.001, 0.001] * 2,
        }
    )
    return scores, market_state


def test_residual_momentum_is_fixed_versioned_and_scores_only_declared_features() -> None:
    strategy = _strategy()
    train = _feature_frame(1_500)
    fitted = strategy.fit(
        TrainingDataset(
            features=select_feature_view(
                train,
                required_features=strategy.required_features(),
            ),
            target=None,
        ),
        context=_context(),
    )

    scoring = _feature_frame(3_000).with_columns(
        pl.lit(0.99).alias("future_return_1h"),
        pl.lit(0.01).alias("outcome_funding_rate_1h"),
    )
    result = fitted.score(scoring, context=_context()).frame.sort("symbol")

    assert strategy.strategy_id == "residual_momentum"
    assert strategy.strategy_version == "1"
    assert strategy.required_features() == RESIDUAL_MOMENTUM_FEATURES
    assert strategy.target_column() is None
    assert result.columns == ["decision_time_ms", "symbol", "score"]
    assert result.filter(pl.col("symbol") == "BTCUSDT")["score"].item() > 0
    assert result.filter(pl.col("symbol") == "SOLUSDT")["score"].item() < 0


@pytest.mark.parametrize(
    "weights",
    [(-0.1, 0.5, 0.6), (0.2, 0.3, 0.4), (float("nan"), 0.0, 1.0)],
)
def test_residual_momentum_rejects_invalid_parameters(
    weights: tuple[float, float, float],
) -> None:
    with pytest.raises(ResearchError, match="residual momentum weights"):
        ResidualMomentumParameters(
            momentum_weight_1h=weights[0],
            momentum_weight_4h=weights[1],
            momentum_weight_24h=weights[2],
        )


def test_funding_carry_factory_is_strict_fixed_and_directional() -> None:
    strategy = build_strategy(
        "funding_carry",
        "v1",
        {
            "funding_rate_weight": 1.0,
            "funding_change_weight": 0.5,
            "momentum_confirmation_weight": 0.25,
        },
    )
    train = pl.DataFrame(
        {
            "decision_time_ms": [1_500] * 3,
            "symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "funding_rate_current": [-0.001, 0.0, 0.001],
            "funding_rate_change": [-0.0001, 0.0, 0.0001],
            "residual_momentum_4h": [1.0, 0.0, -1.0],
        }
    )
    scoring = train.with_columns(pl.lit(3_000).alias("decision_time_ms"))
    fitted = strategy.fit(TrainingDataset(features=train, target=None), context=_context())
    scores = fitted.score(scoring, context=_context()).frame.sort("symbol")

    assert strategy.required_features() == FUNDING_CARRY_FEATURES
    assert strategy.target_column() is None
    assert scores.filter(pl.col("symbol") == "BTCUSDT")["score"].item() > 0
    assert scores.filter(pl.col("symbol") == "SOLUSDT")["score"].item() < 0
    with pytest.raises(ResearchError, match="extra_forbidden"):
        build_strategy(
            "funding_carry",
            "1",
            {
                "funding_rate_weight": 1.0,
                "funding_change_weight": 0.5,
                "momentum_confirmation_weight": 0.25,
                "leverage": 10,
            },
        )


def test_residual_mean_reversion_factory_is_strict_fixed_and_directional() -> None:
    strategy = build_strategy(
        "residual_mean_reversion",
        "1",
        {
            "momentum_weight_1h": 0.75,
            "momentum_weight_4h": 0.25,
            "volatility_adjustment": 0.5,
        },
    )
    train = pl.DataFrame(
        {
            "decision_time_ms": [1_500] * 3,
            "symbol": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "residual_momentum_1h": [1.0, 0.0, -1.0],
            "residual_momentum_4h": [2.0, 0.0, -2.0],
            "realized_volatility_24h": [0.02, 0.01, 0.02],
        }
    )
    scoring = train.with_columns(pl.lit(3_000).alias("decision_time_ms"))
    fitted = strategy.fit(TrainingDataset(features=train, target=None), context=_context())
    scores = fitted.score(scoring, context=_context()).frame.sort("symbol")

    assert strategy.required_features() == RESIDUAL_MEAN_REVERSION_FEATURES
    assert strategy.target_column() is None
    assert scores.filter(pl.col("symbol") == "BTCUSDT")["score"].item() < 0
    assert scores.filter(pl.col("symbol") == "SOLUSDT")["score"].item() > 0
    with pytest.raises(ResearchError, match="weights must sum to one"):
        build_strategy(
            "residual_mean_reversion",
            "v1",
            {
                "momentum_weight_1h": 0.8,
                "momentum_weight_4h": 0.3,
                "volatility_adjustment": 0.0,
            },
        )


def test_sma_crossover_factory_is_causal_fixed_and_directional() -> None:
    hour_ms = 3_600_000
    start_ms = 1_700_000_000_000
    periods = 80
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    times = start_ms + np.arange(periods, dtype=np.int64) * hour_ms
    times[60:] += hour_ms
    hourly_returns = np.tile(np.asarray([0.005, 0.0, -0.005]), (periods, 1))
    frame = pl.DataFrame(
        {
            "decision_time_ms": np.repeat(times, len(symbols)),
            "symbol": np.tile(np.asarray(symbols), periods),
            "log_return_1h": hourly_returns.reshape(-1),
        }
    )
    panel = PanelData.from_frame(frame, feature_columns=("log_return_1h",))
    context = FoldContext(
        fold=1,
        train_start_ms=int(times[0]),
        train_end_ms=int(times[47]),
        test_start_ms=int(times[72]),
        test_end_ms=int(times[-1]),
        embargo_bars=1,
        random_seed=42,
    )
    strategy = build_strategy(
        "sma_crossover",
        "v1",
        {"fast_window_hours": 4, "slow_window_hours": 24},
    )
    fitted = strategy.fit_panel(panel, target=None, context=context)
    scores = fitted.score_panel(panel, context=context).frame
    first = scores.filter(pl.col("decision_time_ms") == int(times[72])).sort("symbol")

    assert strategy.required_features() == ("log_return_1h",)
    assert strategy.target_column() is None
    assert first.filter(pl.col("symbol") == "BTCUSDT")["score"].item() > 0
    assert first.filter(pl.col("symbol") == "SOLUSDT")["score"].item() < 0
    strength_strategy = build_strategy(
        "sma_trend_strength",
        "v1",
        {"fast_window_hours": 4, "slow_window_hours": 24},
    )
    strength_scores = (
        strength_strategy.fit_panel(panel, target=None, context=context)
        .score_panel(panel, context=context)
        .frame
    )
    first_strength = strength_scores.filter(pl.col("decision_time_ms") == int(times[72])).sort(
        "symbol"
    )
    assert first_strength.filter(pl.col("symbol") == "BTCUSDT")["score"].item() > 0
    assert first_strength.filter(pl.col("symbol") == "SOLUSDT")["score"].item() < 0
    assert float(first_strength["score"].std()) < 1
    with pytest.raises(ResearchError, match="shorter than"):
        build_strategy(
            "sma_crossover",
            "1",
            {"fast_window_hours": 24, "slow_window_hours": 12},
        )
    with pytest.raises(ResearchError, match="extra_forbidden"):
        build_strategy(
            "sma_crossover",
            "1",
            {"fast_window_hours": 4, "slow_window_hours": 24, "centered": True},
        )


def test_non_trainable_strategy_rejects_target_and_test_rows_during_fit() -> None:
    strategy = _strategy()
    train = _feature_frame(1_500)
    features = select_feature_view(train, required_features=strategy.required_features())
    target = train.select("decision_time_ms", "symbol").with_columns(
        pl.lit(0.1).alias("future_return_1h")
    )

    with pytest.raises(ResearchError, match="does not accept a target"):
        strategy.fit(TrainingDataset(features=features, target=target), context=_context())
    with pytest.raises(ResearchError, match="outside its fold context"):
        strategy.fit(
            TrainingDataset(
                features=select_feature_view(
                    _feature_frame(3_000),
                    required_features=strategy.required_features(),
                ),
                target=None,
            ),
            context=_context(),
        )


def test_neutral_long_short_respects_exposures_beta_limits_and_no_trade_band() -> None:
    scores, market_state = _portfolio_inputs()

    targets = _policy().target_weights(scores, market_state, context=_context())

    for decision_time in (3_000, 4_000):
        period = targets.filter(pl.col("decision_time_ms") == decision_time).sort("symbol")
        weights = period["target_weight"].to_list()
        betas = [0.8, 1.0, 1.2]
        assert math.isclose(sum(weights), 0.0, abs_tol=1e-12)
        assert sum(abs(weight) for weight in weights) <= 0.5 + 1e-12
        assert max(abs(weight) for weight in weights) <= 0.25 + 1e-12
        assert math.isclose(
            sum(w * beta for w, beta in zip(weights, betas, strict=True)),
            0.0,
            abs_tol=1e-12,
        )

    second = targets.filter(pl.col("decision_time_ms") == 4_000)
    assert second.filter(pl.col("symbol") == "BTCUSDT")["target_weight"].item() > 0
    assert second.filter(pl.col("symbol") == "ETHUSDT")["target_weight"].item() < 0


def test_neutral_long_short_fails_on_degenerate_scores_or_misaligned_state() -> None:
    scores, market_state = _portfolio_inputs()
    degenerate = scores.with_columns(pl.lit(0.0).alias("score"))

    with pytest.raises(ResearchError, match="did not produce distinct tails"):
        _policy().target_weights(degenerate, market_state, context=_context())
    with pytest.raises(ResearchError, match="keys must align exactly"):
        _policy().target_weights(scores, market_state.head(5), context=_context())


def test_buffered_neutral_long_short_holds_weights_between_rebalances() -> None:
    scores, market_state = _portfolio_inputs()
    policy = BufferedNeutralLongShortPolicy(
        BufferedNeutralLongShortParameters(
            no_trade_score_band=0.0,
            rebalance_interval_hours=2,
            gross_exposure=0.5,
            annual_volatility_target=0.15,
            max_symbol_weight=0.25,
        )
    )

    targets = policy.target_weights(scores, market_state, context=_context())
    first = targets.filter(pl.col("decision_time_ms") == 3_000).sort("symbol")
    second = targets.filter(pl.col("decision_time_ms") == 4_000).sort("symbol")

    assert first["target_weight"].to_list() == second["target_weight"].to_list()
    assert first.filter(pl.col("symbol") == "BTCUSDT")["target_weight"].item() > 0

    gated = BufferedNeutralLongShortPolicy(
        BufferedNeutralLongShortParameters(
            no_trade_score_band=0.0,
            rebalance_interval_hours=2,
            gross_exposure=0.5,
            annual_volatility_target=0.15,
            max_symbol_weight=0.25,
            minimum_score_spread=5.0,
        )
    ).target_weights(scores, market_state, context=_context())
    assert gated["target_weight"].to_list() == [0.0] * 6


def test_buffered_neutral_long_short_factory_is_strict() -> None:
    parameters = {
        "no_trade_score_band": 0.5,
        "rebalance_interval_hours": 12,
        "gross_exposure": 0.5,
        "annual_volatility_target": 0.15,
        "max_symbol_weight": 0.25,
        "minimum_score_spread": 0.05,
    }

    policy = build_portfolio_policy("buffered_neutral_long_short", "v1", parameters)

    assert policy.policy_id == "buffered_neutral_long_short"
    assert policy.parameters.rebalance_interval_hours == 12
    assert policy.parameters.minimum_score_spread == 0.05
    with pytest.raises(ResearchError, match="extra_forbidden"):
        build_portfolio_policy(
            "buffered_neutral_long_short",
            "1",
            {**parameters, "future_leak": True},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("no_trade_score_band", -0.1),
        ("gross_exposure", 1.1),
        ("annual_volatility_target", 0.0),
        ("max_symbol_weight", float("inf")),
    ],
)
def test_neutral_long_short_rejects_invalid_parameters(field: str, value: float) -> None:
    payload = {
        "no_trade_score_band": 0.25,
        "gross_exposure": 0.5,
        "annual_volatility_target": 0.15,
        "max_symbol_weight": 0.25,
    }
    payload[field] = value

    with pytest.raises(ResearchError):
        NeutralLongShortParameters(**payload)
