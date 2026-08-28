from __future__ import annotations

import math

import polars as pl
import pytest

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext, TrainingDataset, select_feature_view
from binance_algo.research.portfolio.neutral_long_short import (
    NeutralLongShortParameters,
    NeutralLongShortPolicy,
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
