from __future__ import annotations

from dataclasses import dataclass

import polars as pl
import pytest

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import (
    FoldContext,
    StrategyScores,
    TrainingDataset,
    select_feature_view,
)
from binance_algo.research.portfolio.base import PortfolioPolicy
from binance_algo.research.strategies.base import FittedStrategy, Strategy


def _context() -> FoldContext:
    return FoldContext(
        fold=1,
        train_start_ms=1_000,
        train_end_ms=1_999,
        test_start_ms=3_000,
        test_end_ms=3_999,
        embargo_bars=1,
        random_seed=42,
    )


def _source_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "decision_time_ms": [3_000, 3_000],
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "residual_momentum_24h": [0.2, -0.1],
            "rolling_beta": [1.0, 0.8],
            "future_return_1h": [0.01, -0.02],
            "outcome_funding_rate_1h": [0.0001, 0.0002],
            "label_end_time_ms": [3_600, 3_600],
        }
    )


@dataclass(slots=True)
class _SpyFittedStrategy:
    seen_columns: tuple[str, ...] = ()

    def score(self, features: pl.DataFrame, *, context: FoldContext) -> StrategyScores:
        self.seen_columns = tuple(features.columns)
        return StrategyScores(
            features.select("decision_time_ms", "symbol").with_columns(
                pl.lit(float(context.fold)).alias("score")
            )
        )


@dataclass(frozen=True, slots=True)
class _FixedStrategy:
    strategy_id: str = "test_fixed"
    strategy_version: str = "1"

    def required_features(self) -> tuple[str, ...]:
        return ("residual_momentum_24h",)

    def target_column(self) -> str | None:
        return None

    def fit(self, train: TrainingDataset, *, context: FoldContext) -> FittedStrategy:
        assert context.train_end_ms < context.test_start_ms
        assert train.target is None
        return _SpyFittedStrategy()


@dataclass(frozen=True, slots=True)
class _FixedPortfolioPolicy:
    policy_id: str = "test_neutral"
    policy_version: str = "1"

    def required_features(self) -> tuple[str, ...]:
        return ("rolling_beta",)

    def target_weights(
        self,
        scores: pl.DataFrame,
        market_state: pl.DataFrame,
        *,
        context: FoldContext,
    ) -> pl.DataFrame:
        del market_state, context
        return scores.select("decision_time_ms", "symbol").with_columns(
            pl.lit(0.0).alias("target_weight")
        )


def test_contracts_are_structural_and_keep_strategy_and_portfolio_separate() -> None:
    strategy = _FixedStrategy()
    fitted = strategy.fit(
        TrainingDataset(
            features=select_feature_view(
                _source_frame(), required_features=strategy.required_features()
            ),
            target=None,
        ),
        context=_context(),
    )
    policy = _FixedPortfolioPolicy()

    assert isinstance(strategy, Strategy)
    assert isinstance(fitted, FittedStrategy)
    assert isinstance(policy, PortfolioPolicy)


def test_scoring_view_excludes_undeclared_features_labels_and_outcomes() -> None:
    source = _source_frame()
    view = select_feature_view(source, required_features=("residual_momentum_24h",))
    fitted = _SpyFittedStrategy()

    scores = fitted.score(view, context=_context())

    assert fitted.seen_columns == (
        "decision_time_ms",
        "symbol",
        "residual_momentum_24h",
    )
    assert scores.frame.columns == ["decision_time_ms", "symbol", "score"]


@pytest.mark.parametrize(
    "forbidden_name",
    ["future_return_1h", "outcome_funding_rate_1h", "label_end_time_ms"],
)
def test_outcome_or_label_cannot_be_declared_as_a_feature(forbidden_name: str) -> None:
    with pytest.raises(ResearchError, match="outcome or label columns"):
        select_feature_view(_source_frame(), required_features=(forbidden_name,))


def test_training_dataset_rejects_outcomes_mixed_into_features() -> None:
    contaminated = _source_frame().select(
        "decision_time_ms", "symbol", "residual_momentum_24h", "future_return_1h"
    )

    with pytest.raises(ResearchError, match="training features contains outcome or label"):
        TrainingDataset(features=contaminated, target=None)


def test_training_target_is_separate_and_key_aligned() -> None:
    source = _source_frame()
    features = select_feature_view(source, required_features=("residual_momentum_24h",))
    target = source.select("decision_time_ms", "symbol", "future_return_1h")

    training = TrainingDataset(features=features, target=target)

    assert training.target is not None
    assert "future_return_1h" not in training.features.columns
    with pytest.raises(ResearchError, match="keys must align exactly"):
        TrainingDataset(features=features, target=target.reverse())


def test_score_contract_rejects_duplicate_or_non_finite_scores() -> None:
    duplicate = pl.DataFrame(
        {
            "decision_time_ms": [3_000, 3_000],
            "symbol": ["BTCUSDT", "BTCUSDT"],
            "score": [0.1, 0.2],
        }
    )
    non_finite = pl.DataFrame(
        {"decision_time_ms": [3_000], "symbol": ["BTCUSDT"], "score": [float("nan")]}
    )

    with pytest.raises(ResearchError, match="duplicate research keys"):
        StrategyScores(duplicate)
    with pytest.raises(ResearchError, match="finite and non-null"):
        StrategyScores(non_finite)


def test_fold_context_rejects_training_overlap_with_test() -> None:
    with pytest.raises(ResearchError, match="train must precede test"):
        FoldContext(
            fold=1,
            train_start_ms=1_000,
            train_end_ms=3_000,
            test_start_ms=3_000,
            test_end_ms=4_000,
            embargo_bars=1,
            random_seed=42,
        )
