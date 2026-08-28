from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext, TrainingDataset
from binance_algo.research.panel import PanelData
from binance_algo.research.strategies.linear_cross_sectional import (
    LINEAR_CROSS_SECTIONAL_FEATURES,
    LINEAR_CROSS_SECTIONAL_TARGET,
)
from binance_algo.research.strategies.registry import build_strategy


def _frames() -> tuple[pl.DataFrame, pl.DataFrame, FoldContext]:
    symbols = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    train_times = np.arange(1_000, 7_000, 1_000, dtype=np.int64)
    test_times = np.arange(8_000, 11_000, 1_000, dtype=np.int64)
    times = np.concatenate((train_times, test_times))
    cross_section = np.tile(np.asarray([1.0, 0.0, -1.0]), len(times))
    columns: dict[str, object] = {
        "decision_time_ms": np.repeat(times, len(symbols)),
        "symbol": np.tile(np.asarray(symbols), len(times)),
    }
    for index, feature in enumerate(LINEAR_CROSS_SECTIONAL_FEATURES):
        direction = 1.0 if index == 0 else (-1.0 if index == 1 else 0.1)
        columns[feature] = direction * cross_section
    features = pl.DataFrame(columns)
    target = (
        features.filter(pl.col("decision_time_ms") <= int(train_times[-1]))
        .select("decision_time_ms", "symbol", LINEAR_CROSS_SECTIONAL_FEATURES[0])
        .rename({LINEAR_CROSS_SECTIONAL_FEATURES[0]: LINEAR_CROSS_SECTIONAL_TARGET})
    )
    context = FoldContext(
        fold=1,
        train_start_ms=int(train_times[0]),
        train_end_ms=int(train_times[-1]),
        test_start_ms=int(test_times[0]),
        test_end_ms=int(test_times[-1]),
        embargo_bars=1,
        random_seed=42,
    )
    return features, target, context


def test_linear_cross_sectional_is_supervised_causal_and_directional() -> None:
    features, target, context = _frames()
    strategy = build_strategy("linear_cross_sectional", "v1", {"ridge_alpha": 0.1})
    train = features.filter(pl.col("decision_time_ms") <= context.train_end_ms)
    test = features.filter(pl.col("decision_time_ms") >= context.test_start_ms)

    fitted = strategy.fit(TrainingDataset(features=train, target=target), context=context)
    scores = fitted.score(test, context=context).frame
    first = scores.filter(pl.col("decision_time_ms") == context.test_start_ms).sort("symbol")

    assert strategy.required_features() == LINEAR_CROSS_SECTIONAL_FEATURES
    assert strategy.target_column() == LINEAR_CROSS_SECTIONAL_TARGET
    assert first.filter(pl.col("symbol") == "BTCUSDT")["score"].item() > 0
    assert first.filter(pl.col("symbol") == "SOLUSDT")["score"].item() < 0


def test_linear_cross_sectional_panel_and_frame_paths_match() -> None:
    features, target, context = _frames()
    strategy = build_strategy("linear_cross_sectional", "1", {"ridge_alpha": 1.0})
    train = features.filter(pl.col("decision_time_ms") <= context.train_end_ms)
    test = features.filter(pl.col("decision_time_ms") >= context.test_start_ms)
    panel = PanelData.from_frame(features, feature_columns=LINEAR_CROSS_SECTIONAL_FEATURES)

    frame_scores = (
        strategy.fit(TrainingDataset(features=train, target=target), context=context)
        .score(test, context=context)
        .frame
    )
    panel_scores = (
        strategy.fit_panel(panel, target=target, context=context)
        .score_panel(panel, context=context)
        .frame
    )

    assert frame_scores.equals(panel_scores, null_equal=True)


def test_linear_cross_sectional_requires_aligned_target_and_strict_parameters() -> None:
    features, target, context = _frames()
    strategy = build_strategy("linear_cross_sectional", "1", {"ridge_alpha": 0.01})
    train = features.filter(pl.col("decision_time_ms") <= context.train_end_ms)

    with pytest.raises(ResearchError, match="requires its forward-return target"):
        strategy.fit(TrainingDataset(features=train, target=None), context=context)
    panel = PanelData.from_frame(features, feature_columns=LINEAR_CROSS_SECTIONAL_FEATURES)
    with pytest.raises(ResearchError, match="keys do not align"):
        strategy.fit_panel(panel, target=target.head(-3), context=context)
    with pytest.raises(ResearchError, match="extra_forbidden"):
        build_strategy(
            "linear_cross_sectional",
            "1",
            {"ridge_alpha": 0.1, "future_feature": True},
        )
