"""Fixed residual-momentum strategy extracted from the Phase 3 baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import (
    FoldContext,
    StrategyScores,
    TrainingDataset,
    select_feature_view,
)

RESIDUAL_MOMENTUM_FEATURES = (
    "residual_momentum_1h",
    "residual_momentum_4h",
    "residual_momentum_24h",
)


@dataclass(frozen=True, slots=True)
class ResidualMomentumParameters:
    """Immutable weights for the Phase 3 residual-momentum score."""

    momentum_weight_1h: float
    momentum_weight_4h: float
    momentum_weight_24h: float

    def __post_init__(self) -> None:
        weights = self.as_tuple()
        if any(not math.isfinite(weight) or not 0 <= weight <= 1 for weight in weights):
            raise ResearchError("residual momentum weights must be finite and between zero and one")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-12):
            raise ResearchError("residual momentum weights must sum to one")

    def as_tuple(self) -> tuple[float, float, float]:
        return (
            self.momentum_weight_1h,
            self.momentum_weight_4h,
            self.momentum_weight_24h,
        )


def _validate_context_range(
    frame: pl.DataFrame,
    *,
    start_ms: int,
    end_ms: int,
    role: str,
) -> None:
    if frame.is_empty():
        raise ResearchError(f"{role} cannot be empty")
    minimum = frame["decision_time_ms"].min()
    maximum = frame["decision_time_ms"].max()
    if not isinstance(minimum, int) or not isinstance(maximum, int):
        raise ResearchError(f"{role} has invalid decision timestamps")
    if minimum < start_ms or maximum > end_ms:
        raise ResearchError(f"{role} contains decisions outside its fold context")


def _cross_sectional_zscore(
    values: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    standard_deviation = float(np.std(values))
    if standard_deviation <= 1e-15:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / standard_deviation


def _feature_panel(
    features: pl.DataFrame,
) -> tuple[
    tuple[str, ...],
    np.ndarray[Any, np.dtype[np.int64]],
    dict[str, np.ndarray[Any, np.dtype[np.float64]]],
]:
    symbols = tuple(sorted(str(value) for value in features["symbol"].unique().to_list()))
    times = np.asarray(sorted(features["decision_time_ms"].unique().to_list()), dtype=np.int64)
    if features.height != len(times) * len(symbols):
        raise ResearchError("residual momentum feature panel is incomplete")
    arrays = {
        feature: np.full((len(times), len(symbols)), np.nan, dtype=np.float64)
        for feature in RESIDUAL_MOMENTUM_FEATURES
    }
    time_index = {int(value): index for index, value in enumerate(times)}
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    for row in features.iter_rows(named=True):
        x = time_index[int(row["decision_time_ms"])]
        y = symbol_index[str(row["symbol"])]
        for feature in RESIDUAL_MOMENTUM_FEATURES:
            arrays[feature][x, y] = float(row[feature])
    if any(np.any(~np.isfinite(values)) for values in arrays.values()):
        raise ResearchError("residual momentum features are incomplete or non-finite")
    return symbols, times, arrays


@dataclass(frozen=True, slots=True)
class FittedResidualMomentumStrategy:
    """Frozen, non-trainable residual-momentum strategy for one fold."""

    parameters: ResidualMomentumParameters

    def score(self, features: pl.DataFrame, *, context: FoldContext) -> StrategyScores:
        projected = select_feature_view(
            features,
            required_features=RESIDUAL_MOMENTUM_FEATURES,
        )
        _validate_context_range(
            projected,
            start_ms=context.test_start_ms,
            end_ms=context.test_end_ms,
            role="residual momentum scoring frame",
        )
        symbols, times, arrays = _feature_panel(projected)
        weights = self.parameters.as_tuple()
        score_matrix = np.empty_like(arrays["residual_momentum_1h"])
        for period in range(len(times)):
            score_matrix[period] = (
                weights[0] * _cross_sectional_zscore(arrays["residual_momentum_1h"][period])
                + weights[1] * _cross_sectional_zscore(arrays["residual_momentum_4h"][period])
                + weights[2] * _cross_sectional_zscore(arrays["residual_momentum_24h"][period])
            )
        rows = [
            {
                "decision_time_ms": int(decision_time),
                "symbol": symbol,
                "score": float(score_matrix[time_index, symbol_index]),
            }
            for time_index, decision_time in enumerate(times)
            for symbol_index, symbol in enumerate(symbols)
        ]
        return StrategyScores(pl.DataFrame(rows))


@dataclass(frozen=True, slots=True)
class ResidualMomentumStrategy:
    """Versioned Phase 3 strategy definition; ``fit`` never calibrates on OOS data."""

    parameters: ResidualMomentumParameters
    strategy_id: str = field(default="residual_momentum", init=False)
    strategy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return RESIDUAL_MOMENTUM_FEATURES

    def target_column(self) -> str | None:
        return None

    def fit(
        self,
        train: TrainingDataset,
        *,
        context: FoldContext,
    ) -> FittedResidualMomentumStrategy:
        projected = select_feature_view(
            train.features,
            required_features=self.required_features(),
        )
        _validate_context_range(
            projected,
            start_ms=context.train_start_ms,
            end_ms=context.train_end_ms,
            role="residual momentum training frame",
        )
        _feature_panel(projected)
        if train.target is not None:
            raise ResearchError("residual momentum is non-trainable and does not accept a target")
        return FittedResidualMomentumStrategy(parameters=self.parameters)


__all__ = [
    "RESIDUAL_MOMENTUM_FEATURES",
    "FittedResidualMomentumStrategy",
    "ResidualMomentumParameters",
    "ResidualMomentumStrategy",
]
