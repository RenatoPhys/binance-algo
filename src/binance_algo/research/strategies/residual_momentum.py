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
)
from binance_algo.research.datasets.views import build_feature_view
from binance_algo.research.panel import PanelData, matrix_to_long_frame

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
    if values.ndim == 1:
        standard_deviation = float(np.std(values))
        if standard_deviation <= 1e-15:
            return np.zeros_like(values)
        return (values - float(np.mean(values))) / standard_deviation
    means = np.mean(values, axis=1, keepdims=True)
    standard_deviations = np.std(values, axis=1, keepdims=True)
    return np.divide(
        values - means,
        standard_deviations,
        out=np.zeros_like(values),
        where=standard_deviations > 1e-15,
    )


def _feature_panel(
    features: pl.DataFrame,
) -> tuple[
    tuple[str, ...],
    np.ndarray[Any, np.dtype[np.int64]],
    dict[str, np.ndarray[Any, np.dtype[np.float64]]],
]:
    panel = PanelData.from_frame(features, feature_columns=RESIDUAL_MOMENTUM_FEATURES)
    panel.require_complete(role="residual momentum feature")
    return panel.symbols, panel.times, dict(panel.features)


def _score_panel_data(
    panel: PanelData,
    *,
    parameters: ResidualMomentumParameters,
    context: FoldContext,
) -> StrategyScores:
    panel.require_complete_range(
        context.test_start_ms,
        context.test_end_ms,
        role="residual momentum scoring",
    )
    time_slice = panel.time_slice(context.test_start_ms, context.test_end_ms)
    times = panel.times[time_slice]
    arrays = {
        feature: np.asarray(
            panel.matrix(
                feature,
                start_ms=context.test_start_ms,
                end_ms=context.test_end_ms,
            ),
            dtype=np.float64,
        )
        for feature in RESIDUAL_MOMENTUM_FEATURES
    }
    weights = parameters.as_tuple()
    score_matrix = (
        weights[0] * _cross_sectional_zscore(arrays["residual_momentum_1h"])
        + weights[1] * _cross_sectional_zscore(arrays["residual_momentum_4h"])
        + weights[2] * _cross_sectional_zscore(arrays["residual_momentum_24h"])
    )
    return StrategyScores(
        matrix_to_long_frame(
            times=times,
            symbols=panel.symbols,
            value_name="score",
            values=score_matrix,
        )
    )


@dataclass(frozen=True, slots=True)
class FittedResidualMomentumStrategy:
    """Frozen, non-trainable residual-momentum strategy for one fold."""

    parameters: ResidualMomentumParameters

    def score(self, features: pl.DataFrame, *, context: FoldContext) -> StrategyScores:
        projected = build_feature_view(
            features,
            required_features=RESIDUAL_MOMENTUM_FEATURES,
        )
        _validate_context_range(
            projected,
            start_ms=context.test_start_ms,
            end_ms=context.test_end_ms,
            role="residual momentum scoring frame",
        )
        return _score_panel_data(
            PanelData.from_frame(projected, feature_columns=RESIDUAL_MOMENTUM_FEATURES),
            parameters=self.parameters,
            context=context,
        )

    def score_panel(self, features: PanelData, *, context: FoldContext) -> StrategyScores:
        return _score_panel_data(features, parameters=self.parameters, context=context)


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
        projected = build_feature_view(
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

    def fit_panel(
        self,
        train: PanelData,
        *,
        target: pl.DataFrame | None,
        context: FoldContext,
    ) -> FittedResidualMomentumStrategy:
        if target is not None:
            raise ResearchError("residual momentum is non-trainable and does not accept a target")
        train.require_complete_range(
            context.train_start_ms,
            context.train_end_ms,
            role="residual momentum training",
        )
        for feature in self.required_features():
            train.matrix(
                feature,
                start_ms=context.train_start_ms,
                end_ms=context.train_end_ms,
            )
        return FittedResidualMomentumStrategy(parameters=self.parameters)


__all__ = [
    "RESIDUAL_MOMENTUM_FEATURES",
    "FittedResidualMomentumStrategy",
    "ResidualMomentumParameters",
    "ResidualMomentumStrategy",
]
