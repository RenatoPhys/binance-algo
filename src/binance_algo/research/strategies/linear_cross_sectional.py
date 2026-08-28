"""Training-only ridge model for causal cross-sectional return forecasts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext, StrategyScores, TrainingDataset
from binance_algo.research.datasets.views import build_feature_view
from binance_algo.research.panel import PanelData, matrix_to_long_frame
from binance_algo.research.strategies.fixed import cross_sectional_zscore

LINEAR_CROSS_SECTIONAL_FEATURES = (
    "residual_momentum_4h",
    "residual_momentum_24h",
    "realized_volatility_24h",
    "intraday_range_4h",
    "quote_volume_zscore_24h",
    "taker_buy_imbalance_1h",
)
LINEAR_CROSS_SECTIONAL_TARGET = "future_return_1h"


@dataclass(frozen=True, slots=True)
class LinearCrossSectionalParameters:
    """Regularization for the fixed-feature ridge regression."""

    ridge_alpha: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.ridge_alpha) or not 1e-6 <= self.ridge_alpha <= 100:
            raise ResearchError("linear cross-sectional ridge alpha must be in [1e-6, 100]")


def _feature_tensor(
    panel: PanelData,
    *,
    start_ms: int,
    end_ms: int,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    arrays = [
        cross_sectional_zscore(
            np.asarray(
                panel.matrix(feature, start_ms=start_ms, end_ms=end_ms),
                dtype=np.float64,
            )
        )
        for feature in LINEAR_CROSS_SECTIONAL_FEATURES
    ]
    return np.stack(arrays, axis=2)


def _aligned_training_target(
    target: pl.DataFrame | None,
    *,
    feature_panel: PanelData,
    context: FoldContext,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    if target is None:
        raise ResearchError("linear cross-sectional training requires its forward-return target")
    target_panel = PanelData.from_frame(
        target,
        outcome_columns=(LINEAR_CROSS_SECTIONAL_TARGET,),
    )
    target_panel.require_complete(role="linear cross-sectional training target")
    feature_slice = feature_panel.time_slice(context.train_start_ms, context.train_end_ms)
    if target_panel.symbols != feature_panel.symbols or not np.array_equal(
        target_panel.times, feature_panel.times[feature_slice]
    ):
        raise ResearchError("linear cross-sectional training target keys do not align")
    values = np.asarray(
        target_panel.outcomes[LINEAR_CROSS_SECTIONAL_TARGET],
        dtype=np.float64,
    )
    centered = values - np.mean(values, axis=1, keepdims=True)
    scale = float(np.std(centered))
    if scale <= 1e-15:
        raise ResearchError("linear cross-sectional training target has no dispersion")
    return np.asarray(np.clip(centered / scale, -5.0, 5.0), dtype=np.float64)


def _fit_coefficients(
    panel: PanelData,
    target: pl.DataFrame | None,
    *,
    parameters: LinearCrossSectionalParameters,
    context: FoldContext,
) -> tuple[float, ...]:
    panel.require_complete_range(
        context.train_start_ms,
        context.train_end_ms,
        role="linear cross-sectional training",
    )
    tensor = _feature_tensor(
        panel,
        start_ms=context.train_start_ms,
        end_ms=context.train_end_ms,
    )
    target_values = _aligned_training_target(
        target,
        feature_panel=panel,
        context=context,
    )
    design = tensor.reshape(-1, len(LINEAR_CROSS_SECTIONAL_FEATURES))
    response = target_values.reshape(-1)
    observation_count = design.shape[0]
    gram = design.T @ design / observation_count
    right_hand_side = design.T @ response / observation_count
    coefficients = np.linalg.solve(
        gram + parameters.ridge_alpha * np.eye(design.shape[1]),
        right_hand_side,
    )
    if not np.all(np.isfinite(coefficients)):
        raise ResearchError("linear cross-sectional fit produced non-finite coefficients")
    return tuple(float(value) for value in coefficients)


def _score_panel(
    panel: PanelData,
    *,
    coefficients: tuple[float, ...],
    context: FoldContext,
) -> StrategyScores:
    panel.require_complete_range(
        context.test_start_ms,
        context.test_end_ms,
        role="linear cross-sectional scoring",
    )
    tensor = _feature_tensor(
        panel,
        start_ms=context.test_start_ms,
        end_ms=context.test_end_ms,
    )
    raw_score = np.tensordot(
        tensor,
        np.asarray(coefficients, dtype=np.float64),
        axes=([2], [0]),
    )
    score = cross_sectional_zscore(raw_score)
    time_slice = panel.time_slice(context.test_start_ms, context.test_end_ms)
    return StrategyScores(
        matrix_to_long_frame(
            times=panel.times[time_slice],
            symbols=panel.symbols,
            value_name="score",
            values=score,
        )
    )


@dataclass(frozen=True, slots=True)
class FittedLinearCrossSectionalStrategy:
    coefficients: tuple[float, ...]

    def score(self, features: pl.DataFrame, *, context: FoldContext) -> StrategyScores:
        projected = build_feature_view(
            features,
            required_features=LINEAR_CROSS_SECTIONAL_FEATURES,
        )
        panel = PanelData.from_frame(
            projected,
            feature_columns=LINEAR_CROSS_SECTIONAL_FEATURES,
        )
        return _score_panel(panel, coefficients=self.coefficients, context=context)

    def score_panel(self, features: PanelData, *, context: FoldContext) -> StrategyScores:
        return _score_panel(features, coefficients=self.coefficients, context=context)


@dataclass(frozen=True, slots=True)
class LinearCrossSectionalStrategy:
    """Fit a regularized linear ranking model independently inside each outer fold."""

    parameters: LinearCrossSectionalParameters
    strategy_id: str = field(default="linear_cross_sectional", init=False)
    strategy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return LINEAR_CROSS_SECTIONAL_FEATURES

    def target_column(self) -> str | None:
        return LINEAR_CROSS_SECTIONAL_TARGET

    def fit(
        self,
        train: TrainingDataset,
        *,
        context: FoldContext,
    ) -> FittedLinearCrossSectionalStrategy:
        projected = build_feature_view(
            train.features,
            required_features=self.required_features(),
        )
        panel = PanelData.from_frame(
            projected,
            feature_columns=self.required_features(),
        )
        coefficients = _fit_coefficients(
            panel,
            train.target,
            parameters=self.parameters,
            context=context,
        )
        return FittedLinearCrossSectionalStrategy(coefficients)

    def fit_panel(
        self,
        train: PanelData,
        *,
        target: pl.DataFrame | None,
        context: FoldContext,
    ) -> FittedLinearCrossSectionalStrategy:
        coefficients = _fit_coefficients(
            train,
            target,
            parameters=self.parameters,
            context=context,
        )
        return FittedLinearCrossSectionalStrategy(coefficients)


__all__ = [
    "LINEAR_CROSS_SECTIONAL_FEATURES",
    "LINEAR_CROSS_SECTIONAL_TARGET",
    "FittedLinearCrossSectionalStrategy",
    "LinearCrossSectionalParameters",
    "LinearCrossSectionalStrategy",
]
