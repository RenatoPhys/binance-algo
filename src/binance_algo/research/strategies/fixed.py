"""Shared validation helpers for fixed, non-supervised research strategies."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext
from binance_algo.research.datasets.views import build_feature_view
from binance_algo.research.panel import PanelData


def cross_sectional_zscore(
    values: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    means = np.mean(values, axis=1, keepdims=True)
    standard_deviations = np.std(values, axis=1, keepdims=True)
    return np.divide(
        values - means,
        standard_deviations,
        out=np.zeros_like(values),
        where=standard_deviations > 1e-15,
    )


def projected_panel(
    frame: pl.DataFrame,
    *,
    required_features: Iterable[str],
    start_ms: int,
    end_ms: int,
    role: str,
) -> PanelData:
    features = tuple(required_features)
    projected = build_feature_view(frame, required_features=features)
    if projected.is_empty():
        raise ResearchError(f"{role} cannot be empty")
    minimum = projected["decision_time_ms"].min()
    maximum = projected["decision_time_ms"].max()
    if not isinstance(minimum, int) or not isinstance(maximum, int):
        raise ResearchError(f"{role} has invalid decision timestamps")
    if minimum < start_ms or maximum > end_ms:
        raise ResearchError(f"{role} contains decisions outside its fold context")
    panel = PanelData.from_frame(projected, feature_columns=features)
    panel.require_complete(role=role)
    return panel


def validate_fixed_panel_fit(
    panel: PanelData,
    *,
    required_features: Iterable[str],
    target: pl.DataFrame | None,
    context: FoldContext,
    role: str,
) -> None:
    if target is not None:
        raise ResearchError(f"{role} is non-trainable and does not accept a target")
    panel.require_complete_range(context.train_start_ms, context.train_end_ms, role=role)
    for feature in required_features:
        panel.matrix(feature, start_ms=context.train_start_ms, end_ms=context.train_end_ms)


__all__ = ["cross_sectional_zscore", "projected_panel", "validate_fixed_panel_fit"]
