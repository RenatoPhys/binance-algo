"""Leakage-resistant dataset projections backed by registries and schema roles."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FEATURE_KEY_COLUMNS, select_feature_view
from binance_algo.research.datasets.schemas import (
    RESEARCH_DATASET_SCHEMA_V2,
    ColumnRole,
    DatasetSchema,
)
from binance_algo.research.features.registry import PHASE3_FEATURE_REGISTRY, FeatureRegistry
from binance_algo.research.labels.base import LabelDefinition, LabelRegistry
from binance_algo.research.labels.forward_returns import PHASE3_LABEL_REGISTRY


def build_feature_view(
    frame: pl.DataFrame,
    *,
    required_features: Iterable[str],
    feature_registry: FeatureRegistry = PHASE3_FEATURE_REGISTRY,
    schema: DatasetSchema = RESEARCH_DATASET_SCHEMA_V2,
) -> pl.DataFrame:
    """Project keys plus active, registered FEATURE columns and nothing else."""

    names = tuple(required_features)
    for name in names:
        feature_registry.resolve_name(name)
        try:
            role = schema.role_for(name)
        except ResearchError:
            role = ColumnRole.FEATURE
        if role is not ColumnRole.FEATURE:
            raise ResearchError(f"scoring column is not a FEATURE: {name}")
    return select_feature_view(frame, required_features=names)


def resolve_label(
    *,
    label_id: str | None = None,
    target_column: str | None = None,
    label_registry: LabelRegistry = PHASE3_LABEL_REGISTRY,
) -> LabelDefinition:
    if (label_id is None) == (target_column is None):
        raise ResearchError("select exactly one label_id or target_column")
    if label_id is not None:
        return label_registry.resolve_id(label_id)
    assert target_column is not None
    return label_registry.resolve_target_column(target_column)


def build_target_view(
    frame: pl.DataFrame,
    *,
    label_id: str | None = None,
    target_column: str | None = None,
    label_registry: LabelRegistry = PHASE3_LABEL_REGISTRY,
    schema: DatasetSchema = RESEARCH_DATASET_SCHEMA_V2,
) -> pl.DataFrame:
    """Select one registered TARGET independently from all feature projections."""

    label = resolve_label(
        label_id=label_id,
        target_column=target_column,
        label_registry=label_registry,
    )
    if schema.role_for(label.target_column) is not ColumnRole.TARGET:
        raise ResearchError(f"registered label column is not a TARGET: {label.target_column}")
    required = (*FEATURE_KEY_COLUMNS, label.target_column)
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ResearchError(f"research source frame is missing target columns: {missing}")
    selected = frame.select(required)
    if any(selected[column].null_count() for column in FEATURE_KEY_COLUMNS):
        raise ResearchError("target view contains null research keys")
    if selected.select(FEATURE_KEY_COLUMNS).is_duplicated().any():
        raise ResearchError("target view contains duplicate research keys")
    return selected


__all__ = ["build_feature_view", "build_target_view", "resolve_label"]
