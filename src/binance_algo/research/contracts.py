"""Shared, offline contracts for leakage-controlled research components."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import polars as pl

from binance_algo.common.errors import ResearchError

FEATURE_KEY_COLUMNS = ("decision_time_ms", "symbol")
SCORE_REQUIRED_COLUMNS = (*FEATURE_KEY_COLUMNS, "score")
FORBIDDEN_FEATURE_PREFIXES = ("future_", "outcome_", "label_")


def validate_feature_names(feature_names: Iterable[str]) -> tuple[str, ...]:
    """Validate a strategy's explicit feature declaration and preserve its order."""

    names = tuple(feature_names)
    if any(not name for name in names):
        raise ResearchError("feature names must be non-empty")
    if len(set(names)) != len(names):
        raise ResearchError("feature names must be unique")
    key_names = sorted(name for name in names if name in FEATURE_KEY_COLUMNS)
    if key_names:
        raise ResearchError(f"key columns cannot be declared as features: {key_names}")
    forbidden = sorted(name for name in names if name.startswith(FORBIDDEN_FEATURE_PREFIXES))
    if forbidden:
        raise ResearchError(f"outcome or label columns cannot be research features: {forbidden}")
    return names


def _require_columns(frame: pl.DataFrame, required: Iterable[str], *, role: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ResearchError(f"{role} is missing required columns: {missing}")


def _validate_keys(frame: pl.DataFrame, *, role: str) -> None:
    _require_columns(frame, FEATURE_KEY_COLUMNS, role=role)
    if any(frame[column].null_count() for column in FEATURE_KEY_COLUMNS):
        raise ResearchError(f"{role} contains null research keys")
    if frame.select(FEATURE_KEY_COLUMNS).is_duplicated().any():
        raise ResearchError(f"{role} contains duplicate research keys")


def _validate_feature_frame(frame: pl.DataFrame, *, role: str) -> None:
    _validate_keys(frame, role=role)
    forbidden = sorted(
        column for column in frame.columns if column.startswith(FORBIDDEN_FEATURE_PREFIXES)
    )
    if forbidden:
        raise ResearchError(f"{role} contains outcome or label columns: {forbidden}")


def select_feature_view(
    frame: pl.DataFrame,
    *,
    required_features: Iterable[str],
) -> pl.DataFrame:
    """Project source data to keys plus declared features, excluding outcomes by construction."""

    feature_names = validate_feature_names(required_features)
    selected_columns = (*FEATURE_KEY_COLUMNS, *feature_names)
    _require_columns(frame, selected_columns, role="research source frame")
    selected = frame.select(selected_columns)
    _validate_feature_frame(selected, role="feature view")
    return selected


@dataclass(frozen=True, slots=True)
class FoldContext:
    """Immutable outer walk-forward boundaries supplied to fit and score."""

    fold: int
    train_start_ms: int
    train_end_ms: int
    test_start_ms: int
    test_end_ms: int
    embargo_bars: int
    random_seed: int

    def __post_init__(self) -> None:
        if self.fold < 1:
            raise ResearchError("fold number must be positive")
        if not (self.train_start_ms <= self.train_end_ms < self.test_start_ms <= self.test_end_ms):
            raise ResearchError("fold boundaries must be ordered and train must precede test")
        if self.embargo_bars < 0:
            raise ResearchError("fold embargo cannot be negative")


@dataclass(frozen=True, slots=True)
class TrainingDataset:
    """Training features and an optional, separately held target frame."""

    features: pl.DataFrame
    target: pl.DataFrame | None

    def __post_init__(self) -> None:
        _validate_feature_frame(self.features, role="training features")
        if self.target is None:
            return
        _validate_keys(self.target, role="training target")
        target_columns = set(self.target.columns).difference(FEATURE_KEY_COLUMNS)
        if not target_columns:
            raise ResearchError("training target must contain an explicit target column")
        feature_keys = self.features.select(FEATURE_KEY_COLUMNS)
        target_keys = self.target.select(FEATURE_KEY_COLUMNS)
        if not feature_keys.equals(target_keys, null_equal=True):
            raise ResearchError("training feature and target keys must align exactly")


@dataclass(frozen=True, slots=True)
class StrategyScores:
    """Validated long-form strategy output for one scoring interval."""

    frame: pl.DataFrame

    def __post_init__(self) -> None:
        _validate_keys(self.frame, role="strategy scores")
        _require_columns(self.frame, SCORE_REQUIRED_COLUMNS, role="strategy scores")
        try:
            scores = self.frame["score"].cast(pl.Float64, strict=True)
        except (TypeError, ValueError, pl.exceptions.PolarsError) as exc:
            raise ResearchError("strategy scores must be numeric") from exc
        if scores.null_count() or not scores.is_finite().all():
            raise ResearchError("strategy scores must be finite and non-null")


__all__ = [
    "FEATURE_KEY_COLUMNS",
    "FORBIDDEN_FEATURE_PREFIXES",
    "SCORE_REQUIRED_COLUMNS",
    "FoldContext",
    "StrategyScores",
    "TrainingDataset",
    "select_feature_view",
    "validate_feature_names",
]
