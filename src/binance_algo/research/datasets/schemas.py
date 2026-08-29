"""Explicit column roles for the Phase 3 research dataset schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from binance_algo.common.errors import ResearchError
from binance_algo.research.features.registry import PHASE3_FEATURE_NAMES


class ColumnRole(StrEnum):
    KEY = "KEY"
    FEATURE = "FEATURE"
    TARGET = "TARGET"
    OUTCOME = "OUTCOME"
    METADATA = "METADATA"


@dataclass(frozen=True, slots=True)
class DatasetSchema:
    version: int
    column_roles: Mapping[str, ColumnRole]

    def __post_init__(self) -> None:
        if self.version <= 0 or not self.column_roles:
            raise ResearchError("dataset schema must have a positive version and declared columns")
        object.__setattr__(self, "column_roles", MappingProxyType(dict(self.column_roles)))

    def role_for(self, column: str) -> ColumnRole:
        try:
            return self.column_roles[column]
        except KeyError as exc:
            raise ResearchError(f"column has no registered schema role: {column}") from exc

    def feature_columns(self) -> tuple[str, ...]:
        return tuple(
            column for column, role in self.column_roles.items() if role is ColumnRole.FEATURE
        )

    def to_manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "column_roles": {
                column: self.column_roles[column].value for column in sorted(self.column_roles)
            },
        }


RESEARCH_DATASET_SCHEMA_V2 = DatasetSchema(
    version=2,
    column_roles={
        "decision_time_ms": ColumnRole.KEY,
        "symbol": ColumnRole.KEY,
        **{name: ColumnRole.FEATURE for name in PHASE3_FEATURE_NAMES},
        "future_return_1h": ColumnRole.TARGET,
        "future_residual_return_1h": ColumnRole.TARGET,
        "outcome_quote_volume_1h": ColumnRole.OUTCOME,
        "outcome_funding_rate_1h": ColumnRole.OUTCOME,
        "feature_cutoff_ms": ColumnRole.METADATA,
        "feature_source_min_ms": ColumnRole.METADATA,
        "feature_source_max_ms": ColumnRole.METADATA,
        "execution_time_ms": ColumnRole.METADATA,
        "label_end_time_ms": ColumnRole.METADATA,
        "universe_version": ColumnRole.METADATA,
        "feature_version": ColumnRole.METADATA,
        "execution_lag_bars": ColumnRole.METADATA,
        "dataset_schema_version": ColumnRole.METADATA,
    },
)


def research_dataset_schema(feature_columns: tuple[str, ...]) -> DatasetSchema:
    """Build the explicit schema for a registered feature set without changing v2."""

    if feature_columns == PHASE3_FEATURE_NAMES:
        return RESEARCH_DATASET_SCHEMA_V2
    return DatasetSchema(
        version=3,
        column_roles={
            "decision_time_ms": ColumnRole.KEY,
            "symbol": ColumnRole.KEY,
            **{name: ColumnRole.FEATURE for name in feature_columns},
            "future_return_1h": ColumnRole.TARGET,
            "future_residual_return_1h": ColumnRole.TARGET,
            "outcome_quote_volume_1h": ColumnRole.OUTCOME,
            "outcome_funding_rate_1h": ColumnRole.OUTCOME,
            "feature_cutoff_ms": ColumnRole.METADATA,
            "feature_source_min_ms": ColumnRole.METADATA,
            "feature_source_max_ms": ColumnRole.METADATA,
            "execution_time_ms": ColumnRole.METADATA,
            "label_end_time_ms": ColumnRole.METADATA,
            "universe_version": ColumnRole.METADATA,
            "feature_version": ColumnRole.METADATA,
            "execution_lag_bars": ColumnRole.METADATA,
            "dataset_schema_version": ColumnRole.METADATA,
        },
    )


__all__ = [
    "RESEARCH_DATASET_SCHEMA_V2",
    "ColumnRole",
    "DatasetSchema",
    "research_dataset_schema",
]
