"""Immutable contracts for registered, point-in-time research features."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError


class FeatureStatus(StrEnum):
    ACTIVE = "ACTIVE"
    DEPRECATED = "DEPRECATED"
    SUPERSEDED = "SUPERSEDED"


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    feature_id: str
    name: str
    version: str
    description: str
    dtype: str
    lookback: str
    timestamp_semantics: str
    required_datasets: tuple[str, ...]
    required_columns: tuple[str, ...]
    implementation_path: str
    parameters: Mapping[str, Any]
    status: FeatureStatus = FeatureStatus.ACTIVE

    def __post_init__(self) -> None:
        if not self.feature_id or not self.name or not self.version:
            raise ResearchError("feature identity fields must be non-empty")
        if self.name not in self.feature_id or self.version not in self.feature_id:
            raise ResearchError("feature_id must include the feature name and version")
        if not self.required_datasets or not self.required_columns:
            raise ResearchError(f"feature {self.feature_id} must declare its source dependencies")
        try:
            status = FeatureStatus(self.status)
        except ValueError as exc:
            raise ResearchError(f"feature has an invalid status: {self.status}") from exc
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))

    def to_manifest(self) -> dict[str, object]:
        return {
            "feature_id": self.feature_id,
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "dtype": self.dtype,
            "lookback": self.lookback,
            "timestamp_semantics": self.timestamp_semantics,
            "required_datasets": self.required_datasets,
            "required_columns": self.required_columns,
            "implementation_path": self.implementation_path,
            "parameters": dict(self.parameters),
            "status": self.status.value,
        }


FeatureArray = np.ndarray[Any, np.dtype[np.float64]]


@dataclass(frozen=True, slots=True)
class FeatureComputeContext:
    """Raw causal inputs plus outputs produced by earlier explicit bundles."""

    symbols: tuple[str, ...]
    decision_indices: np.ndarray[Any, np.dtype[np.int64]]
    decision_times: np.ndarray[Any, np.dtype[np.int64]]
    log_close: FeatureArray
    minute_log_returns: FeatureArray
    highs: FeatureArray
    lows: FeatureArray
    quote_volume: FeatureArray
    taker_quote_volume: FeatureArray
    funding: pl.DataFrame
    prior_outputs: Mapping[str, FeatureArray]

    @property
    def output_shape(self) -> tuple[int, int]:
        return (len(self.decision_indices), len(self.symbols))

    def require_output(self, name: str) -> FeatureArray:
        try:
            return self.prior_outputs[name]
        except KeyError as exc:
            raise ResearchError(f"feature bundle dependency is unavailable: {name}") from exc


class FeatureBundle(Protocol):
    bundle_id: str
    version: str

    def definitions(self) -> tuple[FeatureDefinition, ...]: ...

    def compute(
        self,
        context: FeatureComputeContext,
        parameters: Mapping[str, Any],
    ) -> Mapping[str, FeatureArray]: ...


__all__ = [
    "FeatureArray",
    "FeatureBundle",
    "FeatureComputeContext",
    "FeatureDefinition",
    "FeatureStatus",
]
