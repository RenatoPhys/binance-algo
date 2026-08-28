"""Explicit feature registry and content-addressed feature-set specifications."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import orjson

from binance_algo.common.errors import ResearchError
from binance_algo.config import ResearchConfig
from binance_algo.research.features.base import FeatureDefinition, FeatureStatus
from binance_algo.research.features.funding import FUNDING_FEATURES
from binance_algo.research.features.microstructure import MICROSTRUCTURE_FEATURES
from binance_algo.research.features.momentum import MOMENTUM_FEATURES
from binance_algo.research.features.volatility import VOLATILITY_FEATURES
from binance_algo.research.features.volume import VOLUME_FEATURES


class FeatureRegistry:
    def __init__(self, definitions: Iterable[FeatureDefinition]) -> None:
        active_by_name: dict[str, FeatureDefinition] = {}
        by_id: dict[str, FeatureDefinition] = {}
        for definition in definitions:
            if definition.feature_id in by_id:
                raise ResearchError(f"duplicate feature registration: {definition.feature_id}")
            by_id[definition.feature_id] = definition
            if definition.status is FeatureStatus.ACTIVE:
                if definition.name in active_by_name:
                    raise ResearchError(
                        f"multiple active feature versions registered: {definition.name}"
                    )
                active_by_name[definition.name] = definition
        self._active_by_name = MappingProxyType(active_by_name)
        self._by_id = MappingProxyType(by_id)

    def resolve_name(self, name: str, *, require_active: bool = True) -> FeatureDefinition:
        try:
            definition = self._active_by_name[name]
        except KeyError as exc:
            matches = tuple(item for item in self._by_id.values() if item.name == name)
            if not matches:
                raise ResearchError(f"feature is not registered: {name}") from exc
            if require_active:
                raise ResearchError(f"feature has no active version: {name}") from exc
            return sorted(matches, key=lambda item: item.feature_id)[-1]
        else:
            return definition

    def resolve_id(self, feature_id: str, *, require_active: bool = True) -> FeatureDefinition:
        try:
            definition = self._by_id[feature_id]
        except KeyError as exc:
            raise ResearchError(f"feature id is not registered: {feature_id}") from exc
        if require_active and definition.status is not FeatureStatus.ACTIVE:
            raise ResearchError(f"feature is not active: {definition.feature_id}")
        return definition

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return tuple(self._by_id[key] for key in sorted(self._by_id))


@dataclass(frozen=True, slots=True)
class FeatureSetSpec:
    feature_set_id: str
    feature_ids: tuple[str, ...]
    per_feature_parameters: Mapping[str, Mapping[str, Any]]
    version: str
    description: str
    canonical_checksum: str = ""

    def __post_init__(self) -> None:
        if not self.feature_set_id or not self.version or not self.feature_ids:
            raise ResearchError("feature set identity and members must be non-empty")
        if len(set(self.feature_ids)) != len(self.feature_ids):
            raise ResearchError("feature set cannot contain duplicate feature ids")
        parameters = {
            feature_id: MappingProxyType(dict(values))
            for feature_id, values in self.per_feature_parameters.items()
        }
        unknown_parameters = set(parameters).difference(self.feature_ids)
        if unknown_parameters:
            raise ResearchError(
                f"feature set parameters reference unknown members: {sorted(unknown_parameters)}"
            )
        object.__setattr__(self, "per_feature_parameters", MappingProxyType(parameters))
        checksum = hashlib.sha256(
            orjson.dumps(self.identity_payload(), option=orjson.OPT_SORT_KEYS)
        ).hexdigest()
        if self.canonical_checksum and self.canonical_checksum != checksum:
            raise ResearchError("feature set canonical checksum does not match its contents")
        object.__setattr__(self, "canonical_checksum", checksum)

    def identity_payload(self) -> dict[str, object]:
        return {
            "feature_set_id": self.feature_set_id,
            "feature_ids": sorted(self.feature_ids),
            "per_feature_parameters": {
                key: dict(self.per_feature_parameters[key])
                for key in sorted(self.per_feature_parameters)
            },
            "version": self.version,
            "description": self.description,
        }

    def to_manifest(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "declared_feature_order": self.feature_ids,
            "canonical_checksum": self.canonical_checksum,
        }


PHASE3_FEATURE_REGISTRY = FeatureRegistry(
    (
        *MOMENTUM_FEATURES,
        *VOLATILITY_FEATURES,
        *VOLUME_FEATURES,
        *MICROSTRUCTURE_FEATURES,
        *FUNDING_FEATURES,
    )
)

PHASE3_FEATURE_NAMES = (
    "log_return_5m",
    "log_return_15m",
    "log_return_1h",
    "log_return_4h",
    "log_return_24h",
    "realized_volatility_24h",
    "intraday_range_4h",
    "quote_volume_1h",
    "quote_volume_zscore_24h",
    "taker_buy_imbalance_1h",
    "benchmark_return_1h",
    "rolling_beta",
    "residual_momentum_1h",
    "residual_momentum_4h",
    "residual_momentum_24h",
    "market_volatility_regime",
    "funding_rate_current",
    "funding_rate_change",
)


def phase3_feature_set(config: ResearchConfig) -> FeatureSetSpec:
    feature_ids = tuple(
        PHASE3_FEATURE_REGISTRY.resolve_name(name).feature_id for name in PHASE3_FEATURE_NAMES
    )
    rolling_beta_id = PHASE3_FEATURE_REGISTRY.resolve_name("rolling_beta").feature_id
    return FeatureSetSpec(
        feature_set_id="phase3_baseline_features:v1",
        feature_ids=feature_ids,
        per_feature_parameters={
            rolling_beta_id: {"beta_window_hours": config.beta_window_hours},
        },
        version="v1",
        description="Feature set extracted without numerical changes from the Phase 3 baseline.",
    )


__all__ = [
    "PHASE3_FEATURE_NAMES",
    "PHASE3_FEATURE_REGISTRY",
    "FeatureRegistry",
    "FeatureSetSpec",
    "phase3_feature_set",
]
