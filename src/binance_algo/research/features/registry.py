"""Explicit feature registry and content-addressed feature-set specifications."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import orjson
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from binance_algo.common.errors import ResearchError
from binance_algo.config import ResearchConfig
from binance_algo.research.features.base import (
    FeatureArray,
    FeatureBundle,
    FeatureComputeContext,
    FeatureDefinition,
    FeatureStatus,
)
from binance_algo.research.features.clock_phase import ClockPhaseBundle
from binance_algo.research.features.flow import FlowBundle
from binance_algo.research.features.funding import FundingBundle
from binance_algo.research.features.microstructure import MicrostructureBundle
from binance_algo.research.features.momentum import ReturnsMomentumBundle
from binance_algo.research.features.pair_state import PairStateBundle
from binance_algo.research.features.path_state import PathStateBundle
from binance_algo.research.features.volatility import VolatilityBundle
from binance_algo.research.features.volume import VolumeBundle


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


class _StrictFeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FeatureBundleDeclaration(_StrictFeatureConfig):
    bundle_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    outputs: tuple[str, ...] = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    output_parameters: dict[str, dict[str, Any]] = Field(default_factory=dict)


class FeatureSetDeclaration(_StrictFeatureConfig):
    feature_set_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    outputs: tuple[str, ...] = Field(min_length=1)
    bundles: tuple[FeatureBundleDeclaration, ...] = Field(min_length=1)


class FeatureBundleRegistry:
    """Closed registry of explicit bundles; arbitrary imports are intentionally unsupported."""

    def __init__(self, bundles: Iterable[FeatureBundle]) -> None:
        by_key: dict[tuple[str, str], FeatureBundle] = {}
        for bundle in bundles:
            key = (bundle.bundle_id, bundle.version)
            if key in by_key:
                raise ResearchError(
                    f"duplicate feature bundle: {bundle.bundle_id}:{bundle.version}"
                )
            by_key[key] = bundle
        self._by_key = MappingProxyType(by_key)

    def resolve(self, bundle_id: str, version: str) -> FeatureBundle:
        try:
            return self._by_key[(bundle_id, version)]
        except KeyError as exc:
            raise ResearchError(f"unsupported feature bundle: {bundle_id}:{version}") from exc

    def bundles(self) -> tuple[FeatureBundle, ...]:
        return tuple(self._by_key[key] for key in sorted(self._by_key))


@dataclass(frozen=True, slots=True)
class ResolvedFeatureBundle:
    bundle: FeatureBundle
    outputs: tuple[str, ...]
    parameters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ResolvedFeaturePlan:
    bundles: tuple[ResolvedFeatureBundle, ...]
    feature_set: FeatureSetSpec


PHASE3_BUNDLE_REGISTRY = FeatureBundleRegistry(
    (
        ReturnsMomentumBundle(),
        VolatilityBundle(),
        VolumeBundle(),
        MicrostructureBundle(),
        FundingBundle(),
        ClockPhaseBundle(),
        FlowBundle(),
        PathStateBundle(),
        PairStateBundle(),
    )
)
PHASE3_FEATURE_REGISTRY = FeatureRegistry(
    definition for bundle in PHASE3_BUNDLE_REGISTRY.bundles() for definition in bundle.definitions()
)
PHASE3_FEATURE_SET_PATH = (
    Path(__file__).resolve().parents[4] / "configs" / "feature_sets" / "phase3_baseline.yaml"
)
ALPHA_REBOOT_FEATURE_SET_PATH = (
    Path(__file__).resolve().parents[4] / "configs" / "feature_sets" / "alpha_reboot_v1.yaml"
)
BUILTIN_FEATURE_SET_PATHS = MappingProxyType(
    {
        "phase3_baseline_features:v1": PHASE3_FEATURE_SET_PATH,
        "alpha_reboot_features:v1": ALPHA_REBOOT_FEATURE_SET_PATH,
    }
)


@lru_cache(maxsize=8)
def load_feature_set_declaration(path: Path) -> FeatureSetDeclaration:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ResearchError("feature-set YAML root must be a mapping")
        return FeatureSetDeclaration.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ResearchError(f"cannot load feature-set file {path}: {exc}") from exc


def _resolve_parameter(value: Any, config: ResearchConfig) -> Any:
    if value == "$research.beta_window_hours":
        return config.beta_window_hours
    if isinstance(value, dict):
        return {str(key): _resolve_parameter(item, config) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_parameter(item, config) for item in value]
    return value


def resolve_feature_plan(
    declaration: FeatureSetDeclaration,
    *,
    config: ResearchConfig,
) -> ResolvedFeaturePlan:
    resolved_bundles: list[ResolvedFeatureBundle] = []
    definitions_by_output: dict[str, FeatureDefinition] = {}
    per_feature_parameters: dict[str, Mapping[str, Any]] = {}
    output_names: list[str] = []
    for item in declaration.bundles:
        bundle = PHASE3_BUNDLE_REGISTRY.resolve(item.bundle_id, item.version)
        definitions = {definition.name: definition for definition in bundle.definitions()}
        unknown = set(item.outputs).difference(definitions)
        if unknown:
            raise ResearchError(
                f"bundle {item.bundle_id}:{item.version} declares unknown outputs: "
                f"{sorted(unknown)}"
            )
        duplicate = set(output_names).intersection(item.outputs)
        if duplicate:
            raise ResearchError(f"feature outputs are declared more than once: {sorted(duplicate)}")
        unknown_parameters = set(item.output_parameters).difference(item.outputs)
        if unknown_parameters:
            raise ResearchError(
                f"bundle output parameters reference unknown outputs: {sorted(unknown_parameters)}"
            )
        parameters = _resolve_parameter(item.parameters, config)
        resolved_bundles.append(
            ResolvedFeatureBundle(
                bundle=bundle,
                outputs=item.outputs,
                parameters=MappingProxyType(dict(parameters)),
            )
        )
        for output in item.outputs:
            definition = definitions[output]
            output_names.append(output)
            definitions_by_output[output] = definition
            if output in item.output_parameters:
                resolved = _resolve_parameter(item.output_parameters[output], config)
                per_feature_parameters[definition.feature_id] = MappingProxyType(dict(resolved))
    if len(set(declaration.outputs)) != len(declaration.outputs):
        raise ResearchError("feature-set output order contains duplicates")
    if set(declaration.outputs) != set(output_names):
        raise ResearchError("feature-set output order differs from its bundle outputs")
    feature_set = FeatureSetSpec(
        feature_set_id=declaration.feature_set_id,
        feature_ids=tuple(definitions_by_output[name].feature_id for name in declaration.outputs),
        per_feature_parameters=per_feature_parameters,
        version=declaration.version,
        description=declaration.description,
    )
    return ResolvedFeaturePlan(bundles=tuple(resolved_bundles), feature_set=feature_set)


def phase3_feature_plan(config: ResearchConfig) -> ResolvedFeaturePlan:
    return resolve_feature_plan(
        load_feature_set_declaration(PHASE3_FEATURE_SET_PATH),
        config=config,
    )


def builtin_feature_plan(
    name: str,
    version: str,
    *,
    config: ResearchConfig,
) -> ResolvedFeaturePlan:
    """Resolve one explicitly registered feature-set declaration by identity."""

    normalized_version = version if version.startswith("v") else f"v{version}"
    identifier = f"{name}:{normalized_version}"
    try:
        path = BUILTIN_FEATURE_SET_PATHS[identifier]
    except KeyError as exc:
        raise ResearchError(f"unsupported feature set: {identifier}") from exc
    plan = resolve_feature_plan(load_feature_set_declaration(path), config=config)
    if plan.feature_set.feature_set_id != identifier:
        raise ResearchError(f"feature-set file identity differs from registry key: {identifier}")
    return plan


def alpha_reboot_feature_plan(config: ResearchConfig) -> ResolvedFeaturePlan:
    return builtin_feature_plan("alpha_reboot_features", "v1", config=config)


PHASE3_FEATURE_NAMES = tuple(load_feature_set_declaration(PHASE3_FEATURE_SET_PATH).outputs)
ALPHA_REBOOT_FEATURE_NAMES = tuple(
    load_feature_set_declaration(ALPHA_REBOOT_FEATURE_SET_PATH).outputs
)


def compute_feature_plan(
    context: FeatureComputeContext,
    plan: ResolvedFeaturePlan,
) -> Mapping[str, FeatureArray]:
    outputs: dict[str, FeatureArray] = {}
    for resolved in plan.bundles:
        bundle_context = replace(context, prior_outputs=MappingProxyType(dict(outputs)))
        computed = resolved.bundle.compute(bundle_context, resolved.parameters)
        definition_names = {definition.name for definition in resolved.bundle.definitions()}
        unknown_computed = set(computed).difference(definition_names)
        if unknown_computed:
            raise ResearchError(
                f"bundle {resolved.bundle.bundle_id} returned undefined outputs: "
                f"{sorted(unknown_computed)}"
            )
        missing = set(resolved.outputs).difference(computed)
        if missing:
            raise ResearchError(
                f"bundle {resolved.bundle.bundle_id} omitted configured outputs: {sorted(missing)}"
            )
        for name in resolved.outputs:
            array = np.asarray(computed[name], dtype=np.float64)
            if array.shape != context.output_shape:
                raise ResearchError(
                    f"feature {name} has shape {array.shape}, expected {context.output_shape}"
                )
            if np.any(np.isinf(array)):
                raise ResearchError(f"feature {name} contains infinite values")
            if not np.any(np.isfinite(array)):
                raise ResearchError(f"feature {name} contains no finite values")
            array.setflags(write=False)
            outputs[name] = array
    return MappingProxyType(outputs)


def phase3_feature_set(config: ResearchConfig) -> FeatureSetSpec:
    """Compatibility wrapper around the explicit Phase 3 bundle declaration."""

    return phase3_feature_plan(config).feature_set


def alpha_reboot_feature_set(config: ResearchConfig) -> FeatureSetSpec:
    return alpha_reboot_feature_plan(config).feature_set


__all__ = [
    "ALPHA_REBOOT_FEATURE_NAMES",
    "ALPHA_REBOOT_FEATURE_SET_PATH",
    "BUILTIN_FEATURE_SET_PATHS",
    "PHASE3_BUNDLE_REGISTRY",
    "PHASE3_FEATURE_NAMES",
    "PHASE3_FEATURE_REGISTRY",
    "PHASE3_FEATURE_SET_PATH",
    "FeatureBundleDeclaration",
    "FeatureBundleRegistry",
    "FeatureRegistry",
    "FeatureSetDeclaration",
    "FeatureSetSpec",
    "ResolvedFeatureBundle",
    "ResolvedFeaturePlan",
    "alpha_reboot_feature_plan",
    "alpha_reboot_feature_set",
    "builtin_feature_plan",
    "compute_feature_plan",
    "load_feature_set_declaration",
    "phase3_feature_plan",
    "phase3_feature_set",
    "resolve_feature_plan",
]
