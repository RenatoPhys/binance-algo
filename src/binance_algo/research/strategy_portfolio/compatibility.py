"""Compatibility identities and explicit alignment for strategy sleeves."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from binance_algo.research.experiments.canonical import canonical_sha256
from binance_algo.research.strategy_portfolio.loader import LoadedStrategyComponent
from binance_algo.research.strategy_portfolio.models import (
    AccountingMode,
    AlignmentPolicy,
    AlignmentSpec,
)


@dataclass(frozen=True, slots=True)
class CompatibilityIdentity:
    compatibility_group: str
    grid_sha256: str
    decision_frequency_ms: int
    symbols: tuple[str, ...]
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AlignmentCoverage:
    experiment_id: str
    original_periods: int
    aligned_periods: int
    discarded_periods: int
    coverage: float


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    valid: bool
    policy: AlignmentPolicy
    compatibility_group: str
    issues: tuple[str, ...]
    warnings: tuple[str, ...]
    decision_times: tuple[int, ...]
    coverage: tuple[AlignmentCoverage, ...]


def _grid_sha256(component: LoadedStrategyComponent) -> str:
    digest = hashlib.sha256()
    for fold, decision, execution in component.oos_curve.select(
        "fold", "decision_time_ms", "execution_time_ms"
    ).iter_rows():
        digest.update(f"{int(fold)}\x1f{int(decision)}\x1f{int(execution)}\n".encode())
    return digest.hexdigest()


def _decision_frequency_ms(component: LoadedStrategyComponent) -> int:
    frequencies: set[int] = set()
    for fold in component.oos_curve["fold"].unique(maintain_order=True).to_list():
        values = np.asarray(
            component.oos_curve.filter(component.oos_curve["fold"] == fold)[
                "decision_time_ms"
            ].to_numpy(),
            dtype=np.int64,
        )
        frequencies.update(int(value) for value in np.diff(values) if value > 0)
    return math.gcd(*frequencies) if frequencies else 0


def compatibility_identity(component: LoadedStrategyComponent) -> CompatibilityIdentity:
    frequency = _decision_frequency_ms(component)
    grid_sha256 = _grid_sha256(component)
    payload = {
        "dataset_reference": component.spec.dataset_reference.model_dump(mode="json"),
        "label": component.spec.label.model_dump(mode="json"),
        "execution_model": component.spec.execution_model.model_dump(mode="json"),
        "cost_model": component.spec.cost_model.model_dump(mode="json"),
        "split_plan": component.spec.split_plan.model_dump(mode="json"),
        "grid_sha256": grid_sha256,
        "decision_frequency_ms": frequency,
        "symbols": list(component.symbols),
    }
    return CompatibilityIdentity(
        compatibility_group=canonical_sha256(payload),
        grid_sha256=grid_sha256,
        decision_frequency_ms=frequency,
        symbols=component.symbols,
        payload=payload,
    )


def _different(values: Sequence[object]) -> bool:
    first = values[0]
    return any(value != first for value in values[1:])


def _static_issues(
    components: tuple[LoadedStrategyComponent, ...],
    alignment: AlignmentSpec,
    accounting_mode: AccountingMode,
) -> list[str]:
    issues: list[str] = []
    requirements = {
        "dataset_reference": alignment.require_same_dataset,
        "label": alignment.require_same_label,
        "execution_model": alignment.require_same_execution_model,
        "cost_model": alignment.require_same_cost_model,
        "split_plan": alignment.require_same_split_plan,
    }
    if accounting_mode is AccountingMode.NETTED:
        requirements = dict.fromkeys(requirements, True)
    for field, required in requirements.items():
        if required and _different([getattr(component.spec, field) for component in components]):
            issues.append(f"{field} differs between components")
    identities = [compatibility_identity(component) for component in components]
    if _different([item.decision_frequency_ms for item in identities]):
        issues.append("decision frequency differs between components")
    if any(item.decision_frequency_ms <= 0 for item in identities):
        issues.append("decision frequency is not stable within every component")
    if _different([item.symbols for item in identities]):
        issues.append("weights_json symbol sets differ between components")
    return issues


def _warnings(components: tuple[LoadedStrategyComponent, ...]) -> list[str]:
    warnings: list[str] = []
    if _different([component.spec.code_fingerprint for component in components]):
        warnings.append("components were produced by different code fingerprints")
    if _different([component.spec.validation_plan for component in components]):
        warnings.append("components use different validation profiles/plans")
    if _different([component.research_stage for component in components]):
        warnings.append("components are at different research stages")
    campaign_classes = [
        tuple(
            sorted(
                "confirmation"
                if "confirmation" in name.lower()
                else "full"
                if "full" in name.lower()
                else "development"
                if "development" in name.lower()
                else "other"
                for name in component.campaigns
            )
        )
        for component in components
    ]
    if _different(campaign_classes):
        warnings.append("components mix development, confirmation, full, or unrelated campaigns")
    return warnings


def assess_compatibility(
    components: tuple[LoadedStrategyComponent, ...],
    *,
    alignment: AlignmentSpec,
    accounting_mode: AccountingMode,
) -> CompatibilityReport:
    if not components:
        raise ValueError("compatibility requires at least one component")
    issues = _static_issues(components, alignment, accounting_mode)
    time_lists = [
        tuple(int(value) for value in item.oos_curve["decision_time_ms"]) for item in components
    ]
    if alignment.policy is AlignmentPolicy.STRICT:
        decision_times = time_lists[0]
        if _different(time_lists):
            issues.append("strict alignment requires identical decision timestamps")
        grids = [
            tuple(
                (int(fold), int(execution))
                for fold, execution in item.oos_curve.select(
                    "fold", "execution_time_ms"
                ).iter_rows()
            )
            for item in components
        ]
        if _different(grids):
            issues.append("strict alignment requires identical folds and execution timestamps")
    else:
        common = set(time_lists[0])
        for component_times in time_lists[1:]:
            common.intersection_update(component_times)
        decision_times = tuple(sorted(common))
        if not decision_times:
            issues.append("intersection alignment has no common decision timestamps")
        else:
            observed: list[dict[int, tuple[int, int]]] = []
            for component in components:
                observed.append(
                    {
                        int(decision): (int(fold), int(execution))
                        for fold, decision, execution in component.oos_curve.select(
                            "fold", "decision_time_ms", "execution_time_ms"
                        ).iter_rows()
                        if int(decision) in common
                    }
                )
            for decision in decision_times:
                grid_values = [item[decision] for item in observed]
                if _different(grid_values):
                    issues.append(
                        "intersection alignment has different fold/execution values at "
                        f"decision_time_ms={decision}"
                    )
                    break
    coverage = tuple(
        AlignmentCoverage(
            experiment_id=component.run.experiment_id,
            original_periods=len(times),
            aligned_periods=len(decision_times),
            discarded_periods=len(times) - len(decision_times),
            coverage=len(decision_times) / len(times),
        )
        for component, times in zip(components, time_lists, strict=True)
    )
    identities = [compatibility_identity(component) for component in components]
    group = (
        identities[0].compatibility_group
        if not issues and alignment.policy is AlignmentPolicy.STRICT
        else canonical_sha256(
            {
                "policy": alignment.policy.value,
                "source_groups": sorted(item.compatibility_group for item in identities),
                "decision_times": list(decision_times),
            }
        )
    )
    warnings = _warnings(components)
    if alignment.policy is AlignmentPolicy.INTERSECTION:
        warnings.append("intersection / exploratory: discarded periods invalidate primary ranking")
    return CompatibilityReport(
        valid=not issues,
        policy=alignment.policy,
        compatibility_group=group,
        issues=tuple(dict.fromkeys(issues)),
        warnings=tuple(dict.fromkeys(warnings)),
        decision_times=decision_times,
        coverage=coverage,
    )


__all__ = [
    "AlignmentCoverage",
    "CompatibilityIdentity",
    "CompatibilityReport",
    "assess_compatibility",
    "compatibility_identity",
]
