"""Explicit synchronization of built-in research definitions into the durable registry."""

from __future__ import annotations

from dataclasses import dataclass

from binance_algo.config import ResearchConfig
from binance_algo.research.experiments.store import ResearchStore
from binance_algo.research.features.registry import (
    PHASE3_FEATURE_REGISTRY,
    alpha_reboot_feature_set,
    phase3_feature_set,
)


@dataclass(frozen=True, slots=True)
class RegistrySyncResult:
    feature_count: int
    feature_set_id: str
    feature_set_ids: tuple[str, ...] = ()


def sync_builtin_registry(
    store: ResearchStore,
    *,
    research_config: ResearchConfig,
) -> RegistrySyncResult:
    """Register known definitions explicitly; importing modules never mutates the database."""

    definitions = PHASE3_FEATURE_REGISTRY.definitions()
    for definition in definitions:
        store.register_feature_definition(definition)
    feature_sets = (
        phase3_feature_set(research_config),
        alpha_reboot_feature_set(research_config),
    )
    for feature_set in feature_sets:
        store.register_feature_set(feature_set)
    return RegistrySyncResult(
        feature_count=len(definitions),
        feature_set_id=feature_sets[0].feature_set_id,
        feature_set_ids=tuple(item.feature_set_id for item in feature_sets),
    )


__all__ = ["RegistrySyncResult", "sync_builtin_registry"]
