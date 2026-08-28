"""Immutable contracts and lookup registry for supervised research labels."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from types import MappingProxyType

from binance_algo.common.errors import ResearchError


@dataclass(frozen=True, slots=True)
class LabelDefinition:
    label_id: str
    name: str
    version: str
    horizon_minutes: int
    execution_lag_bars: int
    semantics: str
    target_column: str

    def __post_init__(self) -> None:
        if not self.label_id or not self.name or not self.version or not self.target_column:
            raise ResearchError("label identity fields must be non-empty")
        if self.name not in self.label_id or self.version not in self.label_id:
            raise ResearchError("label_id must include the label name and version")
        if self.horizon_minutes <= 0 or self.execution_lag_bars < 0:
            raise ResearchError("label horizon must be positive and lag must be non-negative")

    def to_manifest(self) -> dict[str, object]:
        return {
            "label_id": self.label_id,
            "name": self.name,
            "version": self.version,
            "horizon_minutes": self.horizon_minutes,
            "execution_lag_bars": self.execution_lag_bars,
            "semantics": self.semantics,
            "target_column": self.target_column,
        }


class LabelRegistry:
    def __init__(self, definitions: Iterable[LabelDefinition]) -> None:
        by_id: dict[str, LabelDefinition] = {}
        by_column: dict[str, LabelDefinition] = {}
        for definition in definitions:
            if definition.label_id in by_id or definition.target_column in by_column:
                raise ResearchError(f"duplicate label registration: {definition.label_id}")
            by_id[definition.label_id] = definition
            by_column[definition.target_column] = definition
        self._by_id = MappingProxyType(by_id)
        self._by_column = MappingProxyType(by_column)

    def resolve_id(self, label_id: str) -> LabelDefinition:
        try:
            return self._by_id[label_id]
        except KeyError as exc:
            raise ResearchError(f"label id is not registered: {label_id}") from exc

    def resolve_target_column(self, target_column: str) -> LabelDefinition:
        try:
            return self._by_column[target_column]
        except KeyError as exc:
            raise ResearchError(f"target column is not registered: {target_column}") from exc

    def definitions(self) -> tuple[LabelDefinition, ...]:
        return tuple(self._by_id[key] for key in sorted(self._by_id))


__all__ = ["LabelDefinition", "LabelRegistry"]
