"""Portable references to current and legacy research dataset manifests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import orjson

from binance_algo.common.errors import ResearchError
from binance_algo.research.datasets.fingerprints import sha256_path


@dataclass(frozen=True, slots=True)
class DatasetReference:
    dataset_id: str
    manifest_path: str
    dataset_schema_version: int
    feature_set_id: str
    label_id: str
    universe_version: str
    start_time_ms: int
    end_time_ms: int
    row_count: int
    content_checksum: str
    fingerprint_method: str

    def identity_payload(self) -> dict[str, object]:
        """Return experiment identity fields; absolute paths are intentionally excluded."""

        return {
            "dataset_id": self.dataset_id,
            "dataset_schema_version": self.dataset_schema_version,
            "feature_set_id": self.feature_set_id,
            "label_id": self.label_id,
            "universe_version": self.universe_version,
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "row_count": self.row_count,
            "content_checksum": self.content_checksum,
            "fingerprint_method": self.fingerprint_method,
        }


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ResearchError(f"dataset manifest field is not an object: {field}")
    return cast(Mapping[str, Any], value)


def load_dataset_reference(manifest_path: Path) -> DatasetReference:
    """Read lineage_v2 manifests and adapt Phase 3 manifests as legacy_content_hash."""

    manifest_path = manifest_path.resolve()
    try:
        raw = orjson.loads(manifest_path.read_bytes())
    except (OSError, orjson.JSONDecodeError) as exc:
        raise ResearchError(
            f"cannot read research dataset manifest {manifest_path}: {exc}"
        ) from exc
    manifest = _mapping(raw, field="root")
    audit = _mapping(manifest.get("audit", {}), field="audit")
    fingerprint_method = str(manifest.get("fingerprint_method", "legacy_content_hash"))
    dataset_id = str(manifest.get("dataset_id", manifest.get("dataset_version", "")))
    if not dataset_id:
        raise ResearchError("dataset manifest has no dataset identity")
    content_checksum = manifest.get("content_checksum")
    if content_checksum is None:
        parquet_path = manifest_path.with_suffix(".parquet")
        if not parquet_path.is_file():
            raise ResearchError("legacy dataset Parquet is missing beside its manifest")
        content_checksum = sha256_path(parquet_path)
    feature_set = manifest.get("feature_set")
    if isinstance(feature_set, dict):
        feature_set_id = str(feature_set.get("feature_set_id", ""))
    else:
        feature_set_id = str(manifest.get("feature_set_id", manifest.get("feature_version", "")))
    label_definition = manifest.get("label_definition")
    if isinstance(label_definition, dict):
        label_id = str(label_definition.get("label_id", ""))
    else:
        label_id = str(manifest.get("label_id", "legacy:gross_forward_return_1h:v1"))
    return DatasetReference(
        dataset_id=dataset_id,
        manifest_path=str(manifest_path),
        dataset_schema_version=int(manifest.get("dataset_schema_version", 1)),
        feature_set_id=feature_set_id,
        label_id=label_id,
        universe_version=str(manifest.get("universe_version", "")),
        start_time_ms=int(manifest.get("start_time_ms", audit.get("min_decision_time_ms", 0))),
        end_time_ms=int(manifest.get("end_time_ms", audit.get("max_decision_time_ms", 0))),
        row_count=int(manifest.get("row_count", audit.get("row_count", 0))),
        content_checksum=str(content_checksum),
        fingerprint_method=fingerprint_method,
    )


__all__ = ["DatasetReference", "load_dataset_reference"]
