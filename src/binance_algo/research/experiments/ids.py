"""Deterministic identifiers for experiment definitions and completed results."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from binance_algo.research.experiments.canonical import canonical_sha256
from binance_algo.research.experiments.models import ExperimentSpec


def experiment_id(spec: ExperimentSpec) -> str:
    """Hash only immutable scientific inputs; metrics and run timestamps are absent."""

    return canonical_sha256(spec)


def deterministic_run_id(*, experiment_id_value: str, attempt: int) -> str:
    if not experiment_id_value or attempt < 1:
        raise ValueError("run identity requires an experiment and positive attempt")
    payload = f"research-run-v1\x1f{experiment_id_value}\x1f{attempt}".encode()
    return hashlib.sha256(payload).hexdigest()


def result_digest(
    *,
    metrics: Mapping[str, Any],
    artifact_checksums: Mapping[str, str],
) -> str:
    """Hash deterministic outputs separately from the experiment definition."""

    return canonical_sha256(
        {
            "metrics": dict(metrics),
            "artifact_checksums": dict(artifact_checksums),
        }
    )


__all__ = ["deterministic_run_id", "experiment_id", "result_digest"]
