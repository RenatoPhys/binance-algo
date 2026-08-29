"""Deterministic scaffold for explicitly supplied experiment identifiers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from binance_algo.common.errors import ResearchError
from binance_algo.research.strategy_portfolio.io import replace_bytes
from binance_algo.research.strategy_portfolio.models import MAXIMUM_COMPONENTS


def scaffold_payload(experiment_ids: tuple[str, ...]) -> dict[str, Any]:
    if not experiment_ids:
        raise ResearchError("portfolio scaffold requires at least one --experiment-id")
    if len(experiment_ids) > MAXIMUM_COMPONENTS:
        raise ResearchError(f"portfolio scaffold supports at most {MAXIMUM_COMPONENTS} components")
    if any(not identifier.strip() for identifier in experiment_ids):
        raise ResearchError("portfolio scaffold experiment IDs must be non-empty")
    if len(set(experiment_ids)) != len(experiment_ids):
        raise ResearchError("portfolio scaffold experiment IDs must be unique")
    return {
        "schema_version": 1,
        "portfolios": [
            {
                "portfolio_id": "declared_equal_weight_v1",
                "title": "Declared equal-weight research comparison",
                "description": (
                    "Exploratory allocation declared by the researcher; this is not promoted "
                    "alpha or independent out-of-sample validation."
                ),
                "accounting_mode": "netted",
                "weighting": "equal_weight",
                "components": [
                    {
                        "experiment_id": identifier,
                        "run_id": None,
                        "label": f"Declared component {index}",
                    }
                    for index, identifier in enumerate(experiment_ids, start=1)
                ],
                "alignment": {
                    "policy": "strict",
                    "require_same_dataset": True,
                    "require_same_label": True,
                    "require_same_execution_model": True,
                    "require_same_cost_model": True,
                    "require_same_split_plan": True,
                },
                "analytics": {
                    "correlation_frequency": "daily",
                    "rolling_windows_hours": [720, 2160],
                    "trade_epsilon": 1.0e-10,
                },
            }
        ],
    }


def write_scaffold(path: Path, experiment_ids: tuple[str, ...]) -> Path:
    payload = scaffold_payload(experiment_ids)
    text = yaml.safe_dump(
        payload,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    replace_bytes(path.resolve(), text.encode("utf-8"))
    return path.resolve()


__all__ = ["scaffold_payload", "write_scaffold"]
