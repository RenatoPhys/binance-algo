"""Deterministic local inventory of successful research experiments and artifacts."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

from binance_algo.research.experiments.canonical import canonical_json
from binance_algo.research.experiments.models import MetricScope
from binance_algo.research.experiments.store import ResearchMetricRecord, ResearchStore
from binance_algo.research.strategy_portfolio.compatibility import compatibility_identity
from binance_algo.research.strategy_portfolio.io import replace_bytes
from binance_algo.research.strategy_portfolio.loader import (
    REQUIRED_ARTIFACT_TYPES,
    load_strategy_component,
)
from binance_algo.research.strategy_portfolio.models import StrategyPortfolioComponent

INVENTORY_SCHEMA_VERSION = 1


def _metric_inventory(metrics: tuple[ResearchMetricRecord, ...]) -> dict[str, Any]:
    summary: dict[str, float] = {}
    stress: dict[str, dict[str, float]] = {}
    for metric in metrics:
        if metric.scope is MetricScope.TEST and metric.fold is None and metric.regime is None:
            summary[metric.metric_name] = metric.metric_value
        elif metric.scope is MetricScope.STRESS and metric.regime is not None:
            stress.setdefault(metric.regime, {})[metric.metric_name] = metric.metric_value
    return {
        "summary": {name: summary[name] for name in sorted(summary)},
        "stress": {
            scenario: {name: values[name] for name in sorted(values)}
            for scenario, values in sorted(stress.items())
        },
    }


def _validation_profile(parameters: dict[str, Any]) -> str:
    profile = parameters.get("profile")
    if isinstance(profile, str):
        return profile
    return "full" if parameters.get("bootstrap_samples") is not None else "discovery"


def build_strategy_portfolio_inventory(
    *,
    store: ResearchStore,
    data_root: Path,
) -> dict[str, Any]:
    """Audit the newest verified successful run for every successful experiment."""

    records: list[dict[str, Any]] = []
    for experiment_id in sorted(store.list_experiment_ids()):
        successful = store.list_successful_runs(experiment_id)
        if not successful:
            continue
        spec = store.get_experiment(experiment_id)
        if spec is None:
            continue
        declaration = StrategyPortfolioComponent(
            experiment_id=experiment_id,
            label=f"{spec.strategy.component_id}:{spec.strategy.version}",
            capital_weight=Decimal(1),
        )
        campaigns = store.campaigns_for_experiment(experiment_id)
        promotions = store.list_promotions(experiment_id)
        base: dict[str, Any] = {
            "experiment_id": experiment_id,
            "strategy": {
                "component_id": spec.strategy.component_id,
                "version": spec.strategy.version,
                "parameters": spec.model_dump(mode="json")["strategy_parameters"],
            },
            "portfolio_policy": {
                "component_id": spec.portfolio_policy.component_id,
                "version": spec.portfolio_policy.version,
                "parameters": spec.model_dump(mode="json")["portfolio_parameters"],
            },
            "hypothesis_id": spec.hypothesis_id,
            "campaigns": [
                {"campaign_id": campaign.campaign_id, "name": campaign.name}
                for campaign in campaigns
            ],
            "validation_profile": _validation_profile(spec.validation_plan.parameters),
            "dataset_reference": spec.dataset_reference.model_dump(mode="json"),
            "label": spec.label.model_dump(mode="json"),
            "split_plan": spec.split_plan.model_dump(mode="json"),
            "execution_model": spec.execution_model.model_dump(mode="json"),
            "cost_model": spec.cost_model.model_dump(mode="json"),
            "code_fingerprint": spec.code_fingerprint.model_dump(mode="json"),
            "promotion_events": [event.to_spec().model_dump(mode="json") for event in promotions],
        }
        try:
            component = load_strategy_component(
                store=store,
                data_root=data_root,
                declaration=declaration,
            )
        except Exception as exc:
            latest = successful[-1]
            records.append(
                {
                    **base,
                    "status": "INVALID",
                    "run_id": latest.run_id,
                    "run_status": latest.status.value,
                    "error": str(exc),
                    "research_stage": "UNKNOWN",
                    "metrics": _metric_inventory(store.list_metrics(latest.run_id)),
                    "artifacts": {},
                    "verification": {"valid": False, "issues": [str(exc)]},
                    "window": None,
                    "compatibility_group": None,
                }
            )
            continue
        artifact_types = {artifact.artifact_type for artifact in component.artifacts}
        availability = {
            name: name in artifact_types for name in sorted((*REQUIRED_ARTIFACT_TYPES, "positions"))
        }
        records.append(
            {
                **base,
                "status": "VALID",
                "run_id": component.run.run_id,
                "run_status": component.run.status.value,
                "result_digest": component.run.result_digest,
                "research_stage": component.research_stage.value,
                "metrics": _metric_inventory(store.list_metrics(component.run.run_id)),
                "artifacts": {
                    "available": availability,
                    "checksums": {
                        name: component.source_checksums[name]
                        for name in sorted(component.source_checksums)
                    },
                },
                "verification": {
                    "valid": True,
                    "issues": [],
                    "checked_files": len(component.artifacts),
                },
                "window": {
                    "start_time_ms": component.start_time_ms,
                    "end_time_ms": component.end_time_ms,
                    "periods": component.oos_curve.height,
                    "folds": component.oos_curve["fold"].n_unique(),
                },
                "compatibility_group": compatibility_identity(component).compatibility_group,
            }
        )
    return {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "successful_experiments": len(records),
        "verified_experiments": sum(item["status"] == "VALID" for item in records),
        "invalid_experiments": sum(item["status"] != "VALID" for item in records),
        "experiments": records,
    }


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_inventory_markdown(inventory: dict[str, Any]) -> str:
    lines = [
        "# Research strategy portfolio inventory",
        "",
        "> Local audit output. It is not alpha promotion evidence and is not part of any "
        "run digest.",
        "",
        f"- Successful experiments: {inventory['successful_experiments']}",
        f"- Verified experiments: {inventory['verified_experiments']}",
        f"- Invalid experiments: {inventory['invalid_experiments']}",
        "",
        "| Experiment | Run | Strategy | Policy | Profile | Stage | Periods | Folds | Group | "
        "Status |",
        "|---|---|---|---|---|---|---:|---:|---|---|",
    ]
    for item in inventory["experiments"]:
        window = item["window"] or {}
        strategy = item["strategy"]
        policy = item["portfolio_policy"]
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    item["experiment_id"],
                    item["run_id"],
                    f"{strategy['component_id']}:{strategy['version']}",
                    f"{policy['component_id']}:{policy['version']}",
                    item["validation_profile"],
                    item["research_stage"],
                    window.get("periods", "—"),
                    window.get("folds", "—"),
                    item["compatibility_group"] or "—",
                    item["status"],
                )
            )
            + " |"
        )
        if item.get("error"):
            lines.extend(("", f"- `{item['experiment_id']}`: {_cell(item['error'])}", ""))
    return "\n".join(lines) + "\n"


def write_strategy_portfolio_inventory(
    *,
    store: ResearchStore,
    data_root: Path,
    reports_root: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    inventory = build_strategy_portfolio_inventory(store=store, data_root=data_root)
    json_path = reports_root.resolve() / "research_strategy_portfolio_inventory.json"
    markdown_path = reports_root.resolve() / "research_strategy_portfolio_inventory.md"
    replace_bytes(json_path, canonical_json(inventory) + b"\n")
    replace_bytes(markdown_path, render_inventory_markdown(inventory).encode("utf-8"))
    return json_path, markdown_path, inventory


__all__ = [
    "INVENTORY_SCHEMA_VERSION",
    "build_strategy_portfolio_inventory",
    "render_inventory_markdown",
    "write_strategy_portfolio_inventory",
]
