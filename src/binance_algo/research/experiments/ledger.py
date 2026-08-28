"""Derived JSON/Markdown views over the durable feature-evaluation ledger."""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from binance_algo.common.errors import ResearchError
from binance_algo.research.experiments.store import (
    CampaignRecord,
    FeatureEvaluationRecord,
    ResearchStore,
)


@dataclass(frozen=True, slots=True)
class LedgerReportResult:
    subject: str
    evaluation_count: int
    report_json_path: Path
    report_markdown_path: Path


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        if temporary.exists():
            temporary.unlink()
        raise ResearchError(f"cannot write research-ledger report {path}: {exc}") from exc


def _evaluation_payload(record: FeatureEvaluationRecord) -> dict[str, Any]:
    return {
        "evaluation_id": record.evaluation_id,
        "run_id": record.run_id,
        "experiment_id": record.experiment_id,
        "hypothesis_id": record.hypothesis_id,
        "feature_id": record.feature_id,
        "evaluation_type": record.evaluation_type.value,
        "scope": record.scope,
        "metric_name": record.metric_name,
        "metric_value": record.metric_value,
        "decision": record.decision.value,
        "decision_reason": record.decision_reason,
        "context": dict(record.context),
        "created_at_ms": record.created_at_ms,
    }


def _campaign_payload(campaign: CampaignRecord) -> dict[str, Any]:
    return {
        "campaign_id": campaign.campaign_id,
        "name": campaign.name,
        "hypothesis_id": campaign.hypothesis_id,
        "status": campaign.status.value,
        "trial_count": campaign.trial_count,
    }


def _write_report(
    *,
    subject: str,
    title: str,
    evaluations: tuple[FeatureEvaluationRecord, ...],
    campaigns: tuple[CampaignRecord, ...],
    directory: Path,
) -> LedgerReportResult:
    json_path = directory / "history.json"
    markdown_path = directory / "history.md"
    payload = {
        "subject": subject,
        "evaluation_count": len(evaluations),
        "campaigns": [_campaign_payload(campaign) for campaign in campaigns],
        "evaluations": [_evaluation_payload(record) for record in evaluations],
        "latest_decision": evaluations[-1].decision.value if evaluations else None,
        "latest_decision_reason": evaluations[-1].decision_reason if evaluations else None,
    }
    _atomic_write(json_path, orjson.dumps(payload, option=orjson.OPT_SORT_KEYS) + b"\n")
    lines = [
        f"# {title}",
        "",
        "> Derived report. The SQLite research registry is the source of truth.",
        "",
        f"- Evaluations: {len(evaluations)}",
        f"- Campaigns: {len(campaigns)}",
        f"- Latest decision: {payload['latest_decision'] or '-'}",
        f"- Latest reason: {payload['latest_decision_reason'] or '-'}",
        "",
        "| Feature | Hypothesis | Metric | Delta/value | Decision | Run |",
        "|---|---|---|---:|---|---|",
    ]
    for record in evaluations:
        value = "-" if record.metric_value is None else f"{record.metric_value:.8g}"
        lines.append(
            f"| `{record.feature_id}` | `{record.hypothesis_id}` | "
            f"{record.metric_name} | {value} | {record.decision.value} | "
            f"`{record.run_id[:16]}` |"
        )
    _atomic_write(markdown_path, ("\n".join(lines) + "\n").encode())
    return LedgerReportResult(
        subject=subject,
        evaluation_count=len(evaluations),
        report_json_path=json_path,
        report_markdown_path=markdown_path,
    )


def write_feature_history_report(
    *,
    store: ResearchStore,
    feature_id: str,
    reports_root: Path,
) -> LedgerReportResult:
    if store.get_feature_definition(feature_id) is None:
        raise ResearchError(f"unknown feature: {feature_id}")
    evaluations = store.list_feature_evaluations(feature_id=feature_id)
    campaigns = store.campaigns_using_feature(feature_id)
    identifier = hashlib.sha256(feature_id.encode()).hexdigest()[:16]
    return _write_report(
        subject=feature_id,
        title=f"Feature history — {feature_id}",
        evaluations=evaluations,
        campaigns=campaigns,
        directory=reports_root.resolve() / "research_ledger" / f"feature_id={identifier}",
    )


def write_hypothesis_history_report(
    *,
    store: ResearchStore,
    hypothesis_id: str,
    reports_root: Path,
) -> LedgerReportResult:
    if store.get_hypothesis(hypothesis_id) is None:
        raise ResearchError(f"unknown hypothesis: {hypothesis_id}")
    evaluations = store.list_feature_evaluations(hypothesis_id=hypothesis_id)
    campaigns_by_id: dict[str, CampaignRecord] = {}
    for record in evaluations:
        campaign_id = record.context.get("campaign_id")
        if isinstance(campaign_id, str):
            campaign = store.get_campaign(campaign_id)
            if campaign is not None:
                campaigns_by_id[campaign.campaign_id] = campaign
    campaigns = tuple(campaigns_by_id[key] for key in sorted(campaigns_by_id))
    identifier = hashlib.sha256(hypothesis_id.encode()).hexdigest()[:16]
    return _write_report(
        subject=hypothesis_id,
        title=f"Hypothesis history — {hypothesis_id}",
        evaluations=evaluations,
        campaigns=campaigns,
        directory=(reports_root.resolve() / "research_ledger" / f"hypothesis_id={identifier}"),
    )


__all__ = [
    "LedgerReportResult",
    "write_feature_history_report",
    "write_hypothesis_history_report",
]
