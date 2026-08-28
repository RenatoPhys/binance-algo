"""Contextual feature ablations backed by the durable research ledger."""

from __future__ import annotations

import hashlib
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Self, cast

import orjson
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from binance_algo.common.errors import ResearchError
from binance_algo.research.experiments.canonical import canonicalize
from binance_algo.research.experiments.models import (
    FeatureDecision,
    FeatureEvaluationSpec,
    FeatureEvaluationType,
    MetricScope,
)
from binance_algo.research.experiments.store import (
    CampaignRecord,
    ExperimentRunRecord,
    FeatureEvaluationRecord,
    ResearchMetricRecord,
    ResearchStore,
)


class StrictAblationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AblationChange(StrEnum):
    ADDED = "ADDED"
    REMOVED = "REMOVED"


class TrialSelector(StrictAblationModel):
    strategy_parameters: dict[str, Any] = Field(default_factory=dict)
    portfolio_parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_selector(self) -> Self:
        if not self.strategy_parameters and not self.portfolio_parameters:
            raise ValueError("an ablation trial selector cannot be empty")
        strategy = canonicalize(
            self.strategy_parameters,
            field_path="ablation.selector.strategy_parameters",
        )
        portfolio = canonicalize(
            self.portfolio_parameters,
            field_path="ablation.selector.portfolio_parameters",
        )
        object.__setattr__(self, "strategy_parameters", cast(dict[str, Any], strategy))
        object.__setattr__(self, "portfolio_parameters", cast(dict[str, Any], portfolio))
        return self


class AblationDecisionRule(StrictAblationModel):
    min_delta_sharpe: float = 0.0
    min_delta_total_return: float = 0.0
    min_improved_folds: int = Field(default=2, ge=1)
    require_positive_cost_1_5x: bool = True

    @model_validator(mode="after")
    def validate_thresholds(self) -> Self:
        if not math.isfinite(self.min_delta_sharpe) or not math.isfinite(
            self.min_delta_total_return
        ):
            raise ValueError("ablation decision thresholds must be finite")
        return self


class AblationDeclaration(StrictAblationModel):
    feature_id: str = Field(min_length=1)
    change: AblationChange
    baseline: TrialSelector
    candidate: TrialSelector
    evaluation_type: FeatureEvaluationType = FeatureEvaluationType.ABLATION
    rule: AblationDecisionRule = Field(default_factory=AblationDecisionRule)
    decision_override: FeatureDecision | None = None
    decision_reason: str | None = None

    @model_validator(mode="after")
    def validate_override(self) -> Self:
        if (self.decision_override is None) != (self.decision_reason is None):
            raise ValueError("decision_override and decision_reason must be declared together")
        return self


@dataclass(frozen=True, slots=True)
class IncrementalMetrics:
    delta_total_return: float
    delta_sharpe: float
    delta_max_drawdown: float
    delta_rank_ic: float
    delta_turnover: float
    delta_explicit_cost: float
    delta_capacity: float
    delta_month_concentration: float


@dataclass(frozen=True, slots=True)
class AblationEvaluationResult:
    feature_id: str
    change: AblationChange
    baseline_run_id: str
    candidate_run_id: str
    with_feature_run_id: str
    without_feature_run_id: str
    metrics: IncrementalMetrics
    decision: FeatureDecision
    decision_reason: str
    evaluations: tuple[FeatureEvaluationRecord, ...]


@dataclass(frozen=True, slots=True)
class AblationCampaignResult:
    campaign_id: str
    evaluations: tuple[AblationEvaluationResult, ...]
    report_json_path: Path
    report_markdown_path: Path


def _metric_map(records: Sequence[ResearchMetricRecord]) -> dict[str, float]:
    return {
        record.metric_name: record.metric_value
        for record in records
        if record.scope is MetricScope.TEST and record.fold is None and record.regime is None
    }


def _explicit_cost(metrics: Mapping[str, float]) -> float:
    return sum(metrics[name] for name in ("trading_fees", "spread_cost", "slippage_cost"))


def _required(metrics: Mapping[str, float], name: str) -> float:
    try:
        value = float(metrics[name])
    except KeyError as exc:
        raise ResearchError(f"ablation requires the metric {name}") from exc
    if not math.isfinite(value):
        raise ResearchError(f"ablation metric is not finite: {name}")
    return value


def calculate_incremental_metrics(
    *,
    with_feature: Mapping[str, float],
    without_feature: Mapping[str, float],
    with_feature_month_concentration: float,
    without_feature_month_concentration: float,
) -> IncrementalMetrics:
    """Return feature-effect deltas, always oriented as with-feature minus without-feature."""

    for value, name in (
        (with_feature_month_concentration, "with_feature_month_concentration"),
        (without_feature_month_concentration, "without_feature_month_concentration"),
    ):
        if not math.isfinite(value):
            raise ResearchError(f"ablation input is not finite: {name}")
    return IncrementalMetrics(
        delta_total_return=_required(with_feature, "total_return")
        - _required(without_feature, "total_return"),
        delta_sharpe=_required(with_feature, "sharpe") - _required(without_feature, "sharpe"),
        delta_max_drawdown=_required(with_feature, "max_drawdown")
        - _required(without_feature, "max_drawdown"),
        delta_rank_ic=_required(with_feature, "mean_cross_sectional_rank_ic")
        - _required(without_feature, "mean_cross_sectional_rank_ic"),
        delta_turnover=_required(with_feature, "turnover") - _required(without_feature, "turnover"),
        delta_explicit_cost=_explicit_cost(with_feature) - _explicit_cost(without_feature),
        delta_capacity=_required(without_feature, "maximum_volume_participation")
        - _required(with_feature, "maximum_volume_participation"),
        delta_month_concentration=(
            with_feature_month_concentration - without_feature_month_concentration
        ),
    )


def _selector_matches(selector: TrialSelector, tags: Mapping[str, Any]) -> bool:
    strategy = tags.get("strategy_parameters")
    portfolio = tags.get("portfolio_parameters")
    if not isinstance(strategy, Mapping) or not isinstance(portfolio, Mapping):
        return False
    return all(
        strategy.get(key) == value for key, value in selector.strategy_parameters.items()
    ) and all(portfolio.get(key) == value for key, value in selector.portfolio_parameters.items())


def _resolve_trial(
    store: ResearchStore,
    campaign: CampaignRecord,
    selector: TrialSelector,
) -> tuple[str, ExperimentRunRecord, Mapping[str, Any]]:
    matches = [
        (identifier, tags)
        for _, identifier, tags in store.list_campaign_experiments(campaign.campaign_id)
        if _selector_matches(selector, tags)
    ]
    if len(matches) != 1:
        raise ResearchError(
            f"ablation selector matched {len(matches)} trials in campaign {campaign.name}; "
            "selectors must identify exactly one trial"
        )
    identifier, tags = matches[0]
    run = store.latest_successful_run(identifier)
    if run is None:
        raise ResearchError(f"ablation trial has no successful run: {identifier}")
    return identifier, run, tags


def _validate_comparable_experiments(
    store: ResearchStore,
    baseline_experiment_id: str,
    candidate_experiment_id: str,
) -> None:
    baseline = store.get_experiment(baseline_experiment_id)
    candidate = store.get_experiment(candidate_experiment_id)
    if baseline is None or candidate is None:
        raise ResearchError("ablation experiment definition disappeared")
    baseline_payload = baseline.model_dump(mode="python")
    candidate_payload = candidate.model_dump(mode="python")
    for payload in (baseline_payload, candidate_payload):
        payload.pop("strategy_parameters")
        payload.pop("campaign_id")
    if baseline_payload != candidate_payload:
        raise ResearchError(
            "ablation trials differ outside strategy parameters; contextual deltas are invalid"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _month_concentration(store: ResearchStore, data_root: Path, run_id: str) -> float:
    artifacts = [
        artifact
        for artifact in store.list_artifacts(run_id)
        if artifact.artifact_type == "monthly_metrics"
    ]
    if len(artifacts) != 1:
        raise ResearchError(f"run {run_id} must have exactly one monthly_metrics artifact")
    artifact = artifacts[0]
    root = data_root.resolve()
    path = (root / artifact.path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ResearchError(f"monthly metrics artifact is unavailable: {artifact.path}")
    if _sha256(path) != artifact.checksum_sha256:
        raise ResearchError(f"monthly metrics artifact is corrupt: {artifact.path}")
    frame = pl.read_parquet(path, columns=["net_pnl"])
    if artifact.row_count != frame.height or frame.is_empty():
        raise ResearchError(f"monthly metrics artifact has an invalid shape: {artifact.path}")
    absolute = frame["net_pnl"].abs()
    denominator = float(absolute.sum())
    maximum = absolute.max()
    if not isinstance(maximum, (int, float)):
        raise ResearchError("monthly metrics artifact has invalid net P&L values")
    concentration = 0.0 if denominator <= 1e-18 else float(maximum) / denominator
    if not math.isfinite(concentration):
        raise ResearchError("month concentration is not finite")
    return concentration


def _fold_improvements(
    with_records: Sequence[ResearchMetricRecord],
    without_records: Sequence[ResearchMetricRecord],
) -> tuple[int, int]:
    def returns(records: Sequence[ResearchMetricRecord]) -> dict[int, float]:
        return {
            record.fold: record.metric_value
            for record in records
            if record.scope is MetricScope.TEST
            and record.fold is not None
            and record.metric_name == "total_return"
        }

    with_folds = returns(with_records)
    without_folds = returns(without_records)
    if not with_folds or with_folds.keys() != without_folds.keys():
        raise ResearchError("ablation trials do not have comparable fold metrics")
    improved = sum(with_folds[fold] > without_folds[fold] for fold in with_folds)
    return improved, len(with_folds)


def _cost_1_5x_return(records: Sequence[ResearchMetricRecord]) -> float:
    values = [
        record.metric_value
        for record in records
        if record.scope is MetricScope.STRESS
        and record.regime == "cost_1_5x"
        and record.metric_name == "total_return"
    ]
    if len(values) != 1 or not math.isfinite(values[0]):
        raise ResearchError("ablation requires one finite cost_1_5x total return")
    return values[0]


def _decide(
    *,
    declaration: AblationDeclaration,
    incremental: IncrementalMetrics,
    with_metrics: Mapping[str, float],
    without_metrics: Mapping[str, float],
    improved_folds: int,
    with_cost_1_5x_return: float,
) -> tuple[FeatureDecision, str]:
    if declaration.decision_override is not None:
        assert declaration.decision_reason is not None
        return declaration.decision_override, declaration.decision_reason
    rule = declaration.rule
    survives_costs = not rule.require_positive_cost_1_5x or with_cost_1_5x_return > 0
    if (
        incremental.delta_sharpe > rule.min_delta_sharpe
        and incremental.delta_total_return > rule.min_delta_total_return
        and improved_folds >= rule.min_improved_folds
        and survives_costs
    ):
        return (
            FeatureDecision.SUPPORTED,
            "net return and Sharpe improved across the required folds and the "
            "configured cost stress",
        )
    gross_delta = (
        _required(with_metrics, "price_pnl")
        + _required(with_metrics, "funding_pnl")
        - _required(without_metrics, "price_pnl")
        - _required(without_metrics, "funding_pnl")
    )
    if (
        gross_delta > 0
        and incremental.delta_total_return < 0
        and (incremental.delta_turnover > 0 or incremental.delta_explicit_cost > 0)
    ):
        return (
            FeatureDecision.REJECTED,
            "positive gross contribution was consumed by turnover and explicit costs",
        )
    if incremental.delta_sharpe <= 0 and incremental.delta_total_return <= 0:
        return (
            FeatureDecision.REJECTED,
            "the feature did not improve net return or Sharpe in this registered context",
        )
    return (
        FeatureDecision.INCONCLUSIVE,
        "the registered rule was not satisfied; preserve the result for contextual retesting",
    )


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
        raise ResearchError(f"cannot write ablation report {path}: {exc}") from exc


def _write_report(
    *,
    campaign: CampaignRecord,
    results: Sequence[AblationEvaluationResult],
    reports_root: Path,
) -> tuple[Path, Path]:
    directory = (
        reports_root.resolve() / "research_ledger" / f"campaign_id={campaign.campaign_id[:24]}"
    )
    json_path = directory / "ablation.json"
    markdown_path = directory / "ablation.md"
    payload = {
        "campaign_id": campaign.campaign_id,
        "campaign_name": campaign.name,
        "evaluations": [
            {
                "feature_id": result.feature_id,
                "change": result.change.value,
                "baseline_run_id": result.baseline_run_id,
                "candidate_run_id": result.candidate_run_id,
                "with_feature_run_id": result.with_feature_run_id,
                "without_feature_run_id": result.without_feature_run_id,
                "metrics": asdict(result.metrics),
                "decision": result.decision.value,
                "decision_reason": result.decision_reason,
                "evaluation_ids": [item.evaluation_id for item in result.evaluations],
            }
            for result in results
        ],
    }
    _atomic_write(json_path, orjson.dumps(payload, option=orjson.OPT_SORT_KEYS) + b"\n")
    lines = [
        f"# Feature ablations — {campaign.name}",
        "",
        "> Derived report. The research registry is the source of truth; "
        "negative results are retained.",
        "",
        "| Feature | Change | Δ return | Δ Sharpe | Δ turnover | Δ cost | Decision |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        metrics = result.metrics
        lines.append(
            f"| `{result.feature_id}` | {result.change.value} | "
            f"{metrics.delta_total_return:.6%} | {metrics.delta_sharpe:.6f} | "
            f"{metrics.delta_turnover:.6f} | {metrics.delta_explicit_cost:.8f} | "
            f"{result.decision.value} |"
        )
        lines.append(f"\nReason: {result.decision_reason}\n")
    _atomic_write(markdown_path, ("\n".join(lines) + "\n").encode())
    return json_path, markdown_path


class AblationRunner:
    def __init__(
        self,
        *,
        store: ResearchStore,
        data_root: Path,
        reports_root: Path,
    ) -> None:
        self.store = store
        self.data_root = data_root.resolve()
        self.reports_root = reports_root.resolve()

    def evaluate_campaign(
        self,
        campaign: CampaignRecord,
        declarations: Sequence[AblationDeclaration],
    ) -> AblationCampaignResult:
        if not declarations:
            raise ResearchError(f"campaign {campaign.name} declares no feature ablations")
        results = tuple(self._evaluate(campaign, declaration) for declaration in declarations)
        json_path, markdown_path = _write_report(
            campaign=campaign,
            results=results,
            reports_root=self.reports_root,
        )
        return AblationCampaignResult(
            campaign_id=campaign.campaign_id,
            evaluations=results,
            report_json_path=json_path,
            report_markdown_path=markdown_path,
        )

    def _evaluate(
        self,
        campaign: CampaignRecord,
        declaration: AblationDeclaration,
    ) -> AblationEvaluationResult:
        baseline_id, baseline_run, baseline_tags = _resolve_trial(
            self.store, campaign, declaration.baseline
        )
        candidate_id, candidate_run, candidate_tags = _resolve_trial(
            self.store, campaign, declaration.candidate
        )
        if baseline_run.run_id == candidate_run.run_id:
            raise ResearchError("ablation baseline and candidate must be different trials")
        _validate_comparable_experiments(self.store, baseline_id, candidate_id)
        if self.store.get_feature_definition(declaration.feature_id) is None:
            raise ResearchError(f"unknown feature: {declaration.feature_id}")
        if not self.store.experiment_uses_feature(
            baseline_id, declaration.feature_id
        ) or not self.store.experiment_uses_feature(candidate_id, declaration.feature_id):
            raise ResearchError(
                f"ablation feature is not a member of both experiment feature sets: "
                f"{declaration.feature_id}"
            )
        baseline_records = self.store.list_metrics(baseline_run.run_id)
        candidate_records = self.store.list_metrics(candidate_run.run_id)
        if declaration.change is AblationChange.REMOVED:
            with_run, without_run = baseline_run, candidate_run
            with_records, without_records = baseline_records, candidate_records
        else:
            with_run, without_run = candidate_run, baseline_run
            with_records, without_records = candidate_records, baseline_records
        with_metrics = _metric_map(with_records)
        without_metrics = _metric_map(without_records)
        with_month = _month_concentration(self.store, self.data_root, with_run.run_id)
        without_month = _month_concentration(self.store, self.data_root, without_run.run_id)
        incremental = calculate_incremental_metrics(
            with_feature=with_metrics,
            without_feature=without_metrics,
            with_feature_month_concentration=with_month,
            without_feature_month_concentration=without_month,
        )
        improved_folds, fold_count = _fold_improvements(with_records, without_records)
        with_cost_1_5x_return = _cost_1_5x_return(with_records)
        decision, reason = _decide(
            declaration=declaration,
            incremental=incremental,
            with_metrics=with_metrics,
            without_metrics=without_metrics,
            improved_folds=improved_folds,
            with_cost_1_5x_return=with_cost_1_5x_return,
        )
        context = {
            "campaign_id": campaign.campaign_id,
            "campaign_name": campaign.name,
            "hypothesis_id": campaign.hypothesis_id,
            "change": declaration.change.value,
            "delta_orientation": "with_feature_minus_without_feature",
            "baseline_experiment_id": baseline_id,
            "baseline_run_id": baseline_run.run_id,
            "baseline_tags": dict(baseline_tags),
            "candidate_experiment_id": candidate_id,
            "candidate_run_id": candidate_run.run_id,
            "candidate_tags": dict(candidate_tags),
            "with_feature_run_id": with_run.run_id,
            "without_feature_run_id": without_run.run_id,
            "with_feature_metrics": with_metrics,
            "without_feature_metrics": without_metrics,
            "with_feature_month_concentration": with_month,
            "without_feature_month_concentration": without_month,
            "improved_folds": improved_folds,
            "fold_count": fold_count,
            "with_feature_cost_1_5x_return": with_cost_1_5x_return,
            "decision_rule": declaration.rule.model_dump(mode="json"),
        }
        specs = tuple(
            FeatureEvaluationSpec(
                run_id=with_run.run_id,
                feature_id=declaration.feature_id,
                evaluation_type=declaration.evaluation_type,
                scope=MetricScope.TEST.value,
                metric_name=name,
                metric_value=value,
                decision=decision,
                decision_reason=reason,
                context=context,
            )
            for name, value in asdict(incremental).items()
        )
        evaluations = self.store.record_feature_evaluations(specs)
        return AblationEvaluationResult(
            feature_id=declaration.feature_id,
            change=declaration.change,
            baseline_run_id=baseline_run.run_id,
            candidate_run_id=candidate_run.run_id,
            with_feature_run_id=with_run.run_id,
            without_feature_run_id=without_run.run_id,
            metrics=incremental,
            decision=decision,
            decision_reason=reason,
            evaluations=evaluations,
        )


__all__ = [
    "AblationCampaignResult",
    "AblationChange",
    "AblationDecisionRule",
    "AblationDeclaration",
    "AblationEvaluationResult",
    "AblationRunner",
    "IncrementalMetrics",
    "TrialSelector",
    "calculate_incremental_metrics",
]
