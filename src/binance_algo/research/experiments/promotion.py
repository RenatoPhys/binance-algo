"""Auditable promotion gates, rejection events, lockbox policy, and candidate reports."""

from __future__ import annotations

import os
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import orjson

from binance_algo.common.errors import ResearchError
from binance_algo.config import ResearchPlatformConfig
from binance_algo.research.experiments.models import (
    CodeFingerprint,
    PromotionDecision,
    PromotionEventSpec,
    ProvenanceQuality,
    ResearchStage,
)
from binance_algo.research.experiments.runner import ExperimentRunner
from binance_algo.research.experiments.store import (
    PROMOTION_TRANSITIONS,
    PromotionRecord,
    ResearchStore,
)
from binance_algo.research.validation.robustness import (
    CampaignRobustnessResult,
    RobustnessStatus,
    build_campaign_robustness,
)


@dataclass(frozen=True, slots=True)
class PromotionGate:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class CandidateAssessment:
    experiment_id: str
    current_stage: ResearchStage
    requested_stage: ResearchStage
    passed: bool
    gates: tuple[PromotionGate, ...]
    robustness: CampaignRobustnessResult
    report_json_path: Path
    report_markdown_path: Path


@dataclass(frozen=True, slots=True)
class PromotionResult:
    assessment: CandidateAssessment
    event: PromotionRecord


def current_research_stage(events: Sequence[PromotionRecord]) -> ResearchStage:
    stage = ResearchStage.DISCOVERY
    for event in events:
        if event.from_stage is not stage:
            raise ResearchError(
                f"promotion history is inconsistent: expected {stage.value}, "
                f"found {event.from_stage.value}"
            )
        if event.decision in {
            PromotionDecision.APPROVED,
            PromotionDecision.REJECTED,
            PromotionDecision.INVALIDATED,
        }:
            stage = event.to_stage
    return stage


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
        raise ResearchError(f"cannot write candidate report {path}: {exc}") from exc


def _write_candidate_report(
    *,
    experiment_id: str,
    current_stage: ResearchStage,
    requested_stage: ResearchStage,
    gates: Sequence[PromotionGate],
    robustness: CampaignRobustnessResult,
    reports_root: Path,
) -> tuple[Path, Path]:
    directory = (
        reports_root.resolve() / "research_candidates" / f"experiment_id={experiment_id[:24]}"
    )
    json_path = directory / "candidate.json"
    markdown_path = directory / "candidate.md"
    payload = {
        "experiment_id": experiment_id,
        "current_stage": current_stage.value,
        "requested_stage": requested_stage.value,
        "passed": all(gate.passed for gate in gates),
        "gates": [asdict(gate) for gate in gates],
        "campaign_context": {
            "campaign_id": robustness.campaign_id,
            "campaign_name": robustness.campaign_name,
            "planned_trials": robustness.planned_trials,
            "successful_trials": robustness.successful_trials,
            "distinct_strategies": robustness.distinct_strategies,
            "approximate_independent_strategies": (robustness.approximate_independent_strategies),
            "best_experiment_id": robustness.best_experiment_id,
            "sharpe_distribution": dict(robustness.sharpe_distribution),
            "return_distribution": dict(robustness.return_distribution),
            "neighborhood": asdict(robustness.neighborhood),
            "dsr": asdict(robustness.dsr),
            "pbo": asdict(robustness.pbo),
            "lockbox": asdict(robustness.lockbox),
        },
        "trial": asdict(robustness.trial(experiment_id)),
    }
    _atomic_write(json_path, orjson.dumps(payload, option=orjson.OPT_SORT_KEYS) + b"\n")
    lines = [
        f"# Candidate assessment — {experiment_id}",
        "",
        "> This experiment is shown in its full campaign context. The best trial is not "
        "independent OOS evidence.",
        "",
        f"- Current stage: {current_stage.value}",
        f"- Requested stage: {requested_stage.value}",
        f"- Campaign: `{robustness.campaign_id}` ({robustness.planned_trials} planned trials)",
        f"- DSR probability: {robustness.dsr.probability:.6f}",
        f"- PBO: {robustness.pbo.status.value} — {robustness.pbo.reason}",
        f"- Lockbox: {robustness.lockbox.status.value} — {robustness.lockbox.reason}",
        "",
        "| Gate | Result | Detail |",
        "|---|---|---|",
    ]
    for gate in gates:
        lines.append(f"| {gate.name} | {'PASS' if gate.passed else 'FAIL'} | {gate.detail} |")
    _atomic_write(markdown_path, ("\n".join(lines) + "\n").encode())
    return json_path, markdown_path


class PromotionManager:
    def __init__(
        self,
        *,
        store: ResearchStore,
        experiment_runner: ExperimentRunner,
        data_root: Path,
        reports_root: Path,
        platform: ResearchPlatformConfig,
        current_code_fingerprint: CodeFingerprint,
    ) -> None:
        self.store = store
        self.experiment_runner = experiment_runner
        self.data_root = data_root.resolve()
        self.reports_root = reports_root.resolve()
        self.platform = platform
        self.current_code_fingerprint = current_code_fingerprint

    def assess_candidate(self, experiment_id: str) -> CandidateAssessment:
        spec = self.store.get_experiment(experiment_id)
        if spec is None:
            raise ResearchError(f"unknown experiment: {experiment_id}")
        campaigns = self.store.campaigns_for_experiment(experiment_id)
        if len(campaigns) != 1:
            raise ResearchError(
                f"candidate assessment requires exactly one campaign context; got {len(campaigns)}"
            )
        campaign = campaigns[0]
        robustness = build_campaign_robustness(
            store=self.store,
            campaign=campaign,
            data_root=self.data_root,
            reports_root=self.reports_root,
            platform=self.platform,
        )
        trial = robustness.trial(experiment_id)
        run = self.store.latest_successful_run(experiment_id)
        verification_valid = False
        verification_detail = "no successful run"
        if run is not None:
            verification = self.experiment_runner.verify_run(run.run_id)
            verification_valid = verification.valid
            verification_detail = (
                f"{verification.checked_files} artifacts verified"
                if verification.valid
                else "; ".join(verification.issues)
            )
        hypothesis = self.store.get_hypothesis(spec.hypothesis_id)
        neighborhood = robustness.neighborhood
        current_stage = current_research_stage(self.store.list_promotions(experiment_id))
        gates = (
            PromotionGate("successful_run", run is not None, verification_detail),
            PromotionGate("artifacts_integral", verification_valid, verification_detail),
            PromotionGate(
                "experiment_clean_git",
                spec.code_fingerprint.provenance_quality is ProvenanceQuality.GIT_CLEAN,
                spec.code_fingerprint.provenance_quality.value,
            ),
            PromotionGate(
                "promotion_clean_git",
                (
                    not self.platform.promotion_requires_clean_git
                    or self.current_code_fingerprint.provenance_quality
                    is ProvenanceQuality.GIT_CLEAN
                ),
                self.current_code_fingerprint.provenance_quality.value,
            ),
            PromotionGate(
                "preregistered_hypothesis",
                hypothesis is not None and bool(hypothesis.preregistered_success_criteria),
                spec.hypothesis_id,
            ),
            PromotionGate(
                "positive_net_oos",
                trial.total_return > 0,
                f"total_return={trial.total_return:.8f}",
            ),
            PromotionGate(
                "fold_stability",
                trial.profitable_folds >= self.platform.min_profitable_folds,
                f"profitable_folds={trial.profitable_folds}/{trial.fold_count}",
            ),
            PromotionGate(
                "month_concentration",
                trial.month_concentration <= self.platform.max_month_concentration,
                f"concentration={trial.month_concentration:.6f}",
            ),
            PromotionGate(
                "symbol_concentration",
                trial.symbol_concentration <= self.platform.max_symbol_concentration,
                f"concentration={trial.symbol_concentration:.6f}",
            ),
            PromotionGate(
                "cost_1_5x",
                trial.cost_1_5x_return > 0,
                f"total_return={trial.cost_1_5x_return:.8f}",
            ),
            PromotionGate(
                "signal_delay",
                trial.delay_1_bar_return > 0,
                f"total_return={trial.delay_1_bar_return:.8f}",
            ),
            PromotionGate(
                "parameter_neighborhood",
                (
                    neighborhood.status is RobustnessStatus.AVAILABLE
                    and neighborhood.neighbor_positive_fraction is not None
                    and neighborhood.neighbor_positive_fraction
                    >= self.platform.min_neighbor_positive_fraction
                ),
                neighborhood.reason,
            ),
            PromotionGate(
                "multiple_testing_dsr",
                robustness.dsr.probability >= self.platform.min_dsr_probability,
                (
                    f"probability={robustness.dsr.probability:.6f}; "
                    f"trials={robustness.dsr.number_of_trials}"
                ),
            ),
            PromotionGate(
                "pbo_considered",
                True,
                f"{robustness.pbo.status.value}: {robustness.pbo.reason}",
            ),
            PromotionGate(
                "campaign_context",
                experiment_id == robustness.best_experiment_id,
                (
                    f"selected={experiment_id}; best={robustness.best_experiment_id}; "
                    f"planned_trials={robustness.planned_trials}"
                ),
            ),
        )
        json_path, markdown_path = _write_candidate_report(
            experiment_id=experiment_id,
            current_stage=current_stage,
            requested_stage=ResearchStage.CANDIDATE,
            gates=gates,
            robustness=robustness,
            reports_root=self.reports_root,
        )
        return CandidateAssessment(
            experiment_id=experiment_id,
            current_stage=current_stage,
            requested_stage=ResearchStage.CANDIDATE,
            passed=all(gate.passed for gate in gates),
            gates=gates,
            robustness=robustness,
            report_json_path=json_path,
            report_markdown_path=markdown_path,
        )

    def promote_candidate(self, experiment_id: str, *, reason: str) -> PromotionResult:
        assessment = self.assess_candidate(experiment_id)
        if assessment.current_stage is not ResearchStage.DISCOVERY:
            raise ResearchError(
                f"candidate promotion requires DISCOVERY; got {assessment.current_stage.value}"
            )
        decision = PromotionDecision.APPROVED if assessment.passed else PromotionDecision.BLOCKED
        event = self.store.record_promotion_event(
            PromotionEventSpec(
                experiment_id=experiment_id,
                from_stage=assessment.current_stage,
                to_stage=ResearchStage.CANDIDATE,
                decision=decision,
                criteria_snapshot=self._criteria_snapshot(assessment),
                reason=reason,
                code_fingerprint=self.current_code_fingerprint,
            )
        )
        return PromotionResult(assessment=assessment, event=event)

    def promote_phase4(self, experiment_id: str, *, reason: str) -> PromotionRecord:
        events = self.store.list_promotions(experiment_id)
        stage = current_research_stage(events)
        campaigns = self.store.campaigns_for_experiment(experiment_id)
        if len(campaigns) != 1:
            raise ResearchError("phase4 promotion requires exactly one campaign context")
        robustness = build_campaign_robustness(
            store=self.store,
            campaign=campaigns[0],
            data_root=self.data_root,
            reports_root=self.reports_root,
            platform=self.platform,
        )
        passed = (
            stage is ResearchStage.LOCKBOX_EVALUATED
            and robustness.lockbox.status is RobustnessStatus.AVAILABLE
            and self.current_code_fingerprint.provenance_quality is ProvenanceQuality.GIT_CLEAN
        )
        criteria = {
            "current_stage": stage.value,
            "lockbox": asdict(robustness.lockbox),
            "clean_git": (
                self.current_code_fingerprint.provenance_quality is ProvenanceQuality.GIT_CLEAN
            ),
        }
        return self.store.record_promotion_event(
            PromotionEventSpec(
                experiment_id=experiment_id,
                from_stage=stage,
                to_stage=ResearchStage.PHASE4_CANDIDATE,
                decision=PromotionDecision.APPROVED if passed else PromotionDecision.BLOCKED,
                criteria_snapshot=criteria,
                reason=reason,
                code_fingerprint=self.current_code_fingerprint,
            )
        )

    def reject(self, experiment_id: str, *, reason: str) -> PromotionRecord:
        stage = current_research_stage(self.store.list_promotions(experiment_id))
        if ResearchStage.REJECTED not in PROMOTION_TRANSITIONS[stage]:
            raise ResearchError(f"cannot reject experiment from stage {stage.value}")
        return self.store.record_promotion_event(
            PromotionEventSpec(
                experiment_id=experiment_id,
                from_stage=stage,
                to_stage=ResearchStage.REJECTED,
                decision=PromotionDecision.REJECTED,
                criteria_snapshot={"explicit_rejection": True},
                reason=reason,
                code_fingerprint=self.current_code_fingerprint,
            )
        )

    @staticmethod
    def _criteria_snapshot(assessment: CandidateAssessment) -> dict[str, Any]:
        return {
            "passed": assessment.passed,
            "gates": [asdict(gate) for gate in assessment.gates],
            "campaign_id": assessment.robustness.campaign_id,
            "planned_trials": assessment.robustness.planned_trials,
            "successful_trials": assessment.robustness.successful_trials,
            "dsr": asdict(assessment.robustness.dsr),
            "pbo": asdict(assessment.robustness.pbo),
            "lockbox": asdict(assessment.robustness.lockbox),
            "candidate_report": (
                f"research_candidates/experiment_id={assessment.experiment_id[:24]}/candidate.json"
            ),
        }


__all__ = [
    "PROMOTION_TRANSITIONS",
    "CandidateAssessment",
    "PromotionGate",
    "PromotionManager",
    "PromotionResult",
    "current_research_stage",
]
