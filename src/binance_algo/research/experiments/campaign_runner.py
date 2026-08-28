"""Local campaign coordinator with cache, resume, and isolated trial failures."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from binance_algo.common.errors import ResearchError
from binance_algo.config import ResearchConfig
from binance_algo.research.experiments.campaign import CampaignPlan
from binance_algo.research.experiments.compare import (
    CampaignComparisonResult,
    write_campaign_comparison,
)
from binance_algo.research.experiments.models import CampaignStatus, RunStatus
from binance_algo.research.experiments.runner import ExperimentRunner
from binance_algo.research.experiments.store import CampaignRecord, ResearchStore


@dataclass(frozen=True, slots=True)
class TrialWorkerRequest:
    store_path: str
    data_root: str
    research_config: dict[str, Any]
    compression: str
    experiment_id: str


@dataclass(frozen=True, slots=True)
class TrialWorkerResult:
    experiment_id: str
    run_id: str
    result_digest: str


@dataclass(frozen=True, slots=True)
class CampaignRunResult:
    campaign: CampaignRecord
    planned_count: int
    cache_hit_count: int
    executed_count: int
    succeeded_count: int
    failed_count: int
    failed_experiments: tuple[str, ...]
    comparison: CampaignComparisonResult


def _execute_trial_worker(request: TrialWorkerRequest) -> TrialWorkerResult:
    store = ResearchStore(Path(request.store_path))
    config = ResearchConfig.model_validate(request.research_config)
    result = ExperimentRunner(
        store=store,
        data_root=Path(request.data_root),
        research_config=config,
        compression=request.compression,
    ).run(request.experiment_id)
    if result.run.result_digest is None:
        raise ResearchError("successful trial has no result digest")
    return TrialWorkerResult(
        experiment_id=request.experiment_id,
        run_id=result.run.run_id,
        result_digest=result.run.result_digest,
    )


class CampaignRunner:
    def __init__(
        self,
        *,
        store: ResearchStore,
        data_root: Path,
        reports_root: Path,
        research_config: ResearchConfig,
        compression: str,
    ) -> None:
        self.store = store
        self.data_root = data_root.resolve()
        self.reports_root = reports_root.resolve()
        self.research_config = research_config
        self.compression = compression

    def register(self, plan: CampaignPlan) -> CampaignRecord:
        if self.store.get_hypothesis(plan.source.campaign.hypothesis_id) is None:
            raise ResearchError(
                f"campaign hypothesis is not registered: {plan.source.campaign.hypothesis_id}"
            )
        campaign = self.store.register_campaign(
            identifier=plan.campaign_id,
            name=plan.source.campaign.name,
            description=plan.source.campaign.description,
            hypothesis_id=plan.source.campaign.hypothesis_id,
            spec_payload=plan.stored_payload(),
            trial_count=plan.valid_combinations,
        )
        for trial in plan.trials:
            registered = self.store.register_experiment(trial.spec)
            if registered != trial.experiment_id:
                raise ResearchError("campaign trial identity changed during registration")
            self.store.associate_campaign_experiment(
                campaign_id=campaign.campaign_id,
                experiment_id_value=trial.experiment_id,
                ordinal=trial.ordinal,
                tags=trial.tags,
            )
        return campaign

    def run(self, plan: CampaignPlan) -> CampaignRunResult:
        campaign = self.register(plan)
        experiment_runner = ExperimentRunner(
            store=self.store,
            data_root=self.data_root,
            research_config=self.research_config,
            compression=self.compression,
        )
        cache_hits: list[str] = []
        pending: list[str] = []
        for trial in plan.trials:
            succeeded = self.store.latest_successful_run(trial.experiment_id)
            if succeeded is not None and experiment_runner.verify_run(succeeded.run_id).valid:
                cache_hits.append(trial.experiment_id)
            else:
                pending.append(trial.experiment_id)
        if not pending and campaign.status is CampaignStatus.COMPLETED:
            comparison = write_campaign_comparison(
                store=self.store,
                campaign=campaign,
                reports_root=self.reports_root,
                compression=self.compression,
            )
            return CampaignRunResult(
                campaign=campaign,
                planned_count=len(plan.trials),
                cache_hit_count=len(cache_hits),
                executed_count=0,
                succeeded_count=len(cache_hits),
                failed_count=0,
                failed_experiments=(),
                comparison=comparison,
            )
        if campaign.status is CampaignStatus.PLANNED:
            campaign = self.store.transition_campaign(
                campaign.campaign_id,
                CampaignStatus.QUEUED,
            )
        if campaign.status in {CampaignStatus.QUEUED, CampaignStatus.PARTIAL}:
            campaign = self.store.transition_campaign(
                campaign.campaign_id,
                CampaignStatus.RUNNING,
            )
        if campaign.status is not CampaignStatus.RUNNING:
            raise ResearchError(f"campaign cannot run from status {campaign.status.value}")
        try:
            failures = self._execute_pending(plan, pending)
        except KeyboardInterrupt:
            for identifier in pending:
                runs = self.store.list_runs(experiment_id_value=identifier)
                if runs and runs[-1].status is RunStatus.RUNNING:
                    self.store.transition_run(runs[-1].run_id, RunStatus.STALE)
            self.store.transition_campaign(
                campaign.campaign_id,
                CampaignStatus.CANCELLED,
                last_error="campaign interrupted by operator",
            )
            raise
        succeeded_count = len(cache_hits) + len(pending) - len(failures)
        if succeeded_count == len(plan.trials):
            final_status = CampaignStatus.COMPLETED
        elif succeeded_count:
            final_status = CampaignStatus.PARTIAL
        else:
            final_status = CampaignStatus.FAILED
        campaign = self.store.transition_campaign(
            campaign.campaign_id,
            final_status,
            last_error=(
                None
                if not failures
                else f"{len(failures)} trial(s) failed: {', '.join(failures[:3])}"
            ),
        )
        comparison = write_campaign_comparison(
            store=self.store,
            campaign=campaign,
            reports_root=self.reports_root,
            compression=self.compression,
        )
        return CampaignRunResult(
            campaign=campaign,
            planned_count=len(plan.trials),
            cache_hit_count=len(cache_hits),
            executed_count=len(pending),
            succeeded_count=succeeded_count,
            failed_count=len(failures),
            failed_experiments=tuple(failures),
            comparison=comparison,
        )

    def _execute_pending(self, plan: CampaignPlan, pending: list[str]) -> list[str]:
        if not pending:
            return []
        request_payload = self.research_config.model_dump(mode="json")
        requests = [
            TrialWorkerRequest(
                store_path=str(self.store.path),
                data_root=str(self.data_root),
                research_config=request_payload,
                compression=self.compression,
                experiment_id=identifier,
            )
            for identifier in pending
        ]
        failures: list[str] = []
        if plan.source.runner.max_workers == 1:
            for request in requests:
                try:
                    _execute_trial_worker(request)
                except BaseException:
                    failures.append(request.experiment_id)
                    if plan.source.runner.fail_fast:
                        failures.extend(
                            item.experiment_id for item in requests[requests.index(request) + 1 :]
                        )
                        break
            return failures
        executor = ProcessPoolExecutor(
            max_workers=plan.source.runner.max_workers,
            mp_context=get_context("spawn"),
        )
        futures = {executor.submit(_execute_trial_worker, request): request for request in requests}
        try:
            for future in as_completed(futures):
                request = futures[future]
                try:
                    future.result()
                except BaseException:
                    failures.append(request.experiment_id)
                    if plan.source.runner.fail_fast:
                        for pending_future in futures:
                            pending_future.cancel()
                        break
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        return failures


__all__ = [
    "CampaignRunResult",
    "CampaignRunner",
    "TrialWorkerRequest",
    "TrialWorkerResult",
]
