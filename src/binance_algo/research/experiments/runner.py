"""Registry-backed execution for one immutable research experiment."""

from __future__ import annotations

import os
import socket
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import orjson
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from binance_algo.common.errors import ResearchError
from binance_algo.config import FeeScheduleConfig, ResearchConfig
from binance_algo.data.storage import LocalFilesystemStorage
from binance_algo.research.backtest import (
    ACCOUNTING_METADATA_FIELDS,
    ACCOUNTING_OUTCOME_FIELDS,
    run_research_validation,
)
from binance_algo.research.datasets.references import (
    DatasetReference,
    load_dataset_reference,
)
from binance_algo.research.experiments.artifacts import (
    ArtifactVerification,
    ExperimentArtifactPipeline,
    verify_run_artifacts,
)
from binance_algo.research.experiments.ids import result_digest
from binance_algo.research.experiments.models import (
    ArtifactPolicy,
    CodeFingerprint,
    DatasetIdentity,
    ExperimentSpec,
    FeatureSetIdentity,
    HypothesisSpec,
    HypothesisStatus,
    LabelIdentity,
    ParameterizedComponent,
    RunStatus,
    VersionedComponent,
)
from binance_algo.research.experiments.provenance import build_code_fingerprint
from binance_algo.research.experiments.store import ExperimentRunRecord, ResearchStore
from binance_algo.research.features.registry import phase3_feature_set
from binance_algo.research.labels.forward_returns import GROSS_FORWARD_RETURN_1H
from binance_algo.research.panel import WORKER_DATASET_CACHE
from binance_algo.research.portfolio.registry import build_portfolio_policy
from binance_algo.research.strategies.registry import build_strategy


class _StrictParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExecutionParameters(_StrictParameters):
    lag_bars: int = Field(default=1, ge=1, le=1)


class CostParameters(_StrictParameters):
    spread_bps: Decimal = Field(ge=0, le=100)
    slippage_bps: Decimal = Field(ge=0, le=100)
    initial_capital_usdt: Decimal = Field(gt=0)
    fee_schedule: FeeScheduleConfig


class SplitParameters(_StrictParameters):
    train_days: int = Field(ge=7, le=3650)
    test_days: int = Field(ge=1, le=365)
    embargo_bars: int = Field(ge=1, le=24)


class ValidationParameters(_StrictParameters):
    stress_cost_multipliers: tuple[float, ...] = (1.5, 2.0)
    stress_signal_delay_bars: tuple[int, ...] = (1,)
    bootstrap_samples: int = Field(ge=100, le=10_000)
    bootstrap_block_hours: int = Field(ge=2, le=720)


@dataclass(frozen=True, slots=True)
class ExperimentExecutionResult:
    experiment_id: str
    run: ExperimentRunRecord
    artifact_directory: Path
    verification: ArtifactVerification
    deterministic_with_previous: bool | None


def phase3_baseline_hypothesis() -> HypothesisSpec:
    return HypothesisSpec(
        hypothesis_id="HYP-RESMOM-BASELINE-0001",
        title="Phase 3 residual-momentum baseline",
        mechanism=(
            "A fixed cross-sectional residual-momentum specification is retained as a "
            "regression baseline, not as a claim of edge."
        ),
        expected_direction="positive residual-return persistence after explicit costs",
        expected_horizon="1h",
        target_universe="fixed BTCUSDT, ETHUSDT and SOLUSDT seed chosen ex ante",
        preregistered_success_criteria={
            "purpose": "regression_only",
            "promotion_allowed": False,
        },
        status=HypothesisStatus.REJECTED,
        notes=(
            "Historical compatibility hypothesis. The observed 90-day result was negative; "
            "the record is preserved and must not be promoted."
        ),
    )


def build_phase3_experiment_spec(
    *,
    dataset_reference: DatasetReference,
    config: ResearchConfig,
    project_root: Path,
    artifact_policy: ArtifactPolicy = ArtifactPolicy.SUMMARY,
    code_fingerprint: CodeFingerprint | None = None,
) -> ExperimentSpec:
    feature_set = phase3_feature_set(config)
    return ExperimentSpec(
        hypothesis_id=phase3_baseline_hypothesis().hypothesis_id,
        campaign_id=None,
        dataset_reference=DatasetIdentity.from_reference(dataset_reference),
        feature_set=FeatureSetIdentity(
            feature_set_id=feature_set.feature_set_id,
            canonical_checksum=feature_set.canonical_checksum,
        ),
        label=LabelIdentity(
            label_id=GROSS_FORWARD_RETURN_1H.label_id,
            version=GROSS_FORWARD_RETURN_1H.version,
            target_column=GROSS_FORWARD_RETURN_1H.target_column,
        ),
        strategy=VersionedComponent(component_id="residual_momentum", version="1"),
        strategy_parameters={
            "momentum_weight_1h": config.momentum_weight_1h,
            "momentum_weight_4h": config.momentum_weight_4h,
            "momentum_weight_24h": config.momentum_weight_24h,
        },
        portfolio_policy=VersionedComponent(
            component_id="neutral_long_short",
            version="1",
        ),
        portfolio_parameters={
            "no_trade_score_band": config.no_trade_score_band,
            "gross_exposure": config.gross_exposure,
            "annual_volatility_target": config.annual_volatility_target,
            "max_symbol_weight": config.max_symbol_weight,
        },
        execution_model=ParameterizedComponent(
            component_id="bar_next_open",
            version="1",
            parameters={"lag_bars": 1},
        ),
        cost_model=ParameterizedComponent(
            component_id="configured_taker",
            version="1",
            parameters={
                "spread_bps": config.spread_bps,
                "slippage_bps": config.slippage_bps,
                "initial_capital_usdt": config.initial_capital_usdt,
                "fee_schedule": config.fee_schedule.model_dump(mode="json"),
            },
        ),
        split_plan=ParameterizedComponent(
            component_id="expanding_walk_forward",
            version="1",
            parameters={
                "train_days": config.walk_forward_train_days,
                "test_days": config.walk_forward_test_days,
                "embargo_bars": config.embargo_bars,
            },
        ),
        validation_plan=ParameterizedComponent(
            component_id="phase3_validation",
            version="1",
            parameters={
                "stress_cost_multipliers": [1.5, 2.0],
                "stress_signal_delay_bars": [1],
                "bootstrap_samples": config.block_bootstrap_samples,
                "bootstrap_block_hours": config.block_bootstrap_hours,
            },
        ),
        random_seed=config.random_seed,
        code_fingerprint=code_fingerprint or build_code_fingerprint(project_root),
        artifact_policy=artifact_policy,
    )


def resolve_dataset_path(data_root: Path, identity: DatasetIdentity) -> Path:
    candidates = sorted(data_root.glob("gold/binance/usdm/research_dataset/version=*/dataset.json"))
    for manifest_path in candidates:
        reference = load_dataset_reference(manifest_path)
        if DatasetIdentity.from_reference(reference) == identity:
            parquet_path = manifest_path.with_suffix(".parquet")
            if not parquet_path.is_file():
                raise ResearchError(f"dataset Parquet is missing: {parquet_path}")
            return parquet_path
    raise ResearchError(
        f"local dataset is unavailable for dataset_id={identity.dataset_id}; "
        "restore its manifest and Parquet before rerunning"
    )


def _execution_config(spec: ExperimentSpec, base: ResearchConfig) -> ResearchConfig:
    try:
        if (spec.execution_model.component_id, spec.execution_model.version) not in {
            ("bar_next_open", "1"),
            ("bar_next_open", "v1"),
        }:
            raise ResearchError("only the versioned bar_next_open execution model is supported")
        ExecutionParameters.model_validate(spec.execution_model.parameters)
        if (spec.cost_model.component_id, spec.cost_model.version) not in {
            ("configured_taker", "1"),
            ("configured_taker", "v1"),
        }:
            raise ResearchError("unsupported cost model")
        costs = CostParameters.model_validate(spec.cost_model.parameters)
        if (spec.split_plan.component_id, spec.split_plan.version) not in {
            ("expanding_walk_forward", "1"),
            ("expanding_walk_forward", "v1"),
        }:
            raise ResearchError("unsupported split plan")
        splits = SplitParameters.model_validate(spec.split_plan.parameters)
        if (spec.validation_plan.component_id, spec.validation_plan.version) not in {
            ("phase3_validation", "1"),
            ("phase3_validation", "v1"),
        }:
            raise ResearchError("unsupported validation plan")
        validation = ValidationParameters.model_validate(spec.validation_plan.parameters)
    except ValidationError as exc:
        raise ResearchError(f"invalid experiment execution parameters: {exc}") from exc
    if validation.stress_cost_multipliers != (1.5, 2.0):
        raise ResearchError("this runner requires cost stress multipliers [1.5, 2.0]")
    if validation.stress_signal_delay_bars != (1,):
        raise ResearchError("this runner requires the one-bar signal-delay stress")
    return base.model_copy(
        update={
            "spread_bps": costs.spread_bps,
            "slippage_bps": costs.slippage_bps,
            "initial_capital_usdt": costs.initial_capital_usdt,
            "fee_schedule": costs.fee_schedule,
            "walk_forward_train_days": splits.train_days,
            "walk_forward_test_days": splits.test_days,
            "embargo_bars": splits.embargo_bars,
            "block_bootstrap_samples": validation.bootstrap_samples,
            "block_bootstrap_hours": validation.bootstrap_block_hours,
            "random_seed": spec.random_seed,
        }
    )


class ExperimentRunner:
    def __init__(
        self,
        *,
        store: ResearchStore,
        data_root: Path,
        research_config: ResearchConfig,
        compression: str,
    ) -> None:
        self.store = store
        self.data_root = data_root.resolve()
        self.research_config = research_config
        self.pipeline = ExperimentArtifactPipeline(
            self.data_root,
            compression=compression,
        )

    def run(
        self,
        identifier: str,
        *,
        generate_chart: bool = False,
    ) -> ExperimentExecutionResult:
        spec = self.store.get_experiment(identifier)
        if spec is None:
            raise ResearchError(f"unknown experiment: {identifier}")
        previous = self.store.latest_successful_run(identifier)
        created = self.store.create_run(identifier)
        self.store.transition_run(created.run_id, RunStatus.QUEUED)
        running = self.store.transition_run(
            created.run_id,
            RunStatus.RUNNING,
            worker_id="local",
            host_name=socket.gethostname(),
            process_id=os.getpid(),
        )
        artifact_directory: Path | None = None
        try:
            dataset_path = resolve_dataset_path(self.data_root, spec.dataset_reference)
            strategy = build_strategy(
                spec.strategy.component_id,
                spec.strategy.version,
                spec.strategy_parameters,
            )
            portfolio_policy = build_portfolio_policy(
                spec.portfolio_policy.component_id,
                spec.portfolio_policy.version,
                spec.portfolio_parameters,
            )
            loaded_dataset = WORKER_DATASET_CACHE.load(
                dataset_path,
                feature_columns=tuple(
                    dict.fromkeys(
                        (*strategy.required_features(), *portfolio_policy.required_features())
                    )
                ),
                outcome_columns=ACCOUNTING_OUTCOME_FIELDS,
                metadata_columns=ACCOUNTING_METADATA_FIELDS,
            )
            config = _execution_config(spec, self.research_config)
            validation = run_research_validation(
                loaded_dataset.frame,
                config=config,
                strategy=strategy,
                portfolio_policy=portfolio_policy,
                panel_data=loaded_dataset.panel,
            )
            bundle = self.pipeline.persist(
                experiment_id=identifier,
                run_id=running.run_id,
                spec=spec,
                run=validation.run,
                stress=validation.stress,
                bootstrap=validation.bootstrap,
                generate_chart=generate_chart,
            )
            artifact_directory = bundle.final_directory
            digest = result_digest(
                metrics=bundle.metrics_payload,
                artifact_checksums=bundle.artifact_checksums,
            )
            deterministic = None
            if previous is not None:
                deterministic = previous.result_digest == digest
                if not deterministic:
                    raise ResearchError(
                        "determinism violation: rerun result_digest differs from prior success"
                    )
            verification = verify_run_artifacts(
                data_root=self.data_root,
                run_id=running.run_id,
                artifacts=bundle.artifacts,
            )
            if not verification.valid:
                raise ResearchError(
                    "artifact verification failed before completion: "
                    + "; ".join(verification.issues)
                )
            completed = self.store.complete_run(
                running.run_id,
                result_digest_value=digest,
                metrics=bundle.metric_records,
                artifacts=bundle.artifacts,
            )
            return ExperimentExecutionResult(
                experiment_id=identifier,
                run=completed,
                artifact_directory=bundle.final_directory,
                verification=verification,
                deterministic_with_previous=deterministic,
            )
        except BaseException as exc:
            current = self.store.get_run(running.run_id)
            if current is not None and current.status is RunStatus.RUNNING:
                if artifact_directory is not None and artifact_directory.exists():
                    self.pipeline.quarantine(
                        artifact_directory,
                        run_id=running.run_id,
                        reason="failed-after-promotion",
                    )
                traceback_path = self._write_traceback(running.run_id)
                self.store.transition_run(
                    running.run_id,
                    RunStatus.FAILED,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    traceback_path=traceback_path,
                )
            raise

    def verify_experiment(self, identifier: str) -> ArtifactVerification:
        run = self.store.latest_successful_run(identifier)
        if run is None:
            raise ResearchError(f"experiment has no successful run: {identifier}")
        return self.verify_run(run.run_id)

    def verify_run(self, run_id: str) -> ArtifactVerification:
        run = self.store.get_run(run_id)
        if run is None:
            raise ResearchError(f"unknown experiment run: {run_id}")
        artifacts = self.store.list_artifacts(run_id)
        verification = verify_run_artifacts(
            data_root=self.data_root,
            run_id=run_id,
            artifacts=artifacts,
        )
        issues = list(verification.issues)
        metrics_artifact = next(
            (artifact for artifact in artifacts if artifact.artifact_type == "metrics"),
            None,
        )
        if verification.valid and metrics_artifact is not None:
            payload = orjson.loads((self.data_root / metrics_artifact.path).read_bytes())
            if not isinstance(payload, dict):
                issues.append("metrics artifact root is not an object")
            else:
                deterministic_checksums = {
                    Path(artifact.path).name: artifact.checksum_sha256
                    for artifact in artifacts
                    if artifact.artifact_type not in {"manifest", "pnl"}
                }
                calculated = result_digest(
                    metrics=cast(Mapping[str, Any], payload),
                    artifact_checksums=deterministic_checksums,
                )
                if calculated != run.result_digest:
                    issues.append("result_digest mismatch")
        elif metrics_artifact is None:
            issues.append("metrics artifact is not registered")
        return ArtifactVerification(
            run_id=run_id,
            valid=not issues and verification.valid,
            checked_files=verification.checked_files,
            issues=tuple(issues),
        )

    def _write_traceback(self, run_id: str) -> str:
        storage = LocalFilesystemStorage(self.data_root)
        base = storage.path("quarantine", "research", f"{run_id}-traceback")
        directory = base
        ordinal = 1
        while directory.exists():
            ordinal += 1
            directory = base.with_name(f"{base.name}-{ordinal}")
        path = directory / "traceback.txt"
        storage.write_bytes_atomic(path, traceback.format_exc().encode("utf-8"))
        return path.relative_to(self.data_root).as_posix()


__all__ = [
    "CostParameters",
    "ExecutionParameters",
    "ExperimentExecutionResult",
    "ExperimentRunner",
    "SplitParameters",
    "ValidationParameters",
    "build_phase3_experiment_spec",
    "phase3_baseline_hypothesis",
    "resolve_dataset_path",
]
