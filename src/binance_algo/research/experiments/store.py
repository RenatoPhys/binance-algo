"""Transactional SQLite research registry with immutable definitions and explicit states."""

from __future__ import annotations

import hashlib
import math
import sqlite3
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from binance_algo.common.errors import InvalidResearchTransition, ResearchStoreError
from binance_algo.research.experiments.canonical import (
    canonical_json_text,
    canonical_sha256,
)
from binance_algo.research.experiments.ids import deterministic_run_id, experiment_id
from binance_algo.research.experiments.migrations import (
    LATEST_SCHEMA_VERSION,
    apply_migrations,
)
from binance_algo.research.experiments.models import (
    CampaignStatus,
    CodeFingerprint,
    ExperimentSpec,
    FeatureDecision,
    FeatureEvaluationSpec,
    FeatureEvaluationType,
    HypothesisSpec,
    HypothesisStatus,
    MetricScope,
    PromotionDecision,
    PromotionEventSpec,
    ResearchStage,
    RunStatus,
)
from binance_algo.research.features.base import FeatureDefinition
from binance_algo.research.features.registry import FeatureSetSpec

RUN_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.PENDING: frozenset({RunStatus.QUEUED}),
    RunStatus.QUEUED: frozenset({RunStatus.RUNNING, RunStatus.CANCELLED}),
    RunStatus.RUNNING: frozenset({RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.STALE}),
    RunStatus.STALE: frozenset({RunStatus.QUEUED}),
    RunStatus.SUCCEEDED: frozenset(),
    RunStatus.FAILED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
}

HYPOTHESIS_TRANSITIONS: dict[HypothesisStatus, frozenset[HypothesisStatus]] = {
    HypothesisStatus.DRAFT: frozenset({HypothesisStatus.READY, HypothesisStatus.INVALIDATED}),
    HypothesisStatus.READY: frozenset({HypothesisStatus.TESTED, HypothesisStatus.INVALIDATED}),
    HypothesisStatus.TESTED: frozenset(
        {
            HypothesisStatus.SUPPORTED,
            HypothesisStatus.REJECTED,
            HypothesisStatus.INCONCLUSIVE,
            HypothesisStatus.INVALIDATED,
        }
    ),
    HypothesisStatus.SUPPORTED: frozenset({HypothesisStatus.INVALIDATED}),
    HypothesisStatus.REJECTED: frozenset({HypothesisStatus.INVALIDATED}),
    HypothesisStatus.INCONCLUSIVE: frozenset(
        {HypothesisStatus.READY, HypothesisStatus.INVALIDATED}
    ),
    HypothesisStatus.INVALIDATED: frozenset(),
}

CAMPAIGN_TRANSITIONS: dict[CampaignStatus, frozenset[CampaignStatus]] = {
    CampaignStatus.PLANNED: frozenset({CampaignStatus.QUEUED, CampaignStatus.CANCELLED}),
    CampaignStatus.QUEUED: frozenset({CampaignStatus.RUNNING, CampaignStatus.CANCELLED}),
    CampaignStatus.RUNNING: frozenset(
        {
            CampaignStatus.COMPLETED,
            CampaignStatus.PARTIAL,
            CampaignStatus.FAILED,
            CampaignStatus.CANCELLED,
        }
    ),
    CampaignStatus.PARTIAL: frozenset({CampaignStatus.RUNNING}),
    CampaignStatus.COMPLETED: frozenset(),
    CampaignStatus.FAILED: frozenset(),
    CampaignStatus.CANCELLED: frozenset(),
}

PROMOTION_TRANSITIONS: dict[ResearchStage, frozenset[ResearchStage]] = {
    ResearchStage.DISCOVERY: frozenset(
        {ResearchStage.CANDIDATE, ResearchStage.REJECTED, ResearchStage.INVALIDATED}
    ),
    ResearchStage.CANDIDATE: frozenset(
        {ResearchStage.LOCKBOX_EVALUATED, ResearchStage.REJECTED, ResearchStage.INVALIDATED}
    ),
    ResearchStage.LOCKBOX_EVALUATED: frozenset(
        {ResearchStage.PHASE4_CANDIDATE, ResearchStage.REJECTED, ResearchStage.INVALIDATED}
    ),
    ResearchStage.PHASE4_CANDIDATE: frozenset({ResearchStage.INVALIDATED}),
    ResearchStage.REJECTED: frozenset({ResearchStage.INVALIDATED}),
    ResearchStage.INVALIDATED: frozenset(),
}


def now_ms() -> int:
    return time.time_ns() // 1_000_000


@dataclass(frozen=True, slots=True)
class ExperimentRunRecord:
    run_id: str
    experiment_id: str
    attempt: int
    status: RunStatus
    worker_id: str | None
    host_name: str | None
    process_id: int | None
    started_at_ms: int | None
    heartbeat_at_ms: int | None
    finished_at_ms: int | None
    runtime_seconds: float | None
    result_digest: str | None
    error_type: str | None
    error_message: str | None
    traceback_path: str | None
    created_at_ms: int


@dataclass(frozen=True, slots=True)
class ResearchRegistryStatus:
    database_path: str
    schema_version: int
    latest_schema_version: int
    journal_mode: str
    foreign_keys: bool
    counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class ResearchMetricRecord:
    scope: MetricScope
    metric_name: str
    metric_value: float
    fold: int | None = None
    regime: str | None = None
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ResearchArtifactRecord:
    artifact_type: str
    path: str
    checksum_sha256: str
    row_count: int | None
    size_bytes: int
    schema_version: int


@dataclass(frozen=True, slots=True)
class CampaignRecord:
    campaign_id: str
    name: str
    description: str
    hypothesis_id: str
    spec_json: str
    spec_sha256: str
    status: CampaignStatus
    created_at_ms: int
    started_at_ms: int | None
    finished_at_ms: int | None
    trial_count: int
    last_error: str | None


@dataclass(frozen=True, slots=True)
class FeatureEvaluationRecord:
    evaluation_id: str
    run_id: str
    experiment_id: str
    hypothesis_id: str
    feature_id: str
    evaluation_type: FeatureEvaluationType
    scope: str
    metric_name: str
    metric_value: float | None
    decision: FeatureDecision
    decision_reason: str
    context: Mapping[str, Any]
    created_at_ms: int

    def to_spec(self) -> FeatureEvaluationSpec:
        return FeatureEvaluationSpec(
            run_id=self.run_id,
            feature_id=self.feature_id,
            evaluation_type=self.evaluation_type,
            scope=self.scope,
            metric_name=self.metric_name,
            metric_value=self.metric_value,
            decision=self.decision,
            decision_reason=self.decision_reason,
            context=dict(self.context),
        )


@dataclass(frozen=True, slots=True)
class PromotionRecord:
    promotion_id: str
    experiment_id: str
    from_stage: ResearchStage
    to_stage: ResearchStage
    decision: PromotionDecision
    criteria_snapshot: Mapping[str, Any]
    reason: str
    created_at_ms: int
    code_fingerprint: CodeFingerprint

    def to_spec(self) -> PromotionEventSpec:
        return PromotionEventSpec(
            experiment_id=self.experiment_id,
            from_stage=self.from_stage,
            to_stage=self.to_stage,
            decision=self.decision,
            criteria_snapshot=dict(self.criteria_snapshot),
            reason=self.reason,
            code_fingerprint=self.code_fingerprint,
        )


class ResearchStore:
    def __init__(self, path: Path, *, busy_timeout_ms: int = 5_000) -> None:
        self.path = path.resolve()
        self._busy_timeout_ms = busy_timeout_ms

    def initialize(self, *, target_version: int | None = None) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with closing(self._connect()) as connection:
                return apply_migrations(connection, target_version=target_version)
        except ResearchStoreError:
            raise
        except sqlite3.Error as exc:
            raise ResearchStoreError(
                f"cannot initialize research store {self.path}: {exc}"
            ) from exc

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def status(self) -> ResearchRegistryStatus:
        tables = (
            "research_hypotheses",
            "research_campaigns",
            "research_feature_definitions",
            "research_feature_sets",
            "research_feature_set_members",
            "research_experiments",
            "research_campaign_experiments",
            "research_experiment_runs",
            "research_metrics",
            "research_artifacts",
            "research_feature_evaluations",
            "research_promotions",
        )
        try:
            with closing(self._connect()) as connection:
                version_row = connection.execute(
                    "SELECT MAX(version) FROM research_schema_migrations"
                ).fetchone()
                journal_row = connection.execute("PRAGMA journal_mode").fetchone()
                foreign_key_row = connection.execute("PRAGMA foreign_keys").fetchone()
                counts = {
                    table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                    for table in tables
                }
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot read research registry status: {exc}") from exc
        return ResearchRegistryStatus(
            database_path=str(self.path),
            schema_version=(
                int(version_row[0]) if version_row is not None and version_row[0] is not None else 0
            ),
            latest_schema_version=LATEST_SCHEMA_VERSION,
            journal_mode=str(journal_row[0]) if journal_row is not None else "unknown",
            foreign_keys=bool(foreign_key_row[0]) if foreign_key_row is not None else False,
            counts=counts,
        )

    def register_hypothesis(self, spec: HypothesisSpec) -> HypothesisSpec:
        criteria_json = canonical_json_text(spec.preregistered_success_criteria)
        timestamp = now_ms()
        values = (
            spec.hypothesis_id,
            spec.title,
            spec.mechanism,
            spec.expected_direction,
            spec.expected_horizon,
            spec.target_universe,
            criteria_json,
            spec.status.value,
            timestamp,
            timestamp,
            spec.notes,
        )
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO research_hypotheses(
                        hypothesis_id, title, mechanism, expected_direction,
                        expected_horizon, target_universe,
                        preregistered_success_criteria_json, status,
                        created_at_ms, updated_at_ms, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )
                row = connection.execute(
                    "SELECT * FROM research_hypotheses WHERE hypothesis_id = ?",
                    (spec.hypothesis_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ResearchStoreError(
                f"cannot register hypothesis {spec.hypothesis_id}: {exc}"
            ) from exc
        if row is None:
            raise ResearchStoreError(f"registered hypothesis disappeared: {spec.hypothesis_id}")
        existing = self._hypothesis_from_row(row)
        if self._hypothesis_definition(existing) != self._hypothesis_definition(spec):
            raise ResearchStoreError(f"immutable hypothesis conflict: {spec.hypothesis_id}")
        return existing

    def get_hypothesis(self, hypothesis_id: str) -> HypothesisSpec | None:
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT * FROM research_hypotheses WHERE hypothesis_id = ?",
                    (hypothesis_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot read hypothesis {hypothesis_id}: {exc}") from exc
        return self._hypothesis_from_row(row) if row is not None else None

    def list_hypotheses(self) -> tuple[HypothesisSpec, ...]:
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    "SELECT * FROM research_hypotheses ORDER BY created_at_ms, hypothesis_id"
                ).fetchall()
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot list research hypotheses: {exc}") from exc
        return tuple(self._hypothesis_from_row(row) for row in rows)

    def transition_hypothesis(
        self,
        hypothesis_id: str,
        status: HypothesisStatus,
    ) -> HypothesisSpec:
        try:
            with self.transaction() as connection:
                row = connection.execute(
                    "SELECT * FROM research_hypotheses WHERE hypothesis_id = ?",
                    (hypothesis_id,),
                ).fetchone()
                if row is None:
                    raise ResearchStoreError(f"unknown hypothesis: {hypothesis_id}")
                current = HypothesisStatus(str(row["status"]))
                if status not in HYPOTHESIS_TRANSITIONS[current]:
                    raise InvalidResearchTransition(
                        f"invalid hypothesis transition {current.value} -> {status.value}"
                    )
                connection.execute(
                    """
                    UPDATE research_hypotheses SET status = ?, updated_at_ms = ?
                    WHERE hypothesis_id = ?
                    """,
                    (status.value, now_ms(), hypothesis_id),
                )
                updated = connection.execute(
                    "SELECT * FROM research_hypotheses WHERE hypothesis_id = ?",
                    (hypothesis_id,),
                ).fetchone()
        except (ResearchStoreError, InvalidResearchTransition):
            raise
        except sqlite3.Error as exc:
            raise ResearchStoreError(
                f"cannot transition hypothesis {hypothesis_id}: {exc}"
            ) from exc
        assert updated is not None
        return self._hypothesis_from_row(updated)

    def register_feature_definition(self, definition: FeatureDefinition) -> str:
        manifest = definition.to_manifest()
        timestamp = now_ms()
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO research_feature_definitions(
                        feature_id, name, version, description, dtype, lookback,
                        timestamp_semantics, required_datasets_json, required_columns_json,
                        implementation_path, parameters_json, status, created_at_ms,
                        deprecated_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (
                        definition.feature_id,
                        definition.name,
                        definition.version,
                        definition.description,
                        definition.dtype,
                        definition.lookback,
                        definition.timestamp_semantics,
                        canonical_json_text(definition.required_datasets),
                        canonical_json_text(definition.required_columns),
                        definition.implementation_path,
                        canonical_json_text(dict(definition.parameters)),
                        definition.status.value,
                        timestamp,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM research_feature_definitions WHERE feature_id = ?",
                    (definition.feature_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ResearchStoreError(
                f"cannot register feature {definition.feature_id}: {exc}"
            ) from exc
        if row is None or self._feature_manifest_from_row(row) != manifest:
            raise ResearchStoreError(f"immutable feature conflict: {definition.feature_id}")
        return definition.feature_id

    def register_feature_set(self, spec: FeatureSetSpec) -> str:
        name = spec.feature_set_id.rsplit(":", 1)[0]
        spec_json = canonical_json_text(spec.identity_payload())
        timestamp = now_ms()
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO research_feature_sets(
                        feature_set_id, name, version, description,
                        spec_json, spec_sha256, created_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        spec.feature_set_id,
                        name,
                        spec.version,
                        spec.description,
                        spec_json,
                        spec.canonical_checksum,
                        timestamp,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM research_feature_sets WHERE feature_set_id = ?",
                    (spec.feature_set_id,),
                ).fetchone()
                if row is None:
                    raise ResearchStoreError(
                        f"registered feature set disappeared: {spec.feature_set_id}"
                    )
                try:
                    existing_payload = orjson.loads(str(row["spec_json"]))
                except orjson.JSONDecodeError as exc:
                    raise ResearchStoreError(
                        f"stored feature set is not valid JSON: {spec.feature_set_id}"
                    ) from exc
                if isinstance(existing_payload, dict):
                    existing_payload.pop("canonical_checksum", None)
                    existing_payload.pop("declared_feature_order", None)
                if (
                    existing_payload != orjson.loads(spec_json)
                    or str(row["spec_sha256"]) != spec.canonical_checksum
                ):
                    raise ResearchStoreError(
                        f"immutable feature set conflict: {spec.feature_set_id}"
                    )
                for ordinal, feature_id in enumerate(sorted(spec.feature_ids)):
                    parameters = canonical_json_text(
                        dict(spec.per_feature_parameters.get(feature_id, {}))
                    )
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO research_feature_set_members(
                            feature_set_id, feature_id, ordinal, parameters_json
                        ) VALUES (?, ?, ?, ?)
                        """,
                        (spec.feature_set_id, feature_id, ordinal, parameters),
                    )
                    member = connection.execute(
                        """
                        SELECT ordinal, parameters_json FROM research_feature_set_members
                        WHERE feature_set_id = ? AND feature_id = ?
                        """,
                        (spec.feature_set_id, feature_id),
                    ).fetchone()
                    if member is None or str(member[1]) != parameters:
                        raise ResearchStoreError(
                            f"immutable feature-set member conflict: {feature_id}"
                        )
        except ResearchStoreError:
            raise
        except sqlite3.Error as exc:
            raise ResearchStoreError(
                f"cannot register feature set {spec.feature_set_id}: {exc}"
            ) from exc
        return spec.feature_set_id

    def list_feature_definitions(self) -> tuple[dict[str, object], ...]:
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    """
                    SELECT * FROM research_feature_definitions
                    ORDER BY name, version, feature_id
                    """
                ).fetchall()
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot list feature definitions: {exc}") from exc
        return tuple(self._feature_manifest_from_row(row) for row in rows)

    def get_feature_definition(self, feature_id: str) -> dict[str, object] | None:
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT * FROM research_feature_definitions WHERE feature_id = ?",
                    (feature_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot read feature {feature_id}: {exc}") from exc
        return self._feature_manifest_from_row(row) if row is not None else None

    def register_experiment(self, spec: ExperimentSpec) -> str:
        identifier = experiment_id(spec)
        spec_json = canonical_json_text(spec)
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO research_experiments(
                        experiment_id, experiment_sha256, hypothesis_id, spec_json,
                        dataset_id, universe_version, feature_set_id, label_id,
                        strategy_id, strategy_version, portfolio_policy_id,
                        execution_model_id, cost_model_id, split_plan_id,
                        random_seed, code_fingerprint_json, created_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        identifier,
                        spec.hypothesis_id,
                        spec_json,
                        spec.dataset_reference.dataset_id,
                        spec.dataset_reference.universe_version,
                        spec.feature_set.feature_set_id,
                        spec.label.label_id,
                        spec.strategy.component_id,
                        spec.strategy.version,
                        spec.portfolio_policy.component_id,
                        spec.execution_model.component_id,
                        spec.cost_model.component_id,
                        spec.split_plan.component_id,
                        spec.random_seed,
                        canonical_json_text(spec.code_fingerprint),
                        now_ms(),
                    ),
                )
                row = connection.execute(
                    "SELECT experiment_sha256, spec_json FROM research_experiments "
                    "WHERE experiment_id = ?",
                    (identifier,),
                ).fetchone()
        except sqlite3.IntegrityError as exc:
            raise ResearchStoreError(f"cannot register experiment {identifier}: {exc}") from exc
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot register experiment {identifier}: {exc}") from exc
        if row is None or (str(row[0]), str(row[1])) != (identifier, spec_json):
            raise ResearchStoreError(f"immutable experiment conflict: {identifier}")
        return identifier

    def register_campaign(
        self,
        *,
        identifier: str,
        name: str,
        description: str,
        hypothesis_id: str,
        spec_payload: Mapping[str, Any],
        trial_count: int,
    ) -> CampaignRecord:
        if not identifier or not name or not description or trial_count < 0:
            raise ResearchStoreError("campaign definition is incomplete")
        try:
            spec_json = orjson.dumps(spec_payload, option=orjson.OPT_SORT_KEYS).decode()
        except (TypeError, orjson.JSONEncodeError) as exc:
            raise ResearchStoreError(f"campaign spec is not serializable: {exc}") from exc
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO research_campaigns(
                        campaign_id, name, description, hypothesis_id, spec_json,
                        spec_sha256, status, created_at_ms, trial_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        name,
                        description,
                        hypothesis_id,
                        spec_json,
                        identifier,
                        CampaignStatus.PLANNED.value,
                        now_ms(),
                        trial_count,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM research_campaigns WHERE campaign_id = ? OR name = ?",
                    (identifier, name),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot register campaign {name}: {exc}") from exc
        if row is None:
            raise ResearchStoreError(f"registered campaign disappeared: {name}")
        record = self._campaign_from_row(row)
        if any(
            (
                record.campaign_id != identifier,
                record.name != name,
                record.description != description,
                record.hypothesis_id != hypothesis_id,
                record.spec_json != spec_json,
                record.spec_sha256 != identifier,
                record.trial_count != trial_count,
            )
        ):
            raise ResearchStoreError(f"immutable campaign conflict: {name}")
        return record

    def get_campaign(self, identifier_or_name: str) -> CampaignRecord | None:
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT * FROM research_campaigns WHERE campaign_id = ? OR name = ?",
                    (identifier_or_name, identifier_or_name),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot read campaign {identifier_or_name}: {exc}") from exc
        return self._campaign_from_row(row) if row is not None else None

    def transition_campaign(
        self,
        identifier_or_name: str,
        status: CampaignStatus,
        *,
        last_error: str | None = None,
    ) -> CampaignRecord:
        try:
            with self.transaction() as connection:
                row = connection.execute(
                    "SELECT * FROM research_campaigns WHERE campaign_id = ? OR name = ?",
                    (identifier_or_name, identifier_or_name),
                ).fetchone()
                if row is None:
                    raise ResearchStoreError(f"unknown campaign: {identifier_or_name}")
                current = CampaignStatus(str(row["status"]))
                if status not in CAMPAIGN_TRANSITIONS[current]:
                    raise InvalidResearchTransition(
                        f"invalid campaign transition {current.value} -> {status.value}"
                    )
                timestamp = now_ms()
                started_at = row["started_at_ms"]
                if status is CampaignStatus.RUNNING and started_at is None:
                    started_at = timestamp
                finished_at = (
                    timestamp
                    if status
                    in {
                        CampaignStatus.COMPLETED,
                        CampaignStatus.FAILED,
                        CampaignStatus.CANCELLED,
                    }
                    else None
                )
                connection.execute(
                    """
                    UPDATE research_campaigns SET status = ?, started_at_ms = ?,
                        finished_at_ms = ?, last_error = ? WHERE campaign_id = ?
                    """,
                    (
                        status.value,
                        started_at,
                        finished_at,
                        last_error,
                        str(row["campaign_id"]),
                    ),
                )
                updated = connection.execute(
                    "SELECT * FROM research_campaigns WHERE campaign_id = ?",
                    (str(row["campaign_id"]),),
                ).fetchone()
        except (ResearchStoreError, InvalidResearchTransition):
            raise
        except sqlite3.Error as exc:
            raise ResearchStoreError(
                f"cannot transition campaign {identifier_or_name}: {exc}"
            ) from exc
        assert updated is not None
        return self._campaign_from_row(updated)

    def associate_campaign_experiment(
        self,
        *,
        campaign_id: str,
        experiment_id_value: str,
        ordinal: int,
        tags: Mapping[str, Any] | None = None,
    ) -> None:
        tags_json = canonical_json_text(dict(tags or {}))
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO research_campaign_experiments(
                        campaign_id, experiment_id, ordinal, tags_json
                    ) VALUES (?, ?, ?, ?)
                    """,
                    (campaign_id, experiment_id_value, ordinal, tags_json),
                )
                row = connection.execute(
                    """
                    SELECT ordinal, tags_json FROM research_campaign_experiments
                    WHERE campaign_id = ? AND experiment_id = ?
                    """,
                    (campaign_id, experiment_id_value),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ResearchStoreError(
                f"cannot associate experiment {experiment_id_value} with {campaign_id}: {exc}"
            ) from exc
        if row is None or (int(row["ordinal"]), str(row["tags_json"])) != (
            ordinal,
            tags_json,
        ):
            raise ResearchStoreError(
                f"immutable campaign experiment conflict: {campaign_id}/{experiment_id_value}"
            )

    def list_campaign_experiments(
        self,
        identifier_or_name: str,
    ) -> tuple[tuple[int, str, Mapping[str, Any]], ...]:
        campaign = self.get_campaign(identifier_or_name)
        if campaign is None:
            raise ResearchStoreError(f"unknown campaign: {identifier_or_name}")
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    """
                    SELECT ordinal, experiment_id, tags_json
                    FROM research_campaign_experiments WHERE campaign_id = ?
                    ORDER BY ordinal
                    """,
                    (campaign.campaign_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise ResearchStoreError(
                f"cannot list campaign experiments for {identifier_or_name}: {exc}"
            ) from exc
        return tuple(
            (int(row["ordinal"]), str(row["experiment_id"]), orjson.loads(str(row["tags_json"])))
            for row in rows
        )

    def get_experiment(self, identifier: str) -> ExperimentSpec | None:
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT spec_json FROM research_experiments WHERE experiment_id = ?",
                    (identifier,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot read experiment {identifier}: {exc}") from exc
        if row is None:
            return None
        return ExperimentSpec.model_validate(orjson.loads(str(row[0])))

    def list_experiment_ids(self) -> tuple[str, ...]:
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    "SELECT experiment_id FROM research_experiments ORDER BY created_at_ms"
                ).fetchall()
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot list experiments: {exc}") from exc
        return tuple(str(row[0]) for row in rows)

    def create_run(self, identifier: str) -> ExperimentRunRecord:
        try:
            with self.transaction() as connection:
                exists = connection.execute(
                    "SELECT 1 FROM research_experiments WHERE experiment_id = ?",
                    (identifier,),
                ).fetchone()
                if exists is None:
                    raise ResearchStoreError(f"unknown experiment: {identifier}")
                row = connection.execute(
                    "SELECT COALESCE(MAX(attempt), 0) + 1 FROM research_experiment_runs "
                    "WHERE experiment_id = ?",
                    (identifier,),
                ).fetchone()
                attempt = int(row[0])
                run_id = deterministic_run_id(
                    experiment_id_value=identifier,
                    attempt=attempt,
                )
                connection.execute(
                    """
                    INSERT INTO research_experiment_runs(
                        run_id, experiment_id, attempt, status, created_at_ms
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (run_id, identifier, attempt, RunStatus.PENDING.value, now_ms()),
                )
                created = connection.execute(
                    "SELECT * FROM research_experiment_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
        except ResearchStoreError:
            raise
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot create run for {identifier}: {exc}") from exc
        assert created is not None
        return self._run_from_row(created)

    def get_run(self, run_id: str) -> ExperimentRunRecord | None:
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT * FROM research_experiment_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot read experiment run {run_id}: {exc}") from exc
        return self._run_from_row(row) if row is not None else None

    def transition_run(
        self,
        run_id: str,
        status: RunStatus,
        *,
        result_digest_value: str | None = None,
        worker_id: str | None = None,
        host_name: str | None = None,
        process_id: int | None = None,
        error_type: str | None = None,
        error_message: str | None = None,
        traceback_path: str | None = None,
    ) -> ExperimentRunRecord:
        try:
            with self.transaction() as connection:
                row = connection.execute(
                    "SELECT * FROM research_experiment_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    raise ResearchStoreError(f"unknown experiment run: {run_id}")
                current = RunStatus(str(row["status"]))
                if status not in RUN_TRANSITIONS[current]:
                    raise InvalidResearchTransition(
                        f"invalid experiment run transition {current.value} -> {status.value}"
                    )
                if status is RunStatus.SUCCEEDED and not result_digest_value:
                    raise InvalidResearchTransition(
                        "experiment run cannot succeed without a result digest"
                    )
                if status is RunStatus.SUCCEEDED:
                    metric_count = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM research_metrics WHERE run_id = ?",
                            (run_id,),
                        ).fetchone()[0]
                    )
                    artifact_count = int(
                        connection.execute(
                            "SELECT COUNT(*) FROM research_artifacts WHERE run_id = ?",
                            (run_id,),
                        ).fetchone()[0]
                    )
                    if metric_count == 0 or artifact_count == 0:
                        raise InvalidResearchTransition(
                            "experiment run cannot succeed before metrics and artifacts exist"
                        )
                if status is not RunStatus.SUCCEEDED and result_digest_value is not None:
                    raise InvalidResearchTransition(
                        "result digest is valid only for a succeeded experiment run"
                    )
                timestamp = now_ms()
                started = int(row["started_at_ms"]) if row["started_at_ms"] is not None else None
                if status is RunStatus.RUNNING and started is None:
                    started = timestamp
                finished = (
                    timestamp
                    if status
                    in {
                        RunStatus.SUCCEEDED,
                        RunStatus.FAILED,
                        RunStatus.CANCELLED,
                    }
                    else None
                )
                runtime = (
                    (finished - started) / 1_000
                    if finished is not None and started is not None
                    else None
                )
                heartbeat = timestamp if status is RunStatus.RUNNING else row["heartbeat_at_ms"]
                connection.execute(
                    """
                    UPDATE research_experiment_runs SET
                        status = ?, worker_id = COALESCE(?, worker_id),
                        host_name = COALESCE(?, host_name),
                        process_id = COALESCE(?, process_id),
                        started_at_ms = ?, heartbeat_at_ms = ?, finished_at_ms = ?,
                        runtime_seconds = ?, result_digest = ?, error_type = ?,
                        error_message = ?, traceback_path = ?
                    WHERE run_id = ?
                    """,
                    (
                        status.value,
                        worker_id,
                        host_name,
                        process_id,
                        started,
                        heartbeat,
                        finished,
                        runtime,
                        result_digest_value,
                        error_type,
                        error_message,
                        traceback_path,
                        run_id,
                    ),
                )
                updated = connection.execute(
                    "SELECT * FROM research_experiment_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
        except (ResearchStoreError, InvalidResearchTransition):
            raise
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot transition experiment run {run_id}: {exc}") from exc
        assert updated is not None
        return self._run_from_row(updated)

    def heartbeat_run(self, run_id: str, *, timestamp_ms: int | None = None) -> None:
        timestamp = now_ms() if timestamp_ms is None else timestamp_ms
        try:
            with self.transaction() as connection:
                row = connection.execute(
                    "SELECT status FROM research_experiment_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    raise ResearchStoreError(f"unknown experiment run: {run_id}")
                if RunStatus(str(row[0])) is not RunStatus.RUNNING:
                    raise InvalidResearchTransition("only a running experiment can heartbeat")
                connection.execute(
                    "UPDATE research_experiment_runs SET heartbeat_at_ms = ? WHERE run_id = ?",
                    (timestamp, run_id),
                )
        except (ResearchStoreError, InvalidResearchTransition):
            raise
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot heartbeat experiment run {run_id}: {exc}") from exc

    def mark_stale_runs(self, *, stale_before_ms: int) -> tuple[str, ...]:
        try:
            with self.transaction() as connection:
                rows = connection.execute(
                    """
                    SELECT run_id FROM research_experiment_runs
                    WHERE status = 'RUNNING'
                      AND COALESCE(heartbeat_at_ms, started_at_ms, created_at_ms) < ?
                    ORDER BY run_id
                    """,
                    (stale_before_ms,),
                ).fetchall()
                run_ids = tuple(str(row[0]) for row in rows)
                if run_ids:
                    placeholders = ", ".join("?" for _ in run_ids)
                    connection.execute(
                        f"UPDATE research_experiment_runs SET status = 'STALE' "
                        f"WHERE run_id IN ({placeholders})",
                        run_ids,
                    )
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot recover stale experiment runs: {exc}") from exc
        return run_ids

    def record_metric(
        self,
        *,
        run_id: str,
        scope: MetricScope,
        metric_name: str,
        metric_value: float,
        fold: int | None = None,
        regime: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not metric_name or not math.isfinite(metric_value):
            raise ResearchStoreError("research metric name and value must be valid and finite")
        metadata_json = canonical_json_text(dict(metadata or {}))
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO research_metrics(
                        run_id, scope, fold, regime, metric_name, metric_value, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        scope.value,
                        fold,
                        regime,
                        metric_name,
                        metric_value,
                        metadata_json,
                    ),
                )
                row = connection.execute(
                    """
                    SELECT metric_value, metadata_json FROM research_metrics
                    WHERE run_id = ? AND scope = ? AND fold IS ? AND regime IS ?
                      AND metric_name = ?
                    """,
                    (run_id, scope.value, fold, regime, metric_name),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot record metric for {run_id}: {exc}") from exc
        if row is None or (float(row[0]), str(row[1])) != (metric_value, metadata_json):
            raise ResearchStoreError(
                f"immutable metric conflict: {run_id}/{scope.value}/{metric_name}"
            )

    def record_artifact(
        self,
        *,
        run_id: str,
        artifact_type: str,
        path: str,
        checksum_sha256: str,
        row_count: int | None,
        size_bytes: int,
        schema_version: int,
    ) -> str:
        payload = "\x1f".join((run_id, artifact_type, path, checksum_sha256)).encode()
        artifact_id = hashlib.sha256(payload).hexdigest()
        try:
            with self.transaction() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO research_artifacts(
                        artifact_id, run_id, artifact_type, path, checksum_sha256,
                        row_count, size_bytes, schema_version, created_at_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact_id,
                        run_id,
                        artifact_type,
                        path,
                        checksum_sha256,
                        row_count,
                        size_bytes,
                        schema_version,
                        now_ms(),
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM research_artifacts WHERE artifact_id = ?",
                    (artifact_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot record artifact for {run_id}: {exc}") from exc
        if row is None or any(
            (
                str(row["run_id"]) != run_id,
                str(row["artifact_type"]) != artifact_type,
                str(row["path"]) != path,
                str(row["checksum_sha256"]) != checksum_sha256,
                (int(row["row_count"]) if row["row_count"] is not None else None) != row_count,
                int(row["size_bytes"]) != size_bytes,
                int(row["schema_version"]) != schema_version,
            )
        ):
            raise ResearchStoreError(f"immutable artifact conflict: {artifact_id}")
        return artifact_id

    def complete_run(
        self,
        run_id: str,
        *,
        result_digest_value: str,
        metrics: Sequence[ResearchMetricRecord],
        artifacts: Sequence[ResearchArtifactRecord],
    ) -> ExperimentRunRecord:
        """Register validated outputs and success atomically in one database transaction."""

        if not result_digest_value or not metrics or not artifacts:
            raise ResearchStoreError(
                "run completion requires a result digest, metrics, and artifacts"
            )
        try:
            with self.transaction() as connection:
                row = connection.execute(
                    "SELECT * FROM research_experiment_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    raise ResearchStoreError(f"unknown experiment run: {run_id}")
                current = RunStatus(str(row["status"]))
                if current is not RunStatus.RUNNING:
                    raise InvalidResearchTransition(
                        f"only a running experiment can complete, not {current.value}"
                    )
                for metric in metrics:
                    if not metric.metric_name or not math.isfinite(metric.metric_value):
                        raise ResearchStoreError(
                            "research metric name and value must be valid and finite"
                        )
                    connection.execute(
                        """
                        INSERT INTO research_metrics(
                            run_id, scope, fold, regime, metric_name,
                            metric_value, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            metric.scope.value,
                            metric.fold,
                            metric.regime,
                            metric.metric_name,
                            metric.metric_value,
                            canonical_json_text(dict(metric.metadata or {})),
                        ),
                    )
                for artifact in artifacts:
                    payload = "\x1f".join(
                        (
                            run_id,
                            artifact.artifact_type,
                            artifact.path,
                            artifact.checksum_sha256,
                        )
                    ).encode()
                    connection.execute(
                        """
                        INSERT INTO research_artifacts(
                            artifact_id, run_id, artifact_type, path, checksum_sha256,
                            row_count, size_bytes, schema_version, created_at_ms
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            hashlib.sha256(payload).hexdigest(),
                            run_id,
                            artifact.artifact_type,
                            artifact.path,
                            artifact.checksum_sha256,
                            artifact.row_count,
                            artifact.size_bytes,
                            artifact.schema_version,
                            now_ms(),
                        ),
                    )
                timestamp = now_ms()
                started = (
                    int(row["started_at_ms"]) if row["started_at_ms"] is not None else timestamp
                )
                connection.execute(
                    """
                    UPDATE research_experiment_runs SET
                        status = ?, finished_at_ms = ?, runtime_seconds = ?,
                        result_digest = ?, error_type = NULL, error_message = NULL,
                        traceback_path = NULL
                    WHERE run_id = ?
                    """,
                    (
                        RunStatus.SUCCEEDED.value,
                        timestamp,
                        (timestamp - started) / 1_000,
                        result_digest_value,
                        run_id,
                    ),
                )
                updated = connection.execute(
                    "SELECT * FROM research_experiment_runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
        except (ResearchStoreError, InvalidResearchTransition):
            raise
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot complete experiment run {run_id}: {exc}") from exc
        assert updated is not None
        return self._run_from_row(updated)

    def list_runs(
        self, *, experiment_id_value: str | None = None
    ) -> tuple[ExperimentRunRecord, ...]:
        query = "SELECT * FROM research_experiment_runs"
        parameters: tuple[str, ...] = ()
        if experiment_id_value is not None:
            query += " WHERE experiment_id = ?"
            parameters = (experiment_id_value,)
        query += " ORDER BY created_at_ms, attempt"
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(query, parameters).fetchall()
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot list experiment runs: {exc}") from exc
        return tuple(self._run_from_row(row) for row in rows)

    def latest_successful_run(self, identifier: str) -> ExperimentRunRecord | None:
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    """
                    SELECT * FROM research_experiment_runs
                    WHERE experiment_id = ? AND status = 'SUCCEEDED'
                    ORDER BY attempt DESC LIMIT 1
                    """,
                    (identifier,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot read successful run for {identifier}: {exc}") from exc
        return self._run_from_row(row) if row is not None else None

    def list_artifacts(self, run_id: str) -> tuple[ResearchArtifactRecord, ...]:
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    """
                    SELECT artifact_type, path, checksum_sha256, row_count,
                           size_bytes, schema_version
                    FROM research_artifacts WHERE run_id = ?
                    ORDER BY artifact_type, path
                    """,
                    (run_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot list artifacts for {run_id}: {exc}") from exc
        return tuple(
            ResearchArtifactRecord(
                artifact_type=str(row["artifact_type"]),
                path=str(row["path"]),
                checksum_sha256=str(row["checksum_sha256"]),
                row_count=int(row["row_count"]) if row["row_count"] is not None else None,
                size_bytes=int(row["size_bytes"]),
                schema_version=int(row["schema_version"]),
            )
            for row in rows
        )

    def list_metrics(self, run_id: str) -> tuple[ResearchMetricRecord, ...]:
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    """
                    SELECT scope, fold, regime, metric_name, metric_value, metadata_json
                    FROM research_metrics WHERE run_id = ?
                    ORDER BY scope, fold, regime, metric_name
                    """,
                    (run_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot list metrics for {run_id}: {exc}") from exc
        return tuple(
            ResearchMetricRecord(
                scope=MetricScope(str(row["scope"])),
                fold=int(row["fold"]) if row["fold"] is not None else None,
                regime=str(row["regime"]) if row["regime"] is not None else None,
                metric_name=str(row["metric_name"]),
                metric_value=float(row["metric_value"]),
                metadata=orjson.loads(str(row["metadata_json"])),
            )
            for row in rows
        )

    def record_feature_evaluation(
        self,
        spec: FeatureEvaluationSpec,
    ) -> FeatureEvaluationRecord:
        return self.record_feature_evaluations((spec,))[0]

    def record_feature_evaluations(
        self,
        specs: Sequence[FeatureEvaluationSpec],
    ) -> tuple[FeatureEvaluationRecord, ...]:
        if not specs:
            raise ResearchStoreError("feature evaluation batch cannot be empty")
        identifiers = tuple(canonical_sha256(spec.model_dump(mode="python")) for spec in specs)
        if len(set(identifiers)) != len(identifiers):
            raise ResearchStoreError("feature evaluation batch contains duplicates")
        created: list[FeatureEvaluationRecord] = []
        try:
            with self.transaction() as connection:
                for identifier, spec in zip(identifiers, specs, strict=True):
                    run = connection.execute(
                        """
                        SELECT r.status, r.experiment_id, e.hypothesis_id
                        FROM research_experiment_runs AS r
                        JOIN research_experiments AS e
                          ON e.experiment_id = r.experiment_id
                        WHERE r.run_id = ?
                        """,
                        (spec.run_id,),
                    ).fetchone()
                    if run is None:
                        raise ResearchStoreError(
                            f"feature evaluation references an unknown run: {spec.run_id}"
                        )
                    if RunStatus(str(run["status"])) is not RunStatus.SUCCEEDED:
                        raise ResearchStoreError(
                            "feature evaluations require a succeeded experiment run"
                        )
                    feature = connection.execute(
                        "SELECT 1 FROM research_feature_definitions WHERE feature_id = ?",
                        (spec.feature_id,),
                    ).fetchone()
                    if feature is None:
                        raise ResearchStoreError(
                            f"feature evaluation references an unknown feature: {spec.feature_id}"
                        )
                    context_json = canonical_json_text(spec.context)
                    timestamp = now_ms()
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO research_feature_evaluations(
                            evaluation_id, run_id, feature_id, evaluation_type,
                            scope, metric_name, metric_value, decision,
                            decision_reason, context_json, created_at_ms
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            identifier,
                            spec.run_id,
                            spec.feature_id,
                            spec.evaluation_type.value,
                            spec.scope,
                            spec.metric_name,
                            spec.metric_value,
                            spec.decision.value,
                            spec.decision_reason,
                            context_json,
                            timestamp,
                        ),
                    )
                    row = connection.execute(
                        """
                        SELECT fe.*, r.experiment_id, e.hypothesis_id
                        FROM research_feature_evaluations AS fe
                        JOIN research_experiment_runs AS r ON r.run_id = fe.run_id
                        JOIN research_experiments AS e
                          ON e.experiment_id = r.experiment_id
                        WHERE fe.evaluation_id = ?
                        """,
                        (identifier,),
                    ).fetchone()
                    if row is None:
                        raise ResearchStoreError(
                            f"registered feature evaluation disappeared: {identifier}"
                        )
                    record = self._feature_evaluation_from_row(row)
                    if record.to_spec() != spec:
                        raise ResearchStoreError(
                            f"immutable feature evaluation conflict: {identifier}"
                        )
                    created.append(record)
        except ResearchStoreError:
            raise
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot record feature evaluation: {exc}") from exc
        return tuple(created)

    def list_feature_evaluations(
        self,
        *,
        feature_id: str | None = None,
        run_id: str | None = None,
        hypothesis_id: str | None = None,
    ) -> tuple[FeatureEvaluationRecord, ...]:
        filters: list[str] = []
        parameters: list[str] = []
        if feature_id is not None:
            filters.append("fe.feature_id = ?")
            parameters.append(feature_id)
        if run_id is not None:
            filters.append("fe.run_id = ?")
            parameters.append(run_id)
        if hypothesis_id is not None:
            filters.append("e.hypothesis_id = ?")
            parameters.append(hypothesis_id)
        query = """
            SELECT fe.*, r.experiment_id, e.hypothesis_id
            FROM research_feature_evaluations AS fe
            JOIN research_experiment_runs AS r ON r.run_id = fe.run_id
            JOIN research_experiments AS e ON e.experiment_id = r.experiment_id
        """
        if filters:
            query += " WHERE " + " AND ".join(filters)
        query += " ORDER BY fe.created_at_ms, fe.evaluation_id"
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(query, tuple(parameters)).fetchall()
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot list feature evaluations: {exc}") from exc
        return tuple(self._feature_evaluation_from_row(row) for row in rows)

    def campaigns_using_feature(self, feature_id: str) -> tuple[CampaignRecord, ...]:
        try:
            with closing(self._connect()) as connection:
                exists = connection.execute(
                    "SELECT 1 FROM research_feature_definitions WHERE feature_id = ?",
                    (feature_id,),
                ).fetchone()
                if exists is None:
                    raise ResearchStoreError(f"unknown feature: {feature_id}")
                rows = connection.execute(
                    """
                    SELECT DISTINCT c.*
                    FROM research_feature_set_members AS fsm
                    JOIN research_experiments AS e
                      ON e.feature_set_id = fsm.feature_set_id
                    JOIN research_campaign_experiments AS ce
                      ON ce.experiment_id = e.experiment_id
                    JOIN research_campaigns AS c ON c.campaign_id = ce.campaign_id
                    WHERE fsm.feature_id = ?
                    ORDER BY c.created_at_ms, c.campaign_id
                    """,
                    (feature_id,),
                ).fetchall()
        except ResearchStoreError:
            raise
        except sqlite3.Error as exc:
            raise ResearchStoreError(
                f"cannot list campaigns for feature {feature_id}: {exc}"
            ) from exc
        return tuple(self._campaign_from_row(row) for row in rows)

    def experiment_uses_feature(self, identifier: str, feature_id: str) -> bool:
        try:
            with closing(self._connect()) as connection:
                experiment = connection.execute(
                    "SELECT 1 FROM research_experiments WHERE experiment_id = ?",
                    (identifier,),
                ).fetchone()
                if experiment is None:
                    raise ResearchStoreError(f"unknown experiment: {identifier}")
                row = connection.execute(
                    """
                    SELECT 1
                    FROM research_experiments AS e
                    JOIN research_feature_set_members AS fsm
                      ON fsm.feature_set_id = e.feature_set_id
                    WHERE e.experiment_id = ? AND fsm.feature_id = ?
                    """,
                    (identifier, feature_id),
                ).fetchone()
        except ResearchStoreError:
            raise
        except sqlite3.Error as exc:
            raise ResearchStoreError(
                f"cannot inspect experiment feature membership: {exc}"
            ) from exc
        return row is not None

    def record_promotion_event(self, spec: PromotionEventSpec) -> PromotionRecord:
        identifier = canonical_sha256(spec.model_dump(mode="python"))
        criteria_json = canonical_json_text(spec.criteria_snapshot)
        fingerprint_json = canonical_json_text(spec.code_fingerprint)
        try:
            with self.transaction() as connection:
                experiment = connection.execute(
                    "SELECT 1 FROM research_experiments WHERE experiment_id = ?",
                    (spec.experiment_id,),
                ).fetchone()
                if experiment is None:
                    raise ResearchStoreError(
                        f"promotion references an unknown experiment: {spec.experiment_id}"
                    )
                prior = connection.execute(
                    """
                    SELECT from_stage, to_stage, decision FROM research_promotions
                    WHERE experiment_id = ? ORDER BY created_at_ms, rowid
                    """,
                    (spec.experiment_id,),
                ).fetchall()
                current_stage = ResearchStage.DISCOVERY
                for event in prior:
                    if ResearchStage(str(event["from_stage"])) is not current_stage:
                        raise ResearchStoreError("stored promotion history is inconsistent")
                    if PromotionDecision(str(event["decision"])) is not PromotionDecision.BLOCKED:
                        current_stage = ResearchStage(str(event["to_stage"]))
                if spec.from_stage is not current_stage:
                    existing = connection.execute(
                        "SELECT * FROM research_promotions WHERE promotion_id = ?",
                        (identifier,),
                    ).fetchone()
                    if existing is not None:
                        record = self._promotion_from_row(existing)
                        if record.to_spec() != spec:
                            raise ResearchStoreError(f"immutable promotion conflict: {identifier}")
                        return record
                    raise InvalidResearchTransition(
                        f"promotion event expected {current_stage.value}, "
                        f"got {spec.from_stage.value}"
                    )
                if (
                    spec.decision is not PromotionDecision.BLOCKED
                    and spec.to_stage not in PROMOTION_TRANSITIONS[current_stage]
                ):
                    raise InvalidResearchTransition(
                        f"invalid promotion transition {current_stage.value} -> "
                        f"{spec.to_stage.value}"
                    )
                if (
                    spec.decision is PromotionDecision.REJECTED
                    and spec.to_stage is not ResearchStage.REJECTED
                ) or (
                    spec.decision is PromotionDecision.INVALIDATED
                    and spec.to_stage is not ResearchStage.INVALIDATED
                ):
                    raise InvalidResearchTransition(
                        "promotion decision and destination stage are inconsistent"
                    )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO research_promotions(
                        promotion_id, experiment_id, from_stage, to_stage,
                        decision, criteria_snapshot_json, reason,
                        created_at_ms, code_fingerprint_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identifier,
                        spec.experiment_id,
                        spec.from_stage.value,
                        spec.to_stage.value,
                        spec.decision.value,
                        criteria_json,
                        spec.reason,
                        now_ms(),
                        fingerprint_json,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM research_promotions WHERE promotion_id = ?",
                    (identifier,),
                ).fetchone()
        except (ResearchStoreError, InvalidResearchTransition):
            raise
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot record promotion event: {exc}") from exc
        if row is None:
            raise ResearchStoreError(f"registered promotion disappeared: {identifier}")
        record = self._promotion_from_row(row)
        if record.to_spec() != spec:
            raise ResearchStoreError(f"immutable promotion conflict: {identifier}")
        return record

    def list_promotions(self, experiment_id: str) -> tuple[PromotionRecord, ...]:
        try:
            with closing(self._connect()) as connection:
                exists = connection.execute(
                    "SELECT 1 FROM research_experiments WHERE experiment_id = ?",
                    (experiment_id,),
                ).fetchone()
                if exists is None:
                    raise ResearchStoreError(f"unknown experiment: {experiment_id}")
                rows = connection.execute(
                    """
                    SELECT * FROM research_promotions
                    WHERE experiment_id = ?
                    ORDER BY created_at_ms, rowid
                    """,
                    (experiment_id,),
                ).fetchall()
        except ResearchStoreError:
            raise
        except sqlite3.Error as exc:
            raise ResearchStoreError(f"cannot list promotion events: {exc}") from exc
        return tuple(self._promotion_from_row(row) for row in rows)

    def campaigns_for_experiment(self, experiment_id: str) -> tuple[CampaignRecord, ...]:
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    """
                    SELECT c.*
                    FROM research_campaign_experiments AS ce
                    JOIN research_campaigns AS c ON c.campaign_id = ce.campaign_id
                    WHERE ce.experiment_id = ?
                    ORDER BY c.created_at_ms, c.campaign_id
                    """,
                    (experiment_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise ResearchStoreError(
                f"cannot list campaigns for experiment {experiment_id}: {exc}"
            ) from exc
        return tuple(self._campaign_from_row(row) for row in rows)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self._busy_timeout_ms / 1_000,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @staticmethod
    def _hypothesis_definition(spec: HypothesisSpec) -> dict[str, object]:
        payload = spec.model_dump(mode="python")
        payload.pop("status")
        return payload

    @staticmethod
    def _hypothesis_from_row(row: sqlite3.Row) -> HypothesisSpec:
        return HypothesisSpec(
            hypothesis_id=str(row["hypothesis_id"]),
            title=str(row["title"]),
            mechanism=str(row["mechanism"]),
            expected_direction=(
                str(row["expected_direction"]) if row["expected_direction"] is not None else None
            ),
            expected_horizon=(
                str(row["expected_horizon"]) if row["expected_horizon"] is not None else None
            ),
            target_universe=(
                str(row["target_universe"]) if row["target_universe"] is not None else None
            ),
            preregistered_success_criteria=orjson.loads(
                str(row["preregistered_success_criteria_json"])
            ),
            status=HypothesisStatus(str(row["status"])),
            notes=str(row["notes"]) if row["notes"] is not None else None,
        )

    @staticmethod
    def _feature_manifest_from_row(row: sqlite3.Row) -> dict[str, object]:
        return {
            "feature_id": str(row["feature_id"]),
            "name": str(row["name"]),
            "version": str(row["version"]),
            "description": str(row["description"]),
            "dtype": str(row["dtype"]),
            "lookback": str(row["lookback"]),
            "timestamp_semantics": str(row["timestamp_semantics"]),
            "required_datasets": tuple(orjson.loads(str(row["required_datasets_json"]))),
            "required_columns": tuple(orjson.loads(str(row["required_columns_json"]))),
            "implementation_path": str(row["implementation_path"]),
            "parameters": orjson.loads(str(row["parameters_json"])),
            "status": str(row["status"]),
        }

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> ExperimentRunRecord:
        return ExperimentRunRecord(
            run_id=str(row["run_id"]),
            experiment_id=str(row["experiment_id"]),
            attempt=int(row["attempt"]),
            status=RunStatus(str(row["status"])),
            worker_id=str(row["worker_id"]) if row["worker_id"] is not None else None,
            host_name=str(row["host_name"]) if row["host_name"] is not None else None,
            process_id=int(row["process_id"]) if row["process_id"] is not None else None,
            started_at_ms=(int(row["started_at_ms"]) if row["started_at_ms"] is not None else None),
            heartbeat_at_ms=(
                int(row["heartbeat_at_ms"]) if row["heartbeat_at_ms"] is not None else None
            ),
            finished_at_ms=(
                int(row["finished_at_ms"]) if row["finished_at_ms"] is not None else None
            ),
            runtime_seconds=(
                float(row["runtime_seconds"]) if row["runtime_seconds"] is not None else None
            ),
            result_digest=(str(row["result_digest"]) if row["result_digest"] is not None else None),
            error_type=str(row["error_type"]) if row["error_type"] is not None else None,
            error_message=(str(row["error_message"]) if row["error_message"] is not None else None),
            traceback_path=(
                str(row["traceback_path"]) if row["traceback_path"] is not None else None
            ),
            created_at_ms=int(row["created_at_ms"]),
        )

    @staticmethod
    def _campaign_from_row(row: sqlite3.Row) -> CampaignRecord:
        return CampaignRecord(
            campaign_id=str(row["campaign_id"]),
            name=str(row["name"]),
            description=str(row["description"]),
            hypothesis_id=str(row["hypothesis_id"]),
            spec_json=str(row["spec_json"]),
            spec_sha256=str(row["spec_sha256"]),
            status=CampaignStatus(str(row["status"])),
            created_at_ms=int(row["created_at_ms"]),
            started_at_ms=(int(row["started_at_ms"]) if row["started_at_ms"] is not None else None),
            finished_at_ms=(
                int(row["finished_at_ms"]) if row["finished_at_ms"] is not None else None
            ),
            trial_count=int(row["trial_count"]),
            last_error=str(row["last_error"]) if row["last_error"] is not None else None,
        )

    @staticmethod
    def _feature_evaluation_from_row(row: sqlite3.Row) -> FeatureEvaluationRecord:
        return FeatureEvaluationRecord(
            evaluation_id=str(row["evaluation_id"]),
            run_id=str(row["run_id"]),
            experiment_id=str(row["experiment_id"]),
            hypothesis_id=str(row["hypothesis_id"]),
            feature_id=str(row["feature_id"]),
            evaluation_type=FeatureEvaluationType(str(row["evaluation_type"])),
            scope=str(row["scope"]),
            metric_name=str(row["metric_name"]),
            metric_value=(float(row["metric_value"]) if row["metric_value"] is not None else None),
            decision=FeatureDecision(str(row["decision"])),
            decision_reason=str(row["decision_reason"]),
            context=orjson.loads(str(row["context_json"])),
            created_at_ms=int(row["created_at_ms"]),
        )

    @staticmethod
    def _promotion_from_row(row: sqlite3.Row) -> PromotionRecord:
        return PromotionRecord(
            promotion_id=str(row["promotion_id"]),
            experiment_id=str(row["experiment_id"]),
            from_stage=ResearchStage(str(row["from_stage"])),
            to_stage=ResearchStage(str(row["to_stage"])),
            decision=PromotionDecision(str(row["decision"])),
            criteria_snapshot=orjson.loads(str(row["criteria_snapshot_json"])),
            reason=str(row["reason"]),
            created_at_ms=int(row["created_at_ms"]),
            code_fingerprint=CodeFingerprint.model_validate(
                orjson.loads(str(row["code_fingerprint_json"]))
            ),
        )


__all__ = [
    "CAMPAIGN_TRANSITIONS",
    "HYPOTHESIS_TRANSITIONS",
    "PROMOTION_TRANSITIONS",
    "RUN_TRANSITIONS",
    "CampaignRecord",
    "ExperimentRunRecord",
    "FeatureEvaluationRecord",
    "PromotionRecord",
    "ResearchArtifactRecord",
    "ResearchMetricRecord",
    "ResearchRegistryStatus",
    "ResearchStore",
]
