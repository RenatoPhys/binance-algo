from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pytest

from binance_algo.common.errors import InvalidResearchTransition, ResearchStoreError
from binance_algo.config import load_settings
from binance_algo.research.experiments.canonical import canonical_json_text
from binance_algo.research.experiments.migrations import (
    MIGRATION_1,
    Migration,
    apply_migrations,
)
from binance_algo.research.experiments.models import (
    CodeFingerprint,
    DatasetIdentity,
    ExperimentSpec,
    FeatureSetIdentity,
    HypothesisSpec,
    HypothesisStatus,
    LabelIdentity,
    MetricScope,
    ParameterizedComponent,
    ProvenanceQuality,
    RunStatus,
    VersionedComponent,
)
from binance_algo.research.experiments.registry import sync_builtin_registry
from binance_algo.research.experiments.store import (
    ResearchArtifactRecord,
    ResearchMetricRecord,
    ResearchStore,
)
from binance_algo.research.features.registry import FeatureSetSpec, phase3_feature_set
from binance_algo.research.labels.forward_returns import GROSS_FORWARD_RETURN_1H

PROJECT_ROOT = Path(__file__).parents[2]
BASE_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


def _hypothesis(*, title: str = "Residual momentum persists") -> HypothesisSpec:
    return HypothesisSpec(
        hypothesis_id="HYP-RESMOM-0001",
        title=title,
        mechanism="Cross-sectional residual returns may persist after beta removal.",
        expected_direction="positive",
        expected_horizon="1h",
        target_universe="BTCUSDT,ETHUSDT,SOLUSDT",
        preregistered_success_criteria={"net_sharpe_min": Decimal("0.5")},
        status=HypothesisStatus.DRAFT,
        notes=None,
    )


def _experiment() -> ExperimentSpec:
    settings = load_settings(BASE_CONFIG)
    feature_set = phase3_feature_set(settings.research)
    return ExperimentSpec(
        hypothesis_id="HYP-RESMOM-0001",
        campaign_id=None,
        dataset_reference=DatasetIdentity(
            dataset_id="dataset-v2",
            dataset_schema_version=2,
            feature_set_id=feature_set.feature_set_id,
            label_id=GROSS_FORWARD_RETURN_1H.label_id,
            universe_version="universe-v1",
            start_time_ms=1,
            end_time_ms=2,
            row_count=3,
            content_checksum="c" * 64,
            fingerprint_method="lineage_v2",
        ),
        feature_set=FeatureSetIdentity(
            feature_set_id=feature_set.feature_set_id,
            canonical_checksum=feature_set.canonical_checksum,
        ),
        label=LabelIdentity(
            label_id=GROSS_FORWARD_RETURN_1H.label_id,
            version=GROSS_FORWARD_RETURN_1H.version,
            target_column=GROSS_FORWARD_RETURN_1H.target_column,
        ),
        strategy=VersionedComponent(component_id="residual_momentum", version="v1"),
        strategy_parameters={"weights": [0.2, 0.3, 0.5]},
        portfolio_policy=VersionedComponent(component_id="neutral_long_short", version="v1"),
        portfolio_parameters={"gross_exposure": Decimal("0.5")},
        execution_model=ParameterizedComponent(
            component_id="next_open", version="v1", parameters={"lag_bars": 1}
        ),
        cost_model=ParameterizedComponent(
            component_id="phase3_costs",
            version="v1",
            parameters={"spread_bps": Decimal("1")},
        ),
        split_plan=ParameterizedComponent(
            component_id="walk_forward",
            version="v1",
            parameters={"train_days": 30, "test_days": 14},
        ),
        validation_plan=ParameterizedComponent(
            component_id="phase3_validation",
            version="v1",
            parameters={"embargo_bars": 1},
        ),
        random_seed=42,
        code_fingerprint=CodeFingerprint(
            git_commit="a" * 40,
            git_dirty=False,
            git_diff_sha256=None,
            source_tree_sha256=None,
            provenance_quality=ProvenanceQuality.GIT_CLEAN,
        ),
        artifact_policy="summary",
    )


def _initialized_store(path: Path) -> ResearchStore:
    settings = load_settings(BASE_CONFIG)
    store = ResearchStore(path)
    store.initialize()
    sync_builtin_registry(store, research_config=settings.research)
    store.register_hypothesis(_hypothesis())
    store.register_experiment(_experiment())
    return store


def test_new_database_is_wal_foreign_keyed_and_idempotent(tmp_path: Path) -> None:
    settings = load_settings(BASE_CONFIG)
    store = ResearchStore(tmp_path / "research.sqlite3")
    assert store.initialize() == 3
    assert store.initialize() == 3
    first = sync_builtin_registry(store, research_config=settings.research)
    second = sync_builtin_registry(store, research_config=settings.research)

    status = store.status()
    assert status.schema_version == status.latest_schema_version == 3
    assert status.journal_mode.lower() == "wal"
    assert status.foreign_keys
    assert status.counts["research_feature_definitions"] == first.feature_count
    assert first == second

    feature_set = phase3_feature_set(settings.research)
    reordered = FeatureSetSpec(
        feature_set_id=feature_set.feature_set_id,
        feature_ids=tuple(reversed(feature_set.feature_ids)),
        per_feature_parameters=feature_set.per_feature_parameters,
        version=feature_set.version,
        description=feature_set.description,
    )
    assert store.register_feature_set(reordered) == feature_set.feature_set_id
    assert len(status.counts) == 12


def test_migration_upgrade_and_failure_rollback(tmp_path: Path) -> None:
    path = tmp_path / "research.sqlite3"
    store = ResearchStore(path)
    assert store.initialize(target_version=1) == 1

    bad_migration = Migration(
        version=2,
        name="forced_failure",
        statements=("CREATE TABLE must_rollback(value TEXT)", "INVALID SQL"),
    )
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        with pytest.raises(sqlite3.Error):
            apply_migrations(connection, migrations=(MIGRATION_1, bad_migration))
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='must_rollback'"
            ).fetchone()
            is None
        )
        assert (
            connection.execute("SELECT MAX(version) FROM research_schema_migrations").fetchone()[0]
            == 1
        )
    finally:
        connection.close()

    assert store.initialize() == 3


def test_feature_set_sync_accepts_legacy_manifest_and_declared_ordinals(tmp_path: Path) -> None:
    settings = load_settings(BASE_CONFIG)
    store = ResearchStore(tmp_path / "research.sqlite3")
    store.initialize()
    sync_builtin_registry(store, research_config=settings.research)
    feature_set = phase3_feature_set(settings.research)
    with store.transaction() as connection:
        connection.execute(
            "UPDATE research_feature_sets SET spec_json = ? WHERE feature_set_id = ?",
            (canonical_json_text(feature_set.to_manifest()), feature_set.feature_set_id),
        )
        connection.execute(
            "UPDATE research_feature_set_members SET ordinal = ordinal + 100 "
            "WHERE feature_set_id = ?",
            (feature_set.feature_set_id,),
        )
        for ordinal, feature_id in enumerate(feature_set.feature_ids):
            connection.execute(
                "UPDATE research_feature_set_members SET ordinal = ? "
                "WHERE feature_set_id = ? AND feature_id = ?",
                (ordinal, feature_set.feature_set_id, feature_id),
            )

    synced = sync_builtin_registry(store, research_config=settings.research)
    assert synced.feature_set_id == feature_set.feature_set_id


def test_hypothesis_and_experiment_registration_are_immutable_and_idempotent(
    tmp_path: Path,
) -> None:
    store = _initialized_store(tmp_path / "research.sqlite3")
    identifier = store.register_experiment(_experiment())
    assert store.register_experiment(_experiment()) == identifier
    assert store.register_hypothesis(_hypothesis()) == _hypothesis()
    ready = store.transition_hypothesis("HYP-RESMOM-0001", HypothesisStatus.READY)
    assert store.register_hypothesis(_hypothesis()).status is ready.status
    with pytest.raises(InvalidResearchTransition, match="READY -> SUPPORTED"):
        store.transition_hypothesis("HYP-RESMOM-0001", HypothesisStatus.SUPPORTED)

    with pytest.raises(ResearchStoreError, match="immutable hypothesis conflict"):
        store.register_hypothesis(_hypothesis(title="Conflicting title"))
    with store.transaction() as connection:
        connection.execute(
            "UPDATE research_experiments SET spec_json = '{}' WHERE experiment_id = ?",
            (identifier,),
        )
    with pytest.raises(ResearchStoreError, match="immutable experiment conflict"):
        store.register_experiment(_experiment())


def test_run_state_machine_attempts_success_digest_and_stale_recovery(tmp_path: Path) -> None:
    store = _initialized_store(tmp_path / "research.sqlite3")
    identifier = store.list_experiment_ids()[0]
    first = store.create_run(identifier)
    assert first.attempt == 1 and first.status is RunStatus.PENDING
    store.transition_run(first.run_id, RunStatus.QUEUED)
    store.transition_run(first.run_id, RunStatus.RUNNING)

    with pytest.raises(InvalidResearchTransition, match="without a result digest"):
        store.transition_run(first.run_id, RunStatus.SUCCEEDED)
    store.transition_run(first.run_id, RunStatus.FAILED, error_type="ExpectedFailure")
    with pytest.raises(InvalidResearchTransition, match="FAILED -> QUEUED"):
        store.transition_run(first.run_id, RunStatus.QUEUED)

    second = store.create_run(identifier)
    assert second.attempt == 2 and second.run_id != first.run_id
    store.transition_run(second.run_id, RunStatus.QUEUED)
    store.transition_run(second.run_id, RunStatus.RUNNING)
    store.heartbeat_run(second.run_id, timestamp_ms=10)
    assert store.mark_stale_runs(stale_before_ms=11) == (second.run_id,)
    store.transition_run(second.run_id, RunStatus.QUEUED)
    store.transition_run(second.run_id, RunStatus.RUNNING)
    store.record_metric(
        run_id=second.run_id,
        scope=MetricScope.TEST,
        metric_name="sharpe",
        metric_value=1.0,
    )
    store.record_artifact(
        run_id=second.run_id,
        artifact_type="metrics",
        path="relative/metrics.json",
        checksum_sha256="a" * 64,
        row_count=None,
        size_bytes=10,
        schema_version=1,
    )
    succeeded = store.transition_run(
        second.run_id,
        RunStatus.SUCCEEDED,
        result_digest_value="d" * 64,
    )
    assert succeeded.result_digest == "d" * 64


def test_foreign_keys_metrics_and_artifacts_are_enforced(tmp_path: Path) -> None:
    store = _initialized_store(tmp_path / "research.sqlite3")
    with pytest.raises(ResearchStoreError, match="FOREIGN KEY"):
        store.record_artifact(
            run_id="missing",
            artifact_type="curve",
            path="relative/curve.parquet",
            checksum_sha256="a" * 64,
            row_count=1,
            size_bytes=10,
            schema_version=1,
        )

    run = store.create_run(store.list_experiment_ids()[0])
    store.record_metric(
        run_id=run.run_id,
        scope=MetricScope.TEST,
        metric_name="sharpe",
        metric_value=1.5,
    )
    with pytest.raises(ResearchStoreError, match="immutable metric conflict"):
        store.record_metric(
            run_id=run.run_id,
            scope=MetricScope.TEST,
            metric_name="sharpe",
            metric_value=1.6,
        )
    store.record_metric(
        run_id=run.run_id,
        scope=MetricScope.TEST,
        metric_name="sharpe",
        metric_value=1.5,
    )
    artifact_id = store.record_artifact(
        run_id=run.run_id,
        artifact_type="curve",
        path="relative/curve.parquet",
        checksum_sha256="a" * 64,
        row_count=10,
        size_bytes=100,
        schema_version=1,
    )
    assert (
        store.record_artifact(
            run_id=run.run_id,
            artifact_type="curve",
            path="relative/curve.parquet",
            checksum_sha256="a" * 64,
            row_count=10,
            size_bytes=100,
            schema_version=1,
        )
        == artifact_id
    )
    with pytest.raises(ResearchStoreError, match="immutable artifact conflict"):
        store.record_artifact(
            run_id=run.run_id,
            artifact_type="curve",
            path="relative/curve.parquet",
            checksum_sha256="b" * 64,
            row_count=10,
            size_bytes=100,
            schema_version=1,
        )
    with pytest.raises(ResearchStoreError, match="finite"):
        store.record_metric(
            run_id=run.run_id,
            scope=MetricScope.TEST,
            metric_name="invalid",
            metric_value=float("nan"),
        )


def test_complete_run_rolls_back_all_outputs_on_conflict(tmp_path: Path) -> None:
    store = _initialized_store(tmp_path / "research.sqlite3")
    run = store.create_run(store.list_experiment_ids()[0])
    store.transition_run(run.run_id, RunStatus.QUEUED)
    store.transition_run(run.run_id, RunStatus.RUNNING)
    duplicate = ResearchMetricRecord(
        scope=MetricScope.TEST,
        metric_name="sharpe",
        metric_value=1.0,
    )

    with pytest.raises(ResearchStoreError, match="UNIQUE"):
        store.complete_run(
            run.run_id,
            result_digest_value="d" * 64,
            metrics=(duplicate, duplicate),
            artifacts=(
                ResearchArtifactRecord(
                    artifact_type="metrics",
                    path="relative/metrics.json",
                    checksum_sha256="a" * 64,
                    row_count=None,
                    size_bytes=10,
                    schema_version=1,
                ),
            ),
        )

    current = store.get_run(run.run_id)
    assert current is not None and current.status is RunStatus.RUNNING
    assert store.list_metrics(run.run_id) == ()
    assert store.list_artifacts(run.run_id) == ()


def test_concurrent_run_creation_allocates_distinct_attempts(tmp_path: Path) -> None:
    store = _initialized_store(tmp_path / "research.sqlite3")
    identifier = store.list_experiment_ids()[0]
    with ThreadPoolExecutor(max_workers=2) as executor:
        runs = tuple(executor.map(lambda _: store.create_run(identifier), range(2)))

    assert {run.attempt for run in runs} == {1, 2}
    assert len({run.run_id for run in runs}) == 2
