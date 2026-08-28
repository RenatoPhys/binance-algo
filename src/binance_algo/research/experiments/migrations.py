"""Versioned, transactional migrations for the dedicated research registry."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

from binance_algo.common.errors import ResearchStoreError


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    statements: tuple[str, ...]


MIGRATION_1 = Migration(
    version=1,
    name="research_registry_core",
    statements=(
        """
        CREATE TABLE research_hypotheses (
            hypothesis_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            mechanism TEXT NOT NULL,
            expected_direction TEXT,
            expected_horizon TEXT,
            target_universe TEXT,
            preregistered_success_criteria_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN (
                'DRAFT', 'READY', 'TESTED', 'SUPPORTED', 'REJECTED',
                'INCONCLUSIVE', 'INVALIDATED'
            )),
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            notes TEXT
        )
        """,
        """
        CREATE TABLE research_campaigns (
            campaign_id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL,
            hypothesis_id TEXT NOT NULL,
            spec_json TEXT NOT NULL,
            spec_sha256 TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN (
                'PLANNED', 'QUEUED', 'RUNNING', 'COMPLETED',
                'PARTIAL', 'FAILED', 'CANCELLED'
            )),
            created_at_ms INTEGER NOT NULL,
            started_at_ms INTEGER,
            finished_at_ms INTEGER,
            trial_count INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            FOREIGN KEY(hypothesis_id) REFERENCES research_hypotheses(hypothesis_id)
        )
        """,
        """
        CREATE TABLE research_feature_definitions (
            feature_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            description TEXT NOT NULL,
            dtype TEXT NOT NULL,
            lookback TEXT NOT NULL,
            timestamp_semantics TEXT NOT NULL,
            required_datasets_json TEXT NOT NULL,
            required_columns_json TEXT NOT NULL,
            implementation_path TEXT NOT NULL,
            parameters_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('ACTIVE', 'DEPRECATED', 'SUPERSEDED')),
            created_at_ms INTEGER NOT NULL,
            deprecated_at_ms INTEGER,
            UNIQUE(name, version)
        )
        """,
        """
        CREATE TABLE research_feature_sets (
            feature_set_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            version TEXT NOT NULL,
            description TEXT NOT NULL,
            spec_json TEXT NOT NULL,
            spec_sha256 TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            UNIQUE(name, version)
        )
        """,
        """
        CREATE TABLE research_feature_set_members (
            feature_set_id TEXT NOT NULL,
            feature_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            parameters_json TEXT NOT NULL,
            PRIMARY KEY(feature_set_id, feature_id),
            UNIQUE(feature_set_id, ordinal),
            FOREIGN KEY(feature_set_id) REFERENCES research_feature_sets(feature_set_id),
            FOREIGN KEY(feature_id) REFERENCES research_feature_definitions(feature_id)
        )
        """,
        """
        CREATE TABLE research_experiments (
            experiment_id TEXT PRIMARY KEY,
            experiment_sha256 TEXT NOT NULL UNIQUE,
            hypothesis_id TEXT NOT NULL,
            spec_json TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            universe_version TEXT NOT NULL,
            feature_set_id TEXT NOT NULL,
            label_id TEXT NOT NULL,
            strategy_id TEXT NOT NULL,
            strategy_version TEXT NOT NULL,
            portfolio_policy_id TEXT NOT NULL,
            execution_model_id TEXT NOT NULL,
            cost_model_id TEXT NOT NULL,
            split_plan_id TEXT NOT NULL,
            random_seed INTEGER NOT NULL,
            code_fingerprint_json TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            FOREIGN KEY(hypothesis_id) REFERENCES research_hypotheses(hypothesis_id),
            FOREIGN KEY(feature_set_id) REFERENCES research_feature_sets(feature_set_id)
        )
        """,
        """
        CREATE TABLE research_campaign_experiments (
            campaign_id TEXT NOT NULL,
            experiment_id TEXT NOT NULL,
            ordinal INTEGER NOT NULL,
            tags_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY(campaign_id, experiment_id),
            UNIQUE(campaign_id, ordinal),
            FOREIGN KEY(campaign_id) REFERENCES research_campaigns(campaign_id),
            FOREIGN KEY(experiment_id) REFERENCES research_experiments(experiment_id)
        )
        """,
        """
        CREATE TABLE research_experiment_runs (
            run_id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL,
            attempt INTEGER NOT NULL CHECK (attempt >= 1),
            status TEXT NOT NULL CHECK (status IN (
                'PENDING', 'QUEUED', 'RUNNING', 'SUCCEEDED',
                'FAILED', 'CANCELLED', 'STALE'
            )),
            worker_id TEXT,
            host_name TEXT,
            process_id INTEGER,
            started_at_ms INTEGER,
            heartbeat_at_ms INTEGER,
            finished_at_ms INTEGER,
            runtime_seconds REAL,
            result_digest TEXT,
            error_type TEXT,
            error_message TEXT,
            traceback_path TEXT,
            created_at_ms INTEGER NOT NULL,
            UNIQUE(experiment_id, attempt),
            FOREIGN KEY(experiment_id) REFERENCES research_experiments(experiment_id)
        )
        """,
        """
        CREATE TABLE research_metrics (
            run_id TEXT NOT NULL,
            scope TEXT NOT NULL CHECK (scope IN (
                'TRAIN', 'INNER_VALIDATION', 'TEST', 'LOCKBOX',
                'STRESS', 'CAMPAIGN'
            )),
            fold INTEGER,
            regime TEXT,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY(run_id, scope, fold, regime, metric_name),
            FOREIGN KEY(run_id) REFERENCES research_experiment_runs(run_id)
        )
        """,
        """
        CREATE TABLE research_artifacts (
            artifact_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            artifact_type TEXT NOT NULL,
            path TEXT NOT NULL,
            checksum_sha256 TEXT NOT NULL,
            row_count INTEGER,
            size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
            schema_version INTEGER NOT NULL CHECK (schema_version >= 1),
            created_at_ms INTEGER NOT NULL,
            UNIQUE(run_id, artifact_type, path),
            FOREIGN KEY(run_id) REFERENCES research_experiment_runs(run_id)
        )
        """,
        """
        CREATE TABLE research_feature_evaluations (
            evaluation_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            feature_id TEXT NOT NULL,
            evaluation_type TEXT NOT NULL CHECK (evaluation_type IN (
                'UNIVARIATE', 'INCREMENTAL', 'ABLATION', 'PERMUTATION',
                'REGIME_STABILITY', 'TURNOVER_IMPACT', 'COST_SENSITIVITY'
            )),
            scope TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL,
            decision TEXT NOT NULL CHECK (decision IN (
                'SUPPORTED', 'CONDITIONAL', 'REJECTED', 'INCONCLUSIVE',
                'INVALIDATED', 'RETEST_REQUIRED'
            )),
            decision_reason TEXT NOT NULL,
            context_json TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            FOREIGN KEY(run_id) REFERENCES research_experiment_runs(run_id),
            FOREIGN KEY(feature_id) REFERENCES research_feature_definitions(feature_id)
        )
        """,
        """
        CREATE TABLE research_promotions (
            promotion_id TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL,
            from_stage TEXT NOT NULL,
            to_stage TEXT NOT NULL,
            decision TEXT NOT NULL,
            criteria_snapshot_json TEXT NOT NULL,
            reason TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            code_fingerprint_json TEXT NOT NULL,
            FOREIGN KEY(experiment_id) REFERENCES research_experiments(experiment_id)
        )
        """,
    ),
)

MIGRATION_2 = Migration(
    version=2,
    name="research_registry_indexes",
    statements=(
        """
        CREATE UNIQUE INDEX idx_research_metrics_identity
        ON research_metrics(
            run_id, scope, COALESCE(fold, -1), COALESCE(regime, ''), metric_name
        )
        """,
        """
        CREATE INDEX idx_research_runs_status
        ON research_experiment_runs(status, heartbeat_at_ms)
        """,
        """
        CREATE INDEX idx_research_experiments_hypothesis
        ON research_experiments(hypothesis_id, created_at_ms)
        """,
        """
        CREATE INDEX idx_research_artifacts_run
        ON research_artifacts(run_id, artifact_type)
        """,
    ),
)

MIGRATION_3 = Migration(
    version=3,
    name="research_ledger_indexes",
    statements=(
        """
        CREATE INDEX idx_research_feature_evaluations_feature
        ON research_feature_evaluations(feature_id, created_at_ms, evaluation_id)
        """,
        """
        CREATE INDEX idx_research_feature_evaluations_run
        ON research_feature_evaluations(run_id, created_at_ms, evaluation_id)
        """,
        """
        CREATE INDEX idx_research_campaign_experiments_experiment
        ON research_campaign_experiments(experiment_id, campaign_id)
        """,
    ),
)

MIGRATIONS = (MIGRATION_1, MIGRATION_2, MIGRATION_3)
LATEST_SCHEMA_VERSION = MIGRATIONS[-1].version


def apply_migrations(
    connection: sqlite3.Connection,
    *,
    migrations: tuple[Migration, ...] = MIGRATIONS,
    target_version: int | None = None,
) -> int:
    """Apply missing migrations in one immediate transaction and roll back on any failure."""

    ordered = tuple(sorted(migrations, key=lambda migration: migration.version))
    if len({migration.version for migration in ordered}) != len(ordered):
        raise ResearchStoreError("research migrations contain duplicate versions")
    latest = ordered[-1].version if ordered else 0
    target = latest if target_version is None else target_version
    if target < 0 or target > latest:
        raise ResearchStoreError(f"invalid research migration target: {target}")
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS research_schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at_ms INTEGER NOT NULL
            )
            """
        )
        applied = {
            int(row[0])
            for row in connection.execute("SELECT version FROM research_schema_migrations")
        }
        if any(version > latest for version in applied):
            raise ResearchStoreError("research database schema is newer than this application")
        for migration in ordered:
            if migration.version > target or migration.version in applied:
                continue
            for statement in migration.statements:
                connection.execute(statement)
            connection.execute(
                """
                INSERT INTO research_schema_migrations(version, name, applied_at_ms)
                VALUES (?, ?, ?)
                """,
                (migration.version, migration.name, time.time_ns() // 1_000_000),
            )
        connection.commit()
    except BaseException:
        connection.rollback()
        raise
    row = connection.execute("SELECT MAX(version) FROM research_schema_migrations").fetchone()
    return int(row[0]) if row is not None and row[0] is not None else 0


__all__ = [
    "LATEST_SCHEMA_VERSION",
    "MIGRATIONS",
    "Migration",
    "apply_migrations",
]
