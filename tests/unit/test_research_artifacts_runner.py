from __future__ import annotations

from pathlib import Path
from threading import Event
from time import sleep

import orjson
import polars as pl
import pytest

from binance_algo.common.errors import ResearchError
from binance_algo.config import load_settings
from binance_algo.research.datasets.references import load_dataset_reference
from binance_algo.research.experiments.models import (
    ArtifactPolicy,
    CodeFingerprint,
    ProvenanceQuality,
    RunStatus,
)
from binance_algo.research.experiments.registry import sync_builtin_registry
from binance_algo.research.experiments.runner import (
    ExperimentRunner,
    build_phase3_experiment_spec,
    phase3_baseline_hypothesis,
)
from binance_algo.research.experiments.store import ResearchStore
from binance_algo.research.portfolio.registry import build_portfolio_policy
from binance_algo.research.strategies.registry import build_strategy

from ..research_fixtures import SYMBOLS, research_frame

PROJECT_ROOT = Path(__file__).parents[2]
BASE_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


def _runner(
    tmp_path: Path,
    *,
    artifact_policy: ArtifactPolicy = ArtifactPolicy.SUMMARY,
    heartbeat_seconds: float = 30,
) -> tuple[ExperimentRunner, ResearchStore, str]:
    settings = load_settings(BASE_CONFIG)
    config = settings.research.model_copy(
        update={
            "walk_forward_train_days": 7,
            "walk_forward_test_days": 1,
            "block_bootstrap_samples": 100,
        }
    )
    data_root = tmp_path / "data"
    dataset_directory = (
        data_root / "gold" / "binance" / "usdm" / "research_dataset" / "version=synthetic"
    )
    dataset_directory.mkdir(parents=True)
    frame = research_frame()
    frame.write_parquet(dataset_directory / "dataset.parquet")
    store = ResearchStore(tmp_path / "research.sqlite3")
    store.initialize()
    synced_feature_set = sync_builtin_registry(store, research_config=config).feature_set_id
    manifest = {
        "dataset_id": "synthetic-runner-v1",
        "dataset_schema_version": 2,
        "feature_set_id": synced_feature_set,
        "label_id": "gross_forward_return_1h:v1",
        "universe_version": "fixed-three-v1",
        "start_time_ms": int(frame["decision_time_ms"].min()),
        "end_time_ms": int(frame["decision_time_ms"].max()),
        "row_count": frame.height,
        "content_checksum": "c" * 64,
        "fingerprint_method": "lineage_v2",
    }
    (dataset_directory / "dataset.json").write_bytes(
        orjson.dumps(manifest, option=orjson.OPT_SORT_KEYS)
    )
    reference = load_dataset_reference(dataset_directory / "dataset.json")
    store.register_hypothesis(phase3_baseline_hypothesis())
    spec = build_phase3_experiment_spec(
        dataset_reference=reference,
        config=config,
        project_root=PROJECT_ROOT,
        artifact_policy=artifact_policy,
        code_fingerprint=CodeFingerprint(
            git_commit="a" * 40,
            git_dirty=False,
            git_diff_sha256=None,
            source_tree_sha256=None,
            provenance_quality=ProvenanceQuality.GIT_CLEAN,
        ),
    )
    identifier = store.register_experiment(spec)
    runner = ExperimentRunner(
        store=store,
        data_root=data_root,
        research_config=config,
        compression="zstd",
        heartbeat_seconds=heartbeat_seconds,
    )
    return runner, store, identifier


def test_summary_pipeline_is_atomic_and_rerun_digest_is_deterministic(tmp_path: Path) -> None:
    runner, store, identifier = _runner(tmp_path)

    first = runner.run(identifier)
    first_names = {path.name for path in first.artifact_directory.iterdir()}
    assert first.run.status is RunStatus.SUCCEEDED
    assert first.verification.valid
    assert first.deterministic_with_previous is None
    assert "positions.parquet" not in first_names
    assert "scores.parquet" not in first_names
    assert "pnl.svg" not in first_names
    assert {
        "manifest.json",
        "experiment_spec.json",
        "metrics.json",
        "report.md",
        "oos_curve.parquet",
        "fold_metrics.parquet",
        "regime_metrics.parquet",
        "monthly_metrics.parquet",
        "symbol_metrics.parquet",
    }.issubset(first_names)

    second = runner.run(identifier, generate_chart=True)
    assert second.run.attempt == 2
    assert second.deterministic_with_previous is True
    assert second.run.result_digest == first.run.result_digest
    assert (second.artifact_directory / "pnl.svg").is_file()
    assert runner.verify_experiment(identifier).valid
    assert len(store.list_runs(experiment_id_value=identifier)) == 2


def test_full_policy_persists_long_form_scores_and_positions(tmp_path: Path) -> None:
    runner, _, identifier = _runner(tmp_path, artifact_policy=ArtifactPolicy.FULL)

    result = runner.run(identifier)
    scores = pl.read_parquet(result.artifact_directory / "scores.parquet")
    positions = pl.read_parquet(result.artifact_directory / "positions.parquet")
    curve = pl.read_parquet(result.artifact_directory / "oos_curve.parquet")

    assert scores.columns == [
        "fold",
        "decision_time_ms",
        "symbol",
        "score",
        "score_rank",
    ]
    assert positions.columns == [
        "fold",
        "decision_time_ms",
        "execution_time_ms",
        "symbol",
        "previous_weight",
        "target_weight",
        "trade_weight",
        "gross_contribution",
        "net_contribution",
        "beta_contribution",
        "future_return",
        "funding_rate",
        "price_pnl",
        "funding_pnl",
        "allocated_fee",
        "allocated_spread_cost",
        "allocated_slippage_cost",
    ]
    assert scores.height == positions.height == curve.height * len(SYMBOLS)
    assert positions["allocated_fee"].sum() == pytest.approx(curve["trading_fees"].sum())
    assert positions["net_contribution"].sum() == pytest.approx(curve["net_return"].sum())


def test_runner_heartbeat_stops_after_success_and_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import binance_algo.research.experiments.runner as runner_module

    original_validation = runner_module.run_research_validation

    success_runner, success_store, success_identifier = _runner(
        tmp_path / "success",
        heartbeat_seconds=0.01,
    )
    success_beat = Event()
    success_calls: list[str] = []
    original_success_heartbeat = success_store.heartbeat_run

    def record_success_heartbeat(run_id: str, *, timestamp_ms: int | None = None) -> None:
        original_success_heartbeat(run_id, timestamp_ms=timestamp_ms)
        success_calls.append(run_id)
        success_beat.set()

    def wait_then_validate(*args: object, **kwargs: object):
        assert success_beat.wait(timeout=1)
        return original_validation(*args, **kwargs)

    monkeypatch.setattr(success_store, "heartbeat_run", record_success_heartbeat)
    monkeypatch.setattr(runner_module, "run_research_validation", wait_then_validate)
    success_runner.run(success_identifier)
    calls_after_success = len(success_calls)
    sleep(0.04)
    assert calls_after_success >= 1
    assert len(success_calls) == calls_after_success

    failure_runner, failure_store, failure_identifier = _runner(
        tmp_path / "failure",
        heartbeat_seconds=0.01,
    )
    failure_beat = Event()
    failure_calls: list[str] = []
    original_failure_heartbeat = failure_store.heartbeat_run

    def record_failure_heartbeat(run_id: str, *, timestamp_ms: int | None = None) -> None:
        original_failure_heartbeat(run_id, timestamp_ms=timestamp_ms)
        failure_calls.append(run_id)
        failure_beat.set()

    def wait_then_fail(*_: object, **__: object) -> None:
        assert failure_beat.wait(timeout=1)
        raise ResearchError("simulated validation failure")

    monkeypatch.setattr(failure_store, "heartbeat_run", record_failure_heartbeat)
    monkeypatch.setattr(runner_module, "run_research_validation", wait_then_fail)
    with pytest.raises(ResearchError, match="simulated validation failure"):
        failure_runner.run(failure_identifier)
    calls_after_failure = len(failure_calls)
    sleep(0.04)
    assert calls_after_failure >= 1
    assert len(failure_calls) == calls_after_failure
    assert (
        failure_store.list_runs(experiment_id_value=failure_identifier)[0].status
        is RunStatus.FAILED
    )


def test_corruption_is_detected_and_failed_bundle_is_quarantined(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, store, identifier = _runner(tmp_path)
    succeeded = runner.run(identifier)
    curve_path = succeeded.artifact_directory / "oos_curve.parquet"
    curve_path.write_bytes(b"corrupt")

    verification = runner.verify_experiment(identifier)
    assert not verification.valid
    assert any("checksum mismatch" in issue for issue in verification.issues)

    other_runner, other_store, other_identifier = _runner(tmp_path / "failure")

    def fail_validation(*_: object) -> None:
        raise ResearchError("forced artifact validation failure")

    monkeypatch.setattr(other_runner.pipeline, "_validate_temp", fail_validation)
    with pytest.raises(ResearchError, match="forced artifact validation failure"):
        other_runner.run(other_identifier)
    failed = other_store.list_runs(experiment_id_value=other_identifier)[0]
    assert failed.status is RunStatus.FAILED
    assert not other_store.list_artifacts(failed.run_id)
    quarantine = tmp_path / "failure" / "data" / "quarantine" / "research"
    assert any(path.name.startswith(f"{failed.run_id}-failed") for path in quarantine.iterdir())
    assert other_store is not store


def test_component_factories_are_explicit_and_reject_unknown_parameters() -> None:
    with pytest.raises(ResearchError, match="unsupported strategy"):
        build_strategy("arbitrary.import.path", "1", {})
    with pytest.raises(ResearchError, match="extra_forbidden"):
        build_strategy(
            "residual_momentum",
            "1",
            {
                "momentum_weight_1h": 0.2,
                "momentum_weight_4h": 0.3,
                "momentum_weight_24h": 0.5,
                "hidden_tuning": True,
            },
        )
    with pytest.raises(ResearchError, match="extra_forbidden"):
        build_portfolio_policy(
            "neutral_long_short",
            "1",
            {
                "no_trade_score_band": 0.1,
                "gross_exposure": 0.5,
                "annual_volatility_target": 0.15,
                "max_symbol_weight": 0.25,
                "leverage": 10,
            },
        )
