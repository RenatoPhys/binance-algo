from __future__ import annotations

import math
from pathlib import Path

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

PROJECT_ROOT = Path(__file__).parents[2]
BASE_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
START_MS = 1_767_225_600_000


def _research_frame(days: int = 11) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for hour in range(days * 24):
        decision = START_MS + hour * 3_600_000 + 3_599_999
        common_volatility = 0.012 + 0.002 * math.sin(hour / 24)
        for symbol_index, symbol in enumerate(SYMBOLS):
            phase = math.sin(hour / 9 + symbol_index)
            residual_1h = 0.002 * phase
            rows.append(
                {
                    "decision_time_ms": decision,
                    "execution_time_ms": decision + 1,
                    "label_end_time_ms": decision + 3_600_001,
                    "symbol": symbol,
                    "residual_momentum_1h": residual_1h,
                    "residual_momentum_4h": residual_1h * 2 + symbol_index * 0.0001,
                    "residual_momentum_24h": residual_1h * 4 - symbol_index * 0.0001,
                    "realized_volatility_24h": common_volatility * (1 + symbol_index * 0.1),
                    "rolling_beta": 0.8 + symbol_index * 0.25,
                    "future_return_1h": 0.0015 * phase - 0.0002 * symbol_index,
                    "future_residual_return_1h": 0.0012 * phase,
                    "outcome_funding_rate_1h": (
                        0.0001 * (symbol_index + 1) if hour % 8 == 7 else 0.0
                    ),
                    "outcome_quote_volume_1h": 100_000_000.0,
                    "market_volatility_regime": common_volatility * math.sqrt(365),
                }
            )
    return pl.DataFrame(rows)


def _runner(
    tmp_path: Path,
    *,
    artifact_policy: ArtifactPolicy = ArtifactPolicy.SUMMARY,
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
    frame = _research_frame()
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
