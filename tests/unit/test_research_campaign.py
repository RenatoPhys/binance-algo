from __future__ import annotations

import tempfile
from pathlib import Path

import orjson
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from binance_algo.common.errors import ResearchError
from binance_algo.config import load_settings
from binance_algo.research.experiments.campaign import CampaignSpec, plan_campaign
from binance_algo.research.experiments.campaign_runner import CampaignRunner
from binance_algo.research.experiments.models import (
    CampaignStatus,
    CodeFingerprint,
    HypothesisSpec,
    HypothesisStatus,
    ProvenanceQuality,
)
from binance_algo.research.experiments.registry import sync_builtin_registry
from binance_algo.research.experiments.store import ResearchStore

from ..research_fixtures import research_frame

PROJECT_ROOT = Path(__file__).parents[2]
BASE_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


def _fingerprint() -> CodeFingerprint:
    return CodeFingerprint(
        git_commit="b" * 40,
        git_dirty=False,
        git_diff_sha256=None,
        source_tree_sha256=None,
        provenance_quality=ProvenanceQuality.GIT_CLEAN,
    )


def _dataset(data_root: Path, *, version: str = "synthetic") -> Path:
    directory = data_root / "gold" / "binance" / "usdm" / "research_dataset" / f"version={version}"
    directory.mkdir(parents=True, exist_ok=True)
    frame = research_frame()
    frame.write_parquet(directory / "dataset.parquet")
    manifest = {
        "dataset_id": "synthetic-campaign-v1",
        "dataset_schema_version": 2,
        "feature_set_id": "phase3_baseline_features:v1",
        "label_id": "gross_forward_return_1h:v1",
        "universe_version": "fixed-three-v1",
        "start_time_ms": int(frame["decision_time_ms"].min()),
        "end_time_ms": int(frame["decision_time_ms"].max()),
        "row_count": frame.height,
        "content_checksum": "d" * 64,
        "fingerprint_method": "lineage_v2",
    }
    path = directory / "dataset.json"
    path.write_bytes(orjson.dumps(manifest, option=orjson.OPT_SORT_KEYS))
    return path


def _campaign(manifest: Path, *, max_trials: int = 3, max_workers: int = 1) -> CampaignSpec:
    return CampaignSpec.model_validate(
        {
            "campaign": {
                "name": "synthetic_campaign",
                "description": "Deterministic campaign expansion test.",
                "hypothesis_id": "HYP-CAMPAIGN-0001",
                "artifact_policy": "summary",
                "max_trials": max_trials,
            },
            "dataset": {"manifest": str(manifest)},
            "feature_set": {"name": "phase3_baseline_features", "version": "1"},
            "label": {
                "name": "gross_forward_return_1h",
                "version": "1",
                "horizon_minutes": 60,
            },
            "strategy": {
                "name": "residual_momentum",
                "version": "1",
                "fixed": {"momentum_weight_4h": 0.3},
                "grid": {
                    "momentum_weight_1h": [0.1, 0.2, 0.3],
                    "momentum_weight_24h": [0.4, 0.5, 0.6],
                },
            },
            "portfolio": {
                "name": "neutral_long_short",
                "version": "1",
                "fixed": {
                    "no_trade_score_band": 0.1,
                    "gross_exposure": 0.5,
                    "annual_volatility_target": 0.15,
                    "max_symbol_weight": 0.25,
                },
            },
            "execution": {"name": "bar_next_open", "version": "1"},
            "costs": {
                "name": "configured_taker",
                "version": "1",
                "fixed": {"cost_multiplier": 1.0},
            },
            "validation": {
                "split_plan": "expanding_walk_forward_v1",
                "train_days": 7,
                "test_days": 1,
                "bootstrap_samples": 100,
                "bootstrap_block_hours": 24,
                "stress_cost_multipliers": [1.5, 2.0],
                "stress_signal_delay_bars": [1],
                "require_parameter_sum": {
                    "fields": [
                        "momentum_weight_1h",
                        "momentum_weight_4h",
                        "momentum_weight_24h",
                    ],
                    "equals": 1.0,
                },
            },
            "runner": {"max_workers": max_workers, "fail_fast": False, "resume": True},
        }
    )


def _hypothesis() -> HypothesisSpec:
    return HypothesisSpec(
        hypothesis_id="HYP-CAMPAIGN-0001",
        title="Synthetic campaign",
        mechanism="Test deterministic campaign infrastructure.",
        expected_direction="not applicable",
        expected_horizon="1h",
        target_universe="synthetic fixed three",
        preregistered_success_criteria={"purpose": "infrastructure"},
        status=HypothesisStatus.READY,
    )


def _plan(tmp_path: Path, *, max_trials: int = 3, max_workers: int = 1):
    settings = load_settings(BASE_CONFIG)
    config = settings.research.model_copy(update={"block_bootstrap_samples": 100})
    data_root = tmp_path / "data"
    source = _campaign(
        _dataset(data_root),
        max_trials=max_trials,
        max_workers=max_workers,
    )
    plan = plan_campaign(
        source,
        project_root=PROJECT_ROOT,
        data_root=data_root,
        research_config=config,
        code_fingerprint=_fingerprint(),
    )
    return settings, config, data_root, source, plan


def test_campaign_expansion_is_deterministic_constrained_and_path_independent(
    tmp_path: Path,
) -> None:
    _, config, data_root, source, first = _plan(tmp_path)
    second_manifest = _dataset(data_root, version="relocated")
    relocated = source.model_copy(
        update={"dataset": source.dataset.model_copy(update={"manifest": str(second_manifest)})}
    )
    second = plan_campaign(
        relocated,
        project_root=PROJECT_ROOT,
        data_root=data_root,
        research_config=config,
        code_fingerprint=_fingerprint(),
    )

    assert first.possible_combinations == 9
    assert first.valid_combinations == 3
    assert first.rejected_by_constraints == 6
    assert first.campaign_id == second.campaign_id
    assert [trial.experiment_id for trial in first.trials] == sorted(
        trial.experiment_id for trial in first.trials
    )
    assert [trial.experiment_id for trial in first.trials] == [
        trial.experiment_id for trial in second.trials
    ]

    limited = source.model_copy(
        update={"campaign": source.campaign.model_copy(update={"max_trials": 2})}
    )
    with pytest.raises(ResearchError, match="above max_trials=2"):
        plan_campaign(
            limited,
            project_root=PROJECT_ROOT,
            data_root=data_root,
            research_config=config,
            code_fingerprint=_fingerprint(),
        )


@given(grid_order=st.permutations(("momentum_weight_1h", "momentum_weight_24h")))
def test_campaign_identity_is_invariant_to_grid_key_order(
    grid_order: list[str],
) -> None:
    with tempfile.TemporaryDirectory() as directory:
        _, config, data_root, source, reference = _plan(Path(directory))
        ordered_grid = {key: source.strategy.grid[key] for key in grid_order}
        reordered = source.model_copy(
            update={"strategy": source.strategy.model_copy(update={"grid": ordered_grid})}
        )

        candidate = plan_campaign(
            reordered,
            project_root=PROJECT_ROOT,
            data_root=data_root,
            research_config=config,
            code_fingerprint=_fingerprint(),
        )

        assert candidate.campaign_id == reference.campaign_id
        assert [trial.experiment_id for trial in candidate.trials] == [
            trial.experiment_id for trial in reference.trials
        ]


def test_campaign_run_uses_processes_then_rerun_is_all_cache_hits(tmp_path: Path) -> None:
    settings, config, data_root, _, plan = _plan(tmp_path, max_workers=2)
    store = ResearchStore(tmp_path / "research.sqlite3")
    store.initialize()
    sync_builtin_registry(store, research_config=config)
    store.register_hypothesis(_hypothesis())
    runner = CampaignRunner(
        store=store,
        data_root=data_root,
        reports_root=tmp_path / "reports",
        research_config=config,
        compression=settings.storage.parquet_compression,
    )

    first = runner.run(plan)
    assert first.campaign.status is CampaignStatus.COMPLETED
    assert first.executed_count == first.succeeded_count == 3
    assert first.cache_hit_count == first.failed_count == 0
    assert pl.read_parquet(first.comparison.comparison_path).height == 3
    attempts = {
        identifier: len(store.list_runs(experiment_id_value=identifier))
        for _, identifier, _ in store.list_campaign_experiments(plan.campaign_id)
    }

    second = runner.run(plan)
    assert second.campaign.status is CampaignStatus.COMPLETED
    assert second.cache_hit_count == 3
    assert second.executed_count == second.failed_count == 0
    assert attempts == {
        identifier: len(store.list_runs(experiment_id_value=identifier))
        for _, identifier, _ in store.list_campaign_experiments(plan.campaign_id)
    }


def test_partial_campaign_resumes_without_rerunning_successes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import binance_algo.research.experiments.campaign_runner as runner_module

    settings, config, data_root, _, plan = _plan(tmp_path)
    store = ResearchStore(tmp_path / "research.sqlite3")
    store.initialize()
    sync_builtin_registry(store, research_config=config)
    store.register_hypothesis(_hypothesis())
    runner = CampaignRunner(
        store=store,
        data_root=data_root,
        reports_root=tmp_path / "reports",
        research_config=config,
        compression=settings.storage.parquet_compression,
    )
    original = runner_module._execute_trial_worker
    failed_identifier = plan.trials[0].experiment_id

    def fail_one(request):
        if request.experiment_id == failed_identifier:
            raise ResearchError("isolated synthetic failure")
        return original(request)

    monkeypatch.setattr(runner_module, "_execute_trial_worker", fail_one)
    partial = runner.run(plan)
    assert partial.campaign.status is CampaignStatus.PARTIAL
    assert partial.failed_count == 1
    assert partial.succeeded_count == 2

    monkeypatch.setattr(runner_module, "_execute_trial_worker", original)
    resumed = runner.run(plan)
    assert resumed.campaign.status is CampaignStatus.COMPLETED
    assert resumed.cache_hit_count == 2
    assert resumed.executed_count == resumed.succeeded_count - resumed.cache_hit_count == 1
