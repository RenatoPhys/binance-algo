from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import orjson

from binance_algo.config import load_settings
from binance_algo.research.dashboard import build_research_dashboard
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
from binance_algo.research.features.registry import phase3_feature_set
from binance_algo.research.labels.forward_returns import GROSS_FORWARD_RETURN_1H

PROJECT_ROOT = Path(__file__).parents[2]
BASE_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


def _experiment(hypothesis_id: str) -> ExperimentSpec:
    settings = load_settings(BASE_CONFIG)
    feature_set = phase3_feature_set(settings.research)
    return ExperimentSpec(
        hypothesis_id=hypothesis_id,
        campaign_id=None,
        dataset_reference=DatasetIdentity(
            dataset_id="dataset-dashboard-v1",
            dataset_schema_version=2,
            feature_set_id=feature_set.feature_set_id,
            label_id=GROSS_FORWARD_RETURN_1H.label_id,
            universe_version="universe-dashboard-v1",
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


def test_dashboard_empty_store_is_deterministic(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path / "research.sqlite3")
    reports_root = tmp_path / "reports"
    data_root = tmp_path / "data"

    first = build_research_dashboard(
        store=store,
        reports_root=reports_root,
        data_root=data_root,
    )
    snapshot_bytes = first.snapshot_path.read_bytes()
    html_bytes = first.index_path.read_bytes()
    second = build_research_dashboard(
        store=store,
        reports_root=reports_root,
        data_root=data_root,
    )

    assert second.snapshot_path.read_bytes() == snapshot_bytes
    assert second.index_path.read_bytes() == html_bytes
    assert orjson.loads(snapshot_bytes)["totals"] == {
        "campaigns": 0,
        "experiments": 0,
        "failures": 0,
        "hypotheses": 0,
        "successes": 0,
    }
    assert b"Nenhuma campanha registrada" in html_bytes
    assert b"Nenhum experimento registrado" in html_bytes


def test_dashboard_preserves_failures_negative_results_links_and_escaping(
    tmp_path: Path,
) -> None:
    settings = load_settings(BASE_CONFIG)
    store = ResearchStore(tmp_path / "research.sqlite3")
    store.initialize()
    sync_builtin_registry(store, research_config=settings.research)
    hypothesis = HypothesisSpec(
        hypothesis_id="HYP-DASHBOARD-0001",
        title="Unsafe <title>",
        mechanism="Dashboard coverage",
        preregistered_success_criteria={"sharpe": 0.0},
        status=HypothesisStatus.READY,
    )
    store.register_hypothesis(hypothesis)
    identifier = store.register_experiment(_experiment(hypothesis.hypothesis_id))
    campaign_name = '<img src=x onerror="alert(1)">'
    campaign = store.register_campaign(
        identifier="b" * 64,
        name=campaign_name,
        description="External <description>",
        hypothesis_id=hypothesis.hypothesis_id,
        spec_payload={"source_spec": {"campaign": {"name": campaign_name}}},
        trial_count=1,
    )
    store.associate_campaign_experiment(
        campaign_id=campaign.campaign_id,
        experiment_id_value=identifier,
        ordinal=0,
    )

    succeeded = store.create_run(identifier)
    store.transition_run(succeeded.run_id, RunStatus.QUEUED)
    store.transition_run(succeeded.run_id, RunStatus.RUNNING)
    report_path = Path("gold/research/report.md")
    store.complete_run(
        succeeded.run_id,
        result_digest_value="d" * 64,
        metrics=(
            ResearchMetricRecord(MetricScope.TEST, "total_return", -0.125),
            ResearchMetricRecord(MetricScope.TEST, "sharpe", -1.25),
            ResearchMetricRecord(MetricScope.TEST, "max_drawdown", -0.25),
            ResearchMetricRecord(MetricScope.TEST, "turnover", 12.0),
            ResearchMetricRecord(MetricScope.TEST, "total_return", -0.2, fold=1),
            ResearchMetricRecord(
                MetricScope.STRESS,
                "total_return",
                -0.15,
                regime="cost_1_5x",
            ),
            ResearchMetricRecord(
                MetricScope.STRESS,
                "total_return",
                -0.18,
                regime="signal_delay_1_bar",
            ),
        ),
        artifacts=(
            ResearchArtifactRecord(
                artifact_type="report",
                path=report_path.as_posix(),
                checksum_sha256="e" * 64,
                row_count=None,
                size_bytes=10,
                schema_version=1,
            ),
        ),
    )
    failed = store.create_run(identifier)
    store.transition_run(failed.run_id, RunStatus.QUEUED)
    store.transition_run(failed.run_id, RunStatus.RUNNING)
    store.transition_run(
        failed.run_id,
        RunStatus.FAILED,
        error_type="UnsafeError",
        error_message="<script>alert('failed')</script>",
    )

    result = build_research_dashboard(
        store=store,
        reports_root=tmp_path / "reports",
        data_root=tmp_path / "data",
    )
    snapshot = orjson.loads(result.snapshot_path.read_bytes())
    experiment = snapshot["experiments"][0]
    html_text = result.index_path.read_text(encoding="utf-8")

    assert snapshot["totals"]["successes"] == 1
    assert snapshot["totals"]["failures"] == 1
    assert experiment["latest_status"] == "FAILED"
    assert experiment["total_return"] == -0.125
    assert experiment["worst_fold_return"] == -0.2
    assert experiment["cost_1_5x_return"] == -0.15
    assert experiment["signal_delay_1_bar_return"] == -0.18
    assert experiment["research_stage"] == "DISCOVERY"
    assert experiment["report_href"].endswith("data/gold/research/report.md")
    assert campaign_name not in html_text
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in html_text
    assert "<script>alert('failed')</script>" not in html_text
    assert "&lt;script&gt;alert(&#x27;failed&#x27;)&lt;/script&gt;" in html_text
    assert "-12.5000%" in html_text
