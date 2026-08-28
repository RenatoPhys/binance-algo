from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from binance_algo.common.errors import ResearchError, ResearchStoreError
from binance_algo.research.experiments.ablation import (
    AblationChange,
    AblationDeclaration,
    calculate_incremental_metrics,
)
from binance_algo.research.experiments.ledger import write_feature_history_report
from binance_algo.research.experiments.models import (
    FeatureDecision,
    FeatureEvaluationSpec,
    FeatureEvaluationType,
    MetricScope,
    RunStatus,
)
from binance_algo.research.experiments.store import ResearchStore

from .test_research_store import _initialized_store

FEATURE_ID = "residual_momentum_1h:v1"


def _succeeded_run(store: ResearchStore):
    run = store.create_run(store.list_experiment_ids()[0])
    store.transition_run(run.run_id, RunStatus.QUEUED)
    store.transition_run(run.run_id, RunStatus.RUNNING)
    store.record_metric(
        run_id=run.run_id,
        scope=MetricScope.TEST,
        metric_name="sharpe",
        metric_value=1.0,
    )
    store.record_artifact(
        run_id=run.run_id,
        artifact_type="metrics",
        path=f"relative/{run.run_id}/metrics.json",
        checksum_sha256="a" * 64,
        row_count=None,
        size_bytes=10,
        schema_version=1,
    )
    return store.transition_run(
        run.run_id,
        RunStatus.SUCCEEDED,
        result_digest_value="d" * 64,
    )


def _evaluation(
    run_id: str,
    *,
    decision: FeatureDecision,
    context_name: str,
) -> FeatureEvaluationSpec:
    return FeatureEvaluationSpec(
        run_id=run_id,
        feature_id=FEATURE_ID,
        evaluation_type=FeatureEvaluationType.ABLATION,
        scope=MetricScope.TEST.value,
        metric_name="delta_sharpe",
        metric_value=0.1 if decision is FeatureDecision.SUPPORTED else -0.2,
        decision=decision,
        decision_reason=f"contextual {decision.value.lower()} result",
        context={"context_name": context_name, "rule": {"min_delta_sharpe": 0.0}},
    )


def test_feature_ledger_preserves_distinct_contexts_and_negative_results(
    tmp_path: Path,
) -> None:
    store = _initialized_store(tmp_path / "research.sqlite3")
    first = _succeeded_run(store)
    second = _succeeded_run(store)
    supported = _evaluation(
        first.run_id,
        decision=FeatureDecision.SUPPORTED,
        context_name="slow_rebalance",
    )
    rejected = _evaluation(
        second.run_id,
        decision=FeatureDecision.REJECTED,
        context_name="fast_rebalance",
    )

    first_record = store.record_feature_evaluation(supported)
    assert store.record_feature_evaluation(supported).evaluation_id == first_record.evaluation_id
    store.record_feature_evaluation(rejected)

    history = store.list_feature_evaluations(feature_id=FEATURE_ID)
    assert [record.decision for record in history] == [
        FeatureDecision.SUPPORTED,
        FeatureDecision.REJECTED,
    ]
    assert {record.context["context_name"] for record in history} == {
        "slow_rebalance",
        "fast_rebalance",
    }
    feature = store.get_feature_definition(FEATURE_ID)
    assert feature is not None and feature["status"] == "ACTIVE"

    report = write_feature_history_report(
        store=store,
        feature_id=FEATURE_ID,
        reports_root=tmp_path / "reports",
    )
    assert report.evaluation_count == 2
    assert report.report_json_path.is_file()
    assert "REJECTED" in report.report_markdown_path.read_text(encoding="utf-8")


def test_feature_evaluation_rejects_invalid_run_feature_and_reason(tmp_path: Path) -> None:
    store = _initialized_store(tmp_path / "research.sqlite3")
    succeeded = _succeeded_run(store)
    missing_run = _evaluation(
        "missing",
        decision=FeatureDecision.INCONCLUSIVE,
        context_name="missing_run",
    )
    with pytest.raises(ResearchStoreError, match="unknown run"):
        store.record_feature_evaluation(missing_run)

    unknown_feature = _evaluation(
        succeeded.run_id,
        decision=FeatureDecision.INCONCLUSIVE,
        context_name="missing_feature",
    ).model_copy(update={"feature_id": "missing:v1"})
    with pytest.raises(ResearchStoreError, match="unknown feature"):
        store.record_feature_evaluation(unknown_feature)

    pending = store.create_run(store.list_experiment_ids()[0])
    with pytest.raises(ResearchStoreError, match="succeeded"):
        store.record_feature_evaluation(
            _evaluation(
                pending.run_id,
                decision=FeatureDecision.INCONCLUSIVE,
                context_name="pending",
            )
        )

    with pytest.raises(ValidationError, match="decision_reason"):
        FeatureEvaluationSpec(
            run_id=succeeded.run_id,
            feature_id=FEATURE_ID,
            evaluation_type=FeatureEvaluationType.ABLATION,
            scope="TEST",
            metric_name="delta_sharpe",
            metric_value=0.0,
            decision=FeatureDecision.INCONCLUSIVE,
            decision_reason="",
            context={},
        )


def test_incremental_metrics_have_stable_with_minus_without_orientation() -> None:
    without = {
        "total_return": 0.10,
        "sharpe": 1.0,
        "max_drawdown": -0.20,
        "mean_cross_sectional_rank_ic": 0.01,
        "turnover": 2.0,
        "trading_fees": 0.01,
        "spread_cost": 0.02,
        "slippage_cost": 0.03,
        "maximum_volume_participation": 0.08,
    }
    with_feature = {
        "total_return": 0.13,
        "sharpe": 1.4,
        "max_drawdown": -0.15,
        "mean_cross_sectional_rank_ic": 0.015,
        "turnover": 2.5,
        "trading_fees": 0.02,
        "spread_cost": 0.03,
        "slippage_cost": 0.04,
        "maximum_volume_participation": 0.05,
    }

    metrics = calculate_incremental_metrics(
        with_feature=with_feature,
        without_feature=without,
        with_feature_month_concentration=0.4,
        without_feature_month_concentration=0.3,
    )

    assert metrics.delta_total_return == pytest.approx(0.03)
    assert metrics.delta_sharpe == pytest.approx(0.4)
    assert metrics.delta_max_drawdown == pytest.approx(0.05)
    assert metrics.delta_rank_ic == pytest.approx(0.005)
    assert metrics.delta_turnover == pytest.approx(0.5)
    assert metrics.delta_explicit_cost == pytest.approx(0.03)
    assert metrics.delta_capacity == pytest.approx(0.03)
    assert metrics.delta_month_concentration == pytest.approx(0.1)

    with pytest.raises(ResearchError, match="not finite"):
        calculate_incremental_metrics(
            with_feature=with_feature,
            without_feature=without,
            with_feature_month_concentration=float("nan"),
            without_feature_month_concentration=0.3,
        )


def test_ablation_override_requires_an_explicit_reason() -> None:
    payload = {
        "feature_id": FEATURE_ID,
        "change": AblationChange.REMOVED,
        "baseline": {"strategy_parameters": {"momentum_weight_1h": 0.2}},
        "candidate": {"strategy_parameters": {"momentum_weight_1h": 0.0}},
    }
    with pytest.raises(ValidationError, match="declared together"):
        AblationDeclaration.model_validate(
            {**payload, "decision_override": FeatureDecision.RETEST_REQUIRED}
        )
