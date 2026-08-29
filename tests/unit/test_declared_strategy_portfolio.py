from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import orjson
import polars as pl
import pytest
from pydantic import ValidationError

from binance_algo.config import load_settings
from binance_algo.research.costs import explicit_cost_rates
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
    ResearchStage,
    RunStatus,
    VersionedComponent,
)
from binance_algo.research.experiments.registry import sync_builtin_registry
from binance_algo.research.experiments.store import (
    ExperimentRunRecord,
    ResearchArtifactRecord,
    ResearchMetricRecord,
    ResearchStore,
)
from binance_algo.research.features.registry import phase3_feature_set
from binance_algo.research.strategy_portfolio.accounting import build_portfolio_accounting
from binance_algo.research.strategy_portfolio.analytics import (
    correlation_analytics,
    drawdown_analysis,
    monthly_metrics,
)
from binance_algo.research.strategy_portfolio.compatibility import (
    CompatibilityReport,
    assess_compatibility,
)
from binance_algo.research.strategy_portfolio.loader import (
    LoadedStrategyComponent,
    resolve_component_run,
    validate_oos_curve,
)
from binance_algo.research.strategy_portfolio.models import (
    AccountingMode,
    AlignmentPolicy,
    AlignmentSpec,
    PortfolioFile,
    StrategyPortfolioComponent,
    StrategyPortfolioSpec,
    WeightingMode,
    load_portfolio_file,
)
from binance_algo.research.strategy_portfolio.trading import reconstruct_trading

PROJECT_ROOT = Path(__file__).parents[2]
SETTINGS = load_settings(PROJECT_ROOT / "configs" / "base.yaml")
START_MS = 1_767_225_600_000


def _cost_model() -> ParameterizedComponent:
    return ParameterizedComponent(
        component_id="configured_taker",
        version="1",
        parameters={
            "spread_bps": SETTINGS.research.spread_bps,
            "slippage_bps": SETTINGS.research.slippage_bps,
            "initial_capital_usdt": SETTINGS.research.initial_capital_usdt,
            "fee_schedule": SETTINGS.research.fee_schedule.model_dump(mode="json"),
        },
    )


def _spec(strategy_id: str) -> ExperimentSpec:
    feature_set = phase3_feature_set(SETTINGS.research)
    return ExperimentSpec(
        hypothesis_id="HYP-PORTFOLIO-TEST",
        dataset_reference=DatasetIdentity(
            dataset_id="dataset-portfolio-test",
            dataset_schema_version=2,
            feature_set_id=feature_set.feature_set_id,
            label_id="label-test",
            universe_version="universe-test",
            start_time_ms=START_MS,
            end_time_ms=START_MS + 7_200_000,
            row_count=3,
            content_checksum="c" * 64,
            fingerprint_method="test",
        ),
        feature_set=FeatureSetIdentity(
            feature_set_id=feature_set.feature_set_id,
            canonical_checksum=feature_set.canonical_checksum,
        ),
        label=LabelIdentity(label_id="label-test", version="1", target_column="return"),
        strategy=VersionedComponent(component_id=strategy_id, version="1"),
        portfolio_policy=VersionedComponent(component_id="policy", version="1"),
        execution_model=ParameterizedComponent(
            component_id="bar_next_open", version="1", parameters={"lag_bars": 1}
        ),
        cost_model=_cost_model(),
        split_plan=ParameterizedComponent(
            component_id="expanding_walk_forward",
            version="1",
            parameters={"train_days": 30, "test_days": 14, "embargo_bars": 1},
        ),
        validation_plan=ParameterizedComponent(
            component_id="phase3_validation",
            version="1",
            parameters={"profile": "full"},
        ),
        random_seed=42,
        code_fingerprint=CodeFingerprint(
            git_commit="a" * 40,
            git_dirty=False,
            provenance_quality=ProvenanceQuality.GIT_CLEAN,
        ),
    )


def _component(
    experiment_id: str,
    label: str,
    targets: tuple[float, ...],
    price_pnl: tuple[float, ...],
    *,
    folds: tuple[int, ...] | None = None,
) -> LoadedStrategyComponent:
    configured_folds = folds or tuple(1 for _ in targets)
    cost_model = _cost_model()
    rate = explicit_cost_rates(cost_model, START_MS + 1)
    total_rate = rate.fee_rate + rate.half_spread_rate + rate.slippage_rate
    previous = 0.0
    turnover: list[float] = []
    prior_fold: int | None = None
    for index, (fold, target) in enumerate(zip(configured_folds, targets, strict=True)):
        if fold != prior_fold:
            previous = 0.0
            prior_fold = fold
        last = index == len(targets) - 1 or configured_folds[index + 1] != fold
        turnover.append(abs(target - previous) + (abs(target) if last else 0.0))
        previous = target
    fees = [value * rate.fee_rate for value in turnover]
    spread = [value * rate.half_spread_rate for value in turnover]
    slippage = [value * rate.slippage_rate for value in turnover]
    net = [gross - traded * total_rate for gross, traded in zip(price_pnl, turnover, strict=True)]
    times = [START_MS + index * 3_600_000 for index in range(len(targets))]
    curve = pl.DataFrame(
        {
            "fold": configured_folds,
            "decision_time_ms": times,
            "execution_time_ms": [value + 1 for value in times],
            "price_pnl": price_pnl,
            "funding_pnl": [0.0] * len(targets),
            "trading_fees": fees,
            "spread_cost": spread,
            "slippage_cost": slippage,
            "net_return": net,
            "turnover": turnover,
            "gross_exposure": [abs(value) for value in targets],
            "net_exposure": targets,
            "beta_exposure": targets,
            "market_volatility_regime": [0.1] * len(targets),
            "weights_json": [
                orjson.dumps({"BTCUSDT": value}, option=orjson.OPT_SORT_KEYS).decode()
                for value in targets
            ],
        }
    )
    segmented = pl.DataFrame({"key": ["all"], "total_return": [math.prod(1 + x for x in net) - 1]})
    symbol = pl.DataFrame(
        {
            "symbol": ["BTCUSDT"],
            "price_pnl": [sum(price_pnl)],
            "funding_pnl": [0.0],
            "trading_fees": [sum(fees)],
            "spread_cost": [sum(spread)],
            "slippage_cost": [sum(slippage)],
            "net_pnl": [sum(net)],
            "turnover": [sum(turnover)],
        }
    )
    run_id = experiment_id + "-run"
    return LoadedStrategyComponent(
        declaration=StrategyPortfolioComponent(
            experiment_id=experiment_id,
            label=label,
            capital_weight=Decimal(1),
        ),
        spec=_spec(experiment_id),
        run=ExperimentRunRecord(
            run_id=run_id,
            experiment_id=experiment_id,
            attempt=1,
            status=RunStatus.SUCCEEDED,
            worker_id=None,
            host_name=None,
            process_id=None,
            started_at_ms=None,
            heartbeat_at_ms=None,
            finished_at_ms=None,
            runtime_seconds=None,
            result_digest="d" * 64,
            error_type=None,
            error_message=None,
            traceback_path=None,
            created_at_ms=1,
        ),
        artifacts=(),
        artifact_paths={},
        source_checksums={},
        oos_curve=curve,
        monthly_metrics=segmented.rename({"key": "month"}),
        fold_metrics=pl.DataFrame({"fold": [1], "total_return": [sum(net)]}),
        regime_metrics=segmented.rename({"key": "regime"}),
        symbol_metrics=symbol,
        positions=None,
        weights=tuple({"BTCUSDT": value} for value in targets),
        symbols=("BTCUSDT",),
        campaigns=("synthetic",),
        research_stage=ResearchStage.DISCOVERY,
    )


def _compatibility(
    components: tuple[LoadedStrategyComponent, ...],
) -> CompatibilityReport:
    report = assess_compatibility(
        components,
        alignment=AlignmentSpec(),
        accounting_mode=AccountingMode.NETTED,
    )
    assert report.valid, report.issues
    return report


def _write_verified_run(
    store: ResearchStore,
    data_root: Path,
    experiment_id: str,
    *,
    price_pnl: tuple[float, ...],
) -> tuple[ExperimentRunRecord, dict[str, Path]]:
    source = _component("artifact", "Artifact", (0.2, 0.2), price_pnl)
    frames = {
        "oos_curve": source.oos_curve,
        "monthly_metrics": source.monthly_metrics,
        "fold_metrics": source.fold_metrics,
        "regime_metrics": source.regime_metrics,
        "symbol_metrics": source.symbol_metrics,
    }
    run = store.create_run(experiment_id)
    store.transition_run(run.run_id, RunStatus.QUEUED)
    store.transition_run(run.run_id, RunStatus.RUNNING)
    paths: dict[str, Path] = {}
    artifacts = []
    for artifact_type, frame in frames.items():
        relative = Path("gold") / run.run_id / f"{artifact_type}.parquet"
        target = data_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(target)
        paths[artifact_type] = target
        artifacts.append(
            ResearchArtifactRecord(
                artifact_type=artifact_type,
                path=relative.as_posix(),
                checksum_sha256=sha256(target.read_bytes()).hexdigest(),
                row_count=frame.height,
                size_bytes=target.stat().st_size,
                schema_version=1,
            )
        )
    completed = store.complete_run(
        run.run_id,
        result_digest_value=sha256(run.run_id.encode()).hexdigest(),
        metrics=(ResearchMetricRecord(MetricScope.TEST, "total_return", sum(price_pnl)),),
        artifacts=artifacts,
    )
    return completed, paths


def test_portfolio_models_are_strict_and_weights_are_exact(tmp_path: Path) -> None:
    component = {"experiment_id": "one", "label": "One", "capital_weight": "1.0"}
    model = PortfolioFile.model_validate(
        {
            "schema_version": 1,
            "portfolios": [
                {
                    "portfolio_id": "one_only",
                    "title": "One",
                    "description": "Synthetic",
                    "components": [component],
                }
            ],
        }
    )
    assert model.portfolios[0].resolved_weights() == (Decimal("1.0"),)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        StrategyPortfolioSpec.model_validate(
            {
                "portfolio_id": "bad",
                "title": "Bad",
                "description": "Bad",
                "components": [component],
                "optimizer": "max_sharpe",
            }
        )
    with pytest.raises(ValidationError, match="sum exactly to 1"):
        StrategyPortfolioSpec.model_validate(
            {
                "portfolio_id": "bad_sum",
                "title": "Bad",
                "description": "Bad",
                "components": [{**component, "capital_weight": "0.9"}],
            }
        )
    with pytest.raises(ValidationError, match="unique experiment_id"):
        StrategyPortfolioSpec.model_validate(
            {
                "portfolio_id": "duplicates",
                "title": "Bad",
                "description": "Bad",
                "components": [
                    {**component, "capital_weight": "0.5"},
                    {**component, "label": "Duplicate", "capital_weight": "0.5"},
                ],
            }
        )
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        "schema_version: 1\nportfolios:\n  - portfolio_id: equal\n"
        "    title: Equal\n    description: Synthetic\n    weighting: equal_weight\n"
        "    components:\n      - experiment_id: one\n        label: One\n",
        encoding="utf-8",
    )
    assert load_portfolio_file(path).portfolios[0].resolved_weights() == (Decimal(1),)


def test_one_component_and_identical_components_reproduce_source() -> None:
    source = _component("one", "One", (0.5, 0.5), (0.005, -0.002))
    single = build_portfolio_accounting((source,), (Decimal(1),), _compatibility((source,)))
    assert single.netted_curve["net_return"].to_list() == pytest.approx(
        source.oos_curve["net_return"].to_list(), abs=1.0e-15
    )
    assert single.netted_curve["turnover"].to_list() == pytest.approx(
        source.oos_curve["turnover"].to_list(), abs=1.0e-15
    )

    duplicate = _component("two", "Two", (0.5, 0.5), (0.005, -0.002))
    combined = build_portfolio_accounting(
        (source, duplicate),
        (Decimal("0.5"), Decimal("0.5")),
        _compatibility((source, duplicate)),
    )
    assert combined.netted_curve["net_return"].to_list() == pytest.approx(
        source.oos_curve["net_return"].to_list(), abs=1.0e-15
    )


def test_opposite_sleeves_net_exposure_turnover_and_costs() -> None:
    long = _component("long", "Long", (0.5, 0.5), (0.005, 0.005))
    short = _component("short", "Short", (-0.5, -0.5), (-0.005, -0.005))
    report = _compatibility((long, short))
    accounting = build_portfolio_accounting(
        (long, short),
        (Decimal("0.5"), Decimal("0.5")),
        report,
    )
    assert accounting.netted_curve["gross_exposure"].to_list() == [0.0, 0.0]
    assert accounting.netted_curve["turnover"].to_list() == [0.0, 0.0]
    assert accounting.netted_curve["net_return"].to_list() == [0.0, 0.0]
    assert float(accounting.netted_curve["netting_savings"].sum()) > 0
    assert accounting.sleeve_curve["net_return"].sum() < 0
    trading = reconstruct_trading(accounting, (long, short), report, epsilon=1.0e-10)
    assert trading["summary"]["trade_legs"] == 0
    assert trading["summary"]["unnetted_simulated_traded_weight"] > 0


def test_fold_close_is_reconstructed_separately_and_loader_reconciles() -> None:
    source = _component(
        "folds",
        "Folds",
        (0.5, 0.5, -0.5, -0.5),
        (0.001, 0.001, -0.001, -0.001),
        folds=(1, 1, 2, 2),
    )
    validate_oos_curve(source.oos_curve)
    report = _compatibility((source,))
    accounting = build_portfolio_accounting((source,), (Decimal(1),), report)
    trading = reconstruct_trading(accounting, (source,), report, epsilon=1.0e-10)
    assert trading["summary"]["forced_fold_closes"] == 2
    assert trading["summary"]["rebalance_events"] == 4
    broken = source.oos_curve.with_columns((pl.col("net_return") + 0.01).alias("net_return"))
    with pytest.raises(Exception, match="does not reconcile"):
        validate_oos_curve(broken)


def test_equal_weight_rejects_manual_weights_and_negative_weights() -> None:
    with pytest.raises(ValidationError, match="must not declare"):
        StrategyPortfolioSpec(
            portfolio_id="equal",
            title="Equal",
            description="Synthetic",
            weighting=WeightingMode.EQUAL_WEIGHT,
            components=(
                StrategyPortfolioComponent(
                    experiment_id="one",
                    label="One",
                    capital_weight=Decimal(1),
                ),
            ),
        )
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        StrategyPortfolioComponent(
            experiment_id="one",
            label="One",
            capital_weight=Decimal("-0.1"),
        )


def test_strict_rejects_grid_and_static_identity_mismatches() -> None:
    left = _component("left", "Left", (0.2, 0.2), (0.001, -0.001))
    right = _component("right", "Right", (0.2, 0.2), (0.001, -0.001))

    shifted = replace(
        right,
        oos_curve=right.oos_curve.with_columns(
            (pl.col("decision_time_ms") + 1).alias("decision_time_ms"),
            (pl.col("execution_time_ms") + 1).alias("execution_time_ms"),
        ),
    )
    assert not assess_compatibility(
        (left, shifted), alignment=AlignmentSpec(), accounting_mode=AccountingMode.NETTED
    ).valid

    changed_dataset = replace(
        right,
        spec=right.spec.model_copy(
            update={
                "dataset_reference": right.spec.dataset_reference.model_copy(
                    update={"dataset_id": "different-dataset"}
                )
            }
        ),
    )
    dataset_report = assess_compatibility(
        (left, changed_dataset),
        alignment=AlignmentSpec(),
        accounting_mode=AccountingMode.NETTED,
    )
    assert "dataset_reference differs between components" in dataset_report.issues

    changed_cost = replace(
        right,
        spec=right.spec.model_copy(
            update={
                "cost_model": right.spec.cost_model.model_copy(
                    update={"parameters": {**right.spec.cost_model.parameters, "spread_bps": 9}}
                )
            }
        ),
    )
    assert (
        "cost_model differs between components"
        in assess_compatibility(
            (left, changed_cost),
            alignment=AlignmentSpec(),
            accounting_mode=AccountingMode.NETTED,
        ).issues
    )


def test_intersection_is_explicit_and_reports_discarded_coverage() -> None:
    left = _component("left", "Left", (0.2, 0.2, 0.2), (0.001, 0.002, 0.003))
    right = _component("right", "Right", (0.2, 0.2, 0.2), (0.001, 0.002, 0.003))
    changed_grid = replace(
        right,
        oos_curve=right.oos_curve.with_columns(
            pl.Series(
                "decision_time_ms",
                (START_MS, START_MS + 3_600_000, START_MS + 10_800_000),
            ),
            pl.Series(
                "execution_time_ms",
                (START_MS + 1, START_MS + 3_600_001, START_MS + 10_800_001),
            ),
        ),
    )
    strict = assess_compatibility(
        (left, changed_grid),
        alignment=AlignmentSpec(),
        accounting_mode=AccountingMode.NETTED,
    )
    assert not strict.valid
    intersection = assess_compatibility(
        (left, changed_grid),
        alignment=AlignmentSpec(policy=AlignmentPolicy.INTERSECTION),
        accounting_mode=AccountingMode.NETTED,
    )
    assert intersection.valid
    assert intersection.decision_times == (START_MS, START_MS + 3_600_000)
    assert [item.coverage for item in intersection.coverage] == pytest.approx([2 / 3, 2 / 3])
    assert all(item.discarded_periods == 1 for item in intersection.coverage)
    assert any("exploratory" in item for item in intersection.warnings)


def test_trade_event_taxonomy_and_epsilon() -> None:
    source = _component(
        "events",
        "Events",
        (0.0, 0.2, 0.4, 0.1, -0.1, 0.0),
        (0.0, 0.001, 0.001, 0.001, -0.001, 0.0),
    )
    report = _compatibility((source,))
    accounting = build_portfolio_accounting((source,), (Decimal(1),), report)
    trading = reconstruct_trading(accounting, (source,), report, epsilon=1.0e-10)
    assert trading["summary"]["entries"] == 1
    assert trading["summary"]["increases"] == 1
    assert trading["summary"]["reductions"] == 1
    assert trading["summary"]["flips"] == 1
    assert trading["summary"]["exits"] == 1
    assert trading["summary"]["rebalance_events"] == 5
    assert trading["summary"]["trade_legs"] == 5

    tiny = _component("tiny", "Tiny", (1.0e-12, 0.0), (0.0, 0.0))
    tiny_report = _compatibility((tiny,))
    tiny_accounting = build_portfolio_accounting((tiny,), (Decimal(1),), tiny_report)
    ignored = reconstruct_trading(
        tiny_accounting,
        (tiny,),
        tiny_report,
        epsilon=1.0e-10,
    )
    assert ignored["summary"]["trade_legs"] == 0


def test_multiple_symbols_share_one_rebalance_event_but_have_distinct_legs() -> None:
    source = _component("multi", "Multi", (0.0, 0.0), (0.0, 0.0))
    rates = explicit_cost_rates(source.spec.cost_model, START_MS + 1)
    turnover = 0.3
    price_pnl = 0.001
    fees = turnover * rates.fee_rate
    spread = turnover * rates.half_spread_rate
    slippage = turnover * rates.slippage_rate
    targets = {"BTCUSDT": 0.1, "ETHUSDT": 0.2}
    curve = source.oos_curve.with_columns(
        pl.lit(price_pnl).alias("price_pnl"),
        pl.lit(fees).alias("trading_fees"),
        pl.lit(spread).alias("spread_cost"),
        pl.lit(slippage).alias("slippage_cost"),
        pl.lit(price_pnl - fees - spread - slippage).alias("net_return"),
        pl.lit(turnover).alias("turnover"),
        pl.lit(0.3).alias("gross_exposure"),
        pl.lit(0.3).alias("net_exposure"),
        pl.lit(0.3).alias("beta_exposure"),
        pl.lit(orjson.dumps(targets, option=orjson.OPT_SORT_KEYS).decode()).alias("weights_json"),
    )
    multi = replace(
        source,
        oos_curve=curve,
        weights=(targets, targets),
        symbols=("BTCUSDT", "ETHUSDT"),
    )
    report = _compatibility((multi,))
    accounting = build_portfolio_accounting((multi,), (Decimal(1),), report)
    trading = reconstruct_trading(accounting, (multi,), report, epsilon=1.0e-10)
    assert trading["summary"]["rebalance_events"] == 2
    assert trading["summary"]["trade_legs"] == 4
    assert trading["summary"]["entries"] == 2
    assert trading["summary"]["forced_fold_closes"] == 2


def test_fee_schedule_boundary_is_applied_by_execution_date() -> None:
    start = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1_000)
    rates = explicit_cost_rates(_cost_model(), start)
    assert rates.fee_rate == pytest.approx(float(SETTINGS.research.fee_schedule.taker_fee_rate))
    with pytest.raises(Exception, match="does not cover"):
        explicit_cost_rates(_cost_model(), start - 1)


def test_drawdown_monthly_compounding_and_return_correlation() -> None:
    times = (START_MS, START_MS + 86_400_000, START_MS + 2 * 86_400_000, START_MS + 3 * 86_400_000)

    def daily_component(
        identifier: str,
        label: str,
        returns: tuple[float, ...],
    ) -> LoadedStrategyComponent:
        component = _component(identifier, label, (0.0,) * 4, returns)
        return replace(
            component,
            oos_curve=component.oos_curve.with_columns(
                pl.Series("decision_time_ms", times),
                pl.Series("execution_time_ms", tuple(value + 1 for value in times)),
            ),
        )

    positive = daily_component("positive", "Positive", (0.01, -0.01, 0.02, -0.02))
    negative = daily_component("negative", "Negative", (-0.01, 0.01, -0.02, 0.02))
    flat = daily_component("flat", "Flat", (0.0, 0.0, 0.0, 0.0))
    components = (positive, negative, flat)
    report = _compatibility(components)
    accounting = build_portfolio_accounting(
        components,
        (Decimal("0.4"), Decimal("0.4"), Decimal("0.2")),
        report,
    )
    correlations = correlation_analytics(
        components,
        accounting,
        report,
        epsilon=1.0e-10,
    )
    daily = correlations["daily_net"]["values"]
    assert daily[0][1] == pytest.approx(-1.0)
    assert daily[0][2] is None
    assert correlations["effective_independent_strategies"] >= 1.0
    assert any("singular" in item for item in correlations["warnings"])

    drawdown = drawdown_analysis(
        pl.DataFrame(
            {
                "decision_time_ms": times[:3],
                "net_return": (0.10, -0.10, 0.20),
            }
        )
    )
    assert drawdown["episodes"][0]["recovered"] is True
    unrecovered = drawdown_analysis(
        pl.DataFrame({"decision_time_ms": times[:2], "net_return": (0.10, -0.20)})
    )
    assert unrecovered["unrecovered"] is True

    segmented = monthly_metrics(
        pl.DataFrame(
            {
                "decision_time_ms": (1_767_225_600_000, 1_767_312_000_000),
                "net_return": (0.10, -0.10),
            }
        )
    )
    assert segmented[0]["total_return"] == pytest.approx(-0.01)


def test_run_resolution_uses_latest_verified_success_and_rejects_bad_runs(
    tmp_path: Path,
) -> None:
    store = ResearchStore(tmp_path / "research.sqlite3")
    store.initialize()
    sync_builtin_registry(store, research_config=SETTINGS.research)
    store.register_hypothesis(
        HypothesisSpec(
            hypothesis_id="HYP-PORTFOLIO-TEST",
            title="Portfolio loader fixture",
            mechanism="Artifact resolution",
            preregistered_success_criteria={"total_return": 0.0},
            status=HypothesisStatus.READY,
        )
    )
    experiment_id = store.register_experiment(_spec("loader-one"))
    first, _ = _write_verified_run(
        store,
        tmp_path / "data",
        experiment_id,
        price_pnl=(0.001, -0.001),
    )
    latest, latest_paths = _write_verified_run(
        store,
        tmp_path / "data",
        experiment_id,
        price_pnl=(0.002, -0.001),
    )
    latest_paths["oos_curve"].write_bytes(latest_paths["oos_curve"].read_bytes() + b"corrupt")

    declaration = StrategyPortfolioComponent(
        experiment_id=experiment_id,
        label="Loader",
        capital_weight=Decimal(1),
    )
    _, resolved, _ = resolve_component_run(
        store=store,
        data_root=tmp_path / "data",
        declaration=declaration,
    )
    assert resolved.run_id == first.run_id

    with pytest.raises(Exception, match="invalid artifacts"):
        resolve_component_run(
            store=store,
            data_root=tmp_path / "data",
            declaration=declaration.model_copy(update={"run_id": latest.run_id}),
        )

    pending = store.create_run(experiment_id)
    with pytest.raises(Exception, match="not SUCCEEDED"):
        resolve_component_run(
            store=store,
            data_root=tmp_path / "data",
            declaration=declaration.model_copy(update={"run_id": pending.run_id}),
        )

    other_experiment = store.register_experiment(_spec("loader-two"))
    other_run = store.create_run(other_experiment)
    with pytest.raises(Exception, match="belongs to experiment"):
        resolve_component_run(
            store=store,
            data_root=tmp_path / "data",
            declaration=declaration.model_copy(update={"run_id": other_run.run_id}),
        )
