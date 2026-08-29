"""Pure deterministic dashboard snapshots for declared strategy portfolios."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.experiments.models import MetricScope
from binance_algo.research.experiments.store import ResearchStore
from binance_algo.research.strategy_portfolio.accounting import build_portfolio_accounting
from binance_algo.research.strategy_portfolio.analytics import analyze_portfolio
from binance_algo.research.strategy_portfolio.compatibility import assess_compatibility
from binance_algo.research.strategy_portfolio.loader import (
    LoadedStrategyComponent,
    load_strategy_component,
)
from binance_algo.research.strategy_portfolio.models import (
    AccountingMode,
    StrategyPortfolioSpec,
    load_portfolio_file,
)


def _component_stress(store: ResearchStore, run_id: str) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for metric in store.list_metrics(run_id):
        if metric.scope is MetricScope.STRESS and metric.regime is not None:
            output.setdefault(metric.regime, {})[metric.metric_name] = metric.metric_value
    return {
        scenario: {name: values[name] for name in sorted(values)}
        for scenario, values in sorted(output.items())
    }


def _source_runs(components: tuple[LoadedStrategyComponent, ...]) -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": component.run.experiment_id,
            "run_id": component.run.run_id,
            "result_digest": component.run.result_digest,
            "artifact_checksums": {
                name: component.source_checksums[name]
                for name in sorted(component.source_checksums)
            },
        }
        for component in components
    ]


def _invalid_snapshot(
    portfolio: StrategyPortfolioSpec | None,
    *,
    error: str,
    components: tuple[LoadedStrategyComponent, ...] = (),
) -> dict[str, Any]:
    return {
        "portfolio_id": portfolio.portfolio_id if portfolio is not None else "portfolio_file",
        "title": portfolio.title if portfolio is not None else "Invalid portfolio file",
        "description": portfolio.description if portfolio is not None else "",
        "status": "INVALID",
        "accounting_mode": (portfolio.accounting_mode.value if portfolio is not None else None),
        "warnings": [],
        "errors": [error],
        "source_runs": _source_runs(components),
        "alignment": {},
        "components": [],
        "metrics": {},
        "sleeve_metrics": {},
        "netted_metrics": {},
        "correlations": {},
        "position_similarity": {},
        "drawdown_episodes": [],
        "trading": {},
        "symbol_attribution": [],
        "fold_metrics": [],
        "regime_metrics": [],
        "monthly_metrics": [],
        "chart_series": {},
    }


def _valid_snapshot(
    *,
    store: ResearchStore,
    portfolio: StrategyPortfolioSpec,
    components: tuple[LoadedStrategyComponent, ...],
) -> dict[str, Any]:
    compatibility = assess_compatibility(
        components,
        alignment=portfolio.alignment,
        accounting_mode=portfolio.accounting_mode,
    )
    if not compatibility.valid:
        return _invalid_snapshot(
            portfolio,
            error="; ".join(compatibility.issues),
            components=components,
        )
    accounting = build_portfolio_accounting(
        components,
        portfolio.resolved_weights(),
        compatibility,
    )
    analytics = analyze_portfolio(
        components,
        accounting,
        compatibility,
        trade_epsilon=portfolio.analytics.trade_epsilon,
        rolling_windows=portfolio.analytics.rolling_windows_hours,
    )
    foreground = (
        analytics["netted_metrics"]
        if portfolio.accounting_mode is AccountingMode.NETTED
        else analytics["sleeve_metrics"]
    )
    netting_savings = float(accounting.netted_curve["netting_savings"].sum())
    foreground = {
        **foreground,
        "netting_savings": netting_savings,
        "average_offset_ratio": float(
            accounting.netted_curve.select(pl.col("offset_ratio").mean()).item()
        ),
        "turnover_reduction_fraction": analytics["trading"]["summary"][
            "turnover_reduction_fraction"
        ],
        "effective_independent_strategies": analytics["correlations"][
            "effective_independent_strategies"
        ],
        "rebalance_events": analytics["trading"]["summary"]["rebalance_events"],
        "trade_legs": analytics["trading"]["summary"]["trade_legs"],
    }
    component_rows = []
    for component, weight in zip(
        components,
        portfolio.resolved_weights(),
        strict=True,
    ):
        profile = component.spec.validation_plan.parameters.get("profile")
        component_rows.append(
            {
                "experiment_id": component.run.experiment_id,
                "run_id": component.run.run_id,
                "label": component.declaration.label,
                "capital_weight": float(weight),
                "strategy_id": component.spec.strategy.component_id,
                "strategy_version": component.spec.strategy.version,
                "portfolio_policy_id": component.spec.portfolio_policy.component_id,
                "portfolio_policy_version": component.spec.portfolio_policy.version,
                "validation_profile": profile if isinstance(profile, str) else "legacy",
                "research_stage": component.research_stage.value,
                "campaigns": list(component.campaigns),
                "dataset_id": component.spec.dataset_reference.dataset_id,
                "label_id": component.spec.label.label_id,
                "code_fingerprint": component.spec.code_fingerprint.model_dump(mode="json"),
                "artifact_verified": True,
                "positions_available": component.positions is not None,
                "stress": _component_stress(store, component.run.run_id),
            }
        )
    alignment = {
        "policy": compatibility.policy.value,
        "compatibility_group": compatibility.compatibility_group,
        "issues": list(compatibility.issues),
        "warnings": list(compatibility.warnings),
        "periods": len(compatibility.decision_times),
        "start_time_ms": compatibility.decision_times[0],
        "end_time_ms": compatibility.decision_times[-1],
        "coverage": [
            {
                "experiment_id": item.experiment_id,
                "original_periods": item.original_periods,
                "aligned_periods": item.aligned_periods,
                "discarded_periods": item.discarded_periods,
                "coverage": item.coverage,
            }
            for item in compatibility.coverage
        ],
    }
    warnings = [
        "Research only. No strategy is promoted.",
        "The evaluated final period is not an independent lockbox.",
        "Portfolio visualization is not independent validation.",
        "Simulated trade legs are not real exchange fills.",
        *compatibility.warnings,
    ]
    return {
        "portfolio_id": portfolio.portfolio_id,
        "title": portfolio.title,
        "description": portfolio.description,
        "status": "VALID",
        "accounting_mode": portfolio.accounting_mode.value,
        "weighting": portfolio.weighting.value,
        "warnings": list(dict.fromkeys(warnings)),
        "errors": [],
        "source_runs": _source_runs(components),
        "alignment": alignment,
        "components": component_rows,
        "metrics": foreground,
        "sleeve_metrics": analytics["sleeve_metrics"],
        "netted_metrics": analytics["netted_metrics"],
        "correlations": analytics["correlations"],
        "position_similarity": analytics["position_similarity"],
        "drawdown": analytics["drawdown"],
        "drawdown_episodes": analytics["drawdown"]["episodes"],
        "trading": analytics["trading"],
        "component_attribution": analytics["component_attribution"],
        "symbol_attribution": analytics["symbol_attribution"],
        "concentration": analytics["concentration"],
        "fold_metrics": analytics["fold_metrics"],
        "regime_metrics": analytics["regime_metrics"],
        "monthly_metrics": analytics["monthly_metrics"],
        "chart_series": analytics["chart_series"],
    }


def build_portfolio_snapshots(
    *,
    store: ResearchStore,
    data_root: Path,
    portfolio_file: Path,
) -> list[dict[str, Any]]:
    try:
        declaration_file = load_portfolio_file(portfolio_file)
    except ResearchError as exc:
        return [_invalid_snapshot(None, error=str(exc))]
    snapshots: list[dict[str, Any]] = []
    for portfolio in declaration_file.portfolios:
        components: list[LoadedStrategyComponent] = []
        error: str | None = None
        for declaration in portfolio.components:
            try:
                components.append(
                    load_strategy_component(
                        store=store,
                        data_root=data_root,
                        declaration=declaration,
                    )
                )
            except Exception as exc:
                error = f"{declaration.label}: {exc}"
                break
        if error is not None:
            snapshots.append(
                _invalid_snapshot(portfolio, error=error, components=tuple(components))
            )
            continue
        snapshots.append(
            _valid_snapshot(
                store=store,
                portfolio=portfolio,
                components=tuple(components),
            )
        )
    return snapshots


__all__ = ["build_portfolio_snapshots"]
