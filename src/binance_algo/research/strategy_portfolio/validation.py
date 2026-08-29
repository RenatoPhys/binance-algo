"""Portfolio-file validation without dashboard rendering."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from binance_algo.research.experiments.store import ResearchStore
from binance_algo.research.strategy_portfolio.compatibility import assess_compatibility
from binance_algo.research.strategy_portfolio.loader import load_strategy_component
from binance_algo.research.strategy_portfolio.models import load_portfolio_file


@dataclass(frozen=True, slots=True)
class PortfolioValidationRow:
    valid: bool
    portfolio_id: str
    component_label: str
    experiment_id: str
    run_id: str | None
    artifact_status: str
    compatibility_group: str | None
    window: str
    capital_weight: Decimal
    accounting_mode: str
    message: str


def validate_portfolio_declarations(
    *,
    store: ResearchStore,
    data_root: Path,
    portfolio_file: Path,
) -> tuple[PortfolioValidationRow, ...]:
    declarations = load_portfolio_file(portfolio_file)
    rows: list[PortfolioValidationRow] = []
    for portfolio in declarations.portfolios:
        weights = portfolio.resolved_weights()
        loaded = []
        errors: dict[str, str] = {}
        for component in portfolio.components:
            try:
                loaded.append(
                    load_strategy_component(
                        store=store,
                        data_root=data_root,
                        declaration=component,
                    )
                )
            except Exception as exc:
                errors[component.experiment_id] = str(exc)
        report = (
            assess_compatibility(
                tuple(loaded),
                alignment=portfolio.alignment,
                accounting_mode=portfolio.accounting_mode,
            )
            if loaded and not errors
            else None
        )
        loaded_by_id = {item.run.experiment_id: item for item in loaded}
        for component, weight in zip(portfolio.components, weights, strict=True):
            source = loaded_by_id.get(component.experiment_id)
            message = errors.get(component.experiment_id, "")
            if source is not None and report is not None:
                message = "; ".join((*report.issues, *report.warnings))
            rows.append(
                PortfolioValidationRow(
                    valid=(source is not None and report is not None and report.valid),
                    portfolio_id=portfolio.portfolio_id,
                    component_label=component.label,
                    experiment_id=component.experiment_id,
                    run_id=source.run.run_id if source is not None else component.run_id,
                    artifact_status="VERIFIED" if source is not None else "INVALID",
                    compatibility_group=(
                        report.compatibility_group if report is not None else None
                    ),
                    window=(
                        f"{source.start_time_ms}..{source.end_time_ms} "
                        f"({source.oos_curve.height} periods)"
                        if source is not None
                        else "—"
                    ),
                    capital_weight=weight,
                    accounting_mode=portfolio.accounting_mode.value,
                    message=message,
                )
            )
    return tuple(rows)


__all__ = ["PortfolioValidationRow", "validate_portfolio_declarations"]
