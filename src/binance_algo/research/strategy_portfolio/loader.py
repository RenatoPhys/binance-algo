"""Registry-backed loading and validation of strategy-sleeve artifacts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import orjson
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.experiments.artifacts import verify_run_artifacts
from binance_algo.research.experiments.models import (
    ExperimentSpec,
    PromotionDecision,
    ResearchStage,
    RunStatus,
)
from binance_algo.research.experiments.store import (
    ExperimentRunRecord,
    ResearchArtifactRecord,
    ResearchStore,
)
from binance_algo.research.strategy_portfolio.models import StrategyPortfolioComponent

OOS_RECONCILIATION_TOLERANCE = 1.0e-10
REQUIRED_ARTIFACT_TYPES = frozenset(
    {
        "oos_curve",
        "monthly_metrics",
        "fold_metrics",
        "regime_metrics",
        "symbol_metrics",
    }
)
OOS_REQUIRED_COLUMNS = (
    "fold",
    "decision_time_ms",
    "execution_time_ms",
    "price_pnl",
    "funding_pnl",
    "trading_fees",
    "spread_cost",
    "slippage_cost",
    "net_return",
    "turnover",
    "gross_exposure",
    "net_exposure",
    "beta_exposure",
    "market_volatility_regime",
    "weights_json",
)
_OOS_FINANCIAL_COLUMNS = (
    "price_pnl",
    "funding_pnl",
    "trading_fees",
    "spread_cost",
    "slippage_cost",
    "net_return",
    "turnover",
    "gross_exposure",
    "net_exposure",
    "beta_exposure",
    "market_volatility_regime",
)
_SEGMENT_KEYS = {
    "monthly_metrics": "month",
    "fold_metrics": "fold",
    "regime_metrics": "regime",
    "symbol_metrics": "symbol",
}


@dataclass(frozen=True, slots=True)
class LoadedStrategyComponent:
    declaration: StrategyPortfolioComponent
    spec: ExperimentSpec
    run: ExperimentRunRecord
    artifacts: tuple[ResearchArtifactRecord, ...]
    artifact_paths: dict[str, Path]
    source_checksums: dict[str, str]
    oos_curve: pl.DataFrame
    monthly_metrics: pl.DataFrame
    fold_metrics: pl.DataFrame
    regime_metrics: pl.DataFrame
    symbol_metrics: pl.DataFrame
    positions: pl.DataFrame | None
    weights: tuple[dict[str, float], ...]
    symbols: tuple[str, ...]
    campaigns: tuple[str, ...]
    research_stage: ResearchStage

    @property
    def start_time_ms(self) -> int:
        return int(self.oos_curve["decision_time_ms"][0])

    @property
    def end_time_ms(self) -> int:
        return int(self.oos_curve["decision_time_ms"][-1])


def current_research_stage(store: ResearchStore, experiment_id: str) -> ResearchStage:
    stage = ResearchStage.DISCOVERY
    for event in store.list_promotions(experiment_id):
        if event.decision is not PromotionDecision.BLOCKED:
            stage = event.to_stage
    return stage


def _verified_artifacts(
    *,
    store: ResearchStore,
    data_root: Path,
    run: ExperimentRunRecord,
) -> tuple[ResearchArtifactRecord, ...]:
    artifacts = store.list_artifacts(run.run_id)
    verification = verify_run_artifacts(
        data_root=data_root,
        run_id=run.run_id,
        artifacts=artifacts,
    )
    if not verification.valid:
        raise ResearchError(
            f"run {run.run_id} has invalid artifacts: " + "; ".join(verification.issues)
        )
    available = {artifact.artifact_type for artifact in artifacts}
    missing = sorted(REQUIRED_ARTIFACT_TYPES.difference(available))
    if missing:
        raise ResearchError(f"run {run.run_id} is missing required artifacts: {missing}")
    return artifacts


def resolve_component_run(
    *,
    store: ResearchStore,
    data_root: Path,
    declaration: StrategyPortfolioComponent,
) -> tuple[ExperimentSpec, ExperimentRunRecord, tuple[ResearchArtifactRecord, ...]]:
    """Resolve an explicit run or the newest verified successful attempt."""

    spec = store.get_experiment(declaration.experiment_id)
    if spec is None:
        raise ResearchError(f"unknown experiment: {declaration.experiment_id}")
    if declaration.run_id is not None:
        run = store.get_run(declaration.run_id)
        if run is None:
            raise ResearchError(f"unknown run: {declaration.run_id}")
        if run.experiment_id != declaration.experiment_id:
            raise ResearchError(
                f"run {run.run_id} belongs to experiment {run.experiment_id}, not "
                f"{declaration.experiment_id}"
            )
        if run.status is not RunStatus.SUCCEEDED:
            raise ResearchError(f"run {run.run_id} is {run.status.value}, not SUCCEEDED")
        return spec, run, _verified_artifacts(store=store, data_root=data_root, run=run)

    failures: list[str] = []
    for run in reversed(store.list_successful_runs(declaration.experiment_id)):
        try:
            artifacts = _verified_artifacts(store=store, data_root=data_root, run=run)
        except ResearchError as exc:
            failures.append(str(exc))
            continue
        return spec, run, artifacts
    detail = f": {' | '.join(failures)}" if failures else ""
    raise ResearchError(
        f"experiment {declaration.experiment_id} has no verified successful run{detail}"
    )


def _artifact_paths(
    data_root: Path,
    artifacts: tuple[ResearchArtifactRecord, ...],
) -> dict[str, Path]:
    root = data_root.resolve()
    output: dict[str, Path] = {}
    for artifact in artifacts:
        path = (root / artifact.path).resolve()
        if not path.is_relative_to(root):
            raise ResearchError(f"artifact path escapes data root: {artifact.path}")
        output[artifact.artifact_type] = path
    return output


def _require_finite(frame: pl.DataFrame, columns: tuple[str, ...], *, role: str) -> None:
    for column in columns:
        values = np.asarray(frame[column].to_numpy(), dtype=np.float64)
        if np.any(~np.isfinite(values)):
            raise ResearchError(f"{role} contains NaN or infinity in {column}")


def _validate_segmented_frame(frame: pl.DataFrame, *, artifact_type: str) -> None:
    if frame.is_empty():
        raise ResearchError(f"{artifact_type} artifact is empty")
    key = _SEGMENT_KEYS[artifact_type]
    if key not in frame.columns:
        raise ResearchError(f"{artifact_type} is missing required column: {key}")
    numeric = tuple(
        name for name, data_type in frame.schema.items() if name != key and data_type.is_numeric()
    )
    _require_finite(frame, numeric, role=artifact_type)


def _parse_weights(frame: pl.DataFrame) -> tuple[tuple[dict[str, float], ...], tuple[str, ...]]:
    parsed_rows: list[dict[str, float]] = []
    expected_symbols: tuple[str, ...] | None = None
    for row_number, raw in enumerate(frame["weights_json"].to_list()):
        try:
            payload = orjson.loads(raw)
        except (orjson.JSONDecodeError, TypeError) as exc:
            raise ResearchError(f"weights_json row {row_number} is invalid JSON") from exc
        if not isinstance(payload, dict) or not payload:
            raise ResearchError(f"weights_json row {row_number} must be a non-empty object")
        symbols = tuple(payload)
        if symbols != tuple(sorted(symbols)):
            raise ResearchError(f"weights_json row {row_number} symbols are not sorted")
        if expected_symbols is None:
            expected_symbols = symbols
        elif symbols != expected_symbols:
            raise ResearchError(f"weights_json row {row_number} has an unstable symbol set")
        weights: dict[str, float] = {}
        for symbol, value in payload.items():
            if not isinstance(symbol, str) or not symbol:
                raise ResearchError(f"weights_json row {row_number} has an invalid symbol")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ResearchError(f"weights_json row {row_number} has a non-numeric weight")
            weight = float(value)
            if not math.isfinite(weight):
                raise ResearchError(f"weights_json row {row_number} has a non-finite weight")
            weights[symbol] = weight
        parsed_rows.append(weights)
    assert expected_symbols is not None
    return tuple(parsed_rows), expected_symbols


def validate_oos_curve(
    frame: pl.DataFrame,
) -> tuple[tuple[dict[str, float], ...], tuple[str, ...]]:
    if frame.is_empty():
        raise ResearchError("oos_curve artifact is empty")
    missing = sorted(set(OOS_REQUIRED_COLUMNS).difference(frame.columns))
    if missing:
        raise ResearchError(f"oos_curve is missing required columns: {missing}")
    times = np.asarray(frame["decision_time_ms"].to_numpy(), dtype=np.int64)
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ResearchError("oos_curve decision timestamps must be strictly increasing")
    executions = np.asarray(frame["execution_time_ms"].to_numpy(), dtype=np.int64)
    if len(executions) > 1 and np.any(np.diff(executions) <= 0):
        raise ResearchError("oos_curve execution timestamps must be strictly increasing")
    folds = np.asarray(frame["fold"].to_numpy(), dtype=np.int64)
    if np.any(folds < 1) or (len(folds) > 1 and np.any(np.diff(folds) < 0)):
        raise ResearchError("oos_curve folds must be positive and ordered")
    if tuple(dict.fromkeys(int(value) for value in folds)) != tuple(sorted(set(folds))):
        raise ResearchError("oos_curve folds must not reappear after a later fold")
    _require_finite(frame, _OOS_FINANCIAL_COLUMNS, role="oos_curve")
    returns = np.asarray(frame["net_return"].to_numpy(), dtype=np.float64)
    if np.any(1.0 + returns <= 0):
        raise ResearchError("oos_curve contains a return that makes equity non-positive")
    turnover = np.asarray(frame["turnover"].to_numpy(), dtype=np.float64)
    if np.any(turnover < 0):
        raise ResearchError("oos_curve turnover must be non-negative")
    reconciled = (
        np.asarray(frame["price_pnl"].to_numpy(), dtype=np.float64)
        + np.asarray(frame["funding_pnl"].to_numpy(), dtype=np.float64)
        - np.asarray(frame["trading_fees"].to_numpy(), dtype=np.float64)
        - np.asarray(frame["spread_cost"].to_numpy(), dtype=np.float64)
        - np.asarray(frame["slippage_cost"].to_numpy(), dtype=np.float64)
    )
    maximum_error = float(np.max(np.abs(returns - reconciled)))
    if maximum_error > OOS_RECONCILIATION_TOLERANCE:
        raise ResearchError(
            "oos_curve accounting does not reconcile; "
            f"maximum error {maximum_error:.3e} exceeds {OOS_RECONCILIATION_TOLERANCE:.3e}"
        )
    return _parse_weights(frame)


def load_strategy_component(
    *,
    store: ResearchStore,
    data_root: Path,
    declaration: StrategyPortfolioComponent,
) -> LoadedStrategyComponent:
    spec, run, artifacts = resolve_component_run(
        store=store,
        data_root=data_root,
        declaration=declaration,
    )
    paths = _artifact_paths(data_root, artifacts)
    try:
        oos_curve = pl.read_parquet(paths["oos_curve"])
        monthly_metrics = pl.read_parquet(paths["monthly_metrics"])
        fold_metrics = pl.read_parquet(paths["fold_metrics"])
        regime_metrics = pl.read_parquet(paths["regime_metrics"])
        symbol_metrics = pl.read_parquet(paths["symbol_metrics"])
        positions = pl.read_parquet(paths["positions"]) if "positions" in paths else None
    except (OSError, KeyError, pl.exceptions.PolarsError) as exc:
        raise ResearchError(
            f"cannot read registered artifacts for run {run.run_id}: {exc}"
        ) from exc
    weights, symbols = validate_oos_curve(oos_curve)
    frames = {
        "monthly_metrics": monthly_metrics,
        "fold_metrics": fold_metrics,
        "regime_metrics": regime_metrics,
        "symbol_metrics": symbol_metrics,
    }
    for artifact_type, frame in frames.items():
        _validate_segmented_frame(frame, artifact_type=artifact_type)
    campaigns = tuple(item.name for item in store.campaigns_for_experiment(run.experiment_id))
    return LoadedStrategyComponent(
        declaration=declaration,
        spec=spec,
        run=run,
        artifacts=artifacts,
        artifact_paths=paths,
        source_checksums={
            artifact.artifact_type: artifact.checksum_sha256 for artifact in artifacts
        },
        oos_curve=oos_curve,
        monthly_metrics=monthly_metrics,
        fold_metrics=fold_metrics,
        regime_metrics=regime_metrics,
        symbol_metrics=symbol_metrics,
        positions=positions,
        weights=weights,
        symbols=symbols,
        campaigns=campaigns,
        research_stage=current_research_stage(store, run.experiment_id),
    )


__all__ = [
    "OOS_RECONCILIATION_TOLERANCE",
    "OOS_REQUIRED_COLUMNS",
    "REQUIRED_ARTIFACT_TYPES",
    "LoadedStrategyComponent",
    "current_research_stage",
    "load_strategy_component",
    "resolve_component_run",
    "validate_oos_curve",
]
