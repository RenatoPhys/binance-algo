"""Campaign-wide robustness, parameter-neighborhood, and candidate context reports."""

from __future__ import annotations

import hashlib
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.config import ResearchPlatformConfig
from binance_algo.research.contracts import ValidationProfile
from binance_algo.research.experiments.models import MetricScope
from binance_algo.research.experiments.store import CampaignRecord, ResearchStore
from binance_algo.research.validation.multiple_testing import (
    DeflatedSharpeResult,
    PBOResult,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    return_moments,
)


class RobustnessStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    NOT_AVAILABLE = "NOT_AVAILABLE"


@dataclass(frozen=True, slots=True)
class TrialRobustness:
    experiment_id: str
    run_id: str
    tags: Mapping[str, Any]
    total_return: float
    gross_return: float
    sharpe: float
    max_drawdown: float
    turnover: float
    explicit_cost: float
    rank_ic: float
    worst_fold_return: float
    profitable_folds: int
    fold_count: int
    worst_regime_return: float
    regime_count: int
    month_concentration: float
    symbol_concentration: float
    cost_1_5x_return: float
    delay_1_bar_return: float


@dataclass(frozen=True, slots=True)
class ParameterNeighborhood:
    status: RobustnessStatus
    reason: str
    best_experiment_id: str
    neighbor_experiment_ids: tuple[str, ...]
    neighbor_count: int
    neighbor_median_sharpe: float | None
    neighbor_positive_fraction: float | None
    best_to_neighbor_median_gap: float | None


@dataclass(frozen=True, slots=True)
class LockboxAssessment:
    status: RobustnessStatus
    reason: str
    manifest: str | None


@dataclass(frozen=True, slots=True)
class CampaignRobustnessResult:
    campaign_id: str
    campaign_name: str
    planned_trials: int
    successful_trials: int
    distinct_strategies: int
    approximate_independent_strategies: float
    best_experiment_id: str
    trials: tuple[TrialRobustness, ...]
    neighborhood: ParameterNeighborhood
    dsr: DeflatedSharpeResult
    pbo: PBOResult
    lockbox: LockboxAssessment
    sharpe_distribution: Mapping[str, float]
    return_distribution: Mapping[str, float]
    report_json_path: Path
    report_markdown_path: Path

    def trial(self, experiment_id: str) -> TrialRobustness:
        for trial in self.trials:
            if trial.experiment_id == experiment_id:
                return trial
        raise ResearchError(
            f"experiment is not a successful trial in campaign {self.campaign_name}: "
            f"{experiment_id}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_parquet(
    store: ResearchStore,
    data_root: Path,
    run_id: str,
    artifact_type: str,
) -> pl.DataFrame:
    matches = [
        artifact
        for artifact in store.list_artifacts(run_id)
        if artifact.artifact_type == artifact_type
    ]
    if len(matches) != 1:
        raise ResearchError(f"run {run_id} requires one {artifact_type} artifact")
    artifact = matches[0]
    root = data_root.resolve()
    path = (root / artifact.path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ResearchError(f"robustness artifact is unavailable: {artifact.path}")
    if _sha256(path) != artifact.checksum_sha256:
        raise ResearchError(f"robustness artifact is corrupt: {artifact.path}")
    frame = pl.read_parquet(path)
    if artifact.row_count != frame.height:
        raise ResearchError(f"robustness artifact row count differs: {artifact.path}")
    return frame


def _concentration(frame: pl.DataFrame, column: str) -> float:
    if frame.is_empty() or column not in frame.columns:
        raise ResearchError(f"cannot calculate concentration without {column}")
    absolute = frame[column].abs()
    total = absolute.sum()
    maximum = absolute.max()
    if not isinstance(total, (int, float)) or not isinstance(maximum, (int, float)):
        raise ResearchError(f"concentration input is invalid: {column}")
    return 0.0 if float(total) <= 1e-18 else float(maximum) / float(total)


def _metric_maps(
    store: ResearchStore, run_id: str
) -> tuple[
    dict[str, float],
    dict[int, dict[str, float]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
]:
    summary: dict[str, float] = {}
    folds: dict[int, dict[str, float]] = {}
    regimes: dict[str, dict[str, float]] = {}
    stress: dict[str, dict[str, float]] = {}
    for metric in store.list_metrics(run_id):
        if metric.scope is MetricScope.TEST and metric.fold is None and metric.regime is None:
            summary[metric.metric_name] = metric.metric_value
        elif metric.scope is MetricScope.TEST and metric.fold is not None:
            folds.setdefault(metric.fold, {})[metric.metric_name] = metric.metric_value
        elif metric.scope is MetricScope.TEST and metric.regime is not None:
            regimes.setdefault(metric.regime, {})[metric.metric_name] = metric.metric_value
        elif metric.scope is MetricScope.STRESS and metric.regime is not None:
            stress.setdefault(metric.regime, {})[metric.metric_name] = metric.metric_value
    return summary, folds, regimes, stress


def _required(mapping: Mapping[str, float], key: str) -> float:
    try:
        value = float(mapping[key])
    except KeyError as exc:
        raise ResearchError(f"robustness report requires metric {key}") from exc
    if not math.isfinite(value):
        raise ResearchError(f"robustness metric is not finite: {key}")
    return value


def _trial_robustness(
    *,
    store: ResearchStore,
    data_root: Path,
    experiment_id: str,
    run_id: str,
    tags: Mapping[str, Any],
) -> tuple[TrialRobustness, pl.DataFrame]:
    summary, folds, regimes, stress = _metric_maps(store, run_id)
    if not folds or not regimes:
        raise ResearchError(f"run {run_id} lacks fold or regime metrics")
    monthly = _verified_parquet(store, data_root, run_id, "monthly_metrics")
    symbols = _verified_parquet(store, data_root, run_id, "symbol_metrics")
    curve = _verified_parquet(store, data_root, run_id, "oos_curve").select(
        "decision_time_ms", "net_return"
    )
    fold_returns = tuple(_required(values, "total_return") for values in folds.values())
    regime_returns = tuple(_required(values, "total_return") for values in regimes.values())
    explicit_cost = sum(
        _required(summary, name) for name in ("trading_fees", "spread_cost", "slippage_cost")
    )
    return (
        TrialRobustness(
            experiment_id=experiment_id,
            run_id=run_id,
            tags=dict(tags),
            total_return=_required(summary, "total_return"),
            gross_return=_required(summary, "price_pnl") + _required(summary, "funding_pnl"),
            sharpe=_required(summary, "sharpe"),
            max_drawdown=_required(summary, "max_drawdown"),
            turnover=_required(summary, "turnover"),
            explicit_cost=explicit_cost,
            rank_ic=_required(summary, "mean_cross_sectional_rank_ic"),
            worst_fold_return=min(fold_returns),
            profitable_folds=sum(value > 0 for value in fold_returns),
            fold_count=len(fold_returns),
            worst_regime_return=min(regime_returns),
            regime_count=len(regime_returns),
            month_concentration=_concentration(monthly, "net_pnl"),
            symbol_concentration=_concentration(symbols, "net_pnl"),
            cost_1_5x_return=_required(stress.get("cost_1_5x", {}), "total_return"),
            delay_1_bar_return=_required(stress.get("signal_delay_1_bar", {}), "total_return"),
        ),
        curve,
    )


def _flatten_numeric(tags: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for scope in ("strategy_parameters", "portfolio_parameters"):
        parameters = tags.get(scope)
        if not isinstance(parameters, Mapping):
            continue
        for name, value in parameters.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[f"{scope}.{name}"] = float(value)
    return values


def parameter_neighborhood(
    trials: Sequence[TrialRobustness],
    *,
    best_experiment_id: str,
) -> ParameterNeighborhood:
    best = next(trial for trial in trials if trial.experiment_id == best_experiment_id)
    best_values = _flatten_numeric(best.tags)
    peer_values = {trial.experiment_id: _flatten_numeric(trial.tags) for trial in trials}
    varying = tuple(
        key
        for key in sorted(best_values)
        if len({values.get(key) for values in peer_values.values()}) > 1
    )
    if len(trials) < 2 or not varying:
        return ParameterNeighborhood(
            status=RobustnessStatus.NOT_AVAILABLE,
            reason="campaign has no numeric parameter neighborhood around the best trial",
            best_experiment_id=best_experiment_id,
            neighbor_experiment_ids=(),
            neighbor_count=0,
            neighbor_median_sharpe=None,
            neighbor_positive_fraction=None,
            best_to_neighbor_median_gap=None,
        )
    ranges = {
        key: max(values[key] for values in peer_values.values())
        - min(values[key] for values in peer_values.values())
        for key in varying
    }
    distances = []
    for trial in trials:
        if trial.experiment_id == best_experiment_id:
            continue
        distance = math.sqrt(
            sum(
                ((peer_values[trial.experiment_id][key] - best_values[key]) / ranges[key]) ** 2
                for key in varying
                if ranges[key] > 0
            )
        )
        distances.append((distance, trial.experiment_id, trial.sharpe))
    distances.sort(key=lambda item: (item[0], item[1]))
    neighbor_count = min(3, len(distances))
    neighbors = distances[:neighbor_count]
    sharpes = np.asarray([item[2] for item in neighbors], dtype=np.float64)
    median = float(np.median(sharpes))
    return ParameterNeighborhood(
        status=RobustnessStatus.AVAILABLE,
        reason="nearest trials in normalized numeric parameter space",
        best_experiment_id=best_experiment_id,
        neighbor_experiment_ids=tuple(item[1] for item in neighbors),
        neighbor_count=neighbor_count,
        neighbor_median_sharpe=median,
        neighbor_positive_fraction=float(np.mean(sharpes > 0)),
        best_to_neighbor_median_gap=best.sharpe - median,
    )


def _aligned_return_matrix(
    curves: Sequence[tuple[str, pl.DataFrame]],
) -> tuple[np.ndarray[Any, np.dtype[np.float64]] | None, str | None]:
    if not curves:
        return None, "campaign has no successful return series"
    times = curves[0][1]["decision_time_ms"].to_list()
    columns = []
    for experiment_id, curve in curves:
        if curve["decision_time_ms"].to_list() != times:
            return None, f"trial {experiment_id} does not share the same decision segments"
        columns.append(np.asarray(curve["net_return"].to_numpy(), dtype=np.float64))
    return np.column_stack(columns), None


def effective_strategy_count(matrix: np.ndarray[Any, np.dtype[np.float64]]) -> float:
    """Return the correlation-spectrum participation ratio used by robustness reports."""

    if matrix.shape[1] == 1:
        return 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(matrix, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 1.0)
    eigenvalues = np.clip(np.linalg.eigvalsh(correlation), 0.0, None)
    denominator = float(np.sum(eigenvalues**2))
    return 1.0 if denominator <= 1e-18 else float(np.sum(eigenvalues) ** 2 / denominator)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "p75": float(np.quantile(array, 0.75)),
        "maximum": float(np.max(array)),
    }


def _lockbox(platform: ResearchPlatformConfig) -> LockboxAssessment:
    if platform.lockbox_manifest is None:
        return LockboxAssessment(
            status=RobustnessStatus.NOT_AVAILABLE,
            reason=(
                "no independent lockbox dataset/period is configured; the current 90-day "
                "development history cannot be relabeled as lockbox"
            ),
            manifest=None,
        )
    return LockboxAssessment(
        status=RobustnessStatus.AVAILABLE,
        reason="an explicit lockbox manifest is configured; access still requires a separate event",
        manifest=platform.lockbox_manifest,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        if temporary.exists():
            temporary.unlink()
        raise ResearchError(f"cannot write robustness report {path}: {exc}") from exc


def build_campaign_robustness(
    *,
    store: ResearchStore,
    campaign: CampaignRecord,
    data_root: Path,
    reports_root: Path,
    platform: ResearchPlatformConfig,
    periods_per_year: int = 24 * 365,
) -> CampaignRobustnessResult:
    try:
        campaign_payload = orjson.loads(campaign.spec_json)
        profile = ValidationProfile(
            str(
                campaign_payload.get("source_spec", {})
                .get("validation", {})
                .get("profile", ValidationProfile.FULL)
            )
        )
    except (AttributeError, TypeError, ValueError, orjson.JSONDecodeError) as exc:
        raise ResearchError(f"campaign validation profile is invalid: {exc}") from exc
    if profile is ValidationProfile.DISCOVERY:
        raise ResearchError("discovery profile does not execute DSR, PBO, or lockbox robustness")
    trials: list[TrialRobustness] = []
    curves: list[tuple[str, pl.DataFrame]] = []
    distinct_tags: set[bytes] = set()
    for _, experiment_id, tags in store.list_campaign_experiments(campaign.campaign_id):
        run = store.latest_successful_run(experiment_id)
        if run is None:
            continue
        trial, curve = _trial_robustness(
            store=store,
            data_root=data_root,
            experiment_id=experiment_id,
            run_id=run.run_id,
            tags=tags,
        )
        trials.append(trial)
        curves.append((experiment_id, curve))
        distinct_tags.add(orjson.dumps(tags, option=orjson.OPT_SORT_KEYS))
    if not trials:
        raise ResearchError(f"campaign has no successful trials: {campaign.name}")
    trials.sort(key=lambda trial: (-trial.sharpe, trial.experiment_id))
    best = trials[0]
    neighborhood = parameter_neighborhood(trials, best_experiment_id=best.experiment_id)
    matrix, alignment_error = _aligned_return_matrix(curves)
    if matrix is None:
        raise ResearchError(alignment_error or "campaign return matrix is unavailable")
    moments = return_moments(
        matrix[:, next(index for index, item in enumerate(curves) if item[0] == best.experiment_id)]
    )
    trial_sharpes = np.asarray([trial.sharpe for trial in trials], dtype=np.float64)
    dsr = deflated_sharpe_ratio(
        observed_sharpe=best.sharpe,
        sample_size=moments.sample_size,
        skewness=moments.skewness,
        kurtosis=moments.kurtosis,
        number_of_trials=len(trials),
        trial_sharpes=trial_sharpes,
        periods_per_year=periods_per_year,
    )
    pbo = probability_of_backtest_overfitting(matrix)
    effective = effective_strategy_count(matrix)
    directory = (
        reports_root.resolve() / "research_campaigns" / f"campaign_id={campaign.campaign_id[:24]}"
    )
    json_path = directory / "robustness.json"
    markdown_path = directory / "robustness.md"
    lockbox = _lockbox(platform)
    sharpe_distribution = _distribution([trial.sharpe for trial in trials])
    return_distribution = _distribution([trial.total_return for trial in trials])
    payload = {
        "campaign_id": campaign.campaign_id,
        "campaign_name": campaign.name,
        "planned_trials": campaign.trial_count,
        "successful_trials": len(trials),
        "distinct_strategies": len(distinct_tags),
        "approximate_independent_strategies": effective,
        "best_experiment_id": best.experiment_id,
        "interpretation": (
            "The highest Sharpe is a selected campaign trial, not independent OOS evidence. "
            "A broad coherent neighborhood matters more than an isolated peak."
        ),
        "sharpe_distribution": sharpe_distribution,
        "return_distribution": return_distribution,
        "neighborhood": asdict(neighborhood),
        "dsr": asdict(dsr),
        "pbo": asdict(pbo),
        "lockbox": asdict(lockbox),
        "trials": [asdict(trial) for trial in trials],
    }
    _atomic_write(json_path, orjson.dumps(payload, option=orjson.OPT_SORT_KEYS) + b"\n")
    lines = [
        f"# Campaign robustness — {campaign.name}",
        "",
        "> The highest Sharpe is a selected campaign trial, not independent OOS evidence. "
        "A broad coherent parameter region matters more than an isolated peak.",
        "",
        f"- Planned / successful / distinct trials: {campaign.trial_count} / "
        f"{len(trials)} / {len(distinct_tags)}",
        f"- Approximate independent strategies: {effective:.3f}",
        f"- Best experiment: `{best.experiment_id}`",
        f"- DSR probability: {dsr.probability:.6f} over {dsr.number_of_trials} trials",
        f"- PBO: {pbo.status.value} — {pbo.reason}",
        f"- Lockbox: {lockbox.status.value} — {lockbox.reason}",
        "",
        "| Experiment | Return | Sharpe | Worst fold | Month conc. | Symbol conc. |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for trial in trials:
        lines.append(
            f"| `{trial.experiment_id[:16]}` | {trial.total_return:.6%} | "
            f"{trial.sharpe:.6f} | {trial.worst_fold_return:.6%} | "
            f"{trial.month_concentration:.3f} | {trial.symbol_concentration:.3f} |"
        )
    _atomic_write(markdown_path, ("\n".join(lines) + "\n").encode())
    return CampaignRobustnessResult(
        campaign_id=campaign.campaign_id,
        campaign_name=campaign.name,
        planned_trials=campaign.trial_count,
        successful_trials=len(trials),
        distinct_strategies=len(distinct_tags),
        approximate_independent_strategies=effective,
        best_experiment_id=best.experiment_id,
        trials=tuple(trials),
        neighborhood=neighborhood,
        dsr=dsr,
        pbo=pbo,
        lockbox=lockbox,
        sharpe_distribution=sharpe_distribution,
        return_distribution=return_distribution,
        report_json_path=json_path,
        report_markdown_path=markdown_path,
    )


__all__ = [
    "CampaignRobustnessResult",
    "LockboxAssessment",
    "ParameterNeighborhood",
    "RobustnessStatus",
    "TrialRobustness",
    "build_campaign_robustness",
    "effective_strategy_count",
    "parameter_neighborhood",
]
