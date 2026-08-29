"""Strict consolidated reporting for the preregistered Alpha Reboot Wave 1."""

from __future__ import annotations

import math
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any, cast

import numpy as np
import orjson
import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from binance_algo.common.errors import ResearchError
from binance_algo.research.experiments.artifacts import DEVELOPMENT_SEEN_BANNER
from binance_algo.research.experiments.models import ExperimentSpec, RunStatus
from binance_algo.research.experiments.store import (
    CampaignRecord,
    ExperimentRunRecord,
    ResearchStore,
)
from binance_algo.research.performance import calculate_return_statistics
from binance_algo.research.validation.multiple_testing import (
    deflated_sharpe_ratio,
    return_moments,
)

HOURS_PER_YEAR = 24 * 365
DAYS_PER_YEAR = 365
EXPECTED_CAMPAIGN_COUNT = 4
EXPECTED_TRIAL_COUNT = 18
MINIMUM_CORRELATION_OBSERVATIONS = 30

FAMILY_BY_STRATEGY = {
    "quarter_hour_flow": "quarter_hour_flow",
    "flow_absorption_reversal": "flow_absorption_reversal",
    "volatility_compression_breakout": "volatility_compression_breakout",
    "pair_spread_reversion": "pair_spread_reversion",
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChampionConfig(_StrictModel):
    run_id: str | None = None
    strategy: str
    portfolio: str
    portfolio_weights: dict[str, float]


class AlphaRebootWaveConfig(_StrictModel):
    wave_id: str
    research_context: str
    maximum_total_trials: int = Field(ge=1, le=EXPECTED_TRIAL_COUNT)
    campaigns: tuple[str, ...]
    champion: ChampionConfig

    @model_validator(mode="after")
    def validate_wave_contract(self) -> AlphaRebootWaveConfig:
        if self.wave_id != "alpha_reboot_wave1":
            raise ValueError("wave_id must be alpha_reboot_wave1")
        if self.research_context != "development_seen":
            raise ValueError("Wave 1 research_context must be development_seen")
        if len(self.campaigns) != EXPECTED_CAMPAIGN_COUNT:
            raise ValueError(f"Wave 1 requires exactly {EXPECTED_CAMPAIGN_COUNT} campaigns")
        return self


@dataclass(frozen=True, slots=True)
class AlphaRebootReportResult:
    report_directory: Path
    report_markdown_path: Path
    report_json_path: Path
    report_html_path: Path
    candidates_path: Path
    daily_return_correlation_path: Path
    position_correlation_path: Path
    hourly_returns_path: Path
    daily_returns_path: Path
    daily_positions_path: Path
    aligned_daily_returns_path: Path
    strategy_diagnostics_path: Path
    champion_run_id: str
    trial_count: int


@dataclass(frozen=True, slots=True)
class _ResolvedCandidate:
    campaign: CampaignRecord
    ordinal: int
    experiment_id: str
    run: ExperimentRunRecord
    spec: ExperimentSpec
    tags: Mapping[str, Any]
    artifacts: Mapping[str, Path]
    curve: pl.DataFrame
    daily_returns: pl.DataFrame
    daily_positions: pl.DataFrame


def load_alpha_reboot_wave_config(path: Path) -> AlphaRebootWaveConfig:
    try:
        payload = yaml.safe_load(path.resolve().read_text(encoding="utf-8"))
        return AlphaRebootWaveConfig.model_validate(payload)
    except (OSError, yaml.YAMLError, ValueError) as exc:
        raise ResearchError(f"invalid Alpha Reboot Wave 1 configuration {path}: {exc}") from exc


def _component_reference(value: str, *, field: str) -> tuple[str, str]:
    parts = value.rsplit(":v", maxsplit=1)
    if len(parts) != 2 or not all(parts):
        raise ResearchError(f"{field} must use component:vVERSION syntax")
    return parts[0], parts[1]


def _matches_champion(spec: ExperimentSpec, champion: ChampionConfig) -> bool:
    strategy_name, strategy_version = _component_reference(champion.strategy, field="strategy")
    portfolio_name, portfolio_version = _component_reference(champion.portfolio, field="portfolio")
    if (spec.strategy.component_id, spec.strategy.version) != (strategy_name, strategy_version):
        return False
    if (spec.portfolio_policy.component_id, spec.portfolio_policy.version) != (
        portfolio_name,
        portfolio_version,
    ):
        return False
    return all(
        key in spec.portfolio_parameters
        and math.isclose(float(spec.portfolio_parameters[key]), value, abs_tol=1e-12)
        for key, value in champion.portfolio_weights.items()
    )


def select_champion_run_id(eligible_run_ids: Sequence[str], configured_run_id: str | None) -> str:
    """Require an explicit selector whenever the economic identity is not unique."""

    unique = tuple(sorted(set(eligible_run_ids)))
    if configured_run_id is not None:
        if configured_run_id not in unique:
            raise ResearchError(
                f"configured champion run_id {configured_run_id} is not an eligible successful run"
            )
        return configured_run_id
    if len(unique) != 1:
        detail = ", ".join(unique) if unique else "none"
        raise ResearchError(
            "champion identity is ambiguous or absent; configure champion.run_id explicitly. "
            f"Eligible successful runs: {detail}"
        )
    return unique[0]


def resolve_champion_run(
    store: ResearchStore, champion: ChampionConfig
) -> tuple[ExperimentRunRecord, ExperimentSpec]:
    eligible: list[str] = []
    specs: dict[str, ExperimentSpec] = {}
    for run in store.list_runs():
        if run.status is not RunStatus.SUCCEEDED:
            continue
        spec = store.get_experiment(run.experiment_id)
        if spec is not None and _matches_champion(spec, champion):
            eligible.append(run.run_id)
            specs[run.run_id] = spec
    selected = select_champion_run_id(eligible, champion.run_id)
    selected_run = store.get_run(selected)
    if selected_run is None or selected_run.status is not RunStatus.SUCCEEDED:
        raise ResearchError(f"champion run is not successful: {selected}")
    return selected_run, specs[selected]


def _read_campaign_name(path: Path) -> tuple[str, int]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        root = cast(dict[str, Any], payload)
        campaign = cast(dict[str, Any], root["campaign"])
        return str(campaign["name"]), int(campaign["max_trials"])
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ResearchError(f"cannot read Wave 1 campaign identity from {path}: {exc}") from exc


def _artifact_paths(store: ResearchStore, *, data_root: Path, run_id: str) -> dict[str, Path]:
    root = data_root.resolve()
    output: dict[str, Path] = {}
    for artifact in store.list_artifacts(run_id):
        path = (root / artifact.path).resolve()
        if not path.is_relative_to(root):
            raise ResearchError(f"artifact path escapes data root: {artifact.path}")
        if not path.is_file():
            raise ResearchError(f"missing artifact for run {run_id}: {artifact.path}")
        output[artifact.artifact_type] = path
    return output


def _date_expression(column: str) -> pl.Expr:
    return pl.from_epoch(pl.col(column), time_unit="ms").dt.date().cast(pl.String)


def _daily_returns(curve: pl.DataFrame, *, series_id: str, source: str) -> pl.DataFrame:
    return (
        curve.select("decision_time_ms", "net_return")
        .with_columns(_date_expression("decision_time_ms").alias("date_utc"))
        .group_by("date_utc")
        .agg(((pl.col("net_return") + 1.0).product() - 1.0).alias("net_return"))
        .sort("date_utc")
        .with_columns(pl.lit(series_id).alias("series_id"), pl.lit(source).alias("source"))
        .select("series_id", "source", "date_utc", "net_return")
    )


def _daily_positions_from_curve(
    curve: pl.DataFrame, *, series_id: str, source: str
) -> pl.DataFrame:
    rows: list[dict[str, str | float]] = []
    for timestamp, raw in curve.select("decision_time_ms", "weights_json").iter_rows():
        date = datetime.fromtimestamp(int(timestamp) / 1_000, tz=UTC).date().isoformat()
        parsed = orjson.loads(str(raw))
        if not isinstance(parsed, dict):
            raise ResearchError("weights_json must contain an object")
        rows.extend(
            {
                "series_id": series_id,
                "source": source,
                "date_utc": date,
                "symbol": str(symbol),
                "target_weight": float(weight),
            }
            for symbol, weight in parsed.items()
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "series_id": pl.String,
                "source": pl.String,
                "date_utc": pl.String,
                "symbol": pl.String,
                "mean_target_weight": pl.Float64,
            }
        )
    return (
        pl.DataFrame(rows)
        .group_by("series_id", "source", "date_utc", "symbol")
        .agg(pl.col("target_weight").mean().alias("mean_target_weight"))
        .sort("series_id", "date_utc", "symbol")
    )


def _candidate_daily_positions(frame: pl.DataFrame, *, series_id: str, source: str) -> pl.DataFrame:
    return frame.with_columns(
        pl.lit(series_id).alias("series_id"), pl.lit(source).alias("source")
    ).select("series_id", "source", "date_utc", "symbol", "mean_target_weight")


def _correlation_row(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    left_id: str,
    right_id: str,
    value_column: str,
    keys: tuple[str, ...],
) -> dict[str, Any]:
    aligned = left.select(*keys, pl.col(value_column).alias("left_value")).join(
        right.select(*keys, pl.col(value_column).alias("right_value")), on=list(keys), how="inner"
    )
    count = aligned.height
    correlation: float | None = None
    if count >= MINIMUM_CORRELATION_OBSERVATIONS:
        left_values = np.asarray(aligned["left_value"].to_numpy(), dtype=np.float64)
        right_values = np.asarray(aligned["right_value"].to_numpy(), dtype=np.float64)
        if float(np.std(left_values)) > 1e-18 and float(np.std(right_values)) > 1e-18:
            correlation = float(np.corrcoef(left_values, right_values)[0, 1])
    return {
        "left_series_id": left_id,
        "right_series_id": right_id,
        "overlap_observations": count,
        "correlation": correlation,
        "available": correlation is not None,
    }


def _correlation_table(
    frames: Mapping[str, pl.DataFrame], *, value_column: str, keys: tuple[str, ...]
) -> pl.DataFrame:
    rows = [
        _correlation_row(
            frames[left_id],
            frames[right_id],
            left_id=left_id,
            right_id=right_id,
            value_column=value_column,
            keys=keys,
        )
        for left_id in sorted(frames)
        for right_id in sorted(frames)
    ]
    return pl.DataFrame(
        rows,
        schema={
            "left_series_id": pl.String,
            "right_series_id": pl.String,
            "overlap_observations": pl.Int64,
            "correlation": pl.Float64,
            "available": pl.Boolean,
        },
    )


def _correlation_to(correlations: pl.DataFrame, *, left_id: str, right_id: str) -> float | None:
    selected = correlations.filter(
        (pl.col("left_series_id") == left_id) & (pl.col("right_series_id") == right_id)
    )
    return cast(float | None, selected["correlation"][0]) if selected.height else None


def _overall_trade_metrics(frame: pl.DataFrame) -> dict[str, Any]:
    selected = frame.filter((pl.col("scope") == "overall") & (pl.col("group") == "all"))
    if selected.height != 1:
        raise ResearchError("trade_metrics must contain exactly one overall/all row")
    return selected.row(0, named=True)


def _absolute_concentration(frame: pl.DataFrame, column: str) -> float:
    values = np.abs(np.asarray(frame[column].to_numpy(), dtype=np.float64))
    denominator = float(np.sum(values))
    return float(np.max(values) / denominator) if denominator > 1e-18 else 1.0


def _positive_concentration(frame: pl.DataFrame, column: str) -> float:
    values = np.asarray(frame[column].to_numpy(), dtype=np.float64)
    positive = values[values > 0]
    return float(np.max(positive) / np.sum(positive)) if positive.size else 1.0


def _quarter_concentration(monthly: pl.DataFrame) -> float:
    quarters: dict[str, float] = {}
    for month, pnl in monthly.select("month", "net_pnl").iter_rows():
        year, month_number = str(month).split("-")
        key = f"{year}-Q{(int(month_number) - 1) // 3 + 1}"
        quarters[key] = quarters.get(key, 0.0) + float(pnl)
    return _positive_concentration(pl.DataFrame({"net_pnl": list(quarters.values())}), "net_pnl")


def _load_metrics(path: Path) -> dict[str, Any]:
    try:
        payload = orjson.loads(path.read_bytes())
    except (OSError, orjson.JSONDecodeError) as exc:
        raise ResearchError(f"cannot read metrics artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResearchError(f"metrics artifact is not an object: {path}")
    return cast(dict[str, Any], payload)


def _blend_metrics(
    champion: pl.DataFrame, candidate: pl.DataFrame
) -> dict[str, float | int | bool | None]:
    aligned = champion.select("date_utc", pl.col("net_return").alias("champion")).join(
        candidate.select("date_utc", pl.col("net_return").alias("candidate")),
        on="date_utc",
        how="inner",
    )
    if aligned.height < MINIMUM_CORRELATION_OBSERVATIONS:
        return {
            "available": False,
            "overlap_days": aligned.height,
            "champion_return": None,
            "champion_sharpe": None,
            "champion_max_drawdown": None,
            "blend_80_20_return": None,
            "blend_80_20_sharpe": None,
            "blend_80_20_max_drawdown": None,
            "portfolio_sharpe_increment": None,
            "portfolio_max_drawdown_reduction": None,
            "champion_return_cost": None,
            "blend_50_50_inverse_vol_return": None,
            "blend_50_50_inverse_vol_sharpe": None,
            "blend_50_50_inverse_vol_max_drawdown": None,
        }
    champion_values = np.asarray(aligned["champion"].to_numpy(), dtype=np.float64)
    candidate_values = np.asarray(aligned["candidate"].to_numpy(), dtype=np.float64)
    champion_vol = float(np.std(champion_values, ddof=1))
    candidate_vol = float(np.std(candidate_values, ddof=1))
    if champion_vol <= 1e-18 or candidate_vol <= 1e-18:
        raise ResearchError("blend diagnostics require non-zero candidate and champion volatility")
    risk_scaled_candidate = candidate_values * champion_vol / candidate_vol
    blend_80_20 = 0.80 * champion_values + 0.20 * risk_scaled_candidate
    inverse_vol_weights = np.asarray([1.0 / champion_vol, 1.0 / candidate_vol])
    inverse_vol_weights /= np.sum(inverse_vol_weights)
    blend_inverse_vol = (
        inverse_vol_weights[0] * champion_values + inverse_vol_weights[1] * candidate_values
    )
    champion_stats = calculate_return_statistics(champion_values, periods_per_year=DAYS_PER_YEAR)
    blend_stats = calculate_return_statistics(blend_80_20, periods_per_year=DAYS_PER_YEAR)
    diagnostic_stats = calculate_return_statistics(
        blend_inverse_vol, periods_per_year=DAYS_PER_YEAR
    )
    drawdown_reduction = (
        (abs(champion_stats.max_drawdown) - abs(blend_stats.max_drawdown))
        / abs(champion_stats.max_drawdown)
        if champion_stats.max_drawdown < 0
        else 0.0
    )
    return_cost = (
        max(0.0, champion_stats.total_return - blend_stats.total_return)
        / abs(champion_stats.total_return)
        if abs(champion_stats.total_return) > 1e-18
        else 0.0
    )
    return {
        "available": True,
        "overlap_days": aligned.height,
        "champion_return": champion_stats.total_return,
        "champion_sharpe": champion_stats.sharpe,
        "champion_max_drawdown": champion_stats.max_drawdown,
        "blend_80_20_return": blend_stats.total_return,
        "blend_80_20_sharpe": blend_stats.sharpe,
        "blend_80_20_max_drawdown": blend_stats.max_drawdown,
        "portfolio_sharpe_increment": blend_stats.sharpe - champion_stats.sharpe,
        "portfolio_max_drawdown_reduction": drawdown_reduction,
        "champion_return_cost": return_cost,
        "blend_50_50_inverse_vol_return": diagnostic_stats.total_return,
        "blend_50_50_inverse_vol_sharpe": diagnostic_stats.sharpe,
        "blend_50_50_inverse_vol_max_drawdown": diagnostic_stats.max_drawdown,
    }


def _gate(value: bool, detail: str) -> dict[str, bool | str]:
    return {"passed": value, "detail": detail}


def _evaluate_candidate(
    candidate: _ResolvedCandidate,
    *,
    champion_daily: pl.DataFrame,
    daily_correlations: pl.DataFrame,
    trial_sharpes: np.ndarray[Any, np.dtype[np.float64]],
) -> dict[str, Any]:
    artifacts = candidate.artifacts
    required = {
        "metrics",
        "fold_metrics",
        "monthly_metrics",
        "symbol_metrics",
        "trade_metrics",
        "trade_events",
    }
    missing = sorted(required.difference(artifacts))
    if missing:
        return {
            "family": FAMILY_BY_STRATEGY[candidate.spec.strategy.component_id],
            "campaign_name": candidate.campaign.name,
            "ordinal": candidate.ordinal,
            "experiment_id": candidate.experiment_id,
            "run_id": candidate.run.run_id,
            "classification": "invalid/incomplete",
            "invalid_reason": f"missing artifacts: {missing}",
        }
    payload = _load_metrics(artifacts["metrics"])
    metrics = cast(dict[str, Any], payload["metrics"])
    stress = cast(dict[str, dict[str, Any]], payload["stress"])
    bootstrap = cast(dict[str, Any], payload["bootstrap"])
    folds = pl.read_parquet(artifacts["fold_metrics"])
    months = pl.read_parquet(artifacts["monthly_metrics"])
    symbols = pl.read_parquet(artifacts["symbol_metrics"])
    trade_metrics = _overall_trade_metrics(pl.read_parquet(artifacts["trade_metrics"]))
    fold_returns = np.asarray(folds["total_return"].to_numpy(), dtype=np.float64)
    fold_fraction = float(np.mean(fold_returns > 0))
    symbol_concentration = _absolute_concentration(symbols, "net_pnl")
    month_concentration = _positive_concentration(months, "net_pnl")
    quarter_concentration = _quarter_concentration(months)
    gross_edge = float(trade_metrics["gross_edge_bps_per_turnover"])
    explicit_cost_edge = float(trade_metrics["explicit_cost_bps_per_turnover"])
    completed_trades = int(trade_metrics["completed_trades"])
    correlation = _correlation_to(
        daily_correlations,
        left_id=candidate.experiment_id,
        right_id="champion",
    )
    moments = return_moments(candidate.curve["net_return"].to_numpy())
    dsr = deflated_sharpe_ratio(
        observed_sharpe=float(metrics["sharpe"]),
        sample_size=moments.sample_size,
        skewness=moments.skewness,
        kurtosis=moments.kurtosis,
        number_of_trials=len(trial_sharpes),
        trial_sharpes=trial_sharpes,
        periods_per_year=HOURS_PER_YEAR,
    )
    blend = _blend_metrics(champion_daily, candidate.daily_returns)
    standalone = {
        "net_total_return": _gate(float(metrics["total_return"]) > 0, str(metrics["total_return"])),
        "net_sharpe": _gate(float(metrics["sharpe"]) >= 0.75, str(metrics["sharpe"])),
        "max_drawdown": _gate(
            float(metrics["max_drawdown"]) >= -0.15, str(metrics["max_drawdown"])
        ),
        "cost_1_5x": _gate(
            float(stress["cost_1_5x"]["total_return"]) > 0, str(stress["cost_1_5x"]["total_return"])
        ),
        "cost_2_0x": _gate(
            float(stress["cost_2_0x"]["total_return"]) > 0, str(stress["cost_2_0x"]["total_return"])
        ),
        "profitable_fold_fraction": _gate(fold_fraction >= 0.60, str(fold_fraction)),
        "completed_trades": _gate(completed_trades >= 30, str(completed_trades)),
        "gross_edge_vs_cost": _gate(
            gross_edge >= 1.5 * explicit_cost_edge, f"{gross_edge} vs {explicit_cost_edge}"
        ),
        "symbol_concentration": _gate(symbol_concentration <= 0.60, str(symbol_concentration)),
        "positive_month_concentration": _gate(
            month_concentration <= 0.35, str(month_concentration)
        ),
        "bootstrap_probability_positive": _gate(
            float(bootstrap["probability_positive"]) >= 0.70, str(bootstrap["probability_positive"])
        ),
        "signal_delay_1_bar": _gate(
            float(stress["signal_delay_1_bar"]["total_return"]) >= 0,
            str(stress["signal_delay_1_bar"]["total_return"]),
        ),
    }
    blend_effect = bool(blend["available"]) and (
        float(cast(float, blend["portfolio_sharpe_increment"])) >= 0.10
        or (
            float(cast(float, blend["portfolio_max_drawdown_reduction"])) >= 0.15
            and float(cast(float, blend["champion_return_cost"])) <= 0.10
        )
    )
    diversifier = {
        "net_total_return": _gate(float(metrics["total_return"]) > 0, str(metrics["total_return"])),
        "net_sharpe": _gate(float(metrics["sharpe"]) >= 0.35, str(metrics["sharpe"])),
        "cost_1_5x": _gate(
            float(stress["cost_1_5x"]["total_return"]) >= 0,
            str(stress["cost_1_5x"]["total_return"]),
        ),
        "daily_correlation_to_champion": _gate(
            correlation is not None and correlation <= 0.35,
            "unavailable: non-overlapping periods" if correlation is None else str(correlation),
        ),
        "completed_trades": _gate(completed_trades >= 30, str(completed_trades)),
        "symbol_concentration": _gate(symbol_concentration <= 0.60, str(symbol_concentration)),
        "fixed_blend_effect": _gate(
            blend_effect,
            "unavailable: non-overlapping periods" if not blend["available"] else str(blend),
        ),
    }
    standalone_pass = all(bool(value["passed"]) for value in standalone.values())
    diversifier_pass = all(bool(value["passed"]) for value in diversifier.values())
    gross_pnl = float(trade_metrics["gross_pnl"])
    if standalone_pass:
        classification = "standalone pass"
    elif diversifier_pass:
        classification = "diversifier pass"
    elif gross_pnl > 0:
        classification = "gross signal but not executable"
    else:
        classification = "rejected"
    return {
        "family": FAMILY_BY_STRATEGY[candidate.spec.strategy.component_id],
        "campaign_name": candidate.campaign.name,
        "ordinal": candidate.ordinal,
        "experiment_id": candidate.experiment_id,
        "run_id": candidate.run.run_id,
        "classification": classification,
        "strategy_parameters": candidate.tags.get("strategy_parameters", {}),
        "portfolio_parameters": candidate.tags.get("portfolio_parameters", {}),
        "net_total_return": float(metrics["total_return"]),
        "net_sharpe": float(metrics["sharpe"]),
        "max_drawdown": float(metrics["max_drawdown"]),
        "gross_pnl": gross_pnl,
        "explicit_cost": float(trade_metrics["explicit_cost"]),
        "gross_edge_bps_per_turnover": gross_edge,
        "explicit_cost_bps_per_turnover": explicit_cost_edge,
        "cost_1_5x_total_return": float(stress["cost_1_5x"]["total_return"]),
        "cost_2_0x_total_return": float(stress["cost_2_0x"]["total_return"]),
        "signal_delay_1_bar_total_return": float(stress["signal_delay_1_bar"]["total_return"]),
        "profitable_fold_fraction": fold_fraction,
        "completed_trades": completed_trades,
        "entries": int(trade_metrics["entries"]),
        "exits": int(trade_metrics["exits"]),
        "direction_flips": int(trade_metrics["direction_flips"]),
        "holding_hours_mean": float(trade_metrics["holding_hours_mean"]),
        "holding_hours_median": float(trade_metrics["holding_hours_median"]),
        "win_rate": float(trade_metrics["win_rate"]),
        "profit_factor": float(trade_metrics["profit_factor"]),
        "maximum_symbol_pnl_concentration": symbol_concentration,
        "maximum_positive_month_concentration": month_concentration,
        "maximum_positive_quarter_concentration": quarter_concentration,
        "bootstrap_probability_positive": float(bootstrap["probability_positive"]),
        "daily_return_correlation_to_champion": correlation,
        "dsr_probability": dsr.probability,
        "dsr_gate_0_90": dsr.probability >= 0.90,
        "standalone_pass": standalone_pass,
        "diversifier_pass": diversifier_pass,
        "standalone_gates": standalone,
        "diversifier_gates": diversifier,
        "blend_diagnostics": blend,
        "artifact_paths": {name: str(path) for name, path in sorted(artifacts.items())},
    }


def _candidate_frame(rows: Sequence[Mapping[str, Any]]) -> pl.DataFrame:
    flattened = []
    for row in rows:
        flattened.append(
            {
                key: (
                    orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode()
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in row.items()
                if key != "artifact_paths"
            }
        )
    return pl.DataFrame(flattened, infer_schema_length=None).sort("family", "ordinal")


def _group_trade_diagnostic(
    frame: pl.DataFrame,
    *,
    experiment_id: str,
    family: str,
    diagnostic: str,
    group_column: str,
    value_column: str | None = None,
) -> list[dict[str, Any]]:
    aggregations = [
        pl.len().alias("observations"),
        pl.col("gross_return").sum().alias("gross_pnl"),
        pl.col("net_return").sum().alias("net_pnl"),
    ]
    if value_column is not None:
        aggregations.append(pl.col(value_column).mean().alias("mean_value"))
    grouped = frame.group_by(group_column).agg(*aggregations).sort(group_column)
    return [
        {
            "experiment_id": experiment_id,
            "family": family,
            "diagnostic": diagnostic,
            "group": str(row[group_column]),
            "observations": int(row["observations"]),
            "gross_pnl": float(row["gross_pnl"]),
            "net_pnl": float(row["net_pnl"]),
            "mean_value": (float(row["mean_value"]) if value_column is not None else None),
        }
        for row in grouped.iter_rows(named=True)
    ]


def _strategy_diagnostics(
    candidates: Sequence[_ResolvedCandidate], *, data_root: Path
) -> pl.DataFrame:
    first = candidates[0]
    dataset_path = (
        data_root.resolve()
        / "gold"
        / "binance"
        / "usdm"
        / "research_dataset"
        / f"version={first.spec.dataset_reference.dataset_id[:16]}"
        / "dataset.parquet"
    )
    if not dataset_path.is_file():
        raise ResearchError(f"Wave 1 diagnostic dataset is missing: {dataset_path}")
    feature_columns = (
        "execution_time_ms",
        "symbol",
        "quarter_open_flow_z_168h",
        "quarter_flow_excess_z_168h",
        "signed_taker_flow_z_168h",
        "flow_price_agreement_1h",
        "market_volatility_regime",
    )
    features = pl.read_parquet(dataset_path, columns=list(feature_columns))
    output: list[dict[str, Any]] = []
    for candidate in candidates:
        family = FAMILY_BY_STRATEGY[candidate.spec.strategy.component_id]
        trades = (
            pl.read_parquet(candidate.artifacts["trade_events"])
            .rename({"entry_time_ms": "execution_time_ms"})
            .join(features, on=["execution_time_ms", "symbol"], how="left")
        )
        if trades.is_empty():
            continue
        output.extend(
            _group_trade_diagnostic(
                trades,
                experiment_id=candidate.experiment_id,
                family=family,
                diagnostic="trigger_count_and_pnl_by_symbol",
                group_column="symbol",
            )
        )
        if family == "quarter_hour_flow":
            enriched = trades.with_columns(
                pl.when(pl.col("quarter_flow_excess_z_168h") >= 0)
                .then(pl.lit("positive"))
                .otherwise(pl.lit("negative"))
                .alias("flow_sign")
            )
            output.extend(
                _group_trade_diagnostic(
                    enriched,
                    experiment_id=candidate.experiment_id,
                    family=family,
                    diagnostic="pnl_by_quarter_excess_flow_sign",
                    group_column="flow_sign",
                    value_column="quarter_flow_excess_z_168h",
                )
            )
            correlation = enriched.select(
                pl.corr("quarter_open_flow_z_168h", "quarter_flow_excess_z_168h")
            ).item()
            output.append(
                {
                    "experiment_id": candidate.experiment_id,
                    "family": family,
                    "diagnostic": "generic_vs_excess_flow_correlation",
                    "group": "all_entries",
                    "observations": enriched.height,
                    "gross_pnl": float(enriched["gross_return"].sum()),
                    "net_pnl": float(enriched["net_return"].sum()),
                    "mean_value": float(correlation) if correlation is not None else None,
                }
            )
        elif family == "flow_absorption_reversal":
            row_count = trades.height
            enriched = trades.with_columns(
                (
                    pl.col("signed_taker_flow_z_168h")
                    .abs()
                    .rank(method="ordinal")
                    .mul(10)
                    .truediv(row_count)
                    .ceil()
                    .clip(1, 10)
                    .cast(pl.Int64)
                    .cast(pl.String)
                ).alias("absolute_flow_decile"),
                pl.when(pl.col("signed_taker_flow_z_168h") >= 0)
                .then(pl.lit("positive"))
                .otherwise(pl.lit("negative"))
                .alias("flow_sign"),
                pl.when(pl.col("flow_price_agreement_1h") > 0)
                .then(pl.lit("agreement"))
                .otherwise(pl.lit("disagreement"))
                .alias("agreement_state"),
            )
            for diagnostic, group_column, value_column in (
                ("pnl_by_absolute_flow_decile", "absolute_flow_decile", "signed_taker_flow_z_168h"),
                ("pnl_by_flow_sign", "flow_sign", "signed_taker_flow_z_168h"),
                ("pnl_by_flow_price_agreement", "agreement_state", "flow_price_agreement_1h"),
            ):
                output.extend(
                    _group_trade_diagnostic(
                        enriched,
                        experiment_id=candidate.experiment_id,
                        family=family,
                        diagnostic=diagnostic,
                        group_column=group_column,
                        value_column=value_column,
                    )
                )
        elif family == "volatility_compression_breakout":
            low = cast(float, trades["market_volatility_regime"].quantile(1 / 3))
            high = cast(float, trades["market_volatility_regime"].quantile(2 / 3))
            enriched = trades.with_columns(
                pl.when(pl.col("market_volatility_regime") <= low)
                .then(pl.lit("low"))
                .when(pl.col("market_volatility_regime") <= high)
                .then(pl.lit("middle"))
                .otherwise(pl.lit("high"))
                .alias("volatility_regime")
            )
            output.extend(
                _group_trade_diagnostic(
                    enriched,
                    experiment_id=candidate.experiment_id,
                    family=family,
                    diagnostic="pnl_by_entry_volatility_regime",
                    group_column="volatility_regime",
                    value_column="market_volatility_regime",
                )
            )
        elif family == "pair_spread_reversion":
            fits = pl.read_parquet(candidate.artifacts["pair_fit_metrics"])
            pnl = pl.read_parquet(candidate.artifacts["pair_pnl_metrics"])
            for pair_id in sorted(str(value) for value in fits["pair_id"].unique().to_list()):
                pair_fits = fits.filter(pl.col("pair_id") == pair_id)
                pair_pnl = pnl.filter(pl.col("pair_id") == pair_id).sort("fold")
                cumulative = np.cumsum(np.asarray(pair_pnl["net_pnl"].to_numpy(), dtype=np.float64))
                drawdown = cumulative - np.maximum.accumulate(cumulative)
                disabled = int((~pair_fits["eligible"]).sum())
                for diagnostic, value in (
                    ("pair_beta_median", float(cast(float, pair_fits["beta"].median()))),
                    (
                        "pair_half_life_median_hours",
                        float(cast(float, pair_fits["half_life_hours"].median())),
                    ),
                    ("pair_disabled_folds", float(disabled)),
                    ("pair_episode_drawdown", float(np.min(drawdown))),
                ):
                    output.append(
                        {
                            "experiment_id": candidate.experiment_id,
                            "family": family,
                            "diagnostic": diagnostic,
                            "group": pair_id,
                            "observations": pair_fits.height,
                            "gross_pnl": float(
                                (pair_pnl["price_pnl"] + pair_pnl["funding_pnl"]).sum()
                            ),
                            "net_pnl": float(pair_pnl["net_pnl"].sum()),
                            "mean_value": value,
                        }
                    )
    return pl.DataFrame(
        output,
        schema={
            "experiment_id": pl.String,
            "family": pl.String,
            "diagnostic": pl.String,
            "group": pl.String,
            "observations": pl.Int64,
            "gross_pnl": pl.Float64,
            "net_pnl": pl.Float64,
            "mean_value": pl.Float64,
        },
    ).sort("family", "experiment_id", "diagnostic", "group")


def _family_recommendations(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    recommendations: dict[str, str] = {}
    for family in sorted(set(str(row["family"]) for row in rows)):
        selected = [row for row in rows if row["family"] == family]
        if any(bool(row.get("diversifier_pass")) for row in selected):
            recommendations[family] = "KEEP_AS_DIVERSIFIER"
        elif family in {"quarter_hour_flow", "flow_absorption_reversal"} and all(
            float(row.get("gross_pnl", 0.0)) > 0 for row in selected
        ):
            recommendations[family] = "CONTINUE_TO_PREMIUM_DATA"
        elif family == "pair_spread_reversion":
            recommendations[family] = "CONTINUE_TO_RELATIVE_VALUE"
        else:
            recommendations[family] = "REJECT_FAMILY"
    return recommendations


def _atomic_bytes(path: Path, payload: bytes) -> None:
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
        raise ResearchError(f"cannot update Wave 1 report {path}: {exc}") from exc


def _atomic_parquet(path: Path, frame: pl.DataFrame, *, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{uuid.uuid4().hex}.tmp")
    try:
        frame.write_parquet(temporary, compression=cast(Any, compression))
        validated = pl.read_parquet(temporary)
        if validated.shape != frame.shape or validated.columns != frame.columns:
            raise ResearchError(f"Wave 1 Parquet validation failed: {path.name}")
        os.replace(temporary, path)
    except (OSError, pl.exceptions.PolarsError):
        if temporary.exists():
            temporary.unlink()
        raise


def _report_markdown(payload: Mapping[str, Any]) -> str:
    champion = cast(dict[str, Any], payload["champion"])
    candidates = cast(list[dict[str, Any]], payload["candidates"])
    recommendations = cast(dict[str, str], payload["family_recommendations"])
    lines = [
        "# Alpha Research Reboot — Wave 1",
        "",
        f"> {DEVELOPMENT_SEEN_BANNER}",
        "",
        "## Outcome",
        "",
        f"All {payload['trial_count']} preregistered economic variants completed. No winner was "
        "selected automatically.",
        "",
        f"The strict benchmark is run `{champion['run_id']}` (`{champion['strategy']}` + "
        f"`{champion['portfolio']}` at 60/30/10).",
        "",
        "Daily return and position correlation to that benchmark are unavailable because its "
        "final 728-day evaluation begins after the Wave 1 development_seen candidate window. "
        "The diversifier correlation and fixed-blend gates therefore fail closed; no proxy run "
        "was silently substituted.",
        "",
        "## Candidates",
        "",
        "| Family | Trial | Return | Sharpe | Gross pnl | Cost | Trades | DSR | Classification |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in candidates:
        lines.append(
            f"| {row['family']} | {row['ordinal']} | {float(row['net_total_return']):.2%} | "
            f"{float(row['net_sharpe']):.3f} | {float(row['gross_pnl']):.4f} | "
            f"{float(row['explicit_cost']):.4f} | {row['completed_trades']} | "
            f"{float(row['dsr_probability']):.3f} | {row['classification']} |"
        )
    lines.extend(["", "## Gate buckets", ""])
    for classification in (
        "standalone pass",
        "diversifier pass",
        "gross signal but not executable",
        "rejected",
        "invalid/incomplete",
    ):
        count = sum(row["classification"] == classification for row in candidates)
        lines.append(f"- {classification}: {count}")
    lines.extend(["", "## Family recommendations", ""])
    lines.extend(
        f"- `{family}`: `{recommendation}`" for family, recommendation in recommendations.items()
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Quarter-hour flow showed positive gross P&L across its closed grid, but its edge per "
            "turnover was below explicit taker execution cost and every net/stress result was "
            "negative. It is a statistical measurement candidate for aggTrades/premium data, "
            "not executable alpha in the current engine.",
            "",
            "The absorption, compression-breakout and fold-frozen pair screens did not clear "
            "standalone or diversification gates. The pair result argues against this directional "
            "alt/BTC convergence specification; it does not test structural cash-and-carry or "
            "calendar basis spreads.",
            "",
            "Proceed only to the already specified premium/aggTrades data plane and structural "
            "relative-value work. Do not create technical-signal ensembles from this Wave 1.",
            "",
            "## Artifact notes",
            "",
            "- All campaign artifacts remain immutable and registered in ResearchStore.",
            "- The first quarter-hour campaign attempt contains six infrastructure failures from "
            "a dataset-view contract mismatch. Its corrected preregistered retry contains the six "
            "economic variants counted here; infrastructure failures are disclosed but are not "
            "additional DSR trials.",
            "- Pair P&L attribution is diagnostic because two spreads share the BTC leg and are "
            "netted before portfolio-level costs.",
            "- `strategy_diagnostics.parquet` contains the preregistered flow, breakout and pair "
            "diagnostic segmentations.",
        ]
    )
    return "\n".join(lines) + "\n"


def _report_html(payload: Mapping[str, Any]) -> str:
    candidates = cast(list[dict[str, Any]], payload["candidates"])
    rows = "".join(
        "<tr>"
        f"<td>{escape(str(row['family']))}</td>"
        f"<td>{escape(str(row['ordinal']))}</td>"
        f"<td>{float(row['net_total_return']):.4%}</td>"
        f"<td>{float(row['net_sharpe']):.4f}</td>"
        f"<td>{escape(str(row['classification']))}</td>"
        "</tr>"
        for row in candidates
    )
    return (
        '<!doctype html><html><head><meta charset="utf-8"><title>Alpha Reboot Wave 1</title>'
        "<style>body{font-family:system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        ".banner{padding:1rem;background:#fff1c7;border-left:5px solid #c97800}"
        "table{border-collapse:collapse;width:100%}th,td{padding:.5rem;border:1px solid #ddd;"
        "text-align:left}</style></head><body><h1>Alpha Research Reboot — Wave 1</h1>"
        f'<p class="banner">{escape(DEVELOPMENT_SEEN_BANNER)}</p>'
        f"<p>Completed trials: {payload['trial_count']}. No automatic winner selection.</p>"
        "<table><thead><tr><th>Family</th><th>Trial</th><th>Return</th><th>Sharpe</th>"
        f"<th>Classification</th></tr></thead><tbody>{rows}</tbody></table></body></html>"
    )


def build_alpha_reboot_wave1_report(
    *,
    store: ResearchStore,
    config_path: Path,
    project_root: Path,
    data_root: Path,
    reports_root: Path,
    compression: str,
) -> AlphaRebootReportResult:
    config = load_alpha_reboot_wave_config(config_path)
    champion_run, champion_spec = resolve_champion_run(store, config.champion)
    champion_artifacts = _artifact_paths(store, data_root=data_root, run_id=champion_run.run_id)
    if "oos_curve" not in champion_artifacts:
        raise ResearchError("champion run has no oos_curve artifact")
    champion_curve = pl.read_parquet(champion_artifacts["oos_curve"])
    champion_daily = _daily_returns(champion_curve, series_id="champion", source="final_728d")
    champion_positions = _daily_positions_from_curve(
        champion_curve, series_id="champion", source="final_728d"
    )

    campaign_records: list[CampaignRecord] = []
    planned_trials = 0
    for relative in config.campaigns:
        name, max_trials = _read_campaign_name((project_root / relative).resolve())
        planned_trials += max_trials
        campaign = store.get_campaign(name)
        if campaign is None:
            raise ResearchError(f"Wave 1 campaign is not registered: {name}")
        if campaign.status.value != "COMPLETED":
            raise ResearchError(
                f"Wave 1 campaign is not complete: {name} ({campaign.status.value})"
            )
        campaign_records.append(campaign)
    if planned_trials != EXPECTED_TRIAL_COUNT or planned_trials > config.maximum_total_trials:
        raise ResearchError(
            f"Wave 1 trial budget mismatch: planned={planned_trials}, "
            f"configured_maximum={config.maximum_total_trials}"
        )

    resolved: list[_ResolvedCandidate] = []
    for campaign in campaign_records:
        for ordinal, experiment_id, tags in store.list_campaign_experiments(campaign.campaign_id):
            run = store.latest_successful_run(experiment_id)
            spec = store.get_experiment(experiment_id)
            if run is None or spec is None:
                raise ResearchError(
                    f"Wave 1 trial has no successful immutable run: {experiment_id}"
                )
            if spec.strategy.component_id not in FAMILY_BY_STRATEGY:
                raise ResearchError(f"unexpected Wave 1 strategy: {spec.strategy.component_id}")
            artifacts = _artifact_paths(store, data_root=data_root, run_id=run.run_id)
            if "oos_curve" not in artifacts or "daily_positions" not in artifacts:
                raise ResearchError(
                    f"Wave 1 trial lacks aligned return/position artifacts: {run.run_id}"
                )
            curve = pl.read_parquet(artifacts["oos_curve"])
            resolved.append(
                _ResolvedCandidate(
                    campaign=campaign,
                    ordinal=ordinal,
                    experiment_id=experiment_id,
                    run=run,
                    spec=spec,
                    tags=tags,
                    artifacts=artifacts,
                    curve=curve,
                    daily_returns=_daily_returns(
                        curve, series_id=experiment_id, source="development_seen"
                    ),
                    daily_positions=_candidate_daily_positions(
                        pl.read_parquet(artifacts["daily_positions"]),
                        series_id=experiment_id,
                        source="development_seen",
                    ),
                )
            )
    if len(resolved) != EXPECTED_TRIAL_COUNT:
        raise ResearchError(f"Wave 1 must contain exactly {EXPECTED_TRIAL_COUNT} successful trials")

    daily_frames = {candidate.experiment_id: candidate.daily_returns for candidate in resolved}
    daily_frames["champion"] = champion_daily
    position_frames = {candidate.experiment_id: candidate.daily_positions for candidate in resolved}
    position_frames["champion"] = champion_positions
    daily_correlations = _correlation_table(
        daily_frames, value_column="net_return", keys=("date_utc",)
    )
    position_correlations = _correlation_table(
        position_frames,
        value_column="mean_target_weight",
        keys=("date_utc", "symbol"),
    )
    trial_sharpes = np.asarray(
        [
            float(_load_metrics(candidate.artifacts["metrics"])["metrics"]["sharpe"])
            for candidate in resolved
        ],
        dtype=np.float64,
    )
    candidate_rows = [
        _evaluate_candidate(
            candidate,
            champion_daily=champion_daily,
            daily_correlations=daily_correlations,
            trial_sharpes=trial_sharpes,
        )
        for candidate in resolved
    ]
    recommendations = _family_recommendations(candidate_rows)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "wave_id": config.wave_id,
        "research_context": config.research_context,
        "research_banner": DEVELOPMENT_SEEN_BANNER,
        "trial_count": len(candidate_rows),
        "dsr_trial_count": len(trial_sharpes),
        "automatic_winner_selection": False,
        "champion": {
            "run_id": champion_run.run_id,
            "experiment_id": champion_run.experiment_id,
            "strategy": f"{champion_spec.strategy.component_id}:v{champion_spec.strategy.version}",
            "portfolio": (
                f"{champion_spec.portfolio_policy.component_id}:v"
                f"{champion_spec.portfolio_policy.version}"
            ),
            "portfolio_weights": config.champion.portfolio_weights,
            "source": "explicit configured run_id; documented final 728-day run",
        },
        "correlation_and_blend_note": (
            "Champion and Wave 1 candidates have no common evaluation dates; correlation and "
            "blend gates fail closed, and no development proxy champion was substituted."
        ),
        "campaigns": [
            {
                "name": campaign.name,
                "campaign_id": campaign.campaign_id,
                "trial_count": campaign.trial_count,
                "status": campaign.status.value,
            }
            for campaign in campaign_records
        ],
        "candidates": candidate_rows,
        "classification_counts": {
            classification: sum(row["classification"] == classification for row in candidate_rows)
            for classification in (
                "standalone pass",
                "diversifier pass",
                "gross signal but not executable",
                "rejected",
                "invalid/incomplete",
            )
        },
        "family_recommendations": recommendations,
        "program_recommendations": sorted(
            {
                recommendation
                for recommendation in recommendations.values()
                if recommendation.startswith("CONTINUE_TO_")
            }
        ),
        "known_infrastructure_retry": {
            "campaign_name": "quarter_hour_flow_v1_development_seen",
            "failed_trials": 6,
            "reason": "dataset-view contract rejected newly registered schema-v3 features",
            "economic_trial_counted": False,
            "corrected_campaign_name": "quarter_hour_flow_v1_development_seen_r1",
        },
    }
    directory = reports_root.resolve() / "alpha_reboot_wave1"
    report_md = directory / "report.md"
    report_json = directory / "report.json"
    report_html = directory / "report.html"
    candidates_path = directory / "candidates.parquet"
    daily_correlation_path = directory / "daily_return_correlation.parquet"
    position_correlation_path = directory / "position_correlation.parquet"
    hourly_returns_path = directory / "hourly_returns.parquet"
    daily_returns_path = directory / "daily_returns.parquet"
    daily_positions_path = directory / "daily_positions.parquet"
    aligned_daily_path = directory / "aligned_daily_returns.parquet"
    strategy_diagnostics_path = directory / "strategy_diagnostics.parquet"
    payload["consolidated_artifacts"] = {
        "candidates": candidates_path.relative_to(project_root).as_posix(),
        "daily_return_correlation": daily_correlation_path.relative_to(project_root).as_posix(),
        "position_correlation": position_correlation_path.relative_to(project_root).as_posix(),
        "hourly_returns": hourly_returns_path.relative_to(project_root).as_posix(),
        "daily_returns": daily_returns_path.relative_to(project_root).as_posix(),
        "daily_positions": daily_positions_path.relative_to(project_root).as_posix(),
        "aligned_daily_returns": aligned_daily_path.relative_to(project_root).as_posix(),
        "strategy_diagnostics": strategy_diagnostics_path.relative_to(project_root).as_posix(),
    }

    hourly_returns = pl.concat(
        [
            champion_curve.select("decision_time_ms", "net_return").with_columns(
                pl.lit("champion").alias("series_id"),
                pl.lit("final_728d").alias("source"),
            ),
            *[
                candidate.curve.select("decision_time_ms", "net_return").with_columns(
                    pl.lit(candidate.experiment_id).alias("series_id"),
                    pl.lit("development_seen").alias("source"),
                )
                for candidate in resolved
            ],
        ],
        how="vertical_relaxed",
    ).select("series_id", "source", "decision_time_ms", "net_return")
    all_daily = pl.concat(list(daily_frames.values()), how="vertical_relaxed")
    all_positions = pl.concat(list(position_frames.values()), how="vertical_relaxed")
    aligned_rows = [
        champion_daily.select("date_utc", pl.col("net_return").alias("champion_return"))
        .join(
            candidate.daily_returns.select(
                "date_utc", pl.col("net_return").alias("candidate_return")
            ),
            on="date_utc",
            how="inner",
        )
        .with_columns(pl.lit(candidate.experiment_id).alias("candidate_experiment_id"))
        for candidate in resolved
    ]
    aligned_daily = pl.concat(aligned_rows, how="vertical_relaxed").select(
        "candidate_experiment_id", "date_utc", "champion_return", "candidate_return"
    )
    strategy_diagnostics = _strategy_diagnostics(resolved, data_root=data_root)

    _atomic_parquet(candidates_path, _candidate_frame(candidate_rows), compression=compression)
    _atomic_parquet(daily_correlation_path, daily_correlations, compression=compression)
    _atomic_parquet(position_correlation_path, position_correlations, compression=compression)
    _atomic_parquet(hourly_returns_path, hourly_returns, compression=compression)
    _atomic_parquet(daily_returns_path, all_daily, compression=compression)
    _atomic_parquet(daily_positions_path, all_positions, compression=compression)
    _atomic_parquet(aligned_daily_path, aligned_daily, compression=compression)
    _atomic_parquet(strategy_diagnostics_path, strategy_diagnostics, compression=compression)
    _atomic_bytes(report_json, orjson.dumps(payload, option=orjson.OPT_SORT_KEYS) + b"\n")
    _atomic_bytes(report_md, _report_markdown(payload).encode("utf-8"))
    _atomic_bytes(report_html, _report_html(payload).encode("utf-8"))
    return AlphaRebootReportResult(
        report_directory=directory,
        report_markdown_path=report_md,
        report_json_path=report_json,
        report_html_path=report_html,
        candidates_path=candidates_path,
        daily_return_correlation_path=daily_correlation_path,
        position_correlation_path=position_correlation_path,
        hourly_returns_path=hourly_returns_path,
        daily_returns_path=daily_returns_path,
        daily_positions_path=daily_positions_path,
        aligned_daily_returns_path=aligned_daily_path,
        strategy_diagnostics_path=strategy_diagnostics_path,
        champion_run_id=champion_run.run_id,
        trial_count=len(candidate_rows),
    )


__all__ = [
    "AlphaRebootReportResult",
    "AlphaRebootWaveConfig",
    "ChampionConfig",
    "build_alpha_reboot_wave1_report",
    "load_alpha_reboot_wave_config",
    "resolve_champion_run",
    "select_champion_run_id",
]
