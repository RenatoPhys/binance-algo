"""Deterministic aggregate comparison for every trial in a research campaign."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import orjson
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.experiments.models import MetricScope
from binance_algo.research.experiments.store import CampaignRecord, ResearchStore


@dataclass(frozen=True, slots=True)
class CampaignComparisonResult:
    campaign_id: str
    comparison_path: Path
    report_json_path: Path
    report_markdown_path: Path
    trial_count: int
    succeeded_count: int
    failed_count: int


def _atomic_replace_bytes(path: Path, payload: bytes) -> None:
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
        raise ResearchError(f"cannot update campaign report {path}: {exc}") from exc


def _atomic_replace_parquet(path: Path, frame: pl.DataFrame, *, compression: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{uuid.uuid4().hex}.tmp")
    try:
        frame.write_parquet(temporary, compression=cast(Any, compression))
        validated = pl.read_parquet(temporary)
        if validated.shape != frame.shape or validated.columns != frame.columns:
            raise ResearchError("campaign comparison validation failed")
        os.replace(temporary, path)
    except (OSError, pl.exceptions.PolarsError) as exc:
        if temporary.exists():
            temporary.unlink()
        raise ResearchError(f"cannot update campaign comparison {path}: {exc}") from exc


def build_campaign_comparison(store: ResearchStore, campaign: CampaignRecord) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for ordinal, identifier, tags in store.list_campaign_experiments(campaign.campaign_id):
        runs = store.list_runs(experiment_id_value=identifier)
        latest = runs[-1] if runs else None
        succeeded = store.latest_successful_run(identifier)
        metrics = (
            {
                metric.metric_name: metric.metric_value
                for metric in store.list_metrics(succeeded.run_id)
                if metric.scope is MetricScope.TEST
                and metric.fold is None
                and metric.regime is None
            }
            if succeeded is not None
            else {}
        )
        stress = (
            {
                (metric.regime, metric.metric_name): metric.metric_value
                for metric in store.list_metrics(succeeded.run_id)
                if metric.scope is MetricScope.STRESS
            }
            if succeeded is not None
            else {}
        )
        rows.append(
            {
                "ordinal": ordinal,
                "experiment_id": identifier,
                "strategy_parameters": orjson.dumps(
                    tags.get("strategy_parameters", {}), option=orjson.OPT_SORT_KEYS
                ).decode(),
                "portfolio_parameters": orjson.dumps(
                    tags.get("portfolio_parameters", {}), option=orjson.OPT_SORT_KEYS
                ).decode(),
                "total_return": metrics.get("total_return"),
                "sharpe": metrics.get("sharpe"),
                "max_drawdown": metrics.get("max_drawdown"),
                "turnover": metrics.get("turnover"),
                "explicit_cost": (
                    metrics.get("trading_fees", 0.0)
                    + metrics.get("spread_cost", 0.0)
                    + metrics.get("slippage_cost", 0.0)
                    if metrics
                    else None
                ),
                "rank_ic": metrics.get("mean_cross_sectional_rank_ic"),
                "cost_1_5x_return": stress.get(("cost_1_5x", "total_return")),
                "delay_1_bar_return": stress.get(("signal_delay_1_bar", "total_return")),
                "status": latest.status.value if latest is not None else "NOT_RUN",
                "attempts": len(runs),
                "result_digest": succeeded.result_digest if succeeded is not None else None,
            }
        )
    frame = pl.DataFrame(rows).sort("ordinal")
    successful = frame.filter(pl.col("sharpe").is_not_null()).sort("sharpe", descending=True)
    ranks = {
        str(identifier): rank
        for rank, identifier in enumerate(successful["experiment_id"].to_list(), start=1)
    }
    return frame.with_columns(
        pl.col("experiment_id")
        .replace_strict(ranks, default=None, return_dtype=pl.Int64)
        .alias("rank")
    ).select("rank", *frame.columns)


def write_campaign_comparison(
    *,
    store: ResearchStore,
    campaign: CampaignRecord,
    reports_root: Path,
    compression: str,
) -> CampaignComparisonResult:
    frame = build_campaign_comparison(store, campaign)
    directory = (
        reports_root.resolve() / "research_campaigns" / f"campaign_id={campaign.campaign_id[:24]}"
    )
    comparison_path = directory / "comparison.parquet"
    report_json_path = directory / "report.json"
    report_markdown_path = directory / "report.md"
    succeeded = frame.filter(pl.col("status") == "SUCCEEDED").height
    failed = frame.filter(pl.col("status").is_in(["FAILED", "CANCELLED", "STALE"])).height
    payload = {
        "campaign_id": campaign.campaign_id,
        "name": campaign.name,
        "status": campaign.status.value,
        "trial_count": frame.height,
        "succeeded_count": succeeded,
        "failed_count": failed,
        "trials": frame.to_dicts(),
    }
    _atomic_replace_parquet(comparison_path, frame, compression=compression)
    _atomic_replace_bytes(
        report_json_path,
        orjson.dumps(payload, option=orjson.OPT_SORT_KEYS) + b"\n",
    )
    lines = [
        f"# Campaign — {campaign.name}",
        "",
        f"- Campaign ID: `{campaign.campaign_id}`",
        f"- Status: {campaign.status.value}",
        f"- Trials / succeeded / failed: {frame.height} / {succeeded} / {failed}",
        "",
        "> Every valid trial is preserved. The highest Sharpe is not independent OOS evidence.",
        "",
        "| Rank | Experiment | Return | Sharpe | Turnover | Status |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for row in frame.to_dicts():
        rank = "-" if row["rank"] is None else str(row["rank"])
        total_return = "-" if row["total_return"] is None else f"{row['total_return']:.4%}"
        sharpe = "-" if row["sharpe"] is None else f"{row['sharpe']:.3f}"
        turnover = "-" if row["turnover"] is None else f"{row['turnover']:.3f}"
        lines.append(
            f"| {rank} | `{str(row['experiment_id'])[:16]}` | {total_return} | "
            f"{sharpe} | {turnover} | {row['status']} |"
        )
    _atomic_replace_bytes(report_markdown_path, ("\n".join(lines) + "\n").encode())
    return CampaignComparisonResult(
        campaign_id=campaign.campaign_id,
        comparison_path=comparison_path,
        report_json_path=report_json_path,
        report_markdown_path=report_markdown_path,
        trial_count=frame.height,
        succeeded_count=succeeded,
        failed_count=failed,
    )


__all__ = [
    "CampaignComparisonResult",
    "build_campaign_comparison",
    "write_campaign_comparison",
]
