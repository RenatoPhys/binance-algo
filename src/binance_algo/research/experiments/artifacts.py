"""Atomic, immutable artifact bundles for registered experiment runs."""

from __future__ import annotations

import hashlib
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import orjson
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.data.storage import LocalFilesystemStorage
from binance_algo.research.backtest import BacktestRun, calculate_metrics
from binance_algo.research.experiments.canonical import canonical_json
from binance_algo.research.experiments.models import ArtifactPolicy, ExperimentSpec, MetricScope
from binance_algo.research.experiments.store import (
    ResearchArtifactRecord,
    ResearchMetricRecord,
)
from binance_algo.research.trades import (
    build_trade_metrics,
    daily_positions,
    pair_diagnostics,
    reconstruct_trade_events,
)
from binance_algo.research.visualization import render_pnl_svg

ARTIFACT_SCHEMA_VERSION = 2
DEVELOPMENT_SEEN_BANNER = (
    "EXPLORATORY / DEVELOPMENT_SEEN — the historical window has already informed prior "
    "research decisions and is not an independent lockbox."
)
LEGACY_FULL_POSITION_COLUMNS = (
    "fold",
    "decision_time_ms",
    "execution_time_ms",
    "symbol",
    "previous_weight",
    "target_weight",
    "trade_weight",
    "gross_contribution",
    "net_contribution",
    "beta_contribution",
    "future_return",
    "funding_rate",
    "price_pnl",
    "funding_pnl",
    "allocated_fee",
    "allocated_spread_cost",
    "allocated_slippage_cost",
)


@dataclass(frozen=True, slots=True)
class ArtifactBundle:
    final_directory: Path
    artifacts: tuple[ResearchArtifactRecord, ...]
    artifact_checksums: Mapping[str, str]
    metrics_payload: Mapping[str, Any]
    metric_records: tuple[ResearchMetricRecord, ...]


@dataclass(frozen=True, slots=True)
class ArtifactVerification:
    run_id: str
    valid: bool
    checked_files: int
    issues: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _performance_payload(frame: pl.DataFrame) -> dict[str, float | int]:
    return asdict(calculate_metrics(frame))


def _fold_metrics(run: BacktestRun) -> pl.DataFrame:
    rows = []
    for fold in run.folds:
        frame = run.curve.filter(pl.col("fold") == fold.fold)
        rows.append({"fold": fold.fold, **_performance_payload(frame)})
    return pl.DataFrame(rows).sort("fold")


def _regime_metrics(run: BacktestRun) -> pl.DataFrame:
    values = run.curve["market_volatility_regime"]
    low_value = values.quantile(1 / 3, interpolation="linear")
    high_value = values.quantile(2 / 3, interpolation="linear")
    if low_value is None or high_value is None:
        raise ResearchError("cannot segment an empty volatility-regime series")
    low = float(low_value)
    high = float(high_value)
    selections = {
        "low": pl.col("market_volatility_regime") <= low,
        "middle": (pl.col("market_volatility_regime") > low)
        & (pl.col("market_volatility_regime") <= high),
        "high": pl.col("market_volatility_regime") > high,
    }
    rows = [
        {"regime": regime, **_performance_payload(run.curve.filter(predicate))}
        for regime, predicate in selections.items()
    ]
    return pl.DataFrame(rows).sort("regime")


def _monthly_metrics(run: BacktestRun) -> pl.DataFrame:
    months = sorted(
        {
            datetime.fromtimestamp(value / 1_000, tz=UTC).strftime("%Y-%m")
            for value in run.curve["decision_time_ms"].to_list()
        }
    )
    month_expression = pl.from_epoch("decision_time_ms", time_unit="ms").dt.strftime("%Y-%m")
    with_month = run.curve.with_columns(month_expression.alias("month"))
    rows = [
        {"month": month, **_performance_payload(with_month.filter(pl.col("month") == month))}
        for month in months
    ]
    return pl.DataFrame(rows).sort("month")


def _symbol_metrics(run: BacktestRun) -> pl.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for symbol in sorted(str(value) for value in run.positions["symbol"].unique().to_list()):
        frame = run.positions.filter(pl.col("symbol") == symbol)
        maximum_weight = frame["target_weight"].abs().max()
        if not isinstance(maximum_weight, (int, float)):
            raise ResearchError(f"cannot calculate symbol metrics for {symbol}")
        rows.append(
            {
                "symbol": symbol,
                "periods": frame.height,
                "price_pnl": float(frame["price_pnl"].sum()),
                "funding_pnl": float(frame["funding_pnl"].sum()),
                "trading_fees": float(frame["allocated_fee"].sum()),
                "spread_cost": float(frame["allocated_spread_cost"].sum()),
                "slippage_cost": float(frame["allocated_slippage_cost"].sum()),
                "net_pnl": float(frame["net_contribution"].sum()),
                "turnover": float(frame["trade_weight"].sum()),
                "maximum_symbol_weight": float(maximum_weight),
            }
        )
    return pl.DataFrame(rows).sort("symbol")


def build_metrics_payload(
    run: BacktestRun,
    *,
    stress: Mapping[str, Mapping[str, float | int]],
    bootstrap: Mapping[str, float | int],
    strategy_id: str = "unknown_strategy",
) -> tuple[dict[str, Any], dict[str, pl.DataFrame]]:
    trade_events = reconstruct_trade_events(run.positions, strategy_id=strategy_id)
    trade_metrics = build_trade_metrics(trade_events, run.positions)
    pair_fit_metrics, pair_pnl_metrics = pair_diagnostics(run.positions)
    frames = {
        "fold_metrics": _fold_metrics(run),
        "regime_metrics": _regime_metrics(run),
        "monthly_metrics": _monthly_metrics(run),
        "symbol_metrics": _symbol_metrics(run),
        "trade_events": trade_events,
        "trade_metrics": trade_metrics,
        "daily_positions": daily_positions(run.positions),
    }
    if not pair_fit_metrics.is_empty():
        frames["pair_fit_metrics"] = pair_fit_metrics
        frames["pair_pnl_metrics"] = pair_pnl_metrics
    payload: dict[str, Any] = {
        "metrics": asdict(run.metrics),
        "folds": [asdict(fold) for fold in run.folds],
        "stress": {name: dict(values) for name, values in sorted(stress.items())},
        "bootstrap": dict(bootstrap),
        **{name: frame.to_dicts() for name, frame in frames.items()},
    }
    return payload, frames


def _metric_records(payload: Mapping[str, Any]) -> tuple[ResearchMetricRecord, ...]:
    records: list[ResearchMetricRecord] = []
    for name, value in payload["metrics"].items():
        records.append(
            ResearchMetricRecord(
                scope=MetricScope.TEST,
                metric_name=str(name),
                metric_value=float(value),
            )
        )
    for row in payload["fold_metrics"]:
        for name, value in row.items():
            if name != "fold":
                records.append(
                    ResearchMetricRecord(
                        scope=MetricScope.TEST,
                        fold=int(row["fold"]),
                        metric_name=str(name),
                        metric_value=float(value),
                    )
                )
    for row in payload["regime_metrics"]:
        for name, value in row.items():
            if name != "regime":
                records.append(
                    ResearchMetricRecord(
                        scope=MetricScope.TEST,
                        regime=str(row["regime"]),
                        metric_name=str(name),
                        metric_value=float(value),
                    )
                )
    for scenario, values in payload["stress"].items():
        for name, value in values.items():
            records.append(
                ResearchMetricRecord(
                    scope=MetricScope.STRESS,
                    regime=str(scenario),
                    metric_name=str(name),
                    metric_value=float(value),
                )
            )
    return tuple(records)


def experiment_report_markdown(
    *,
    experiment_id: str,
    run: BacktestRun,
    stress: Mapping[str, Mapping[str, float | int]],
    development_seen: bool = False,
) -> str:
    metrics = run.metrics
    lines = [
        "# Registered research experiment",
        "",
        f"Experiment: `{experiment_id}`",
        "",
        *([f"> {DEVELOPMENT_SEEN_BANNER}", ""] if development_seen else []),
        "> Offline vectorized screening result; this is not authorization to trade.",
        "",
        "## Out-of-sample summary",
        "",
        f"- Folds / periods: {len(run.folds)} / {metrics.periods}",
        f"- Total return: {metrics.total_return:.6%}",
        f"- Sharpe: {metrics.sharpe:.6f}",
        f"- Maximum drawdown: {metrics.max_drawdown:.6%}",
        f"- Turnover: {metrics.turnover:.6f}",
        "- Explicit cost: "
        f"{(metrics.trading_fees + metrics.spread_cost + metrics.slippage_cost):.8f}",
        f"- Accounting error: {metrics.accounting_error_max:.3e}",
        "",
        "## Stress scenarios",
        "",
        "| Scenario | Total return | Sharpe | Turnover |",
        "|---|---:|---:|---:|",
    ]
    for name, values in sorted(stress.items()):
        lines.append(
            f"| {name} | {float(values['total_return']):.6%} | "
            f"{float(values['sharpe']):.6f} | {float(values['turnover']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Artifact policy",
            "",
            "Summary artifacts always preserve the curve and segmented metrics. "
            "Scores and positions are persisted only under the full policy.",
            "",
        ]
    )
    return "\n".join(lines)


class ExperimentArtifactPipeline:
    def __init__(self, data_root: Path, *, compression: str) -> None:
        self.data_root = data_root.resolve()
        self.compression = compression
        self.storage = LocalFilesystemStorage(self.data_root)

    def persist(
        self,
        *,
        experiment_id: str,
        run_id: str,
        spec: ExperimentSpec,
        run: BacktestRun,
        stress: Mapping[str, Mapping[str, float | int]],
        bootstrap: Mapping[str, float | int],
        generate_chart: bool,
    ) -> ArtifactBundle:
        temp_directory = self.storage.path("tmp", "research", run_id)
        final_directory = self.storage.path(
            "gold",
            "binance",
            "usdm",
            "research_experiments",
            f"experiment_id={experiment_id[:24]}",
            f"run_id={run_id[:24]}",
        )
        if final_directory.exists():
            raise ResearchError(f"completed run artifacts already exist: {final_directory}")
        if temp_directory.exists():
            self.quarantine(temp_directory, run_id=run_id, reason="stale-temp")
        temp_directory.mkdir(parents=True)
        try:
            metrics_payload, metric_frames = build_metrics_payload(
                run,
                stress=stress,
                bootstrap=bootstrap,
                strategy_id=spec.strategy.component_id,
            )
            development_seen = spec.feature_set.feature_set_id == "alpha_reboot_features:v1"
            if development_seen:
                metrics_payload["research_banner"] = DEVELOPMENT_SEEN_BANNER
            json_files: dict[str, object] = {
                "experiment_spec.json": spec.model_dump(mode="json"),
                "metrics.json": metrics_payload,
            }
            for name, value in json_files.items():
                self.storage.write_bytes_atomic(
                    temp_directory / name,
                    canonical_json(value) + b"\n",
                )
            self.storage.write_bytes_atomic(
                temp_directory / "report.md",
                experiment_report_markdown(
                    experiment_id=experiment_id,
                    run=run,
                    stress=stress,
                    development_seen=development_seen,
                ).encode("utf-8"),
            )
            parquet_frames = {
                "oos_curve.parquet": run.curve,
                **{f"{name}.parquet": frame for name, frame in metric_frames.items()},
            }
            if spec.artifact_policy is ArtifactPolicy.FULL:
                parquet_frames.update(
                    {
                        "scores.parquet": run.scores,
                        "positions.parquet": run.positions.select(LEGACY_FULL_POSITION_COLUMNS),
                    }
                )
            for name, frame in parquet_frames.items():
                self.storage.write_parquet_atomic(
                    temp_directory / name,
                    frame,
                    compression=self.compression,
                )
            if generate_chart:
                self.storage.write_bytes_atomic(
                    temp_directory / "pnl.svg",
                    render_pnl_svg(run.curve).encode("utf-8"),
                )
            artifacts = self._descriptors(
                temp_directory=temp_directory,
                final_directory=final_directory,
                row_counts={name: frame.height for name, frame in parquet_frames.items()},
            )
            manifest = {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "experiment_id": experiment_id,
                "artifact_policy": spec.artifact_policy.value,
                "artifacts": [
                    {
                        "artifact_type": artifact.artifact_type,
                        "path": Path(artifact.path).name,
                        "checksum_sha256": artifact.checksum_sha256,
                        "row_count": artifact.row_count,
                        "size_bytes": artifact.size_bytes,
                        "schema_version": artifact.schema_version,
                    }
                    for artifact in artifacts
                ],
            }
            self.storage.write_bytes_atomic(
                temp_directory / "manifest.json",
                canonical_json(manifest) + b"\n",
            )
            artifacts = self._descriptors(
                temp_directory=temp_directory,
                final_directory=final_directory,
                row_counts={name: frame.height for name, frame in parquet_frames.items()},
            )
            self._validate_temp(temp_directory, artifacts)
            final_directory.parent.mkdir(parents=True, exist_ok=True)
            temp_directory.replace(final_directory)
        except BaseException:
            source = final_directory if final_directory.exists() else temp_directory
            if source.exists():
                self.quarantine(source, run_id=run_id, reason="failed")
            raise
        checksums = {
            Path(artifact.path).name: artifact.checksum_sha256
            for artifact in artifacts
            if artifact.artifact_type not in {"manifest", "pnl"}
        }
        return ArtifactBundle(
            final_directory=final_directory,
            artifacts=artifacts,
            artifact_checksums=checksums,
            metrics_payload=metrics_payload,
            metric_records=_metric_records(metrics_payload),
        )

    def quarantine(self, source: Path, *, run_id: str, reason: str) -> Path:
        source = source.resolve()
        if not source.is_relative_to(self.data_root):
            raise ResearchError(f"cannot quarantine path outside data root: {source}")
        base = self.storage.path("quarantine", "research", f"{run_id}-{reason}")
        target = base
        ordinal = 1
        while target.exists():
            ordinal += 1
            target = base.with_name(f"{base.name}-{ordinal}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        return target

    def _descriptors(
        self,
        *,
        temp_directory: Path,
        final_directory: Path,
        row_counts: Mapping[str, int],
    ) -> tuple[ResearchArtifactRecord, ...]:
        descriptors = []
        for path in sorted(temp_directory.iterdir(), key=lambda item: item.name):
            if not path.is_file():
                raise ResearchError(f"artifact bundle contains a non-file entry: {path.name}")
            final_path = final_directory / path.name
            descriptors.append(
                ResearchArtifactRecord(
                    artifact_type=path.stem,
                    path=final_path.relative_to(self.data_root).as_posix(),
                    checksum_sha256=_sha256(path),
                    row_count=row_counts.get(path.name),
                    size_bytes=path.stat().st_size,
                    schema_version=ARTIFACT_SCHEMA_VERSION,
                )
            )
        return tuple(descriptors)

    @staticmethod
    def _validate_temp(
        temp_directory: Path,
        artifacts: Sequence[ResearchArtifactRecord],
    ) -> None:
        for artifact in artifacts:
            path = temp_directory / Path(artifact.path).name
            if _sha256(path) != artifact.checksum_sha256:
                raise ResearchError(f"artifact checksum changed before promotion: {path.name}")
            if path.suffix == ".parquet":
                frame = pl.read_parquet(path)
                if artifact.row_count != frame.height:
                    raise ResearchError(f"artifact row count mismatch: {path.name}")
            elif path.suffix == ".json":
                orjson.loads(path.read_bytes())


def verify_run_artifacts(
    *,
    data_root: Path,
    run_id: str,
    artifacts: Sequence[ResearchArtifactRecord],
) -> ArtifactVerification:
    root = data_root.resolve()
    issues: list[str] = []
    for artifact in artifacts:
        path = (root / artifact.path).resolve()
        if not path.is_relative_to(root):
            issues.append(f"path escapes data root: {artifact.path}")
            continue
        if not path.is_file():
            issues.append(f"missing artifact: {artifact.path}")
            continue
        if path.stat().st_size != artifact.size_bytes:
            issues.append(f"size mismatch: {artifact.path}")
        if _sha256(path) != artifact.checksum_sha256:
            issues.append(f"checksum mismatch: {artifact.path}")
        if path.suffix == ".parquet" and artifact.row_count is not None:
            try:
                row_count = pl.scan_parquet(path).select(pl.len()).collect().item()
            except pl.exceptions.PolarsError as exc:
                issues.append(f"unreadable parquet {artifact.path}: {exc}")
            else:
                if int(row_count) != artifact.row_count:
                    issues.append(f"row count mismatch: {artifact.path}")
    return ArtifactVerification(
        run_id=run_id,
        valid=not issues and bool(artifacts),
        checked_files=len(artifacts),
        issues=tuple(issues or (() if artifacts else ("run has no registered artifacts",))),
    )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "DEVELOPMENT_SEEN_BANNER",
    "ArtifactBundle",
    "ArtifactVerification",
    "ExperimentArtifactPipeline",
    "build_metrics_payload",
    "experiment_report_markdown",
    "verify_run_artifacts",
]
