"""Run an observational benchmark through the real local research campaign stack."""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import orjson
import polars as pl

from binance_algo.config import load_settings
from binance_algo.research.backtest import ACCOUNTING_METADATA_FIELDS, ACCOUNTING_OUTCOME_FIELDS
from binance_algo.research.experiments.campaign import CampaignSpec, plan_campaign
from binance_algo.research.experiments.campaign_runner import CampaignRunner
from binance_algo.research.experiments.models import HypothesisSpec, HypothesisStatus
from binance_algo.research.experiments.provenance import build_code_fingerprint
from binance_algo.research.experiments.registry import sync_builtin_registry
from binance_algo.research.experiments.store import ResearchStore
from binance_algo.research.panel import WorkerDatasetCache
from binance_algo.research.portfolio.registry import build_portfolio_policy
from binance_algo.research.strategies.registry import build_strategy

START_MS = 1_767_225_600_000
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("var/reports/research_campaign_benchmark.json"),
    )
    return parser.parse_args()


def _synthetic_frame(days: int = 11) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for hour in range(days * 24):
        decision = START_MS + hour * 3_600_000 + 3_599_999
        common_volatility = 0.012 + 0.002 * math.sin(hour / 24)
        for symbol_index, symbol in enumerate(SYMBOLS):
            phase = math.sin(hour / 9 + symbol_index)
            residual_1h = 0.002 * phase
            rows.append(
                {
                    "decision_time_ms": decision,
                    "execution_time_ms": decision + 1,
                    "label_end_time_ms": decision + 3_600_001,
                    "symbol": symbol,
                    "residual_momentum_1h": residual_1h,
                    "residual_momentum_4h": residual_1h * 2 + symbol_index * 0.0001,
                    "residual_momentum_24h": residual_1h * 4 - symbol_index * 0.0001,
                    "realized_volatility_24h": common_volatility * (1 + symbol_index * 0.1),
                    "rolling_beta": 0.8 + symbol_index * 0.25,
                    "future_return_1h": 0.0015 * phase - 0.0002 * symbol_index,
                    "future_residual_return_1h": 0.0012 * phase,
                    "outcome_funding_rate_1h": (
                        0.0001 * (symbol_index + 1) if hour % 8 == 7 else 0.0
                    ),
                    "outcome_quote_volume_1h": 100_000_000.0,
                    "market_volatility_regime": common_volatility * math.sqrt(365),
                }
            )
    return pl.DataFrame(rows)


def _write_dataset(data_root: Path, feature_set_id: str) -> tuple[Path, pl.DataFrame]:
    frame = _synthetic_frame()
    directory = data_root / "gold" / "binance" / "usdm" / "research_dataset" / "version=benchmark"
    directory.mkdir(parents=True)
    dataset_path = directory / "dataset.parquet"
    frame.write_parquet(dataset_path, compression="zstd")
    checksum = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    manifest = {
        "dataset_id": "synthetic-campaign-benchmark-v1",
        "dataset_schema_version": 2,
        "feature_set_id": feature_set_id,
        "label_id": "gross_forward_return_1h:v1",
        "universe_version": "synthetic-three-v1",
        "start_time_ms": int(frame["decision_time_ms"].min()),
        "end_time_ms": int(frame["decision_time_ms"].max()),
        "row_count": frame.height,
        "content_checksum": checksum,
        "fingerprint_method": "benchmark_sha256",
    }
    manifest_path = directory / "dataset.json"
    manifest_path.write_bytes(orjson.dumps(manifest, option=orjson.OPT_SORT_KEYS) + b"\n")
    return manifest_path, frame


def _campaign(manifest_path: Path, *, trials: int, workers: int) -> CampaignSpec:
    bands = [round((index + 1) * 0.5 / (trials + 1), 10) for index in range(trials)]
    return CampaignSpec.model_validate(
        {
            "campaign": {
                "name": "synthetic_discovery_benchmark",
                "description": "Observational benchmark; not strategy evidence.",
                "hypothesis_id": "HYP-CAMPAIGN-BENCHMARK-0001",
                "artifact_policy": "summary",
                "max_trials": trials,
            },
            "dataset": {"manifest": str(manifest_path)},
            "feature_set": {"name": "phase3_baseline_features", "version": "1"},
            "label": {
                "name": "gross_forward_return_1h",
                "version": "1",
                "horizon_minutes": 60,
            },
            "strategy": {
                "name": "residual_momentum",
                "version": "1",
                "fixed": {
                    "momentum_weight_1h": 0.2,
                    "momentum_weight_4h": 0.3,
                    "momentum_weight_24h": 0.5,
                },
            },
            "portfolio": {
                "name": "neutral_long_short",
                "version": "1",
                "fixed": {
                    "gross_exposure": 0.5,
                    "annual_volatility_target": 0.15,
                    "max_symbol_weight": 0.25,
                },
                "grid": {"no_trade_score_band": bands},
            },
            "execution": {"name": "bar_next_open", "version": "1"},
            "costs": {
                "name": "configured_taker",
                "version": "1",
                "fixed": {"cost_multiplier": 1.0},
            },
            "validation": {
                "profile": "discovery",
                "split_plan": "expanding_walk_forward_v1",
                "train_days": 7,
                "test_days": 1,
            },
            "runner": {"max_workers": workers, "fail_fast": False, "resume": True},
        }
    )


def _estimated_worker_memory(dataset_path: Path) -> int:
    strategy = build_strategy(
        "residual_momentum",
        "1",
        {
            "momentum_weight_1h": 0.2,
            "momentum_weight_4h": 0.3,
            "momentum_weight_24h": 0.5,
        },
    )
    portfolio = build_portfolio_policy(
        "neutral_long_short",
        "1",
        {
            "no_trade_score_band": 0.1,
            "gross_exposure": 0.5,
            "annual_volatility_target": 0.15,
            "max_symbol_weight": 0.25,
        },
    )
    loaded = WorkerDatasetCache(max_entries=1).load(
        dataset_path,
        feature_columns=tuple(
            dict.fromkeys((*strategy.required_features(), *portfolio.required_features()))
        ),
        outcome_columns=ACCOUNTING_OUTCOME_FIELDS,
        metadata_columns=ACCOUNTING_METADATA_FIELDS,
    )
    return loaded.frame.estimated_size() + loaded.panel.estimated_nbytes


def _atomic_report(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS) + b"\n")
    os.replace(temporary, path)


def main() -> None:
    args = _arguments()
    if not 1 <= args.trials <= 1_000:
        raise ValueError("--trials must be between 1 and 1000")
    if not 1 <= args.workers <= 32:
        raise ValueError("--workers must be between 1 and 32")
    settings = load_settings(args.config)
    runtime_parent = settings.project_root / "var"
    runtime_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="research-campaign-benchmark-",
        dir=runtime_parent,
    ) as directory:
        runtime_root = Path(directory)
        data_root = runtime_root / "data"
        reports_root = runtime_root / "reports"
        store = ResearchStore(runtime_root / "research.sqlite3")
        store.initialize()
        feature_set_id = sync_builtin_registry(
            store, research_config=settings.research
        ).feature_set_id
        manifest_path, frame = _write_dataset(data_root, feature_set_id)
        dataset_path = manifest_path.with_suffix(".parquet")
        worker_memory = _estimated_worker_memory(dataset_path)
        store.register_hypothesis(
            HypothesisSpec(
                hypothesis_id="HYP-CAMPAIGN-BENCHMARK-0001",
                title="Campaign infrastructure benchmark",
                mechanism="Exercise the complete local campaign execution path.",
                preregistered_success_criteria={"purpose": "benchmark_only"},
                status=HypothesisStatus.READY,
            )
        )
        plan = plan_campaign(
            _campaign(manifest_path, trials=args.trials, workers=args.workers),
            project_root=settings.project_root,
            data_root=data_root,
            research_config=settings.research,
            code_fingerprint=build_code_fingerprint(settings.project_root),
        )
        runner = CampaignRunner(
            store=store,
            data_root=data_root,
            reports_root=reports_root,
            research_config=settings.research,
            compression=settings.storage.parquet_compression,
            heartbeat_seconds=settings.research_platform.heartbeat_seconds,
        )
        started = perf_counter()
        result = runner.run(plan)
        total_runtime = perf_counter() - started
        runtimes = [
            run.runtime_seconds
            for _, identifier, _ in store.list_campaign_experiments(plan.campaign_id)
            for run in store.list_runs(experiment_id_value=identifier)[-1:]
            if run.runtime_seconds is not None
        ]
        artifact_root = data_root / "gold" / "binance" / "usdm" / "research_experiments"
        artifact_size = sum(
            path.stat().st_size for path in artifact_root.rglob("*") if path.is_file()
        )
        report = {
            "scenario": {
                "trials": args.trials,
                "workers": args.workers,
                "profile": "discovery",
                "dataset_rows": frame.height,
            },
            "measurements": {
                "total_runtime_seconds": total_runtime,
                "trial_runtime_p50_seconds": float(np.quantile(runtimes, 0.50))
                if runtimes
                else None,
                "trial_runtime_p95_seconds": float(np.quantile(runtimes, 0.95))
                if runtimes
                else None,
                "estimated_memory_bytes_per_worker": worker_memory,
                "artifact_size_bytes": artifact_size,
            },
            "outcomes": {
                "cache_hits": result.cache_hit_count,
                "successes": result.succeeded_count,
                "failures": result.failed_count,
            },
            "policy": "observational benchmark; no CI SLA",
        }
    output = args.output if args.output.is_absolute() else settings.project_root / args.output
    _atomic_report(output, report)
    print(orjson.dumps(report, option=orjson.OPT_INDENT_2).decode())


if __name__ == "__main__":
    main()
