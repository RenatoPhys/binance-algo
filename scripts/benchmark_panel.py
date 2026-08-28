"""Non-blocking synthetic benchmark for PanelData loading and parameter trials."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np
import orjson
import polars as pl

from binance_algo.research.panel import WorkerDatasetCache, finite_float

HOURS_PER_YEAR = 24 * 365


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=int, default=30)
    parser.add_argument("--years", type=int, default=2)
    parser.add_argument("--features", type=int, default=20)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("var/reports/panel_benchmark.json"),
    )
    return parser.parse_args()


def _synthetic_frame(*, symbols: int, periods: int, features: int) -> pl.DataFrame:
    if min(symbols, periods, features) < 1:
        raise ValueError("symbols, periods and features must be positive")
    row_count = symbols * periods
    period_axis = np.repeat(np.arange(periods, dtype=np.int64), symbols)
    symbol_axis = np.tile(np.arange(symbols, dtype=np.int64), periods)
    columns: dict[str, object] = {
        "decision_time_ms": 1_700_000_000_000 + period_axis * 3_600_000,
        "symbol": np.asarray([f"S{index:03d}" for index in symbol_axis]),
    }
    for index in range(features):
        columns[f"feature_{index:02d}"] = np.sin(
            period_axis / (8.0 + index) + symbol_axis * (0.03 + index / 1_000)
        ).astype(np.float32)
    frame = pl.DataFrame(columns)
    if frame.height != row_count:
        raise RuntimeError("synthetic benchmark frame has an unexpected row count")
    return frame


def main() -> None:
    args = _arguments()
    periods = args.years * HOURS_PER_YEAR
    feature_names = tuple(f"feature_{index:02d}" for index in range(args.features))
    frame = _synthetic_frame(symbols=args.symbols, periods=periods, features=args.features)
    trial_rows: list[dict[str, float | int]] = []
    with tempfile.TemporaryDirectory(prefix="binance-algo-panel-") as directory:
        temporary_root = Path(directory)
        dataset_path = temporary_root / "synthetic.parquet"
        frame.write_parquet(dataset_path, compression="zstd")
        del frame
        cache = WorkerDatasetCache(max_entries=1)
        loaded = cache.load(
            dataset_path,
            feature_columns=feature_names,
            outcome_columns=(),
            metadata_columns=(),
        )
        cached = cache.load(
            dataset_path,
            feature_columns=feature_names,
            outcome_columns=(),
            metadata_columns=(),
        )
        if cached is not loaded:
            raise RuntimeError("worker dataset cache did not reuse the loaded panel")
        started_trials = perf_counter()
        for trial in range(args.trials):
            started = perf_counter()
            first = loaded.panel.features[feature_names[trial % args.features]]
            second = loaded.panel.features[feature_names[(trial + 7) % args.features]]
            weight = (trial + 1) / (args.trials + 1)
            score = weight * first + (1 - weight) * second
            statistic = finite_float(float(np.mean(np.abs(score))), role="trial statistic")
            trial_rows.append(
                {
                    "trial": trial + 1,
                    "runtime_seconds": perf_counter() - started,
                    "statistic": statistic,
                }
            )
        total_trial_seconds = perf_counter() - started_trials
        artifact_path = temporary_root / "trial_results.parquet"
        pl.DataFrame(trial_rows).write_parquet(artifact_path, compression="zstd")
        report = {
            "scenario": {
                "symbols": args.symbols,
                "years": args.years,
                "hourly_periods": periods,
                "features": args.features,
                "trials": args.trials,
                "rows": args.symbols * periods,
            },
            "measurements": {
                "load_seconds": loaded.load_seconds,
                "total_trial_seconds": total_trial_seconds,
                "mean_seconds_per_trial": total_trial_seconds / args.trials,
                "panel_estimated_memory_bytes": loaded.panel.estimated_nbytes,
                "source_parquet_size_bytes": dataset_path.stat().st_size,
                "trial_artifact_size_bytes": artifact_path.stat().st_size,
            },
            "cache": {
                "entries": cache.info().entries,
                "hits": cache.info().hits,
                "misses": cache.info().misses,
            },
            "policy": "observational benchmark; no CI SLA",
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_bytes(orjson.dumps(report, option=orjson.OPT_SORT_KEYS) + b"\n")
    temporary_output.replace(args.output)
    print(orjson.dumps(report, option=orjson.OPT_INDENT_2).decode("utf-8"))


if __name__ == "__main__":
    main()
