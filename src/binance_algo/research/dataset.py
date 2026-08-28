"""Point-in-time hourly research dataset built from closed one-minute bars."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import orjson
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.config import ResearchConfig
from binance_algo.data.storage import LocalFilesystemStorage

MINUTE_MS = 60_000
HOUR_MINUTES = 60
DAY_MINUTES = 1_440
FEATURE_VERSION = "phase3-v1"

FEATURE_DEFINITIONS: tuple[dict[str, object], ...] = (
    {
        "name": "log_return_5m",
        "window": "5m",
        "semantics": "log(close_t / close_t-5m), closed bars only",
    },
    {
        "name": "log_return_15m",
        "window": "15m",
        "semantics": "log(close_t / close_t-15m), closed bars only",
    },
    {
        "name": "log_return_1h",
        "window": "1h",
        "semantics": "log(close_t / close_t-1h), closed bars only",
    },
    {
        "name": "log_return_4h",
        "window": "4h",
        "semantics": "log(close_t / close_t-4h), closed bars only",
    },
    {
        "name": "log_return_24h",
        "window": "24h",
        "semantics": "log(close_t / close_t-24h), closed bars only",
    },
    {
        "name": "realized_volatility_24h",
        "window": "24h",
        "semantics": "sqrt(sum of squared one-minute log returns through cutoff)",
    },
    {
        "name": "intraday_range_4h",
        "window": "4h",
        "semantics": "rolling high / rolling low - 1 through cutoff",
    },
    {
        "name": "quote_volume_zscore_24h",
        "window": "24 hourly observations",
        "semantics": "z-score of trailing one-hour quote volume, including current hour",
    },
    {
        "name": "taker_buy_imbalance_1h",
        "window": "1h",
        "semantics": "2 * taker-buy quote volume / quote volume - 1",
    },
    {
        "name": "rolling_beta_7d",
        "window": "configured trailing hourly window",
        "semantics": "causal OLS beta to leave-one-anchor-out BTC/ETH benchmark",
    },
    {
        "name": "residual_momentum_1h_4h_24h",
        "window": "1h, 4h, 24h",
        "semantics": "sum of causal residual hourly returns",
    },
    {
        "name": "funding_rate_current_and_change",
        "window": "last public event at or before cutoff",
        "semantics": "as-of join; never backward-filled",
    },
)


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    row_count: int
    decision_count: int
    symbols: tuple[str, ...]
    min_decision_time_ms: int
    max_decision_time_ms: int
    feature_after_cutoff_count: int
    execution_not_after_decision_count: int
    label_not_after_execution_count: int
    duplicate_key_count: int
    null_feature_count: int
    passed: bool


@dataclass(frozen=True, slots=True)
class ResearchDatasetResult:
    parquet_path: str
    manifest_path: str
    dataset_version: str
    feature_version: str
    universe_version: str
    audit: DatasetAudit


def _feature_version(config: ResearchConfig) -> str:
    payload = {
        "base": FEATURE_VERSION,
        "definitions": FEATURE_DEFINITIONS,
        "decision_interval_minutes": config.decision_interval_minutes,
        "beta_window_hours": config.beta_window_hours,
    }
    return hashlib.sha256(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)).hexdigest()[:16]


def _universe_version(symbols: tuple[str, ...]) -> str:
    payload = {
        "policy": "fixed_seed_ex_ante_v1",
        "symbols": symbols,
        "membership": "requires complete trailing data; never filled before first observation",
    }
    return hashlib.sha256(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)).hexdigest()[:16]


def load_research_inputs(
    *,
    database_path: Path,
    symbols: tuple[str, ...],
    start_time_ms: int,
    end_time_ms: int,
    funding_required: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load only explicit canonical views and the requested bounded time range."""

    if end_time_ms < start_time_ms:
        raise ResearchError("research end precedes start")
    placeholders = ", ".join("?" for _ in symbols)
    try:
        with duckdb.connect(str(database_path.resolve()), read_only=True) as connection:
            table_names = {
                str(row[0])
                for row in connection.execute(
                    "SELECT table_name FROM information_schema.tables"
                ).fetchall()
            }
            if "klines" not in table_names:
                raise ResearchError("DuckDB view klines does not exist")
            kline_arrow = connection.execute(
                "SELECT symbol, open_time_ms, close_time_ms, open, high, low, close, "
                "quote_volume, taker_buy_quote_volume, is_closed "
                f"FROM klines WHERE symbol IN ({placeholders}) "
                "AND interval = '1m' AND open_time_ms >= ? AND close_time_ms <= ? "
                "ORDER BY symbol, open_time_ms",
                [*symbols, start_time_ms, end_time_ms],
            ).fetch_arrow_table()
            if "funding_rates" in table_names:
                funding_arrow = connection.execute(
                    "SELECT symbol, funding_time_ms, funding_rate, rate_type, mark_price "
                    f"FROM funding_rates WHERE symbol IN ({placeholders}) "
                    "AND funding_time_ms <= ? ORDER BY symbol, funding_time_ms, rate_type",
                    [*symbols, end_time_ms],
                ).fetch_arrow_table()
            elif funding_required:
                raise ResearchError(
                    "DuckDB view funding_rates does not exist; run `binance-algo funding sync`"
                )
            else:
                funding_arrow = None
    except duckdb.Error as exc:
        raise ResearchError(f"cannot load research inputs from DuckDB: {exc}") from exc
    klines = pl.from_arrow(kline_arrow)
    if not isinstance(klines, pl.DataFrame):
        raise ResearchError("kline query did not produce a table")
    funding = (
        pl.from_arrow(funding_arrow)
        if funding_arrow is not None
        else pl.DataFrame(
            schema={
                "symbol": pl.String,
                "funding_time_ms": pl.Int64,
                "funding_rate": pl.Float64,
                "rate_type": pl.String,
                "mark_price": pl.Float64,
            }
        )
    )
    if not isinstance(funding, pl.DataFrame):
        raise ResearchError("funding query did not produce a table")
    return klines, funding


def _validate_klines(klines: pl.DataFrame, symbols: tuple[str, ...]) -> dict[str, pl.DataFrame]:
    required = {
        "symbol",
        "open_time_ms",
        "close_time_ms",
        "open",
        "high",
        "low",
        "close",
        "quote_volume",
        "taker_buy_quote_volume",
        "is_closed",
    }
    missing = required.difference(klines.columns)
    if missing:
        raise ResearchError(f"klines missing research fields: {sorted(missing)}")
    if klines.is_empty():
        raise ResearchError("no klines match the research range")
    frames: dict[str, pl.DataFrame] = {}
    reference_times: np.ndarray[Any, np.dtype[np.int64]] | None = None
    for symbol in symbols:
        frame = klines.filter(pl.col("symbol") == symbol).sort("open_time_ms")
        if frame.is_empty():
            raise ResearchError(f"no klines for research symbol {symbol}")
        if not bool(frame["is_closed"].all()):
            raise ResearchError(f"open kline reached research dataset for {symbol}")
        times = np.asarray(frame["open_time_ms"].to_numpy(), dtype=np.int64)
        if len(np.unique(times)) != len(times):
            raise ResearchError(f"duplicate kline timestamp for {symbol}")
        if len(times) > 1 and not np.all(np.diff(times) == MINUTE_MS):
            raise ResearchError(f"non-contiguous one-minute klines for {symbol}")
        if reference_times is None:
            reference_times = times
        elif not np.array_equal(reference_times, times):
            raise ResearchError("research symbols do not share an identical minute grid")
        frames[symbol] = frame
    return frames


def _zscore(values: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values))
    if standard_deviation <= 1e-15:
        return np.zeros_like(values)
    return (values - mean) / standard_deviation


def _asof_funding(
    funding: pl.DataFrame,
    *,
    symbol: str,
    decision_times: np.ndarray[Any, np.dtype[np.int64]],
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    events = (
        funding.filter(pl.col("symbol") == symbol)
        .group_by("funding_time_ms")
        .agg(pl.col("funding_rate").sum())
        .sort("funding_time_ms")
    )
    current = np.full(len(decision_times), np.nan, dtype=np.float64)
    change = np.full(len(decision_times), np.nan, dtype=np.float64)
    if events.is_empty():
        return current, change
    event_times = np.asarray(events["funding_time_ms"].to_numpy(), dtype=np.int64)
    event_rates = np.asarray(events["funding_rate"].to_numpy(), dtype=np.float64)
    positions = np.searchsorted(event_times, decision_times, side="right") - 1
    visible = positions >= 0
    current[visible] = event_rates[positions[visible]]
    prior_visible = positions >= 1
    change[prior_visible] = (
        event_rates[positions[prior_visible]] - event_rates[positions[prior_visible] - 1]
    )
    return current, change


def _outcome_funding(
    funding: pl.DataFrame,
    *,
    symbol: str,
    execution_times: np.ndarray[Any, np.dtype[np.int64]],
    label_end_times: np.ndarray[Any, np.dtype[np.int64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Sum events in (execution, label_end], assigning boundary events to old positions."""

    events = (
        funding.filter(pl.col("symbol") == symbol)
        .group_by("funding_time_ms")
        .agg(pl.col("funding_rate").sum())
        .sort("funding_time_ms")
    )
    output = np.zeros(len(execution_times), dtype=np.float64)
    if events.is_empty():
        return output
    event_times = np.asarray(events["funding_time_ms"].to_numpy(), dtype=np.int64)
    event_rates = np.asarray(events["funding_rate"].to_numpy(), dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(event_rates)))
    left = np.searchsorted(event_times, execution_times, side="right")
    right = np.searchsorted(event_times, label_end_times, side="right")
    output[:] = cumulative[right] - cumulative[left]
    return output


def build_point_in_time_frame(
    *,
    klines: pl.DataFrame,
    funding: pl.DataFrame,
    symbols: tuple[str, ...],
    config: ResearchConfig,
) -> tuple[pl.DataFrame, DatasetAudit, str, str]:
    """Create one causal row per decision time and symbol with future-only labels."""

    if tuple(dict.fromkeys(symbols)) != symbols or len(symbols) < 3:
        raise ResearchError("cross-sectional baseline needs at least three unique symbols")
    if "BTCUSDT" not in symbols or "ETHUSDT" not in symbols:
        raise ResearchError("residual baseline requires BTCUSDT and ETHUSDT anchors")
    frames = _validate_klines(klines, symbols)
    first = frames[symbols[0]]
    open_times = np.asarray(first["open_time_ms"].to_numpy(), dtype=np.int64)
    close_times = np.asarray(first["close_time_ms"].to_numpy(), dtype=np.int64)
    decision_stride = config.decision_interval_minutes
    minimum_lookback = DAY_MINUTES
    source_lookback = max(DAY_MINUTES, config.beta_window_hours * HOUR_MINUTES)
    horizon = config.forward_horizon_minutes
    minute_numbers = open_times // MINUTE_MS
    decision_indices = np.flatnonzero(minute_numbers % decision_stride == decision_stride - 1)
    decision_indices = decision_indices[
        (decision_indices >= minimum_lookback) & (decision_indices + 1 + horizon < len(open_times))
    ]
    if len(decision_indices) <= config.beta_window_hours:
        raise ResearchError("insufficient history after feature warm-up for research dataset")

    symbol_count = len(symbols)
    decision_count = len(decision_indices)
    opens = np.column_stack(
        [np.asarray(frames[symbol]["open"].to_numpy(), dtype=np.float64) for symbol in symbols]
    )
    highs = np.column_stack(
        [np.asarray(frames[symbol]["high"].to_numpy(), dtype=np.float64) for symbol in symbols]
    )
    lows = np.column_stack(
        [np.asarray(frames[symbol]["low"].to_numpy(), dtype=np.float64) for symbol in symbols]
    )
    closes = np.column_stack(
        [np.asarray(frames[symbol]["close"].to_numpy(), dtype=np.float64) for symbol in symbols]
    )
    quote_volume = np.column_stack(
        [
            np.asarray(frames[symbol]["quote_volume"].to_numpy(), dtype=np.float64)
            for symbol in symbols
        ]
    )
    taker_quote_volume = np.column_stack(
        [
            np.asarray(frames[symbol]["taker_buy_quote_volume"].to_numpy(), dtype=np.float64)
            for symbol in symbols
        ]
    )
    if not all(
        np.all(np.isfinite(values)) and np.all(values > 0)
        for values in (opens, highs, lows, closes)
    ):
        raise ResearchError("research prices must be finite and positive")

    log_close = np.log(closes)
    minute_log_returns = np.full_like(log_close, np.nan)
    minute_log_returns[1:] = np.diff(log_close, axis=0)
    horizons = (5, 15, 60, 240, 1_440)
    horizon_returns = {
        window: log_close[decision_indices] - log_close[decision_indices - window]
        for window in horizons
    }
    hourly_returns = horizon_returns[60]
    realized_volatility = np.empty((decision_count, symbol_count), dtype=np.float64)
    intraday_range = np.empty_like(realized_volatility)
    hourly_quote_volume = np.empty_like(realized_volatility)
    taker_imbalance = np.empty_like(realized_volatility)
    for row_index, minute_index in enumerate(decision_indices):
        day_slice = slice(minute_index - DAY_MINUTES + 1, minute_index + 1)
        range_slice = slice(minute_index - 240 + 1, minute_index + 1)
        hour_slice = slice(minute_index - 60 + 1, minute_index + 1)
        realized_volatility[row_index] = np.sqrt(
            np.nansum(np.square(minute_log_returns[day_slice]), axis=0)
        )
        intraday_range[row_index] = (
            np.max(highs[range_slice], axis=0) / np.min(lows[range_slice], axis=0) - 1
        )
        hourly_quote_volume[row_index] = np.sum(quote_volume[hour_slice], axis=0)
        total_taker = np.sum(taker_quote_volume[hour_slice], axis=0)
        taker_imbalance[row_index] = (
            np.divide(
                2 * total_taker,
                hourly_quote_volume[row_index],
                out=np.full(symbol_count, np.nan),
                where=hourly_quote_volume[row_index] > 0,
            )
            - 1
        )

    volume_zscore = np.full_like(hourly_quote_volume, np.nan)
    for row_index in range(23, decision_count):
        for symbol_index in range(symbol_count):
            volume_zscore[row_index, symbol_index] = _zscore(
                hourly_quote_volume[row_index - 23 : row_index + 1, symbol_index]
            )[-1]

    btc_index = symbols.index("BTCUSDT")
    eth_index = symbols.index("ETHUSDT")
    market_returns = (hourly_returns[:, btc_index] + hourly_returns[:, eth_index]) / 2
    benchmark_returns = np.tile(market_returns[:, None], (1, symbol_count))
    benchmark_returns[:, btc_index] = hourly_returns[:, eth_index]
    benchmark_returns[:, eth_index] = hourly_returns[:, btc_index]
    beta = np.full_like(hourly_returns, np.nan)
    residual_returns = np.full_like(hourly_returns, np.nan)
    beta_window = config.beta_window_hours
    for row_index in range(beta_window - 1, decision_count):
        start = row_index - beta_window + 1
        for symbol_index in range(symbol_count):
            x = benchmark_returns[start : row_index + 1, symbol_index]
            y = hourly_returns[start : row_index + 1, symbol_index]
            variance = float(np.var(x))
            if variance <= 1e-18:
                continue
            beta[row_index, symbol_index] = float(
                np.mean((x - np.mean(x)) * (y - np.mean(y))) / variance
            )
            residual_returns[row_index, symbol_index] = (
                hourly_returns[row_index, symbol_index]
                - beta[row_index, symbol_index] * benchmark_returns[row_index, symbol_index]
            )

    residual_momentum_4h = np.full_like(residual_returns, np.nan)
    residual_momentum_24h = np.full_like(residual_returns, np.nan)
    for row_index in range(decision_count):
        if row_index >= 3:
            residual_momentum_4h[row_index] = np.sum(
                residual_returns[row_index - 3 : row_index + 1], axis=0
            )
        if row_index >= 23:
            residual_momentum_24h[row_index] = np.sum(
                residual_returns[row_index - 23 : row_index + 1], axis=0
            )

    market_volatility_regime = np.full(decision_count, np.nan, dtype=np.float64)
    for row_index in range(23, decision_count):
        market_volatility_regime[row_index] = float(
            np.std(market_returns[row_index - 23 : row_index + 1]) * math.sqrt(24 * 365)
        )

    decision_times = close_times[decision_indices]
    execution_indices = decision_indices + 1
    label_end_indices = execution_indices + horizon
    future_returns = opens[label_end_indices] / opens[execution_indices] - 1
    future_benchmark = np.tile(
        ((future_returns[:, btc_index] + future_returns[:, eth_index]) / 2)[:, None],
        (1, symbol_count),
    )
    future_benchmark[:, btc_index] = future_returns[:, eth_index]
    future_benchmark[:, eth_index] = future_returns[:, btc_index]
    future_residual = future_returns - beta * future_benchmark
    outcome_quote_volume = np.empty_like(hourly_quote_volume)
    for row_index, entry_index in enumerate(execution_indices):
        outcome_quote_volume[row_index] = np.sum(
            quote_volume[entry_index : entry_index + horizon], axis=0
        )

    funding_current = np.empty_like(hourly_returns)
    funding_change = np.empty_like(hourly_returns)
    outcome_funding = np.empty_like(hourly_returns)
    for symbol_index, symbol in enumerate(symbols):
        current, change = _asof_funding(funding, symbol=symbol, decision_times=decision_times)
        funding_current[:, symbol_index] = current
        funding_change[:, symbol_index] = change
        outcome_funding[:, symbol_index] = _outcome_funding(
            funding,
            symbol=symbol,
            execution_times=open_times[execution_indices],
            label_end_times=open_times[label_end_indices],
        )

    feature_version = _feature_version(config)
    universe_version = _universe_version(symbols)
    rows: list[dict[str, object]] = []
    for row_index, minute_index in enumerate(decision_indices):
        for symbol_index, symbol in enumerate(symbols):
            values = (
                *(horizon_returns[window][row_index, symbol_index] for window in horizons),
                realized_volatility[row_index, symbol_index],
                intraday_range[row_index, symbol_index],
                volume_zscore[row_index, symbol_index],
                taker_imbalance[row_index, symbol_index],
                beta[row_index, symbol_index],
                residual_returns[row_index, symbol_index],
                residual_momentum_4h[row_index, symbol_index],
                residual_momentum_24h[row_index, symbol_index],
                market_volatility_regime[row_index],
                funding_current[row_index, symbol_index],
                funding_change[row_index, symbol_index],
                future_returns[row_index, symbol_index],
                future_residual[row_index, symbol_index],
                outcome_quote_volume[row_index, symbol_index],
            )
            if not all(math.isfinite(float(value)) for value in values):
                continue
            rows.append(
                {
                    "decision_time_ms": int(decision_times[row_index]),
                    "feature_cutoff_ms": int(close_times[minute_index]),
                    "feature_source_min_ms": int(
                        open_times[max(0, int(minute_index) - source_lookback + 1)]
                    ),
                    "feature_source_max_ms": int(close_times[minute_index]),
                    "execution_time_ms": int(open_times[execution_indices[row_index]]),
                    "label_end_time_ms": int(open_times[label_end_indices[row_index]]),
                    "symbol": symbol,
                    "universe_version": universe_version,
                    "feature_version": feature_version,
                    "log_return_5m": float(horizon_returns[5][row_index, symbol_index]),
                    "log_return_15m": float(horizon_returns[15][row_index, symbol_index]),
                    "log_return_1h": float(horizon_returns[60][row_index, symbol_index]),
                    "log_return_4h": float(horizon_returns[240][row_index, symbol_index]),
                    "log_return_24h": float(horizon_returns[1_440][row_index, symbol_index]),
                    "realized_volatility_24h": float(realized_volatility[row_index, symbol_index]),
                    "intraday_range_4h": float(intraday_range[row_index, symbol_index]),
                    "quote_volume_1h": float(hourly_quote_volume[row_index, symbol_index]),
                    "quote_volume_zscore_24h": float(volume_zscore[row_index, symbol_index]),
                    "taker_buy_imbalance_1h": float(taker_imbalance[row_index, symbol_index]),
                    "benchmark_return_1h": float(benchmark_returns[row_index, symbol_index]),
                    "rolling_beta": float(beta[row_index, symbol_index]),
                    "residual_momentum_1h": float(residual_returns[row_index, symbol_index]),
                    "residual_momentum_4h": float(residual_momentum_4h[row_index, symbol_index]),
                    "residual_momentum_24h": float(residual_momentum_24h[row_index, symbol_index]),
                    "market_volatility_regime": float(market_volatility_regime[row_index]),
                    "funding_rate_current": float(funding_current[row_index, symbol_index]),
                    "funding_rate_change": float(funding_change[row_index, symbol_index]),
                    "future_return_1h": float(future_returns[row_index, symbol_index]),
                    "future_residual_return_1h": float(future_residual[row_index, symbol_index]),
                    "outcome_quote_volume_1h": float(outcome_quote_volume[row_index, symbol_index]),
                    "outcome_funding_rate_1h": float(outcome_funding[row_index, symbol_index]),
                    "execution_lag_bars": 1,
                    "dataset_schema_version": 1,
                }
            )
    if not rows:
        raise ResearchError("all research rows were removed by causal availability checks")
    frame = pl.DataFrame(rows).sort("decision_time_ms", "symbol")
    counts = frame.group_by("decision_time_ms").len()
    frame = frame.join(
        counts.filter(pl.col("len") == len(symbols)).select("decision_time_ms"),
        on="decision_time_ms",
        how="inner",
    ).sort("decision_time_ms", "symbol")
    audit = audit_point_in_time_frame(frame, symbols=symbols)
    if not audit.passed:
        raise ResearchError(f"point-in-time dataset audit failed: {audit}")
    return frame, audit, feature_version, universe_version


def audit_point_in_time_frame(frame: pl.DataFrame, *, symbols: tuple[str, ...]) -> DatasetAudit:
    feature_columns = [
        column
        for column in frame.columns
        if column.startswith(("log_return_", "realized_", "intraday_", "quote_volume_zscore"))
        or column.startswith(("taker_buy_", "benchmark_", "rolling_beta"))
        or column.startswith(("residual_momentum_", "market_volatility_", "funding_rate_"))
    ]
    duplicate_count = frame.select(
        pl.struct("decision_time_ms", "symbol").is_duplicated().sum()
    ).item()
    null_feature_count = frame.select(
        pl.sum_horizontal([pl.col(column).is_null().sum() for column in feature_columns])
    ).item()
    feature_after = frame.filter(
        pl.col("feature_source_max_ms") > pl.col("decision_time_ms")
    ).height
    execution_bad = frame.filter(pl.col("execution_time_ms") <= pl.col("decision_time_ms")).height
    label_bad = frame.filter(pl.col("label_end_time_ms") <= pl.col("execution_time_ms")).height
    decision_count = frame["decision_time_ms"].n_unique()
    complete = frame.height == decision_count * len(symbols)
    return DatasetAudit(
        row_count=frame.height,
        decision_count=decision_count,
        symbols=symbols,
        min_decision_time_ms=int(frame.select(pl.col("decision_time_ms").min()).item()),
        max_decision_time_ms=int(frame.select(pl.col("decision_time_ms").max()).item()),
        feature_after_cutoff_count=int(feature_after),
        execution_not_after_decision_count=int(execution_bad),
        label_not_after_execution_count=int(label_bad),
        duplicate_key_count=int(duplicate_count),
        null_feature_count=int(null_feature_count),
        passed=(
            complete
            and feature_after == 0
            and execution_bad == 0
            and label_bad == 0
            and duplicate_count == 0
            and null_feature_count == 0
        ),
    )


def persist_research_dataset(
    *,
    frame: pl.DataFrame,
    audit: DatasetAudit,
    feature_version: str,
    universe_version: str,
    storage: LocalFilesystemStorage,
    compression: str,
    symbols: tuple[str, ...],
    config: ResearchConfig,
) -> ResearchDatasetResult:
    digest_payload = orjson.dumps(frame.to_dicts(), option=orjson.OPT_SORT_KEYS)
    dataset_version = hashlib.sha256(digest_payload).hexdigest()[:16]
    parquet_path = storage.path(
        "gold",
        "binance",
        "usdm",
        "research_dataset",
        f"version={dataset_version}",
        "dataset.parquet",
    )
    manifest_path = parquet_path.with_suffix(".json")
    storage.write_parquet_atomic(parquet_path, frame, compression=compression)
    storage.write_json_atomic(
        manifest_path,
        {
            "dataset_version": dataset_version,
            "dataset_schema_version": 1,
            "feature_version": feature_version,
            "universe_version": universe_version,
            "universe_policy": "fixed seed chosen ex ante by the project specification",
            "symbols": symbols,
            "feature_definitions": FEATURE_DEFINITIONS,
            "label_semantics": (
                "future_return_1h starts at the next one-minute open and ends 60 minutes later"
            ),
            "funding_semantics": "last event at or before decision; no backward fill",
            "audit": asdict(audit),
            "research_config": config.model_dump(mode="json"),
        },
    )
    return ResearchDatasetResult(
        parquet_path=str(parquet_path),
        manifest_path=str(manifest_path),
        dataset_version=dataset_version,
        feature_version=feature_version,
        universe_version=universe_version,
        audit=audit,
    )


def build_and_persist_research_dataset(
    *,
    database_path: Path,
    storage: LocalFilesystemStorage,
    symbols: tuple[str, ...],
    start_time_ms: int,
    end_time_ms: int,
    config: ResearchConfig,
    compression: str,
) -> ResearchDatasetResult:
    klines, funding = load_research_inputs(
        database_path=database_path,
        symbols=symbols,
        start_time_ms=start_time_ms,
        end_time_ms=end_time_ms,
        funding_required=config.funding_required,
    )
    frame, audit, feature_version, universe_version = build_point_in_time_frame(
        klines=klines,
        funding=funding,
        symbols=symbols,
        config=config,
    )
    return persist_research_dataset(
        frame=frame,
        audit=audit,
        feature_version=feature_version,
        universe_version=universe_version,
        storage=storage,
        compression=compression,
        symbols=symbols,
        config=config,
    )
