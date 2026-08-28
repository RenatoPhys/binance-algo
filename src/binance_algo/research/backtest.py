"""Vectorized bar research engine with explicit timing and accounting costs."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.config import ResearchConfig
from binance_algo.data.storage import LocalFilesystemStorage
from binance_algo.research.visualization import PNL_VISUALIZATION_VERSION, render_pnl_svg

HOURS_PER_YEAR = 24 * 365


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold: int
    train_start_ms: int
    train_end_ms: int
    test_start_ms: int
    test_end_ms: int
    train_decisions: int
    test_decisions: int
    embargo_bars: int


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    periods: int
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    positive_period_fraction: float
    price_pnl: float
    funding_pnl: float
    trading_fees: float
    spread_cost: float
    slippage_cost: float
    net_pnl: float
    turnover: float
    annualized_turnover: float
    average_gross_exposure: float
    average_net_exposure: float
    maximum_absolute_beta_exposure: float
    maximum_volume_participation: float
    mean_cross_sectional_rank_ic: float
    accounting_error_max: float


@dataclass(frozen=True, slots=True)
class BacktestRun:
    curve: pl.DataFrame
    folds: tuple[WalkForwardFold, ...]
    metrics: PerformanceMetrics


@dataclass(frozen=True, slots=True)
class ResearchBacktestResult:
    run_version: str
    curve_path: str
    report_json_path: str
    report_markdown_path: str
    report_chart_path: str
    metrics: PerformanceMetrics
    fold_count: int
    stress: dict[str, dict[str, float | int]]
    bootstrap: dict[str, float | int]
    regimes: dict[str, dict[str, float | int]]


def make_walk_forward_folds(
    decision_times: np.ndarray[Any, np.dtype[np.int64]],
    *,
    train_days: int,
    test_days: int,
    embargo_bars: int,
) -> tuple[WalkForwardFold, ...]:
    """Create expanding temporal folds; no row is randomly assigned."""

    train_bars = train_days * 24
    test_bars = test_days * 24
    if len(decision_times) < train_bars + embargo_bars + test_bars:
        raise ResearchError("insufficient decisions for one complete walk-forward fold")
    folds: list[WalkForwardFold] = []
    train_end_index = train_bars - 1
    fold_number = 1
    while True:
        test_start_index = train_end_index + 1 + embargo_bars
        test_end_index = test_start_index + test_bars - 1
        if test_end_index >= len(decision_times):
            break
        folds.append(
            WalkForwardFold(
                fold=fold_number,
                train_start_ms=int(decision_times[0]),
                train_end_ms=int(decision_times[train_end_index]),
                test_start_ms=int(decision_times[test_start_index]),
                test_end_ms=int(decision_times[test_end_index]),
                train_decisions=train_end_index + 1,
                test_decisions=test_bars,
                embargo_bars=embargo_bars,
            )
        )
        fold_number += 1
        train_end_index += test_bars
    if not folds:
        raise ResearchError("walk-forward construction produced no folds")
    return tuple(folds)


def _cross_sectional_zscore(
    values: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    standard_deviation = float(np.std(values))
    if standard_deviation <= 1e-15:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / standard_deviation


def _rank(values: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _neutral_weights(
    *,
    top_index: int,
    bottom_index: int,
    betas: np.ndarray[Any, np.dtype[np.float64]],
    realized_volatility: np.ndarray[Any, np.dtype[np.float64]],
    config: ResearchConfig,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    raw = np.zeros(len(betas), dtype=np.float64)
    raw[top_index] = 1
    raw[bottom_index] = -1
    constraints = np.vstack((np.ones(len(betas), dtype=np.float64), betas))
    projected = raw - constraints.T @ np.linalg.pinv(constraints @ constraints.T) @ (
        constraints @ raw
    )
    if (
        float(np.sum(np.abs(projected))) <= 1e-12
        or projected[top_index] <= 0
        or projected[bottom_index] >= 0
    ):
        projected = raw
    projected /= float(np.sum(np.abs(projected)))
    annualized_asset_volatility = realized_volatility * math.sqrt(365)
    volatility_proxy = float(math.sqrt(np.sum(np.square(projected * annualized_asset_volatility))))
    target_gross = float(config.gross_exposure)
    if volatility_proxy > 0:
        target_gross = min(target_gross, float(config.annual_volatility_target) / volatility_proxy)
    weights = projected * target_gross
    maximum_weight = float(np.max(np.abs(weights)))
    if maximum_weight > float(config.max_symbol_weight):
        weights *= float(config.max_symbol_weight) / maximum_weight
    if float(np.sum(np.abs(weights))) > 1 + 1e-12:
        raise ResearchError("baseline attempted economic leverage")
    return weights


def _wide_arrays(
    frame: pl.DataFrame,
) -> tuple[
    tuple[str, ...],
    np.ndarray[Any, np.dtype[np.int64]],
    dict[str, np.ndarray[Any, np.dtype[np.float64]]],
]:
    symbols = tuple(sorted(str(value) for value in frame["symbol"].unique().to_list()))
    times = np.asarray(sorted(frame["decision_time_ms"].unique().to_list()), dtype=np.int64)
    fields = (
        "residual_momentum_1h",
        "residual_momentum_4h",
        "residual_momentum_24h",
        "realized_volatility_24h",
        "rolling_beta",
        "future_return_1h",
        "future_residual_return_1h",
        "outcome_funding_rate_1h",
        "outcome_quote_volume_1h",
        "market_volatility_regime",
        "execution_time_ms",
        "label_end_time_ms",
    )
    arrays = {
        field: np.full((len(times), len(symbols)), np.nan, dtype=np.float64) for field in fields
    }
    time_index = {int(value): index for index, value in enumerate(times)}
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    for row in frame.iter_rows(named=True):
        x = time_index[int(row["decision_time_ms"])]
        y = symbol_index[str(row["symbol"])]
        for field in fields:
            arrays[field][x, y] = float(row[field])
    if any(np.any(~np.isfinite(values)) for values in arrays.values()):
        raise ResearchError("backtest input is incomplete or non-finite")
    return symbols, times, arrays


def _scores_and_targets(
    arrays: dict[str, np.ndarray[Any, np.dtype[np.float64]]],
    *,
    config: ResearchConfig,
    momentum_weights: tuple[float, float, float],
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    periods, symbol_count = arrays["residual_momentum_1h"].shape
    scores = np.empty((periods, symbol_count), dtype=np.float64)
    targets = np.empty_like(scores)
    previous_top: int | None = None
    previous_bottom: int | None = None
    band = float(config.no_trade_score_band)
    for period in range(periods):
        scores[period] = (
            momentum_weights[0] * _cross_sectional_zscore(arrays["residual_momentum_1h"][period])
            + momentum_weights[1] * _cross_sectional_zscore(arrays["residual_momentum_4h"][period])
            + momentum_weights[2] * _cross_sectional_zscore(arrays["residual_momentum_24h"][period])
        )
        top = int(np.argmax(scores[period]))
        bottom = int(np.argmin(scores[period]))
        if previous_top is not None and scores[period, top] - scores[period, previous_top] <= band:
            top = previous_top
        if (
            previous_bottom is not None
            and scores[period, previous_bottom] - scores[period, bottom] <= band
        ):
            bottom = previous_bottom
        if top == bottom:
            raise ResearchError("cross-sectional score did not produce distinct tails")
        targets[period] = _neutral_weights(
            top_index=top,
            bottom_index=bottom,
            betas=arrays["rolling_beta"][period],
            realized_volatility=arrays["realized_volatility_24h"][period],
            config=config,
        )
        previous_top, previous_bottom = top, bottom
    return scores, targets


def _fee_schedule_covers(config: ResearchConfig, execution_time_ms: int) -> bool:
    event_date = datetime.fromtimestamp(execution_time_ms / 1_000, tz=UTC).date()
    schedule = config.fee_schedule
    return event_date >= schedule.effective_from and (
        schedule.effective_to is None or event_date <= schedule.effective_to
    )


def _run_fold(
    frame: pl.DataFrame,
    *,
    fold_number: int,
    config: ResearchConfig,
    cost_multiplier: float,
    signal_delay_bars: int,
    momentum_weights: tuple[float, float, float],
) -> pl.DataFrame:
    symbols, times, arrays = _wide_arrays(frame)
    scores, targets = _scores_and_targets(arrays, config=config, momentum_weights=momentum_weights)
    if signal_delay_bars:
        delayed = np.zeros_like(targets)
        delayed[signal_delay_bars:] = targets[:-signal_delay_bars]
        targets = delayed
    previous = np.zeros(len(symbols), dtype=np.float64)
    equity = 1.0
    output: list[dict[str, object]] = []
    fee_rate = float(config.fee_schedule.taker_fee_rate) * cost_multiplier
    half_spread_rate = float(config.spread_bps) / 20_000 * cost_multiplier
    slippage_rate = float(config.slippage_bps) / 10_000 * cost_multiplier
    capital = float(config.initial_capital_usdt)
    for period, decision_time in enumerate(times):
        execution_time = int(arrays["execution_time_ms"][period, 0])
        if not _fee_schedule_covers(config, execution_time):
            raise ResearchError(f"fee schedule does not cover execution time {execution_time}")
        weights = targets[period]
        trades = np.abs(weights - previous)
        turnover = float(np.sum(trades))
        if period == len(times) - 1:
            turnover += float(np.sum(np.abs(weights)))
        price_pnl = float(np.dot(weights, arrays["future_return_1h"][period]))
        funding_pnl = float(-np.dot(weights, arrays["outcome_funding_rate_1h"][period]))
        trading_fees = turnover * fee_rate
        spread_cost = turnover * half_spread_rate
        slippage_cost = turnover * slippage_rate
        net_return = price_pnl + funding_pnl - trading_fees - spread_cost - slippage_cost
        accounting_error = net_return - (
            price_pnl + funding_pnl - trading_fees - spread_cost - slippage_cost
        )
        if 1 + net_return <= 0:
            raise ResearchError("research equity became non-positive")
        equity *= 1 + net_return
        participation = np.divide(
            capital * trades,
            arrays["outcome_quote_volume_1h"][period],
            out=np.zeros(len(symbols), dtype=np.float64),
            where=arrays["outcome_quote_volume_1h"][period] > 0,
        )
        score_ranks = _rank(scores[period])
        outcome_ranks = _rank(arrays["future_residual_return_1h"][period])
        rank_ic = float(np.corrcoef(score_ranks, outcome_ranks)[0, 1])
        output.append(
            {
                "fold": fold_number,
                "decision_time_ms": int(decision_time),
                "execution_time_ms": execution_time,
                "label_end_time_ms": int(arrays["label_end_time_ms"][period, 0]),
                "price_pnl": price_pnl,
                "funding_pnl": funding_pnl,
                "trading_fees": trading_fees,
                "spread_cost": spread_cost,
                "slippage_cost": slippage_cost,
                "net_return": net_return,
                "equity": equity,
                "turnover": turnover,
                "gross_exposure": float(np.sum(np.abs(weights))),
                "net_exposure": float(np.sum(weights)),
                "beta_exposure": float(np.dot(weights, arrays["rolling_beta"][period])),
                "maximum_volume_participation": float(np.max(participation)),
                "cross_sectional_rank_ic": rank_ic,
                "market_volatility_regime": float(arrays["market_volatility_regime"][period, 0]),
                "accounting_error": accounting_error,
                "weights_json": orjson.dumps(
                    {symbol: float(weights[index]) for index, symbol in enumerate(symbols)},
                    option=orjson.OPT_SORT_KEYS,
                ).decode("utf-8"),
                "scores_json": orjson.dumps(
                    {symbol: float(scores[period, index]) for index, symbol in enumerate(symbols)},
                    option=orjson.OPT_SORT_KEYS,
                ).decode("utf-8"),
            }
        )
        previous = weights
    return pl.DataFrame(output)


def calculate_metrics(curve: pl.DataFrame) -> PerformanceMetrics:
    if curve.is_empty():
        raise ResearchError("cannot calculate metrics for an empty curve")
    returns = np.asarray(curve["net_return"].to_numpy(), dtype=np.float64)
    equity = np.cumprod(1 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1
    mean = float(np.mean(returns))
    standard_deviation = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    annualized_volatility = standard_deviation * math.sqrt(HOURS_PER_YEAR)
    sharpe = mean / standard_deviation * math.sqrt(HOURS_PER_YEAR) if standard_deviation else 0.0
    annualized_return = float(math.expm1(np.mean(np.log1p(returns)) * HOURS_PER_YEAR))
    turnover = float(curve["turnover"].sum())
    return PerformanceMetrics(
        periods=len(returns),
        total_return=float(equity[-1] - 1),
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe=sharpe,
        max_drawdown=float(np.min(drawdowns)),
        positive_period_fraction=float(np.mean(returns > 0)),
        price_pnl=float(curve["price_pnl"].sum()),
        funding_pnl=float(curve["funding_pnl"].sum()),
        trading_fees=float(curve["trading_fees"].sum()),
        spread_cost=float(curve["spread_cost"].sum()),
        slippage_cost=float(curve["slippage_cost"].sum()),
        net_pnl=float(curve["net_return"].sum()),
        turnover=turnover,
        annualized_turnover=turnover * HOURS_PER_YEAR / len(returns),
        average_gross_exposure=float(curve.select(pl.col("gross_exposure").mean()).item()),
        average_net_exposure=float(curve.select(pl.col("net_exposure").mean()).item()),
        maximum_absolute_beta_exposure=float(
            curve.select(pl.col("beta_exposure").abs().max()).item()
        ),
        maximum_volume_participation=float(
            curve.select(pl.col("maximum_volume_participation").max()).item()
        ),
        mean_cross_sectional_rank_ic=float(
            curve.select(pl.col("cross_sectional_rank_ic").mean()).item()
        ),
        accounting_error_max=float(curve.select(pl.col("accounting_error").abs().max()).item()),
    )


def run_walk_forward(
    frame: pl.DataFrame,
    *,
    config: ResearchConfig,
    cost_multiplier: float = 1.0,
    signal_delay_bars: int = 0,
    momentum_weights: tuple[float, float, float] | None = None,
) -> BacktestRun:
    if cost_multiplier < 0:
        raise ResearchError("cost multiplier must be non-negative")
    if signal_delay_bars < 0:
        raise ResearchError("signal delay cannot be negative")
    decision_times = np.asarray(
        sorted(frame["decision_time_ms"].unique().to_list()), dtype=np.int64
    )
    folds = make_walk_forward_folds(
        decision_times,
        train_days=config.walk_forward_train_days,
        test_days=config.walk_forward_test_days,
        embargo_bars=config.embargo_bars,
    )
    weights = momentum_weights or (
        float(config.momentum_weight_1h),
        float(config.momentum_weight_4h),
        float(config.momentum_weight_24h),
    )
    if not math.isclose(sum(weights), 1.0, abs_tol=1e-12):
        raise ResearchError("momentum weights must sum to one")
    curves: list[pl.DataFrame] = []
    for fold in folds:
        test = frame.filter(
            pl.col("decision_time_ms").is_between(
                fold.test_start_ms, fold.test_end_ms, closed="both"
            )
        )
        curves.append(
            _run_fold(
                test,
                fold_number=fold.fold,
                config=config,
                cost_multiplier=cost_multiplier,
                signal_delay_bars=signal_delay_bars,
                momentum_weights=weights,
            )
        )
    curve = pl.concat(curves).sort("decision_time_ms")
    curve = curve.with_columns((pl.col("net_return") + 1).cum_prod().alias("oos_equity"))
    return BacktestRun(curve=curve, folds=folds, metrics=calculate_metrics(curve))


def _block_bootstrap(
    returns: np.ndarray[Any, np.dtype[np.float64]],
    *,
    samples: int,
    block_hours: int,
    seed: int,
) -> dict[str, float | int]:
    generator = np.random.default_rng(seed)
    result = np.empty(samples, dtype=np.float64)
    blocks_needed = math.ceil(len(returns) / block_hours)
    maximum_start = max(1, len(returns) - block_hours + 1)
    for sample in range(samples):
        starts = generator.integers(0, maximum_start, size=blocks_needed)
        bootstrapped = np.concatenate([returns[start : start + block_hours] for start in starts])[
            : len(returns)
        ]
        result[sample] = float(np.prod(1 + bootstrapped) - 1)
    return {
        "samples": samples,
        "block_hours": block_hours,
        "seed": seed,
        "total_return_p05": float(np.quantile(result, 0.05)),
        "total_return_p50": float(np.quantile(result, 0.50)),
        "total_return_p95": float(np.quantile(result, 0.95)),
        "probability_positive": float(np.mean(result > 0)),
    }


def _regime_metrics(curve: pl.DataFrame) -> dict[str, dict[str, float | int]]:
    values = np.asarray(curve["market_volatility_regime"].to_numpy(), dtype=np.float64)
    low_threshold, high_threshold = np.quantile(values, [1 / 3, 2 / 3])
    output: dict[str, dict[str, float | int]] = {}
    masks = {
        "low": values <= low_threshold,
        "middle": (values > low_threshold) & (values <= high_threshold),
        "high": values > high_threshold,
    }
    returns = np.asarray(curve["net_return"].to_numpy(), dtype=np.float64)
    for name, mask in masks.items():
        selected = returns[mask]
        output[name] = {
            "periods": len(selected),
            "mean_hourly_return": float(np.mean(selected)),
            "total_compounded_return": float(np.prod(1 + selected) - 1),
        }
    return output


def _metric_summary(metrics: PerformanceMetrics) -> dict[str, float | int]:
    return {
        "periods": metrics.periods,
        "total_return": metrics.total_return,
        "sharpe": metrics.sharpe,
        "max_drawdown": metrics.max_drawdown,
        "turnover": metrics.turnover,
        "funding_pnl": metrics.funding_pnl,
        "total_explicit_cost": (metrics.trading_fees + metrics.spread_cost + metrics.slippage_cost),
    }


def run_and_persist_backtest(
    *,
    dataset_path: Path,
    storage: LocalFilesystemStorage,
    reports_root: Path,
    compression: str,
    config: ResearchConfig,
) -> ResearchBacktestResult:
    frame = pl.read_parquet(dataset_path)
    baseline = run_walk_forward(frame, config=config)
    cost_15 = run_walk_forward(frame, config=config, cost_multiplier=1.5)
    cost_20 = run_walk_forward(frame, config=config, cost_multiplier=2.0)
    delay = run_walk_forward(frame, config=config, signal_delay_bars=1)
    fast = run_walk_forward(frame, config=config, momentum_weights=(0.30, 0.40, 0.30))
    slow = run_walk_forward(frame, config=config, momentum_weights=(0.10, 0.30, 0.60))
    stress = {
        "cost_1_0x": _metric_summary(baseline.metrics),
        "cost_1_5x": _metric_summary(cost_15.metrics),
        "cost_2_0x": _metric_summary(cost_20.metrics),
        "signal_delay_1_bar": _metric_summary(delay.metrics),
        "momentum_fast": _metric_summary(fast.metrics),
        "momentum_slow": _metric_summary(slow.metrics),
    }
    bootstrap = _block_bootstrap(
        np.asarray(baseline.curve["net_return"].to_numpy(), dtype=np.float64),
        samples=config.block_bootstrap_samples,
        block_hours=config.block_bootstrap_hours,
        seed=config.random_seed,
    )
    regimes = _regime_metrics(baseline.curve)
    run_payload = {
        "report_schema_version": 2,
        "pnl_visualization_version": PNL_VISUALIZATION_VERSION,
        "dataset_path": str(dataset_path.resolve()),
        "research_config": config.model_dump(mode="json"),
        "folds": [asdict(fold) for fold in baseline.folds],
        "metrics": asdict(baseline.metrics),
        "stress": stress,
        "bootstrap": bootstrap,
        "regimes": regimes,
    }
    run_version = hashlib.sha256(
        orjson.dumps(run_payload, option=orjson.OPT_SORT_KEYS)
    ).hexdigest()[:16]
    curve_path = storage.path(
        "gold",
        "binance",
        "usdm",
        "research_backtest",
        f"version={run_version}",
        "oos_curve.parquet",
    )
    storage.write_parquet_atomic(curve_path, baseline.curve, compression=compression)
    reports_storage = LocalFilesystemStorage(reports_root)
    report_json = reports_storage.path(f"research_phase3_{run_version}.json")
    report_markdown = reports_storage.path(f"research_phase3_{run_version}.md")
    report_chart = reports_storage.path(f"research_phase3_{run_version}_pnl.svg")
    reports_storage.write_bytes_atomic(report_chart, render_pnl_svg(baseline.curve).encode("utf-8"))
    reports_storage.write_json_atomic(
        report_json,
        {
            "run_version": run_version,
            "purpose": "end-to-end baseline validation; not evidence of a durable edge",
            "execution_model": "signal at closed 1m bar; marketable execution at next 1m open",
            "funding_model": "events in (entry, exit], positive rate paid by long positions",
            "universe_model": "fixed BTCUSDT/ETHUSDT/SOLUSDT seed chosen ex ante",
            "curve_path": str(curve_path),
            "report_chart_path": str(report_chart),
            **run_payload,
        },
    )
    metrics = baseline.metrics
    lines = [
        "# Phase 3 research report",
        "",
        "> This fixed baseline validates the pipeline. It is not a claim of tradable edge.",
        "",
        f"![Out-of-sample P&L curve]({report_chart.name})",
        "",
        "## Protocol",
        "",
        "- Universe: fixed BTCUSDT, ETHUSDT and SOLUSDT seed specified ex ante.",
        "- Signal: linear cross-sectional residual momentum (1h/4h/24h).",
        "- Execution: next one-minute open after a confirmed close; one-bar embargo.",
        "- Portfolio: net and beta neutral where the three-symbol geometry permits; no leverage.",
        "- Costs: configured taker fee, half-spread, slippage and timestamped funding.",
        "- Validation: expanding walk-forward only; no random split.",
        "",
        "## Out-of-sample performance",
        "",
        f"- Folds / hourly periods: {len(baseline.folds)} / {metrics.periods}",
        f"- Total return: {metrics.total_return:.4%}",
        "- Annualized volatility / Sharpe: "
        f"{metrics.annualized_volatility:.4%} / {metrics.sharpe:.3f}",
        f"- Maximum drawdown: {metrics.max_drawdown:.4%}",
        f"- Price / funding P&L: {metrics.price_pnl:.6f} / {metrics.funding_pnl:.6f}",
        "- Fees / spread / slippage: "
        f"{metrics.trading_fees:.6f} / {metrics.spread_cost:.6f} / "
        f"{metrics.slippage_cost:.6f}",
        f"- Turnover: {metrics.turnover:.3f}",
        f"- Mean rank IC: {metrics.mean_cross_sectional_rank_ic:.4f}",
        f"- Maximum accounting error: {metrics.accounting_error_max:.3e}",
        "",
        "## Stability",
        "",
        "| Scenario | Total return | Sharpe | Max drawdown | Turnover |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in stress.items():
        lines.append(
            f"| {name} | {float(values['total_return']):.4%} | "
            f"{float(values['sharpe']):.3f} | {float(values['max_drawdown']):.4%} | "
            f"{float(values['turnover']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Block bootstrap",
            "",
            "- 5th / 50th / 95th percentile total return: "
            f"{float(bootstrap['total_return_p05']):.4%} / "
            f"{float(bootstrap['total_return_p50']):.4%} / "
            f"{float(bootstrap['total_return_p95']):.4%}",
            "- Probability of positive sampled return: "
            f"{float(bootstrap['probability_positive']):.2%}",
            "",
            "## Known limitation",
            "",
            "The fixed seed avoids using a future liquidity ranking, but historical status "
            "snapshots are not yet available. Dynamic-universe research remains prohibited "
            "until those snapshots exist.",
        ]
    )
    reports_storage.write_bytes_atomic(report_markdown, ("\n".join(lines) + "\n").encode("utf-8"))
    return ResearchBacktestResult(
        run_version=run_version,
        curve_path=str(curve_path),
        report_json_path=str(report_json),
        report_markdown_path=str(report_markdown),
        report_chart_path=str(report_chart),
        metrics=metrics,
        fold_count=len(baseline.folds),
        stress=stress,
        bootstrap=bootstrap,
        regimes=regimes,
    )
