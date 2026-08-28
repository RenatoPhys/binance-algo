"""Vectorized bar research engine with explicit timing and accounting costs."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
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
from binance_algo.research.contracts import (
    FEATURE_KEY_COLUMNS,
    FoldContext,
    TrainingDataset,
    ValidationProfile,
)
from binance_algo.research.datasets.views import build_feature_view, build_target_view
from binance_algo.research.panel import WORKER_DATASET_CACHE, PanelData
from binance_algo.research.portfolio.base import PanelPortfolioPolicy, PortfolioPolicy
from binance_algo.research.strategies.base import PanelFittedStrategy, PanelStrategy, Strategy
from binance_algo.research.visualization import render_pnl_svg

HOURS_PER_YEAR = 24 * 365
ACCOUNTING_OUTCOME_FIELDS = (
    "future_return_1h",
    "future_residual_return_1h",
    "outcome_funding_rate_1h",
    "outcome_quote_volume_1h",
)
ACCOUNTING_METADATA_FIELDS = (
    "market_volatility_regime",
    "execution_time_ms",
    "label_end_time_ms",
)
ACCOUNTING_FIELDS = ("rolling_beta", *ACCOUNTING_OUTCOME_FIELDS, *ACCOUNTING_METADATA_FIELDS)


def accounting_metadata_columns(
    feature_columns: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(name for name in ACCOUNTING_METADATA_FIELDS if name not in feature_columns)


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
    scores: pl.DataFrame
    positions: pl.DataFrame


@dataclass(frozen=True, slots=True)
class _FoldBacktestRun:
    curve: pl.DataFrame
    scores: pl.DataFrame
    positions: pl.DataFrame


@dataclass(frozen=True, slots=True)
class ResearchBacktestResult:
    run_version: str
    curve_path: str
    report_json_path: str
    report_markdown_path: str
    report_chart_path: str | None
    metrics: PerformanceMetrics
    fold_count: int
    stress: dict[str, dict[str, float | int]]
    bootstrap: dict[str, float | int]
    regimes: dict[str, dict[str, float | int]]


@dataclass(frozen=True, slots=True)
class BacktestValidationResult:
    run: BacktestRun
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


def _rank(values: np.ndarray[Any, np.dtype[np.float64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _wide_arrays(
    frame: pl.DataFrame,
    *,
    fields: Sequence[str],
) -> tuple[
    tuple[str, ...],
    np.ndarray[Any, np.dtype[np.int64]],
    dict[str, np.ndarray[Any, np.dtype[np.float64]]],
]:
    panel = PanelData.from_frame(frame, outcome_columns=fields)
    panel.require_complete(role="backtest input")
    return panel.symbols, panel.times, dict(panel.outcomes)


def _long_value_matrix(
    frame: pl.DataFrame,
    *,
    value_column: str,
    symbols: tuple[str, ...],
    times: np.ndarray[Any, np.dtype[np.int64]],
    role: str,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    required = (*FEATURE_KEY_COLUMNS, value_column)
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise ResearchError(f"{role} is missing required columns: {missing}")
    try:
        panel = PanelData.from_frame(frame.select(required), feature_columns=(value_column,))
    except (KeyError, TypeError, ValueError, pl.exceptions.PolarsError) as exc:
        raise ResearchError(f"{role} keys or values are invalid") from exc
    if panel.symbols != symbols or not np.array_equal(panel.times, times):
        raise ResearchError(f"{role} does not cover the complete backtest panel")
    panel.require_complete(role=role)
    return np.asarray(panel.features[value_column], dtype=np.float64)


def _fee_schedule_covers(config: ResearchConfig, execution_time_ms: int) -> bool:
    event_date = datetime.fromtimestamp(execution_time_ms / 1_000, tz=UTC).date()
    schedule = config.fee_schedule
    return event_date >= schedule.effective_from and (
        schedule.effective_to is None or event_date <= schedule.effective_to
    )


def _run_fold(
    train: pl.DataFrame,
    test: pl.DataFrame,
    *,
    context: FoldContext,
    strategy: Strategy,
    portfolio_policy: PortfolioPolicy,
    config: ResearchConfig,
    cost_multiplier: float,
    signal_delay_bars: int,
    panel_data: PanelData | None = None,
) -> _FoldBacktestRun:
    train_features = build_feature_view(
        train,
        required_features=strategy.required_features(),
    )
    target_column = strategy.target_column()
    target: pl.DataFrame | None = None
    if target_column is not None:
        target = build_target_view(train, target_column=target_column)
    if panel_data is not None and isinstance(strategy, PanelStrategy):
        fitted = strategy.fit_panel(panel_data, target=target, context=context)
    else:
        fitted = strategy.fit(
            TrainingDataset(features=train_features, target=target),
            context=context,
        )
    if panel_data is not None and isinstance(fitted, PanelFittedStrategy):
        strategy_scores = fitted.score_panel(panel_data, context=context)
    else:
        test_features = build_feature_view(
            test,
            required_features=strategy.required_features(),
        )
        strategy_scores = fitted.score(test_features, context=context)
    score_frame = strategy_scores.frame
    if panel_data is not None and isinstance(portfolio_policy, PanelPortfolioPolicy):
        target_weight_frame = portfolio_policy.target_weights_panel(
            strategy_scores,
            panel_data,
            context=context,
        )
    else:
        market_state = build_feature_view(
            test,
            required_features=portfolio_policy.required_features(),
        )
        target_weight_frame = portfolio_policy.target_weights(
            score_frame,
            market_state,
            context=context,
        )
    if panel_data is None:
        symbols, times, arrays = _wide_arrays(test, fields=ACCOUNTING_FIELDS)
    else:
        time_slice = panel_data.time_slice(context.test_start_ms, context.test_end_ms)
        symbols = panel_data.symbols
        times = panel_data.times[time_slice]
        arrays = {
            field: np.asarray(
                panel_data.matrix(
                    field,
                    start_ms=context.test_start_ms,
                    end_ms=context.test_end_ms,
                ),
                dtype=np.float64,
            )
            for field in ACCOUNTING_FIELDS
        }
        if test.height != len(times) * len(symbols):
            raise ResearchError("fold frame and reusable PanelData do not align")
    scores = _long_value_matrix(
        score_frame,
        value_column="score",
        symbols=symbols,
        times=times,
        role="strategy scores",
    )
    targets = _long_value_matrix(
        target_weight_frame,
        value_column="target_weight",
        symbols=symbols,
        times=times,
        role="portfolio target weights",
    )
    if signal_delay_bars:
        delayed = np.zeros_like(targets)
        delayed[signal_delay_bars:] = targets[:-signal_delay_bars]
        targets = delayed
    previous = np.zeros(len(symbols), dtype=np.float64)
    equity = 1.0
    output: list[dict[str, object]] = []
    previous_weights = np.empty_like(targets)
    trade_weights = np.empty_like(targets)
    score_rank_matrix = np.empty_like(scores)
    price_contribution_matrix = np.empty_like(targets)
    funding_contribution_matrix = np.empty_like(targets)
    fee_matrix = np.empty_like(targets)
    spread_matrix = np.empty_like(targets)
    slippage_matrix = np.empty_like(targets)
    fee_rate = float(config.fee_schedule.taker_fee_rate) * cost_multiplier
    half_spread_rate = float(config.spread_bps) / 20_000 * cost_multiplier
    slippage_rate = float(config.slippage_bps) / 10_000 * cost_multiplier
    capital = float(config.initial_capital_usdt)
    for period, decision_time in enumerate(times):
        execution_time = int(arrays["execution_time_ms"][period, 0])
        if not _fee_schedule_covers(config, execution_time):
            raise ResearchError(f"fee schedule does not cover execution time {execution_time}")
        weights = targets[period]
        rebalances = weights - previous
        trades = np.abs(rebalances)
        if period == len(times) - 1:
            trades = trades + np.abs(weights)
        turnover = float(np.sum(trades))
        price_contributions = weights * arrays["future_return_1h"][period]
        funding_contributions = -weights * arrays["outcome_funding_rate_1h"][period]
        allocated_fees = trades * fee_rate
        allocated_spread = trades * half_spread_rate
        allocated_slippage = trades * slippage_rate
        previous_weights[period] = previous
        trade_weights[period] = trades
        price_contribution_matrix[period] = price_contributions
        funding_contribution_matrix[period] = funding_contributions
        fee_matrix[period] = allocated_fees
        spread_matrix[period] = allocated_spread
        slippage_matrix[period] = allocated_slippage
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
            capital * np.abs(rebalances),
            arrays["outcome_quote_volume_1h"][period],
            out=np.zeros(len(symbols), dtype=np.float64),
            where=arrays["outcome_quote_volume_1h"][period] > 0,
        )
        score_ranks = _rank(scores[period])
        score_rank_matrix[period] = score_ranks
        outcome_ranks = _rank(arrays["future_residual_return_1h"][period])
        rank_ic = float(np.corrcoef(score_ranks, outcome_ranks)[0, 1])
        output.append(
            {
                "fold": context.fold,
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
    repeated_times = np.repeat(times, len(symbols))
    tiled_symbols = np.tile(np.asarray(symbols), len(times))
    repeated_execution_times = np.repeat(
        np.asarray(arrays["execution_time_ms"][:, 0], dtype=np.int64), len(symbols)
    )
    explicit_cost_matrix = fee_matrix + spread_matrix + slippage_matrix
    gross_contribution_matrix = price_contribution_matrix + funding_contribution_matrix
    score_output = pl.DataFrame(
        {
            "fold": np.full(scores.size, context.fold, dtype=np.int64),
            "decision_time_ms": repeated_times,
            "symbol": tiled_symbols,
            "score": scores.reshape(-1),
            "score_rank": score_rank_matrix.reshape(-1),
        }
    )
    position_output = pl.DataFrame(
        {
            "fold": np.full(targets.size, context.fold, dtype=np.int64),
            "decision_time_ms": repeated_times,
            "execution_time_ms": repeated_execution_times,
            "symbol": tiled_symbols,
            "previous_weight": previous_weights.reshape(-1),
            "target_weight": targets.reshape(-1),
            "trade_weight": trade_weights.reshape(-1),
            "gross_contribution": gross_contribution_matrix.reshape(-1),
            "net_contribution": (gross_contribution_matrix - explicit_cost_matrix).reshape(-1),
            "beta_contribution": (targets * arrays["rolling_beta"]).reshape(-1),
            "future_return": arrays["future_return_1h"].reshape(-1),
            "funding_rate": arrays["outcome_funding_rate_1h"].reshape(-1),
            "price_pnl": price_contribution_matrix.reshape(-1),
            "funding_pnl": funding_contribution_matrix.reshape(-1),
            "allocated_fee": fee_matrix.reshape(-1),
            "allocated_spread_cost": spread_matrix.reshape(-1),
            "allocated_slippage_cost": slippage_matrix.reshape(-1),
        }
    )
    return _FoldBacktestRun(
        curve=pl.DataFrame(output),
        scores=score_output,
        positions=position_output,
    )


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
    strategy: Strategy,
    portfolio_policy: PortfolioPolicy,
    cost_multiplier: float = 1.0,
    signal_delay_bars: int = 0,
    panel_data: PanelData | None = None,
) -> BacktestRun:
    if cost_multiplier < 0:
        raise ResearchError("cost multiplier must be non-negative")
    if signal_delay_bars < 0:
        raise ResearchError("signal delay cannot be negative")
    if panel_data is None:
        feature_columns = tuple(
            dict.fromkeys((*strategy.required_features(), *portfolio_policy.required_features()))
        )
        panel_data = PanelData.from_frame(
            frame,
            feature_columns=feature_columns,
            outcome_columns=ACCOUNTING_OUTCOME_FIELDS,
            metadata_columns=accounting_metadata_columns(feature_columns),
        )
    panel_data.require_complete(role="walk-forward")
    if frame.height != panel_data.availability.size:
        raise ResearchError("walk-forward frame and PanelData have different shapes")
    decision_times = panel_data.times
    folds = make_walk_forward_folds(
        decision_times,
        train_days=config.walk_forward_train_days,
        test_days=config.walk_forward_test_days,
        embargo_bars=config.embargo_bars,
    )
    fold_runs: list[_FoldBacktestRun] = []
    for fold in folds:
        train = frame.filter(
            pl.col("decision_time_ms").is_between(
                fold.train_start_ms, fold.train_end_ms, closed="both"
            )
        )
        test = frame.filter(
            pl.col("decision_time_ms").is_between(
                fold.test_start_ms, fold.test_end_ms, closed="both"
            )
        )
        fold_runs.append(
            _run_fold(
                train,
                test,
                context=FoldContext(
                    fold=fold.fold,
                    train_start_ms=fold.train_start_ms,
                    train_end_ms=fold.train_end_ms,
                    test_start_ms=fold.test_start_ms,
                    test_end_ms=fold.test_end_ms,
                    embargo_bars=fold.embargo_bars,
                    random_seed=config.random_seed,
                ),
                strategy=strategy,
                portfolio_policy=portfolio_policy,
                config=config,
                cost_multiplier=cost_multiplier,
                signal_delay_bars=signal_delay_bars,
                panel_data=panel_data,
            )
        )
    curve = pl.concat([run.curve for run in fold_runs]).sort("decision_time_ms")
    curve = curve.with_columns((pl.col("net_return") + 1).cum_prod().alias("oos_equity"))
    scores = pl.concat([run.scores for run in fold_runs]).sort("decision_time_ms", "symbol")
    positions = pl.concat([run.positions for run in fold_runs]).sort("decision_time_ms", "symbol")
    return BacktestRun(
        curve=curve,
        folds=folds,
        metrics=calculate_metrics(curve),
        scores=scores,
        positions=positions,
    )


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


def run_research_validation(
    frame: pl.DataFrame,
    *,
    config: ResearchConfig,
    strategy: Strategy,
    portfolio_policy: PortfolioPolicy,
    strategy_stress: Mapping[str, Strategy] | None = None,
    panel_data: PanelData | None = None,
    profile: ValidationProfile = ValidationProfile.FULL,
) -> BacktestValidationResult:
    """Run the baseline and deterministic validation scenarios without persistence."""

    configured_strategy_stress = strategy_stress or {}
    if profile is ValidationProfile.DISCOVERY and configured_strategy_stress:
        raise ResearchError("discovery does not run additional strategy stress scenarios")
    if panel_data is None:
        feature_columns = tuple(
            dict.fromkeys(
                (
                    *strategy.required_features(),
                    *portfolio_policy.required_features(),
                    *(
                        feature
                        for stress_strategy in configured_strategy_stress.values()
                        for feature in stress_strategy.required_features()
                    ),
                )
            )
        )
        panel_data = PanelData.from_frame(
            frame,
            feature_columns=feature_columns,
            outcome_columns=ACCOUNTING_OUTCOME_FIELDS,
            metadata_columns=accounting_metadata_columns(feature_columns),
        )
    baseline = run_walk_forward(
        frame,
        config=config,
        strategy=strategy,
        portfolio_policy=portfolio_policy,
        panel_data=panel_data,
    )
    cost_15 = run_walk_forward(
        frame,
        config=config,
        strategy=strategy,
        portfolio_policy=portfolio_policy,
        cost_multiplier=1.5,
        panel_data=panel_data,
    )
    delay = run_walk_forward(
        frame,
        config=config,
        strategy=strategy,
        portfolio_policy=portfolio_policy,
        signal_delay_bars=1,
        panel_data=panel_data,
    )
    stress = {
        "cost_1_5x": _metric_summary(cost_15.metrics),
        "signal_delay_1_bar": _metric_summary(delay.metrics),
    }
    if profile is ValidationProfile.FULL:
        cost_20 = run_walk_forward(
            frame,
            config=config,
            strategy=strategy,
            portfolio_policy=portfolio_policy,
            cost_multiplier=2.0,
            panel_data=panel_data,
        )
        stress = {
            "cost_1_0x": _metric_summary(baseline.metrics),
            "cost_1_5x": stress["cost_1_5x"],
            "cost_2_0x": _metric_summary(cost_20.metrics),
            "signal_delay_1_bar": stress["signal_delay_1_bar"],
        }
    collisions = sorted(set(stress).intersection(configured_strategy_stress))
    if collisions:
        raise ResearchError(f"strategy stress names collide with engine stresses: {collisions}")
    for name, stress_strategy in configured_strategy_stress.items():
        stress[name] = _metric_summary(
            run_walk_forward(
                frame,
                config=config,
                strategy=stress_strategy,
                portfolio_policy=portfolio_policy,
                panel_data=panel_data,
            ).metrics
        )
    bootstrap = (
        _block_bootstrap(
            np.asarray(baseline.curve["net_return"].to_numpy(), dtype=np.float64),
            samples=config.block_bootstrap_samples,
            block_hours=config.block_bootstrap_hours,
            seed=config.random_seed,
        )
        if profile is ValidationProfile.FULL
        else {}
    )
    return BacktestValidationResult(
        run=baseline,
        stress=stress,
        bootstrap=bootstrap,
        regimes=_regime_metrics(baseline.curve),
    )


def run_and_persist_backtest(
    *,
    dataset_path: Path,
    storage: LocalFilesystemStorage,
    reports_root: Path,
    compression: str,
    config: ResearchConfig,
    strategy: Strategy,
    portfolio_policy: PortfolioPolicy,
    strategy_stress: Mapping[str, Strategy],
    generate_chart: bool = False,
) -> ResearchBacktestResult:
    feature_columns = tuple(
        dict.fromkeys(
            (
                *strategy.required_features(),
                *portfolio_policy.required_features(),
                *(
                    feature
                    for stress_strategy in strategy_stress.values()
                    for feature in stress_strategy.required_features()
                ),
            )
        )
    )
    loaded_dataset = WORKER_DATASET_CACHE.load(
        dataset_path,
        feature_columns=feature_columns,
        outcome_columns=ACCOUNTING_OUTCOME_FIELDS,
        metadata_columns=accounting_metadata_columns(feature_columns),
    )
    validation = run_research_validation(
        loaded_dataset.frame,
        config=config,
        strategy=strategy,
        portfolio_policy=portfolio_policy,
        strategy_stress=strategy_stress,
        panel_data=loaded_dataset.panel,
    )
    baseline = validation.run
    stress = validation.stress
    bootstrap = validation.bootstrap
    regimes = validation.regimes
    run_payload = {
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
    report_chart: Path | None = None
    if generate_chart:
        report_chart = reports_storage.path(f"research_phase3_{run_version}_pnl.svg")
        reports_storage.write_bytes_atomic(
            report_chart, render_pnl_svg(baseline.curve).encode("utf-8")
        )
    reports_storage.write_json_atomic(
        report_json,
        {
            "run_version": run_version,
            "purpose": "end-to-end baseline validation; not evidence of a durable edge",
            "execution_model": "signal at closed 1m bar; marketable execution at next 1m open",
            "funding_model": "events in (entry, exit], positive rate paid by long positions",
            "universe_model": "fixed BTCUSDT/ETHUSDT/SOLUSDT seed chosen ex ante",
            "curve_path": str(curve_path),
            **run_payload,
        },
    )
    metrics = baseline.metrics
    lines = [
        "# Phase 3 research report",
        "",
        "> This fixed baseline validates the pipeline. It is not a claim of tradable edge.",
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
        report_chart_path=str(report_chart) if report_chart is not None else None,
        metrics=metrics,
        fold_count=len(baseline.folds),
        stress=stress,
        bootstrap=bootstrap,
        regimes=regimes,
    )
