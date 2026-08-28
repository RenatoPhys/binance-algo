"""Compatibility orchestration for the fixed Phase 3 baseline on the Phase 3.5 engine."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from binance_algo.config import ResearchConfig
from binance_algo.data.storage import LocalFilesystemStorage
from binance_algo.research.backtest import (
    BacktestRun,
    ResearchBacktestResult,
    run_and_persist_backtest,
    run_walk_forward,
)
from binance_algo.research.portfolio.neutral_long_short import (
    NeutralLongShortParameters,
    NeutralLongShortPolicy,
)
from binance_algo.research.strategies.residual_momentum import (
    ResidualMomentumParameters,
    ResidualMomentumStrategy,
)


@dataclass(frozen=True, slots=True)
class Phase3BaselineComponents:
    strategy: ResidualMomentumStrategy
    portfolio_policy: NeutralLongShortPolicy
    strategy_stress: tuple[tuple[str, ResidualMomentumStrategy], ...]


def _strategy(
    config: ResearchConfig,
    weights: tuple[float, float, float] | None = None,
) -> ResidualMomentumStrategy:
    selected = weights or (
        float(config.momentum_weight_1h),
        float(config.momentum_weight_4h),
        float(config.momentum_weight_24h),
    )
    return ResidualMomentumStrategy(
        parameters=ResidualMomentumParameters(
            momentum_weight_1h=selected[0],
            momentum_weight_4h=selected[1],
            momentum_weight_24h=selected[2],
        )
    )


def build_phase3_baseline_components(config: ResearchConfig) -> Phase3BaselineComponents:
    """Translate legacy global config into explicit strategy and portfolio definitions."""

    policy = NeutralLongShortPolicy(
        parameters=NeutralLongShortParameters(
            no_trade_score_band=float(config.no_trade_score_band),
            gross_exposure=float(config.gross_exposure),
            annual_volatility_target=float(config.annual_volatility_target),
            max_symbol_weight=float(config.max_symbol_weight),
        )
    )
    return Phase3BaselineComponents(
        strategy=_strategy(config),
        portfolio_policy=policy,
        strategy_stress=(
            ("momentum_fast", _strategy(config, (0.30, 0.40, 0.30))),
            ("momentum_slow", _strategy(config, (0.10, 0.30, 0.60))),
        ),
    )


def run_phase3_walk_forward(
    frame: pl.DataFrame,
    *,
    config: ResearchConfig,
    cost_multiplier: float = 1.0,
    signal_delay_bars: int = 0,
    momentum_weights: tuple[float, float, float] | None = None,
) -> BacktestRun:
    """Run the legacy baseline through the generic strategy/policy engine."""

    components = build_phase3_baseline_components(config)
    strategy = _strategy(config, momentum_weights) if momentum_weights else components.strategy
    return run_walk_forward(
        frame,
        config=config,
        strategy=strategy,
        portfolio_policy=components.portfolio_policy,
        cost_multiplier=cost_multiplier,
        signal_delay_bars=signal_delay_bars,
    )


def run_and_persist_phase3_baseline(
    *,
    dataset_path: Path,
    storage: LocalFilesystemStorage,
    reports_root: Path,
    compression: str,
    config: ResearchConfig,
    generate_chart: bool = False,
) -> ResearchBacktestResult:
    """Preserve the Phase 3 CLI behavior while using the generic Phase 3.5 engine."""

    components = build_phase3_baseline_components(config)
    return run_and_persist_backtest(
        dataset_path=dataset_path,
        storage=storage,
        reports_root=reports_root,
        compression=compression,
        config=config,
        strategy=components.strategy,
        portfolio_policy=components.portfolio_policy,
        strategy_stress=dict(components.strategy_stress),
        generate_chart=generate_chart,
    )


__all__ = [
    "Phase3BaselineComponents",
    "build_phase3_baseline_components",
    "run_and_persist_phase3_baseline",
    "run_phase3_walk_forward",
]
