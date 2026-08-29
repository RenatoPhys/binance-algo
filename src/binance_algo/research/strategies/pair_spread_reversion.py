"""Fold-frozen ETH/BTC and SOL/BTC spread convergence."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext, StrategyScores, TrainingDataset
from binance_algo.research.panel import PanelData
from binance_algo.research.state_machine import causal_sparse_state
from binance_algo.research.strategies.fixed import projected_panel

PAIR_SPREAD_FEATURES = ("log_close_level",)
PAIR_DEFINITIONS = (("ETHUSDT", "pair_eth_btc"), ("SOLUSDT", "pair_sol_btc"))
FIT_WINDOW_HOURS = 720
MAX_HOLD_HOURS = 96


@dataclass(frozen=True, slots=True)
class PairSpreadReversionParameters:
    spread_z_window_hours: int
    entry_z: float

    def __post_init__(self) -> None:
        if self.spread_z_window_hours not in {168, 336}:
            raise ResearchError("pair spread z-score window must be 168 or 336 hours")
        if self.entry_z not in {1.5, 2.0}:
            raise ResearchError("pair spread entry z-score must be 1.5 or 2.0")


@dataclass(frozen=True, slots=True)
class FoldFrozenPairModel:
    pair_id: str
    alt_symbol: str
    alpha: float
    beta: float
    residual_mean: float
    residual_std: float
    half_life_hours: float
    eligible: bool


def _fit_pair(
    alt: np.ndarray[Any, np.dtype[np.float64]],
    btc: np.ndarray[Any, np.dtype[np.float64]],
    *,
    pair_id: str,
    alt_symbol: str,
    z_window: int,
) -> FoldFrozenPairModel:
    if len(alt) < FIT_WINDOW_HOURS or len(btc) != len(alt):
        return FoldFrozenPairModel(pair_id, alt_symbol, 0.0, math.nan, 0.0, 0.0, math.inf, False)
    y = alt[-FIT_WINDOW_HOURS:]
    x = btc[-FIT_WINDOW_HOURS:]
    finite = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(finite)) != FIT_WINDOW_HOURS:
        return FoldFrozenPairModel(pair_id, alt_symbol, 0.0, math.nan, 0.0, 0.0, math.inf, False)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    variance = float(np.mean(np.square(x - x_mean)))
    if variance <= 1e-15:
        return FoldFrozenPairModel(pair_id, alt_symbol, 0.0, math.nan, 0.0, 0.0, math.inf, False)
    beta = float(np.mean((x - x_mean) * (y - y_mean)) / variance)
    alpha = y_mean - beta * x_mean
    residual = y - alpha - beta * x
    normalization = residual[-z_window:]
    residual_mean = float(np.mean(normalization))
    residual_std = float(np.std(normalization))
    centered = residual - float(np.mean(residual))
    denominator = float(np.dot(centered[:-1], centered[:-1]))
    phi = (
        float(np.dot(centered[:-1], centered[1:]) / denominator)
        if denominator > 1e-15
        else math.nan
    )
    half_life = -math.log(2.0) / math.log(phi) if 0 < phi < 1 else math.inf
    eligible = 0.20 <= beta <= 3.00 and 2.0 <= half_life <= 168.0 and residual_std > 1e-15
    return FoldFrozenPairModel(
        pair_id=pair_id,
        alt_symbol=alt_symbol,
        alpha=alpha,
        beta=beta,
        residual_mean=residual_mean,
        residual_std=residual_std,
        half_life_hours=half_life,
        eligible=eligible,
    )


def _fit_models(
    panel: PanelData,
    *,
    context: FoldContext,
    parameters: PairSpreadReversionParameters,
) -> tuple[FoldFrozenPairModel, ...]:
    missing = sorted({"BTCUSDT", "ETHUSDT", "SOLUSDT"}.difference(panel.symbols))
    if missing:
        raise ResearchError(f"pair-spread universe is missing symbols: {missing}")
    levels = np.asarray(
        panel.matrix(
            "log_close_level",
            start_ms=context.train_start_ms,
            end_ms=context.train_end_ms,
        ),
        dtype=np.float64,
    )
    btc = levels[:, panel.symbols.index("BTCUSDT")]
    return tuple(
        _fit_pair(
            levels[:, panel.symbols.index(alt_symbol)],
            btc,
            pair_id=pair_id,
            alt_symbol=alt_symbol,
            z_window=parameters.spread_z_window_hours,
        )
        for alt_symbol, pair_id in PAIR_DEFINITIONS
    )


@dataclass(frozen=True, slots=True)
class FittedPairSpreadReversionStrategy:
    parameters: PairSpreadReversionParameters
    symbols: tuple[str, ...]
    models: tuple[FoldFrozenPairModel, ...]

    def _score_panel(self, panel: PanelData, *, context: FoldContext) -> StrategyScores:
        if panel.symbols != self.symbols:
            raise ResearchError("pair-spread symbols differ from fitted fold")
        panel.require_complete_range(
            context.test_start_ms,
            context.test_end_ms,
            role="pair-spread scoring",
        )
        time_slice = panel.time_slice(context.test_start_ms, context.test_end_ms)
        levels = np.asarray(panel.matrix("log_close_level"), dtype=np.float64)[time_slice]
        pair_values: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
        for model in self.models:
            contribution = np.zeros_like(levels)
            if model.eligible:
                btc_index = panel.symbols.index("BTCUSDT")
                alt_index = panel.symbols.index(model.alt_symbol)
                spread = levels[:, alt_index] - model.alpha - model.beta * levels[:, btc_index]
                zscore = (spread - model.residual_mean) / model.residual_std
                entry = np.where(
                    np.abs(zscore) >= self.parameters.entry_z,
                    -np.sign(zscore),
                    0.0,
                )[:, None]
                state = causal_sparse_state(
                    entry,
                    hold_hours=MAX_HOLD_HOURS,
                    explicit_exit=(np.abs(zscore) <= 0.5)[:, None],
                ).values[:, 0]
                contribution[:, alt_index] = state
                contribution[:, btc_index] = -model.beta * state
            pair_values[model.pair_id] = contribution
        score = np.sum(np.stack(tuple(pair_values.values())), axis=0)
        repeated_times = np.repeat(panel.times[time_slice], len(panel.symbols))
        tiled_symbols = np.tile(np.asarray(panel.symbols), len(panel.times[time_slice]))
        frame = pl.DataFrame(
            {
                "decision_time_ms": repeated_times,
                "symbol": tiled_symbols,
                "score": score.reshape(-1),
                **{name: values.reshape(-1) for name, values in sorted(pair_values.items())},
                **{
                    f"{model.pair_id}_{field_name}": np.full(
                        score.size,
                        float(np.nan_to_num(value, nan=0.0, posinf=1e9, neginf=-1e9)),
                        dtype=np.float64,
                    )
                    for model in self.models
                    for field_name, value in (
                        ("beta", model.beta),
                        ("half_life_hours", model.half_life_hours),
                        ("eligible", float(model.eligible)),
                    )
                },
            }
        )
        return StrategyScores(frame)

    def score(self, features: pl.DataFrame, *, context: FoldContext) -> StrategyScores:
        panel = projected_panel(
            features,
            required_features=PAIR_SPREAD_FEATURES,
            start_ms=context.test_start_ms,
            end_ms=context.test_end_ms,
            role="pair-spread scoring",
        )
        return self._score_panel(panel, context=context)

    def score_panel(self, features: PanelData, *, context: FoldContext) -> StrategyScores:
        return self._score_panel(features, context=context)


@dataclass(frozen=True, slots=True)
class PairSpreadReversionStrategy:
    parameters: PairSpreadReversionParameters
    strategy_id: str = field(default="pair_spread_reversion", init=False)
    strategy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return PAIR_SPREAD_FEATURES

    def target_column(self) -> str | None:
        return None

    def fit(
        self,
        train: TrainingDataset,
        *,
        context: FoldContext,
    ) -> FittedPairSpreadReversionStrategy:
        if train.target is not None:
            raise ResearchError("pair-spread reversion does not accept a training target")
        panel = projected_panel(
            train.features,
            required_features=self.required_features(),
            start_ms=context.train_start_ms,
            end_ms=context.train_end_ms,
            role="pair-spread training",
        )
        return FittedPairSpreadReversionStrategy(
            parameters=self.parameters,
            symbols=panel.symbols,
            models=_fit_models(panel, context=context, parameters=self.parameters),
        )

    def fit_panel(
        self,
        train: PanelData,
        *,
        target: pl.DataFrame | None,
        context: FoldContext,
    ) -> FittedPairSpreadReversionStrategy:
        if target is not None:
            raise ResearchError("pair-spread reversion does not accept a training target")
        train.require_complete_range(
            context.train_start_ms,
            context.train_end_ms,
            role="pair-spread training",
        )
        return FittedPairSpreadReversionStrategy(
            parameters=self.parameters,
            symbols=train.symbols,
            models=_fit_models(train, context=context, parameters=self.parameters),
        )


__all__ = [
    "FIT_WINDOW_HOURS",
    "MAX_HOLD_HOURS",
    "PAIR_DEFINITIONS",
    "PAIR_SPREAD_FEATURES",
    "FittedPairSpreadReversionStrategy",
    "FoldFrozenPairModel",
    "PairSpreadReversionParameters",
    "PairSpreadReversionStrategy",
]
