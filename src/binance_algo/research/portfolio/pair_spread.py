"""Whole-spread scaling that preserves fold-frozen pair hedge ratios."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext, StrategyScores
from binance_algo.research.panel import PanelData

PAIR_SCORE_COLUMNS = ("pair_eth_btc", "pair_sol_btc")
PAIR_TARGET_COLUMNS = tuple(f"{name}_target_weight" for name in PAIR_SCORE_COLUMNS)
PAIR_POLICY_FEATURES = ("realized_volatility_24h",)


@dataclass(frozen=True, slots=True)
class BufferedPairSpreadParameters:
    gross_exposure: float
    annual_volatility_target: float
    max_symbol_weight: float
    maximum_active_pairs: int

    def __post_init__(self) -> None:
        if not 0 < self.gross_exposure <= 1:
            raise ResearchError("pair gross exposure must be in (0, 1]")
        if not 0 < self.annual_volatility_target <= 1:
            raise ResearchError("pair volatility target must be in (0, 1]")
        if not 0 < self.max_symbol_weight <= 1:
            raise ResearchError("pair maximum symbol weight must be in (0, 1]")
        if self.maximum_active_pairs != 2:
            raise ResearchError("pair policy v1 requires maximum_active_pairs=2")


def _target_frame(
    scores: StrategyScores,
    market_state: PanelData,
    *,
    context: FoldContext,
    parameters: BufferedPairSpreadParameters,
) -> pl.DataFrame:
    score_panel = PanelData.from_frame(
        scores.frame,
        feature_columns=("score", *PAIR_SCORE_COLUMNS),
    )
    time_slice = market_state.time_slice(context.test_start_ms, context.test_end_ms)
    times = market_state.times[time_slice]
    if score_panel.symbols != market_state.symbols or not np.array_equal(score_panel.times, times):
        raise ResearchError("pair scores and market-state keys must align exactly")
    raw_pairs = {
        name: np.asarray(score_panel.features[name], dtype=np.float64)
        for name in PAIR_SCORE_COLUMNS
    }
    raw = np.asarray(score_panel.features["score"], dtype=np.float64)
    if not np.allclose(raw, np.sum(np.stack(tuple(raw_pairs.values())), axis=0), atol=1e-12):
        raise ResearchError("aggregate pair score differs from its pair sleeves")
    volatility = np.asarray(
        market_state.matrix(
            "realized_volatility_24h",
            start_ms=context.test_start_ms,
            end_ms=context.test_end_ms,
        ),
        dtype=np.float64,
    )
    targets = np.zeros_like(raw)
    pair_targets = {name: np.zeros_like(raw) for name in PAIR_SCORE_COLUMNS}
    for row in range(len(raw)):
        active_pairs = sum(
            float(np.sum(np.abs(values[row]))) > 1e-15 for values in raw_pairs.values()
        )
        if active_pairs > parameters.maximum_active_pairs:
            raise ResearchError("pair strategy exceeded maximum_active_pairs")
        gross = float(np.sum(np.abs(raw[row])))
        if gross <= 1e-15:
            continue
        normalized = raw[row] / gross
        annualized_asset_volatility = volatility[row] * math.sqrt(365.0)
        volatility_proxy = float(
            math.sqrt(np.sum(np.square(normalized * annualized_asset_volatility)))
        )
        target_gross = parameters.gross_exposure
        if volatility_proxy > 1e-15:
            target_gross = min(
                target_gross,
                parameters.annual_volatility_target / volatility_proxy,
            )
        scale = target_gross / gross
        weights = raw[row] * scale
        maximum = float(np.max(np.abs(weights)))
        if maximum > parameters.max_symbol_weight:
            scale *= parameters.max_symbol_weight / maximum
            weights = raw[row] * scale
        targets[row] = weights
        for name, values in raw_pairs.items():
            pair_targets[name][row] = values[row] * scale
        if not np.allclose(
            weights,
            np.sum(np.stack(tuple(values[row] for values in pair_targets.values())), axis=0),
            atol=1e-12,
        ):
            raise ResearchError("global pair scaling changed a hedge sleeve direction")
    repeated_times = np.repeat(times, len(market_state.symbols))
    tiled_symbols = np.tile(np.asarray(market_state.symbols), len(times))
    return pl.DataFrame(
        {
            "decision_time_ms": repeated_times,
            "symbol": tiled_symbols,
            "target_weight": targets.reshape(-1),
            **{
                f"{name}_target_weight": values.reshape(-1)
                for name, values in sorted(pair_targets.items())
            },
        }
    )


@dataclass(frozen=True, slots=True)
class BufferedPairSpreadPolicy:
    parameters: BufferedPairSpreadParameters
    policy_id: str = field(default="buffered_pair_spread", init=False)
    policy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return PAIR_POLICY_FEATURES

    def target_weights(
        self,
        scores: pl.DataFrame,
        market_state: pl.DataFrame,
        *,
        context: FoldContext,
    ) -> pl.DataFrame:
        panel = PanelData.from_frame(market_state, feature_columns=PAIR_POLICY_FEATURES)
        return _target_frame(
            StrategyScores(scores),
            panel,
            context=context,
            parameters=self.parameters,
        )

    def target_weights_panel(
        self,
        scores: StrategyScores,
        market_state: PanelData,
        *,
        context: FoldContext,
    ) -> pl.DataFrame:
        return _target_frame(
            scores,
            market_state,
            context=context,
            parameters=self.parameters,
        )


__all__ = [
    "PAIR_POLICY_FEATURES",
    "PAIR_SCORE_COLUMNS",
    "PAIR_TARGET_COLUMNS",
    "BufferedPairSpreadParameters",
    "BufferedPairSpreadPolicy",
]
