"""Aggressive-flow absorption reversal after a weak or contrary price response."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext, StrategyScores, TrainingDataset
from binance_algo.research.panel import PanelData, matrix_to_long_frame
from binance_algo.research.state_machine import causal_sparse_state
from binance_algo.research.strategies.fixed import projected_panel, validate_fixed_panel_fit

FLOW_ABSORPTION_FEATURES = (
    "signed_taker_flow_z_168h",
    "quote_volume_zscore_24h",
    "flow_price_agreement_1h",
)


@dataclass(frozen=True, slots=True)
class FlowAbsorptionParameters:
    flow_z_threshold: float
    hold_hours: int

    def __post_init__(self) -> None:
        if self.flow_z_threshold not in {1.5, 2.0}:
            raise ResearchError("flow-absorption threshold must be 1.5 or 2.0")
        if self.hold_hours not in {4, 8}:
            raise ResearchError("flow-absorption holding period must be four or eight hours")


def flow_absorption_scores(
    panel: PanelData,
    *,
    parameters: FlowAbsorptionParameters,
    context: FoldContext,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    time_slice = panel.time_slice(context.test_start_ms, context.test_end_ms)
    flow_z = np.asarray(panel.matrix("signed_taker_flow_z_168h"), dtype=np.float64)[time_slice]
    volume_z = np.asarray(panel.matrix("quote_volume_zscore_24h"), dtype=np.float64)[time_slice]
    agreement = np.asarray(panel.matrix("flow_price_agreement_1h"), dtype=np.float64)[time_slice]
    trigger = (
        (np.abs(flow_z) >= parameters.flow_z_threshold) & (volume_z >= 0.0) & (agreement <= 0.25)
    )
    entry = np.where(trigger, -np.clip(flow_z, -3.0, 3.0), 0.0)
    return causal_sparse_state(entry, hold_hours=parameters.hold_hours).values


@dataclass(frozen=True, slots=True)
class FittedFlowAbsorptionStrategy:
    parameters: FlowAbsorptionParameters

    def score(self, features: pl.DataFrame, *, context: FoldContext) -> StrategyScores:
        panel = projected_panel(
            features,
            required_features=FLOW_ABSORPTION_FEATURES,
            start_ms=context.test_start_ms,
            end_ms=context.test_end_ms,
            role="flow-absorption scoring",
        )
        return self.score_panel(panel, context=context)

    def score_panel(self, features: PanelData, *, context: FoldContext) -> StrategyScores:
        features.require_complete_range(
            context.test_start_ms,
            context.test_end_ms,
            role="flow-absorption scoring",
        )
        time_slice = features.time_slice(context.test_start_ms, context.test_end_ms)
        return StrategyScores(
            matrix_to_long_frame(
                times=features.times[time_slice],
                symbols=features.symbols,
                value_name="score",
                values=flow_absorption_scores(
                    features,
                    parameters=self.parameters,
                    context=context,
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class FlowAbsorptionStrategy:
    parameters: FlowAbsorptionParameters
    strategy_id: str = field(default="flow_absorption_reversal", init=False)
    strategy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return FLOW_ABSORPTION_FEATURES

    def target_column(self) -> str | None:
        return None

    def fit(
        self,
        train: TrainingDataset,
        *,
        context: FoldContext,
    ) -> FittedFlowAbsorptionStrategy:
        if train.target is not None:
            raise ResearchError("flow absorption does not accept a training target")
        projected_panel(
            train.features,
            required_features=self.required_features(),
            start_ms=context.train_start_ms,
            end_ms=context.train_end_ms,
            role="flow-absorption training",
        )
        return FittedFlowAbsorptionStrategy(self.parameters)

    def fit_panel(
        self,
        train: PanelData,
        *,
        target: pl.DataFrame | None,
        context: FoldContext,
    ) -> FittedFlowAbsorptionStrategy:
        validate_fixed_panel_fit(
            train,
            required_features=self.required_features(),
            target=target,
            context=context,
            role="flow absorption",
        )
        return FittedFlowAbsorptionStrategy(self.parameters)


__all__ = [
    "FLOW_ABSORPTION_FEATURES",
    "FittedFlowAbsorptionStrategy",
    "FlowAbsorptionParameters",
    "FlowAbsorptionStrategy",
    "flow_absorption_scores",
]
