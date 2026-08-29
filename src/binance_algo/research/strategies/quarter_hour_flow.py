"""Quarter-hour opening-flow continuation with fixed causal holding periods."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import FoldContext, StrategyScores, TrainingDataset
from binance_algo.research.features.rolling import trailing_sum
from binance_algo.research.panel import PanelData, matrix_to_long_frame
from binance_algo.research.state_machine import causal_sparse_state
from binance_algo.research.strategies.fixed import projected_panel, validate_fixed_panel_fit

QUARTER_HOUR_FLOW_FEATURES = ("quarter_flow_excess_z_168h",)


@dataclass(frozen=True, slots=True)
class QuarterHourFlowParameters:
    aggregation_hours: int
    hold_hours: int

    def __post_init__(self) -> None:
        if self.aggregation_hours not in {1, 2}:
            raise ResearchError("quarter-hour aggregation must be one or two hours")
        if self.hold_hours not in {4, 8, 12}:
            raise ResearchError("quarter-hour holding period must be 4, 8 or 12 hours")


def quarter_hour_flow_scores(
    panel: PanelData,
    *,
    parameters: QuarterHourFlowParameters,
    context: FoldContext,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    values = np.asarray(panel.matrix("quarter_flow_excess_z_168h"), dtype=np.float64)
    aggregate = trailing_sum(values, parameters.aggregation_hours) / parameters.aggregation_hours
    time_slice = panel.time_slice(context.test_start_ms, context.test_end_ms)
    raw = aggregate[time_slice]
    entry = np.where(np.abs(raw) >= 1.0, np.clip(raw, -3.0, 3.0), 0.0)
    return causal_sparse_state(entry, hold_hours=parameters.hold_hours).values


@dataclass(frozen=True, slots=True)
class FittedQuarterHourFlowStrategy:
    parameters: QuarterHourFlowParameters

    def score(self, features: pl.DataFrame, *, context: FoldContext) -> StrategyScores:
        panel = projected_panel(
            features,
            required_features=QUARTER_HOUR_FLOW_FEATURES,
            start_ms=context.test_start_ms,
            end_ms=context.test_end_ms,
            role="quarter-hour flow scoring",
        )
        return self.score_panel(panel, context=context)

    def score_panel(self, features: PanelData, *, context: FoldContext) -> StrategyScores:
        features.require_complete_range(
            context.test_start_ms,
            context.test_end_ms,
            role="quarter-hour flow scoring",
        )
        time_slice = features.time_slice(context.test_start_ms, context.test_end_ms)
        return StrategyScores(
            matrix_to_long_frame(
                times=features.times[time_slice],
                symbols=features.symbols,
                value_name="score",
                values=quarter_hour_flow_scores(
                    features,
                    parameters=self.parameters,
                    context=context,
                ),
            )
        )


@dataclass(frozen=True, slots=True)
class QuarterHourFlowStrategy:
    parameters: QuarterHourFlowParameters
    strategy_id: str = field(default="quarter_hour_flow", init=False)
    strategy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return QUARTER_HOUR_FLOW_FEATURES

    def target_column(self) -> str | None:
        return None

    def fit(
        self,
        train: TrainingDataset,
        *,
        context: FoldContext,
    ) -> FittedQuarterHourFlowStrategy:
        if train.target is not None:
            raise ResearchError("quarter-hour flow does not accept a training target")
        projected_panel(
            train.features,
            required_features=self.required_features(),
            start_ms=context.train_start_ms,
            end_ms=context.train_end_ms,
            role="quarter-hour flow training",
        )
        return FittedQuarterHourFlowStrategy(self.parameters)

    def fit_panel(
        self,
        train: PanelData,
        *,
        target: pl.DataFrame | None,
        context: FoldContext,
    ) -> FittedQuarterHourFlowStrategy:
        validate_fixed_panel_fit(
            train,
            required_features=self.required_features(),
            target=target,
            context=context,
            role="quarter-hour flow",
        )
        return FittedQuarterHourFlowStrategy(self.parameters)


__all__ = [
    "QUARTER_HOUR_FLOW_FEATURES",
    "FittedQuarterHourFlowStrategy",
    "QuarterHourFlowParameters",
    "QuarterHourFlowStrategy",
    "quarter_hour_flow_scores",
]
