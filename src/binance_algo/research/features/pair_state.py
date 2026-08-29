"""Causal level features for fold-frozen relative-value models."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from binance_algo.common.errors import ResearchError
from binance_algo.research.features.base import (
    FeatureArray,
    FeatureComputeContext,
    FeatureDefinition,
)

PAIR_STATE_FEATURES = (
    FeatureDefinition(
        feature_id="log_close_level:v1",
        name="log_close_level",
        version="v1",
        description="Natural logarithm of the latest fully closed one-minute close.",
        dtype="Float64",
        lookback="latest closed bar",
        timestamp_semantics="closed one-minute close at decision_time_ms",
        required_datasets=("klines",),
        required_columns=("open_time_ms", "close"),
        implementation_path="binance_algo.research.features.pair_state",
        parameters={},
    ),
)


class PairStateBundle:
    bundle_id = "pair_state"
    version = "v1"

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return PAIR_STATE_FEATURES

    def compute(
        self,
        context: FeatureComputeContext,
        parameters: Mapping[str, Any],
    ) -> Mapping[str, FeatureArray]:
        if parameters:
            raise ResearchError("pair_state:v1 does not accept parameters")
        return {"log_close_level": context.log_close[context.decision_indices]}


__all__ = ["PAIR_STATE_FEATURES", "PairStateBundle"]
