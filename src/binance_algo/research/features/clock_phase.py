"""UTC clock-phase features derived from fully closed one-minute klines."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from binance_algo.common.errors import ResearchError
from binance_algo.research.features.base import (
    FeatureArray,
    FeatureComputeContext,
    FeatureDefinition,
)
from binance_algo.research.features.rolling import rolling_zscore

QUARTER_OPEN_MINUTES = frozenset((0, 15, 30, 45))


def _definition(name: str, *, lookback: str, description: str) -> FeatureDefinition:
    return FeatureDefinition(
        feature_id=f"{name}:v1",
        name=name,
        version="v1",
        description=description,
        dtype="Float64",
        lookback=lookback,
        timestamp_semantics="fully closed UTC one-minute bars at or before decision_time_ms",
        required_datasets=("klines",),
        required_columns=(
            "open_time_ms",
            "open",
            "close",
            "quote_volume",
            "taker_buy_quote_volume",
        ),
        implementation_path="binance_algo.research.features.clock_phase",
        parameters={},
    )


CLOCK_PHASE_FEATURES = (
    _definition(
        "quarter_open_signed_flow_1h",
        lookback="1h",
        description="Signed taker quote notional in UTC minutes 00, 15, 30 and 45.",
    ),
    _definition(
        "quarter_open_quote_volume_1h",
        lookback="1h",
        description="Quote volume in UTC quarter-hour opening minutes.",
    ),
    _definition(
        "quarter_open_flow_share_1h",
        lookback="1h",
        description="Signed taker flow divided by quote volume in quarter-hour opens.",
    ),
    _definition(
        "non_open_flow_share_1h",
        lookback="1h",
        description="Signed taker flow share outside quarter-hour opening minutes.",
    ),
    _definition(
        "quarter_flow_excess_1h",
        lookback="1h",
        description="Quarter-open flow share less the remaining-hour flow share.",
    ),
    _definition(
        "quarter_open_return_1h",
        lookback="1h",
        description="Sum of close/open log returns in quarter-hour opening minutes.",
    ),
    _definition(
        "quarter_open_flow_z_168h",
        lookback="168 hourly observations",
        description="Trailing time-series z-score of quarter-open signed-flow share.",
    ),
    _definition(
        "quarter_flow_excess_z_168h",
        lookback="168 hourly observations",
        description="Trailing time-series z-score of quarter-open excess flow.",
    ),
)


def compute_clock_phase_features(
    context: FeatureComputeContext,
) -> dict[str, FeatureArray]:
    shape = context.output_shape
    quarter_signed = np.full(shape, np.nan, dtype=np.float64)
    quarter_volume = np.full(shape, np.nan, dtype=np.float64)
    quarter_share = np.full(shape, np.nan, dtype=np.float64)
    non_open_share = np.full(shape, np.nan, dtype=np.float64)
    quarter_return = np.full(shape, np.nan, dtype=np.float64)
    signed_notional = 2.0 * context.taker_quote_volume - context.quote_volume
    minute_log_return = context.log_close - context.log_open
    for row, minute_index in enumerate(context.decision_indices):
        indices = np.arange(int(minute_index) - 59, int(minute_index) + 1)
        if len(indices) != 60 or indices[0] < 0:
            continue
        minute_of_hour = (context.open_times[indices] // 60_000) % 60
        quarter_mask = np.isin(minute_of_hour, tuple(QUARTER_OPEN_MINUTES))
        if int(np.sum(quarter_mask)) != 4:
            continue
        quarter_indices = indices[quarter_mask]
        other_indices = indices[~quarter_mask]
        q_signed = np.sum(signed_notional[quarter_indices], axis=0)
        q_volume = np.sum(context.quote_volume[quarter_indices], axis=0)
        o_signed = np.sum(signed_notional[other_indices], axis=0)
        o_volume = np.sum(context.quote_volume[other_indices], axis=0)
        quarter_signed[row] = q_signed
        quarter_volume[row] = q_volume
        quarter_share[row] = np.divide(
            q_signed,
            q_volume,
            out=np.full(shape[1], np.nan),
            where=q_volume > 0,
        )
        non_open_share[row] = np.divide(
            o_signed,
            o_volume,
            out=np.full(shape[1], np.nan),
            where=o_volume > 0,
        )
        quarter_return[row] = np.sum(minute_log_return[quarter_indices], axis=0)
    excess = quarter_share - non_open_share
    return {
        "quarter_open_signed_flow_1h": quarter_signed,
        "quarter_open_quote_volume_1h": quarter_volume,
        "quarter_open_flow_share_1h": quarter_share,
        "non_open_flow_share_1h": non_open_share,
        "quarter_flow_excess_1h": excess,
        "quarter_open_return_1h": quarter_return,
        "quarter_open_flow_z_168h": rolling_zscore(quarter_share, 168),
        "quarter_flow_excess_z_168h": rolling_zscore(excess, 168),
    }


class ClockPhaseBundle:
    bundle_id = "clock_phase"
    version = "v1"

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return CLOCK_PHASE_FEATURES

    def compute(
        self,
        context: FeatureComputeContext,
        parameters: Mapping[str, Any],
    ) -> Mapping[str, FeatureArray]:
        if parameters:
            raise ResearchError("clock_phase:v1 does not accept parameters")
        return compute_clock_phase_features(context)


__all__ = [
    "CLOCK_PHASE_FEATURES",
    "QUARTER_OPEN_MINUTES",
    "ClockPhaseBundle",
    "compute_clock_phase_features",
]
