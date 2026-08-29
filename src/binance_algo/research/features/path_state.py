"""Causal volatility-compression, breakout-level, and path-efficiency state."""

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
from binance_algo.research.features.rolling import (
    kaufman_efficiency_ratio,
    prior_rolling_extrema,
    trailing_realized_volatility,
)


def _definition(name: str, *, lookback: str, description: str) -> FeatureDefinition:
    return FeatureDefinition(
        feature_id=f"{name}:v1",
        name=name,
        version="v1",
        description=description,
        dtype="Float64",
        lookback=lookback,
        timestamp_semantics=(
            "closed bars at or before decision_time_ms; prior levels exclude current hour"
        ),
        required_datasets=("klines",),
        required_columns=("open_time_ms", "high", "low", "close"),
        implementation_path="binance_algo.research.features.path_state",
        parameters={},
    )


PATH_STATE_FEATURES = (
    _definition(
        "realized_volatility_4h",
        lookback="4 hourly returns",
        description="Square root of the trailing four squared hourly log returns.",
    ),
    _definition(
        "realized_volatility_168h",
        lookback="168 hourly returns",
        description="Square root of the trailing 168 squared hourly log returns.",
    ),
    _definition(
        "volatility_compression_4h_168h",
        lookback="168h",
        description="Four-hour realized volatility relative to scaled 168-hour volatility.",
    ),
    _definition(
        "prior_high_72h",
        lookback="72 prior completed hours",
        description="Highest one-minute high in the preceding 72 complete hours.",
    ),
    _definition(
        "prior_low_72h",
        lookback="72 prior completed hours",
        description="Lowest one-minute low in the preceding 72 complete hours.",
    ),
    _definition(
        "range_position_72h",
        lookback="72 prior completed hours",
        description="Current close position in the preceding 72-hour high-low range.",
    ),
    _definition(
        "efficiency_ratio_24h",
        lookback="24 hourly returns",
        description="Kaufman path efficiency over 24 hourly return intervals.",
    ),
)


class PathStateBundle:
    bundle_id = "path_state"
    version = "v1"

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return PATH_STATE_FEATURES

    def compute(
        self,
        context: FeatureComputeContext,
        parameters: Mapping[str, Any],
    ) -> Mapping[str, FeatureArray]:
        if parameters:
            raise ResearchError("path_state:v1 does not accept parameters")
        hourly_returns = context.require_output("log_return_1h")
        hourly_log_close = context.log_close[context.decision_indices]
        realized_4h = trailing_realized_volatility(hourly_returns, 4)
        realized_168h = trailing_realized_volatility(hourly_returns, 168)
        compression = np.divide(
            realized_4h,
            np.maximum(realized_168h * np.sqrt(4.0 / 168.0), 1e-15),
            out=np.full_like(realized_4h, np.nan),
            where=np.isfinite(realized_168h),
        )
        hourly_high = np.empty(context.output_shape, dtype=np.float64)
        hourly_low = np.empty(context.output_shape, dtype=np.float64)
        for row, minute_index in enumerate(context.decision_indices):
            hour = slice(int(minute_index) - 59, int(minute_index) + 1)
            hourly_high[row] = np.max(context.highs[hour], axis=0)
            hourly_low[row] = np.min(context.lows[hour], axis=0)
        prior_high, _ = prior_rolling_extrema(hourly_high, 72)
        _, prior_low = prior_rolling_extrema(hourly_low, 72)
        close = np.exp(hourly_log_close)
        spread = prior_high - prior_low
        range_position = np.divide(
            close - prior_low,
            spread,
            out=np.full_like(close, np.nan),
            where=spread > 1e-15,
        )
        return {
            "realized_volatility_4h": realized_4h,
            "realized_volatility_168h": realized_168h,
            "volatility_compression_4h_168h": compression,
            "prior_high_72h": prior_high,
            "prior_low_72h": prior_low,
            "range_position_72h": range_position,
            "efficiency_ratio_24h": kaufman_efficiency_ratio(hourly_log_close, 24),
        }


__all__ = ["PATH_STATE_FEATURES", "PathStateBundle"]
