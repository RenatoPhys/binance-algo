"""Causal hourly aggressive-flow and price-response features."""

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


def _definition(name: str, *, lookback: str, description: str) -> FeatureDefinition:
    return FeatureDefinition(
        feature_id=f"{name}:v1",
        name=name,
        version="v1",
        description=description,
        dtype="Float64",
        lookback=lookback,
        timestamp_semantics="closed one-minute bars at or before decision_time_ms",
        required_datasets=("klines",),
        required_columns=("open_time_ms", "close", "quote_volume", "taker_buy_quote_volume"),
        implementation_path="binance_algo.research.features.flow",
        parameters={},
    )


FLOW_FEATURES = (
    _definition(
        "signed_taker_flow_share_1h",
        lookback="1h",
        description="Signed taker quote notional divided by hourly quote volume.",
    ),
    _definition(
        "signed_taker_flow_z_168h",
        lookback="168 hourly observations",
        description="Trailing per-symbol z-score of aggressive signed-flow share.",
    ),
    _definition(
        "hourly_volatility_proxy",
        lookback="24h",
        description="Realized 24-hour volatility divided by square root of 24.",
    ),
    _definition(
        "vol_scaled_return_1h",
        lookback="24h",
        description="Hourly log return divided by its causal hourly volatility proxy.",
    ),
    _definition(
        "flow_price_agreement_1h",
        lookback="168h",
        description="Flow direction times the volatility-scaled price response.",
    ),
)


class FlowBundle:
    bundle_id = "flow"
    version = "v1"

    def definitions(self) -> tuple[FeatureDefinition, ...]:
        return FLOW_FEATURES

    def compute(
        self,
        context: FeatureComputeContext,
        parameters: Mapping[str, Any],
    ) -> Mapping[str, FeatureArray]:
        if parameters:
            raise ResearchError("flow:v1 does not accept parameters")
        quote_volume = context.require_output("quote_volume_1h")
        signed_flow = np.empty_like(quote_volume)
        for row, minute_index in enumerate(context.decision_indices):
            hour = slice(int(minute_index) - 59, int(minute_index) + 1)
            signed_flow[row] = np.sum(
                2.0 * context.taker_quote_volume[hour] - context.quote_volume[hour],
                axis=0,
            )
        share = np.divide(
            signed_flow,
            quote_volume,
            out=np.full_like(signed_flow, np.nan),
            where=quote_volume > 0,
        )
        flow_z = rolling_zscore(share, 168)
        volatility = context.require_output("realized_volatility_24h") / np.sqrt(24.0)
        hourly_return = context.require_output("log_return_1h")
        scaled_return = np.divide(
            hourly_return,
            np.maximum(volatility, 1e-15),
            out=np.full_like(hourly_return, np.nan),
            where=np.isfinite(volatility),
        )
        return {
            "signed_taker_flow_share_1h": share,
            "signed_taker_flow_z_168h": flow_z,
            "hourly_volatility_proxy": volatility,
            "vol_scaled_return_1h": scaled_return,
            "flow_price_agreement_1h": np.sign(flow_z) * scaled_return,
        }


__all__ = ["FLOW_FEATURES", "FlowBundle"]
