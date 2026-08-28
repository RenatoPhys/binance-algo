"""Point-in-time funding features derived only from already-published events."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from binance_algo.research.features.base import FeatureDefinition


def compute_asof_funding(
    funding: pl.DataFrame,
    *,
    symbol: str,
    decision_times: np.ndarray[Any, np.dtype[np.int64]],
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], np.ndarray[Any, np.dtype[np.float64]]]:
    events = (
        funding.filter(pl.col("symbol") == symbol)
        .group_by("funding_time_ms")
        .agg(pl.col("funding_rate").sum())
        .sort("funding_time_ms")
    )
    current = np.full(len(decision_times), np.nan, dtype=np.float64)
    change = np.full(len(decision_times), np.nan, dtype=np.float64)
    if events.is_empty():
        return current, change
    event_times = np.asarray(events["funding_time_ms"].to_numpy(), dtype=np.int64)
    event_rates = np.asarray(events["funding_rate"].to_numpy(), dtype=np.float64)
    positions = np.searchsorted(event_times, decision_times, side="right") - 1
    visible = positions >= 0
    current[visible] = event_rates[positions[visible]]
    prior_visible = positions >= 1
    change[prior_visible] = (
        event_rates[positions[prior_visible]] - event_rates[positions[prior_visible] - 1]
    )
    return current, change


def _definition(name: str, description: str) -> FeatureDefinition:
    return FeatureDefinition(
        feature_id=f"{name}:v1",
        name=name,
        version="v1",
        description=description,
        dtype="Float64",
        lookback="latest two published funding events",
        timestamp_semantics="last public event at or before decision_time_ms; never backfilled",
        required_datasets=("funding_rates",),
        required_columns=("symbol", "funding_time_ms", "funding_rate"),
        implementation_path="binance_algo.research.features.funding",
        parameters={},
    )


FUNDING_FEATURES = (
    _definition("funding_rate_current", "Most recently published aggregate funding rate."),
    _definition("funding_rate_change", "Change from the prior published funding rate."),
)


__all__ = ["FUNDING_FEATURES", "compute_asof_funding"]
