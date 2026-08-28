"""Neutral long/short portfolio policy extracted from the Phase 3 baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError
from binance_algo.research.contracts import (
    FEATURE_KEY_COLUMNS,
    FoldContext,
    StrategyScores,
    select_feature_view,
)

NEUTRAL_LONG_SHORT_FEATURES = ("rolling_beta", "realized_volatility_24h")


@dataclass(frozen=True, slots=True)
class NeutralLongShortParameters:
    """Immutable constraints for the baseline neutral long/short policy."""

    no_trade_score_band: float
    gross_exposure: float
    annual_volatility_target: float
    max_symbol_weight: float

    def __post_init__(self) -> None:
        values = (
            self.no_trade_score_band,
            self.gross_exposure,
            self.annual_volatility_target,
            self.max_symbol_weight,
        )
        if any(not math.isfinite(value) for value in values):
            raise ResearchError("portfolio parameters must be finite")
        if self.no_trade_score_band < 0:
            raise ResearchError("no-trade score band cannot be negative")
        if not 0 < self.gross_exposure <= 1:
            raise ResearchError("gross exposure must be in (0, 1]")
        if not 0 < self.annual_volatility_target <= 1:
            raise ResearchError("annual volatility target must be in (0, 1]")
        if not 0 < self.max_symbol_weight <= 1:
            raise ResearchError("maximum symbol weight must be in (0, 1]")


def _neutral_weights(
    *,
    top_index: int,
    bottom_index: int,
    betas: np.ndarray[Any, np.dtype[np.float64]],
    realized_volatility: np.ndarray[Any, np.dtype[np.float64]],
    parameters: NeutralLongShortParameters,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    raw = np.zeros(len(betas), dtype=np.float64)
    raw[top_index] = 1
    raw[bottom_index] = -1
    constraints = np.vstack((np.ones(len(betas), dtype=np.float64), betas))
    projected = raw - constraints.T @ np.linalg.pinv(constraints @ constraints.T) @ (
        constraints @ raw
    )
    if (
        float(np.sum(np.abs(projected))) <= 1e-12
        or projected[top_index] <= 0
        or projected[bottom_index] >= 0
    ):
        projected = raw
    projected /= float(np.sum(np.abs(projected)))
    annualized_asset_volatility = realized_volatility * math.sqrt(365)
    volatility_proxy = float(math.sqrt(np.sum(np.square(projected * annualized_asset_volatility))))
    target_gross = parameters.gross_exposure
    if volatility_proxy > 0:
        target_gross = min(target_gross, parameters.annual_volatility_target / volatility_proxy)
    weights = projected * target_gross
    maximum_weight = float(np.max(np.abs(weights)))
    if maximum_weight > parameters.max_symbol_weight:
        weights *= parameters.max_symbol_weight / maximum_weight
    if float(np.sum(np.abs(weights))) > 1 + 1e-12:
        raise ResearchError("portfolio policy attempted economic leverage")
    return weights


def _aligned_panel(
    scores: pl.DataFrame,
    market_state: pl.DataFrame,
) -> tuple[
    tuple[str, ...],
    np.ndarray[Any, np.dtype[np.int64]],
    dict[str, np.ndarray[Any, np.dtype[np.float64]]],
]:
    score_frame = (
        StrategyScores(scores).frame.select(*FEATURE_KEY_COLUMNS, "score").sort(FEATURE_KEY_COLUMNS)
    )
    market_frame = select_feature_view(
        market_state,
        required_features=NEUTRAL_LONG_SHORT_FEATURES,
    ).sort(FEATURE_KEY_COLUMNS)
    if not score_frame.select(FEATURE_KEY_COLUMNS).equals(
        market_frame.select(FEATURE_KEY_COLUMNS), null_equal=True
    ):
        raise ResearchError("strategy scores and portfolio market state keys must align exactly")
    panel = score_frame.join(market_frame, on=list(FEATURE_KEY_COLUMNS), how="inner")
    symbols = tuple(sorted(str(value) for value in panel["symbol"].unique().to_list()))
    times = np.asarray(sorted(panel["decision_time_ms"].unique().to_list()), dtype=np.int64)
    if panel.height != len(times) * len(symbols):
        raise ResearchError("portfolio input panel is incomplete")
    fields = ("score", *NEUTRAL_LONG_SHORT_FEATURES)
    arrays = {
        name: np.full((len(times), len(symbols)), np.nan, dtype=np.float64) for name in fields
    }
    time_index = {int(value): index for index, value in enumerate(times)}
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    for row in panel.iter_rows(named=True):
        x = time_index[int(row["decision_time_ms"])]
        y = symbol_index[str(row["symbol"])]
        for name in fields:
            arrays[name][x, y] = float(row[name])
    if any(np.any(~np.isfinite(values)) for values in arrays.values()):
        raise ResearchError("portfolio inputs are incomplete or non-finite")
    return symbols, times, arrays


@dataclass(frozen=True, slots=True)
class NeutralLongShortPolicy:
    """Select distinct score tails and impose the Phase 3 portfolio constraints."""

    parameters: NeutralLongShortParameters
    policy_id: str = field(default="neutral_long_short", init=False)
    policy_version: str = field(default="1", init=False)

    def required_features(self) -> tuple[str, ...]:
        return NEUTRAL_LONG_SHORT_FEATURES

    def target_weights(
        self,
        scores: pl.DataFrame,
        market_state: pl.DataFrame,
        *,
        context: FoldContext,
    ) -> pl.DataFrame:
        symbols, times, arrays = _aligned_panel(scores, market_state)
        if int(times[0]) < context.test_start_ms or int(times[-1]) > context.test_end_ms:
            raise ResearchError("portfolio input contains decisions outside its fold context")
        targets = np.empty_like(arrays["score"])
        previous_top: int | None = None
        previous_bottom: int | None = None
        band = self.parameters.no_trade_score_band
        for period in range(len(times)):
            top = int(np.argmax(arrays["score"][period]))
            bottom = int(np.argmin(arrays["score"][period]))
            if (
                previous_top is not None
                and arrays["score"][period, top] - arrays["score"][period, previous_top] <= band
            ):
                top = previous_top
            if (
                previous_bottom is not None
                and arrays["score"][period, previous_bottom] - arrays["score"][period, bottom]
                <= band
            ):
                bottom = previous_bottom
            if top == bottom:
                raise ResearchError("cross-sectional score did not produce distinct tails")
            targets[period] = _neutral_weights(
                top_index=top,
                bottom_index=bottom,
                betas=arrays["rolling_beta"][period],
                realized_volatility=arrays["realized_volatility_24h"][period],
                parameters=self.parameters,
            )
            previous_top, previous_bottom = top, bottom
        rows = [
            {
                "decision_time_ms": int(decision_time),
                "symbol": symbol,
                "target_weight": float(targets[time_index, symbol_index]),
            }
            for time_index, decision_time in enumerate(times)
            for symbol_index, symbol in enumerate(symbols)
        ]
        return pl.DataFrame(rows)


__all__ = [
    "NEUTRAL_LONG_SHORT_FEATURES",
    "NeutralLongShortParameters",
    "NeutralLongShortPolicy",
]
