"""Explicit, strict factories for supported portfolio policies."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from binance_algo.common.errors import ResearchError
from binance_algo.research.portfolio.base import PortfolioPolicy
from binance_algo.research.portfolio.carry_regime import (
    BufferedCarryRegimeParameters,
    BufferedCarryRegimePolicy,
)
from binance_algo.research.portfolio.directional import (
    BufferedDirectionalParameters,
    BufferedDirectionalPolicy,
)
from binance_algo.research.portfolio.long_flat import (
    BufferedLongFlatParameters,
    BufferedLongFlatPolicy,
)
from binance_algo.research.portfolio.neutral_long_short import (
    BufferedNeutralLongShortParameters,
    BufferedNeutralLongShortPolicy,
    NeutralLongShortParameters,
    NeutralLongShortPolicy,
)
from binance_algo.research.portfolio.three_sleeve_neutral import (
    BufferedThreeSleeveNeutralParameters,
    BufferedThreeSleeveNeutralPolicy,
)
from binance_algo.research.portfolio.two_sleeve_neutral import (
    BufferedTwoSleeveNeutralParameters,
    BufferedTwoSleeveNeutralPolicy,
)


class NeutralLongShortSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    no_trade_score_band: float = Field(ge=0)
    gross_exposure: float = Field(gt=0, le=1)
    annual_volatility_target: float = Field(gt=0, le=1)
    max_symbol_weight: float = Field(gt=0, le=1)


class BufferedNeutralLongShortSpec(NeutralLongShortSpec):
    rebalance_interval_hours: int = Field(ge=1, le=24 * 30)
    minimum_score_spread: float = Field(default=0.0, ge=0)


class BufferedDirectionalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    signal_threshold: float = Field(ge=0, le=1)
    rebalance_interval_hours: int = Field(ge=1, le=24 * 30)
    gross_exposure: float = Field(gt=0, le=1)
    annual_volatility_target: float = Field(gt=0, le=1)
    max_symbol_weight: float = Field(gt=0, le=1)


class BufferedLongFlatSpec(BufferedDirectionalSpec):
    pass


class BufferedCarryRegimeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    core_carry_weight: float = Field(ge=0, le=1)
    core_fast_strength_weight: float = Field(ge=0, le=1)
    core_slow_strength_weight: float = Field(ge=0, le=1)
    regime_trend_weight: float = Field(ge=0, le=0.5)
    no_trade_score_band: float = Field(ge=0)
    signal_threshold: float = Field(ge=0, le=1)
    rebalance_interval_hours: int = Field(ge=1, le=24 * 30)
    gross_exposure: float = Field(gt=0, le=1)
    annual_volatility_target: float = Field(gt=0, le=1)
    max_symbol_weight: float = Field(gt=0, le=1)


class BufferedTwoSleeveNeutralSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    carry_weight: float = Field(ge=0, le=1)
    no_trade_score_band: float = Field(ge=0)
    rebalance_interval_hours: int = Field(ge=1, le=24 * 30)
    gross_exposure: float = Field(gt=0, le=1)
    annual_volatility_target: float = Field(gt=0, le=1)
    max_symbol_weight: float = Field(gt=0, le=1)


class BufferedThreeSleeveNeutralSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    carry_weight: float = Field(ge=0, le=1)
    fast_strength_weight: float = Field(ge=0, le=1)
    slow_strength_weight: float = Field(ge=0, le=1)
    no_trade_score_band: float = Field(ge=0)
    rebalance_interval_hours: int = Field(ge=1, le=24 * 30)
    gross_exposure: float = Field(gt=0, le=1)
    annual_volatility_target: float = Field(gt=0, le=1)
    max_symbol_weight: float = Field(gt=0, le=1)


PortfolioPolicyFactory = Callable[[Mapping[str, Any]], PortfolioPolicy]


def build_neutral_long_short(parameters: Mapping[str, Any]) -> NeutralLongShortPolicy:
    try:
        parsed = NeutralLongShortSpec.model_validate(dict(parameters))
        return NeutralLongShortPolicy(parameters=NeutralLongShortParameters(**parsed.model_dump()))
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid neutral_long_short parameters: {exc}") from exc


def build_buffered_neutral_long_short(
    parameters: Mapping[str, Any],
) -> BufferedNeutralLongShortPolicy:
    try:
        parsed = BufferedNeutralLongShortSpec.model_validate(dict(parameters))
        return BufferedNeutralLongShortPolicy(
            parameters=BufferedNeutralLongShortParameters(**parsed.model_dump())
        )
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid buffered_neutral_long_short parameters: {exc}") from exc


def build_buffered_directional(parameters: Mapping[str, Any]) -> BufferedDirectionalPolicy:
    try:
        parsed = BufferedDirectionalSpec.model_validate(dict(parameters))
        return BufferedDirectionalPolicy(
            parameters=BufferedDirectionalParameters(**parsed.model_dump())
        )
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid buffered_directional parameters: {exc}") from exc


def build_buffered_long_flat(parameters: Mapping[str, Any]) -> BufferedLongFlatPolicy:
    try:
        parsed = BufferedLongFlatSpec.model_validate(dict(parameters))
        return BufferedLongFlatPolicy(parameters=BufferedLongFlatParameters(**parsed.model_dump()))
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid buffered_long_flat parameters: {exc}") from exc


def build_buffered_carry_regime(
    parameters: Mapping[str, Any],
) -> BufferedCarryRegimePolicy:
    try:
        parsed = BufferedCarryRegimeSpec.model_validate(dict(parameters))
        return BufferedCarryRegimePolicy(
            parameters=BufferedCarryRegimeParameters(**parsed.model_dump())
        )
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid buffered_carry_regime parameters: {exc}") from exc


def build_buffered_two_sleeve_neutral(
    parameters: Mapping[str, Any],
) -> BufferedTwoSleeveNeutralPolicy:
    try:
        parsed = BufferedTwoSleeveNeutralSpec.model_validate(dict(parameters))
        return BufferedTwoSleeveNeutralPolicy(
            parameters=BufferedTwoSleeveNeutralParameters(**parsed.model_dump())
        )
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid buffered_two_sleeve_neutral parameters: {exc}") from exc


def build_buffered_three_sleeve_neutral(
    parameters: Mapping[str, Any],
) -> BufferedThreeSleeveNeutralPolicy:
    try:
        parsed = BufferedThreeSleeveNeutralSpec.model_validate(dict(parameters))
        return BufferedThreeSleeveNeutralPolicy(
            parameters=BufferedThreeSleeveNeutralParameters(**parsed.model_dump())
        )
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid buffered_three_sleeve_neutral parameters: {exc}") from exc


PORTFOLIO_POLICY_FACTORIES: dict[tuple[str, str], PortfolioPolicyFactory] = {
    ("buffered_carry_regime", "1"): build_buffered_carry_regime,
    ("buffered_carry_regime", "v1"): build_buffered_carry_regime,
    ("buffered_directional", "1"): build_buffered_directional,
    ("buffered_directional", "v1"): build_buffered_directional,
    ("buffered_long_flat", "1"): build_buffered_long_flat,
    ("buffered_long_flat", "v1"): build_buffered_long_flat,
    ("buffered_three_sleeve_neutral", "1"): build_buffered_three_sleeve_neutral,
    ("buffered_three_sleeve_neutral", "v1"): build_buffered_three_sleeve_neutral,
    ("buffered_two_sleeve_neutral", "1"): build_buffered_two_sleeve_neutral,
    ("buffered_two_sleeve_neutral", "v1"): build_buffered_two_sleeve_neutral,
    ("buffered_neutral_long_short", "1"): build_buffered_neutral_long_short,
    ("buffered_neutral_long_short", "v1"): build_buffered_neutral_long_short,
    ("neutral_long_short", "1"): build_neutral_long_short,
    ("neutral_long_short", "v1"): build_neutral_long_short,
}


def build_portfolio_policy(
    policy_id: str,
    version: str,
    parameters: Mapping[str, Any],
) -> PortfolioPolicy:
    try:
        factory = PORTFOLIO_POLICY_FACTORIES[(policy_id, version)]
    except KeyError as exc:
        raise ResearchError(f"unsupported portfolio policy: {policy_id}:{version}") from exc
    return factory(parameters)


__all__ = [
    "PORTFOLIO_POLICY_FACTORIES",
    "BufferedCarryRegimeSpec",
    "BufferedDirectionalSpec",
    "BufferedLongFlatSpec",
    "BufferedNeutralLongShortSpec",
    "BufferedThreeSleeveNeutralSpec",
    "BufferedTwoSleeveNeutralSpec",
    "NeutralLongShortSpec",
    "PortfolioPolicyFactory",
    "build_buffered_carry_regime",
    "build_buffered_directional",
    "build_buffered_long_flat",
    "build_buffered_neutral_long_short",
    "build_buffered_three_sleeve_neutral",
    "build_buffered_two_sleeve_neutral",
    "build_neutral_long_short",
    "build_portfolio_policy",
]
