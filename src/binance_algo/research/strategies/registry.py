"""Explicit, strict factories for supported research strategies."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from binance_algo.common.errors import ResearchError
from binance_algo.research.strategies.base import Strategy
from binance_algo.research.strategies.funding_carry import (
    FundingCarryParameters,
    FundingCarryStrategy,
)
from binance_algo.research.strategies.residual_mean_reversion import (
    ResidualMeanReversionParameters,
    ResidualMeanReversionStrategy,
)
from binance_algo.research.strategies.residual_momentum import (
    ResidualMomentumParameters,
    ResidualMomentumStrategy,
)


class ResidualMomentumSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    momentum_weight_1h: float = Field(ge=0, le=1)
    momentum_weight_4h: float = Field(ge=0, le=1)
    momentum_weight_24h: float = Field(ge=0, le=1)


class FundingCarrySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    funding_rate_weight: float = Field(ge=0, le=5)
    funding_change_weight: float = Field(ge=0, le=5)
    momentum_confirmation_weight: float = Field(ge=0, le=5)


class ResidualMeanReversionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    momentum_weight_1h: float = Field(ge=0, le=1)
    momentum_weight_4h: float = Field(ge=0, le=1)
    volatility_adjustment: float = Field(ge=0, le=2)


StrategyFactory = Callable[[Mapping[str, Any]], Strategy]


def build_residual_momentum(parameters: Mapping[str, Any]) -> ResidualMomentumStrategy:
    try:
        parsed = ResidualMomentumSpec.model_validate(dict(parameters))
        return ResidualMomentumStrategy(
            parameters=ResidualMomentumParameters(**parsed.model_dump())
        )
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid residual_momentum parameters: {exc}") from exc


def build_funding_carry(parameters: Mapping[str, Any]) -> FundingCarryStrategy:
    try:
        parsed = FundingCarrySpec.model_validate(dict(parameters))
        return FundingCarryStrategy(parameters=FundingCarryParameters(**parsed.model_dump()))
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid funding_carry parameters: {exc}") from exc


def build_residual_mean_reversion(
    parameters: Mapping[str, Any],
) -> ResidualMeanReversionStrategy:
    try:
        parsed = ResidualMeanReversionSpec.model_validate(dict(parameters))
        return ResidualMeanReversionStrategy(
            parameters=ResidualMeanReversionParameters(**parsed.model_dump())
        )
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid residual_mean_reversion parameters: {exc}") from exc


STRATEGY_FACTORIES: dict[tuple[str, str], StrategyFactory] = {
    ("funding_carry", "1"): build_funding_carry,
    ("funding_carry", "v1"): build_funding_carry,
    ("residual_momentum", "1"): build_residual_momentum,
    ("residual_momentum", "v1"): build_residual_momentum,
    ("residual_mean_reversion", "1"): build_residual_mean_reversion,
    ("residual_mean_reversion", "v1"): build_residual_mean_reversion,
}


def build_strategy(
    strategy_id: str,
    version: str,
    parameters: Mapping[str, Any],
) -> Strategy:
    try:
        factory = STRATEGY_FACTORIES[(strategy_id, version)]
    except KeyError as exc:
        raise ResearchError(f"unsupported strategy: {strategy_id}:{version}") from exc
    return factory(parameters)


__all__ = [
    "STRATEGY_FACTORIES",
    "FundingCarrySpec",
    "ResidualMeanReversionSpec",
    "ResidualMomentumSpec",
    "StrategyFactory",
    "build_funding_carry",
    "build_residual_mean_reversion",
    "build_residual_momentum",
    "build_strategy",
]
