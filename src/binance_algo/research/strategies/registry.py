"""Explicit, strict factories for supported research strategies."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from binance_algo.common.errors import ResearchError
from binance_algo.research.strategies.base import Strategy
from binance_algo.research.strategies.residual_momentum import (
    ResidualMomentumParameters,
    ResidualMomentumStrategy,
)


class ResidualMomentumSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    momentum_weight_1h: float = Field(ge=0, le=1)
    momentum_weight_4h: float = Field(ge=0, le=1)
    momentum_weight_24h: float = Field(ge=0, le=1)


StrategyFactory = Callable[[Mapping[str, Any]], Strategy]


def build_residual_momentum(parameters: Mapping[str, Any]) -> ResidualMomentumStrategy:
    try:
        parsed = ResidualMomentumSpec.model_validate(dict(parameters))
        return ResidualMomentumStrategy(
            parameters=ResidualMomentumParameters(**parsed.model_dump())
        )
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid residual_momentum parameters: {exc}") from exc


STRATEGY_FACTORIES: dict[tuple[str, str], StrategyFactory] = {
    ("residual_momentum", "1"): build_residual_momentum,
    ("residual_momentum", "v1"): build_residual_momentum,
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
    "ResidualMomentumSpec",
    "StrategyFactory",
    "build_residual_momentum",
    "build_strategy",
]
