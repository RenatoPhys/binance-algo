"""Explicit, strict factories for supported portfolio policies."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from binance_algo.common.errors import ResearchError
from binance_algo.research.portfolio.base import PortfolioPolicy
from binance_algo.research.portfolio.neutral_long_short import (
    NeutralLongShortParameters,
    NeutralLongShortPolicy,
)


class NeutralLongShortSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    no_trade_score_band: float = Field(ge=0)
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


PORTFOLIO_POLICY_FACTORIES: dict[tuple[str, str], PortfolioPolicyFactory] = {
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
    "NeutralLongShortSpec",
    "PortfolioPolicyFactory",
    "build_neutral_long_short",
    "build_portfolio_policy",
]
