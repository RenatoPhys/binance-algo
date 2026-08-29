"""Explicit, strict factories for supported research strategies."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from binance_algo.common.errors import ResearchError
from binance_algo.research.strategies.base import Strategy
from binance_algo.research.strategies.carry_consensus_strength import (
    CarryConsensusStrengthParameters,
    CarryConsensusStrengthStrategy,
)
from binance_algo.research.strategies.carry_dual_trend import (
    CarryDualTrendParameters,
    CarryDualTrendStrategy,
)
from binance_algo.research.strategies.carry_multi_horizon import (
    CarryMultiHorizonParameters,
    CarryMultiHorizonStrategy,
)
from binance_algo.research.strategies.carry_multi_regime import (
    CarryMultiRegimeParameters,
    CarryMultiRegimeStrategy,
)
from binance_algo.research.strategies.carry_relative_strength import (
    CarryRelativeStrengthParameters,
    CarryRelativeStrengthStrategy,
)
from binance_algo.research.strategies.donchian_breakout import (
    DonchianBreakoutParameters,
    DonchianBreakoutStrategy,
)
from binance_algo.research.strategies.funding_carry import (
    FundingCarryParameters,
    FundingCarryStrategy,
)
from binance_algo.research.strategies.linear_cross_sectional import (
    LinearCrossSectionalParameters,
    LinearCrossSectionalStrategy,
)
from binance_algo.research.strategies.market_regime_trend import (
    MarketRegimeTrendParameters,
    MarketRegimeTrendStrategy,
)
from binance_algo.research.strategies.multi_horizon_trend import (
    MultiHorizonTrendParameters,
    MultiHorizonTrendStrategy,
)
from binance_algo.research.strategies.relative_strength import (
    RelativeStrengthParameters,
    RelativeStrengthStrategy,
)
from binance_algo.research.strategies.residual_mean_reversion import (
    ResidualMeanReversionParameters,
    ResidualMeanReversionStrategy,
)
from binance_algo.research.strategies.residual_momentum import (
    ResidualMomentumParameters,
    ResidualMomentumStrategy,
)
from binance_algo.research.strategies.sma_crossover import (
    SmaCrossoverParameters,
    SmaCrossoverStrategy,
)
from binance_algo.research.strategies.sma_trend_strength import SmaTrendStrengthStrategy
from binance_algo.research.strategies.volatility_filtered_sma import (
    VolatilityFilteredSmaParameters,
    VolatilityFilteredSmaStrategy,
)


class ResidualMomentumSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    momentum_weight_1h: float = Field(ge=0, le=1)
    momentum_weight_4h: float = Field(ge=0, le=1)
    momentum_weight_24h: float = Field(ge=0, le=1)


class DonchianBreakoutSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    entry_window_hours: int = Field(ge=24, le=24 * 90)
    exit_window_hours: int = Field(ge=4, le=24 * 30)


class FundingCarrySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    funding_rate_weight: float = Field(ge=0, le=5)
    funding_change_weight: float = Field(ge=0, le=5)
    momentum_confirmation_weight: float = Field(ge=0, le=5)


class CarryRelativeStrengthSpec(FundingCarrySpec):
    relative_strength_lookback_hours: int = Field(ge=24, le=24 * 90)


class CarryMultiHorizonSpec(FundingCarrySpec):
    fast_lookback_hours: int = Field(ge=24, le=24 * 90)
    slow_lookback_hours: int = Field(ge=24, le=24 * 90)


class CarryMultiRegimeSpec(CarryMultiHorizonSpec):
    regime_fast_lookback_hours: int = Field(ge=24, le=24 * 90)
    regime_slow_lookback_hours: int = Field(ge=24, le=24 * 90)


class CarryDualTrendSpec(FundingCarrySpec):
    relative_strength_lookback_hours: int = Field(ge=24, le=24 * 90)
    sma_fast_window_hours: int = Field(ge=2, le=72)
    sma_slow_window_hours: int = Field(ge=4, le=720)


class LinearCrossSectionalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ridge_alpha: float = Field(ge=1e-6, le=100)


class MultiHorizonTrendSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    short_lookback_hours: int = Field(ge=24, le=24 * 90)
    medium_lookback_hours: int = Field(ge=24, le=24 * 90)
    long_lookback_hours: int = Field(ge=24, le=24 * 90)
    short_weight: float = Field(ge=0, le=1)
    medium_weight: float = Field(ge=0, le=1)
    long_weight: float = Field(ge=0, le=1)


class MarketRegimeTrendSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fast_lookback_hours: int = Field(ge=24, le=24 * 90)
    slow_lookback_hours: int = Field(ge=24, le=24 * 90)


class ResidualMeanReversionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    momentum_weight_1h: float = Field(ge=0, le=1)
    momentum_weight_4h: float = Field(ge=0, le=1)
    volatility_adjustment: float = Field(ge=0, le=2)


class RelativeStrengthSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    lookback_hours: int = Field(ge=24, le=24 * 90)


class SmaCrossoverSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fast_window_hours: int = Field(ge=2, le=72)
    slow_window_hours: int = Field(ge=4, le=720)


class VolatilityFilteredSmaSpec(SmaCrossoverSpec):
    maximum_volatility_quantile: float = Field(ge=0.25, le=0.90)


StrategyFactory = Callable[[Mapping[str, Any]], Strategy]


def build_donchian_breakout(parameters: Mapping[str, Any]) -> DonchianBreakoutStrategy:
    try:
        parsed = DonchianBreakoutSpec.model_validate(dict(parameters))
        return DonchianBreakoutStrategy(
            parameters=DonchianBreakoutParameters(**parsed.model_dump())
        )
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid donchian_breakout parameters: {exc}") from exc


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


def build_carry_relative_strength(
    parameters: Mapping[str, Any],
) -> CarryRelativeStrengthStrategy:
    try:
        parsed = CarryRelativeStrengthSpec.model_validate(dict(parameters))
        return CarryRelativeStrengthStrategy(
            parameters=CarryRelativeStrengthParameters(**parsed.model_dump())
        )
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid carry_relative_strength parameters: {exc}") from exc


def build_carry_multi_horizon(
    parameters: Mapping[str, Any],
) -> CarryMultiHorizonStrategy:
    try:
        parsed = CarryMultiHorizonSpec.model_validate(dict(parameters))
        return CarryMultiHorizonStrategy(
            parameters=CarryMultiHorizonParameters(**parsed.model_dump())
        )
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid carry_multi_horizon parameters: {exc}") from exc


def build_carry_multi_regime(
    parameters: Mapping[str, Any],
) -> CarryMultiRegimeStrategy:
    try:
        parsed = CarryMultiRegimeSpec.model_validate(dict(parameters))
        return CarryMultiRegimeStrategy(
            parameters=CarryMultiRegimeParameters(**parsed.model_dump())
        )
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid carry_multi_regime parameters: {exc}") from exc


def build_carry_consensus_strength(
    parameters: Mapping[str, Any],
) -> CarryConsensusStrengthStrategy:
    try:
        parsed = CarryMultiHorizonSpec.model_validate(dict(parameters))
        return CarryConsensusStrengthStrategy(
            parameters=CarryConsensusStrengthParameters(**parsed.model_dump())
        )
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid carry_consensus_strength parameters: {exc}") from exc


def build_carry_dual_trend(
    parameters: Mapping[str, Any],
) -> CarryDualTrendStrategy:
    try:
        parsed = CarryDualTrendSpec.model_validate(dict(parameters))
        return CarryDualTrendStrategy(parameters=CarryDualTrendParameters(**parsed.model_dump()))
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid carry_dual_trend parameters: {exc}") from exc


def build_linear_cross_sectional(
    parameters: Mapping[str, Any],
) -> LinearCrossSectionalStrategy:
    try:
        parsed = LinearCrossSectionalSpec.model_validate(dict(parameters))
        return LinearCrossSectionalStrategy(
            parameters=LinearCrossSectionalParameters(**parsed.model_dump())
        )
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid linear_cross_sectional parameters: {exc}") from exc


def build_multi_horizon_trend(
    parameters: Mapping[str, Any],
) -> MultiHorizonTrendStrategy:
    try:
        parsed = MultiHorizonTrendSpec.model_validate(dict(parameters))
        return MultiHorizonTrendStrategy(
            parameters=MultiHorizonTrendParameters(**parsed.model_dump())
        )
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid multi_horizon_trend parameters: {exc}") from exc


def build_market_regime_trend(
    parameters: Mapping[str, Any],
) -> MarketRegimeTrendStrategy:
    try:
        parsed = MarketRegimeTrendSpec.model_validate(dict(parameters))
        return MarketRegimeTrendStrategy(
            parameters=MarketRegimeTrendParameters(**parsed.model_dump())
        )
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid market_regime_trend parameters: {exc}") from exc


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


def build_relative_strength(parameters: Mapping[str, Any]) -> RelativeStrengthStrategy:
    try:
        parsed = RelativeStrengthSpec.model_validate(dict(parameters))
        return RelativeStrengthStrategy(
            parameters=RelativeStrengthParameters(**parsed.model_dump())
        )
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid relative_strength parameters: {exc}") from exc


def build_sma_crossover(parameters: Mapping[str, Any]) -> SmaCrossoverStrategy:
    try:
        parsed = SmaCrossoverSpec.model_validate(dict(parameters))
        return SmaCrossoverStrategy(parameters=SmaCrossoverParameters(**parsed.model_dump()))
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid sma_crossover parameters: {exc}") from exc


def build_sma_trend_strength(parameters: Mapping[str, Any]) -> SmaTrendStrengthStrategy:
    try:
        parsed = SmaCrossoverSpec.model_validate(dict(parameters))
        return SmaTrendStrengthStrategy(parameters=SmaCrossoverParameters(**parsed.model_dump()))
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid sma_trend_strength parameters: {exc}") from exc


def build_volatility_filtered_sma(parameters: Mapping[str, Any]) -> VolatilityFilteredSmaStrategy:
    try:
        parsed = VolatilityFilteredSmaSpec.model_validate(dict(parameters))
        return VolatilityFilteredSmaStrategy(
            parameters=VolatilityFilteredSmaParameters(**parsed.model_dump())
        )
    except (TypeError, ValueError, ResearchError) as exc:
        raise ResearchError(f"invalid volatility_filtered_sma parameters: {exc}") from exc


STRATEGY_FACTORIES: dict[tuple[str, str], StrategyFactory] = {
    ("carry_consensus_strength", "1"): build_carry_consensus_strength,
    ("carry_consensus_strength", "v1"): build_carry_consensus_strength,
    ("carry_dual_trend", "1"): build_carry_dual_trend,
    ("carry_dual_trend", "v1"): build_carry_dual_trend,
    ("carry_multi_horizon", "1"): build_carry_multi_horizon,
    ("carry_multi_horizon", "v1"): build_carry_multi_horizon,
    ("carry_multi_regime", "1"): build_carry_multi_regime,
    ("carry_multi_regime", "v1"): build_carry_multi_regime,
    ("carry_relative_strength", "1"): build_carry_relative_strength,
    ("carry_relative_strength", "v1"): build_carry_relative_strength,
    ("donchian_breakout", "1"): build_donchian_breakout,
    ("donchian_breakout", "v1"): build_donchian_breakout,
    ("funding_carry", "1"): build_funding_carry,
    ("funding_carry", "v1"): build_funding_carry,
    ("linear_cross_sectional", "1"): build_linear_cross_sectional,
    ("linear_cross_sectional", "v1"): build_linear_cross_sectional,
    ("market_regime_trend", "1"): build_market_regime_trend,
    ("market_regime_trend", "v1"): build_market_regime_trend,
    ("multi_horizon_trend", "1"): build_multi_horizon_trend,
    ("multi_horizon_trend", "v1"): build_multi_horizon_trend,
    ("residual_momentum", "1"): build_residual_momentum,
    ("residual_momentum", "v1"): build_residual_momentum,
    ("relative_strength", "1"): build_relative_strength,
    ("relative_strength", "v1"): build_relative_strength,
    ("residual_mean_reversion", "1"): build_residual_mean_reversion,
    ("residual_mean_reversion", "v1"): build_residual_mean_reversion,
    ("sma_crossover", "1"): build_sma_crossover,
    ("sma_crossover", "v1"): build_sma_crossover,
    ("sma_trend_strength", "1"): build_sma_trend_strength,
    ("sma_trend_strength", "v1"): build_sma_trend_strength,
    ("volatility_filtered_sma", "1"): build_volatility_filtered_sma,
    ("volatility_filtered_sma", "v1"): build_volatility_filtered_sma,
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
    "CarryDualTrendSpec",
    "CarryMultiHorizonSpec",
    "CarryMultiRegimeSpec",
    "CarryRelativeStrengthSpec",
    "DonchianBreakoutSpec",
    "FundingCarrySpec",
    "LinearCrossSectionalSpec",
    "MarketRegimeTrendSpec",
    "MultiHorizonTrendSpec",
    "RelativeStrengthSpec",
    "ResidualMeanReversionSpec",
    "ResidualMomentumSpec",
    "SmaCrossoverSpec",
    "StrategyFactory",
    "VolatilityFilteredSmaSpec",
    "build_carry_consensus_strength",
    "build_carry_dual_trend",
    "build_carry_multi_horizon",
    "build_carry_multi_regime",
    "build_carry_relative_strength",
    "build_donchian_breakout",
    "build_funding_carry",
    "build_linear_cross_sectional",
    "build_market_regime_trend",
    "build_multi_horizon_trend",
    "build_relative_strength",
    "build_residual_mean_reversion",
    "build_residual_momentum",
    "build_sma_crossover",
    "build_sma_trend_strength",
    "build_strategy",
    "build_volatility_filtered_sma",
]
