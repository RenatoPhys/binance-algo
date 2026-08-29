"""Typed strategy contracts and implementations for offline research."""

from binance_algo.research.strategies.base import FittedStrategy, Strategy
from binance_algo.research.strategies.carry_relative_strength import (
    CARRY_RELATIVE_STRENGTH_FEATURES,
    CarryRelativeStrengthParameters,
    CarryRelativeStrengthStrategy,
)
from binance_algo.research.strategies.donchian_breakout import (
    DONCHIAN_BREAKOUT_FEATURES,
    DonchianBreakoutParameters,
    DonchianBreakoutStrategy,
)
from binance_algo.research.strategies.funding_carry import (
    FundingCarryParameters,
    FundingCarryStrategy,
)
from binance_algo.research.strategies.linear_cross_sectional import (
    LINEAR_CROSS_SECTIONAL_FEATURES,
    FittedLinearCrossSectionalStrategy,
    LinearCrossSectionalParameters,
    LinearCrossSectionalStrategy,
)
from binance_algo.research.strategies.registry import build_strategy
from binance_algo.research.strategies.relative_strength import (
    RELATIVE_STRENGTH_FEATURES,
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
    SMA_CROSSOVER_FEATURES,
    FittedSmaCrossoverStrategy,
    SmaCrossoverParameters,
    SmaCrossoverStrategy,
)

__all__ = [
    "CARRY_RELATIVE_STRENGTH_FEATURES",
    "DONCHIAN_BREAKOUT_FEATURES",
    "LINEAR_CROSS_SECTIONAL_FEATURES",
    "RELATIVE_STRENGTH_FEATURES",
    "SMA_CROSSOVER_FEATURES",
    "CarryRelativeStrengthParameters",
    "CarryRelativeStrengthStrategy",
    "DonchianBreakoutParameters",
    "DonchianBreakoutStrategy",
    "FittedLinearCrossSectionalStrategy",
    "FittedSmaCrossoverStrategy",
    "FittedStrategy",
    "FundingCarryParameters",
    "FundingCarryStrategy",
    "LinearCrossSectionalParameters",
    "LinearCrossSectionalStrategy",
    "RelativeStrengthParameters",
    "RelativeStrengthStrategy",
    "ResidualMeanReversionParameters",
    "ResidualMeanReversionStrategy",
    "ResidualMomentumParameters",
    "ResidualMomentumStrategy",
    "SmaCrossoverParameters",
    "SmaCrossoverStrategy",
    "Strategy",
    "build_strategy",
]
