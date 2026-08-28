"""Typed strategy contracts and implementations for offline research."""

from binance_algo.research.strategies.base import FittedStrategy, Strategy
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
    "DONCHIAN_BREAKOUT_FEATURES",
    "LINEAR_CROSS_SECTIONAL_FEATURES",
    "SMA_CROSSOVER_FEATURES",
    "DonchianBreakoutParameters",
    "DonchianBreakoutStrategy",
    "FittedLinearCrossSectionalStrategy",
    "FittedSmaCrossoverStrategy",
    "FittedStrategy",
    "FundingCarryParameters",
    "FundingCarryStrategy",
    "LinearCrossSectionalParameters",
    "LinearCrossSectionalStrategy",
    "ResidualMeanReversionParameters",
    "ResidualMeanReversionStrategy",
    "ResidualMomentumParameters",
    "ResidualMomentumStrategy",
    "SmaCrossoverParameters",
    "SmaCrossoverStrategy",
    "Strategy",
    "build_strategy",
]
