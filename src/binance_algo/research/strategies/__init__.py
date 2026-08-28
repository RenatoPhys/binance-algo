"""Typed strategy contracts and implementations for offline research."""

from binance_algo.research.strategies.base import FittedStrategy, Strategy
from binance_algo.research.strategies.funding_carry import (
    FundingCarryParameters,
    FundingCarryStrategy,
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
    "SMA_CROSSOVER_FEATURES",
    "FittedSmaCrossoverStrategy",
    "FittedStrategy",
    "FundingCarryParameters",
    "FundingCarryStrategy",
    "ResidualMeanReversionParameters",
    "ResidualMeanReversionStrategy",
    "ResidualMomentumParameters",
    "ResidualMomentumStrategy",
    "SmaCrossoverParameters",
    "SmaCrossoverStrategy",
    "Strategy",
    "build_strategy",
]
