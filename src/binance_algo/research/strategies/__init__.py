"""Typed strategy contracts and implementations for offline research."""

from binance_algo.research.strategies.base import FittedStrategy, Strategy
from binance_algo.research.strategies.carry_consensus_strength import (
    CARRY_CONSENSUS_STRENGTH_FEATURES,
    CarryConsensusStrengthParameters,
    CarryConsensusStrengthStrategy,
)
from binance_algo.research.strategies.carry_dual_trend import (
    CARRY_DUAL_TREND_FEATURES,
    CarryDualTrendParameters,
    CarryDualTrendStrategy,
)
from binance_algo.research.strategies.carry_multi_horizon import (
    CARRY_MULTI_HORIZON_FEATURES,
    CarryMultiHorizonParameters,
    CarryMultiHorizonStrategy,
)
from binance_algo.research.strategies.carry_multi_regime import (
    CARRY_MULTI_REGIME_FEATURES,
    CarryMultiRegimeParameters,
    CarryMultiRegimeStrategy,
)
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
from binance_algo.research.strategies.market_regime_trend import (
    MARKET_REGIME_TREND_FEATURES,
    MarketRegimeTrendParameters,
    MarketRegimeTrendStrategy,
)
from binance_algo.research.strategies.multi_horizon_trend import (
    MULTI_HORIZON_TREND_FEATURES,
    MultiHorizonTrendParameters,
    MultiHorizonTrendStrategy,
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
    "CARRY_CONSENSUS_STRENGTH_FEATURES",
    "CARRY_DUAL_TREND_FEATURES",
    "CARRY_MULTI_HORIZON_FEATURES",
    "CARRY_MULTI_REGIME_FEATURES",
    "CARRY_RELATIVE_STRENGTH_FEATURES",
    "DONCHIAN_BREAKOUT_FEATURES",
    "LINEAR_CROSS_SECTIONAL_FEATURES",
    "MARKET_REGIME_TREND_FEATURES",
    "MULTI_HORIZON_TREND_FEATURES",
    "RELATIVE_STRENGTH_FEATURES",
    "SMA_CROSSOVER_FEATURES",
    "CarryConsensusStrengthParameters",
    "CarryConsensusStrengthStrategy",
    "CarryDualTrendParameters",
    "CarryDualTrendStrategy",
    "CarryMultiHorizonParameters",
    "CarryMultiHorizonStrategy",
    "CarryMultiRegimeParameters",
    "CarryMultiRegimeStrategy",
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
    "MarketRegimeTrendParameters",
    "MarketRegimeTrendStrategy",
    "MultiHorizonTrendParameters",
    "MultiHorizonTrendStrategy",
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
