"""Typed strategy contracts and implementations for offline research."""

from binance_algo.research.strategies.base import FittedStrategy, Strategy
from binance_algo.research.strategies.registry import build_strategy
from binance_algo.research.strategies.residual_momentum import (
    ResidualMomentumParameters,
    ResidualMomentumStrategy,
)

__all__ = [
    "FittedStrategy",
    "ResidualMomentumParameters",
    "ResidualMomentumStrategy",
    "Strategy",
    "build_strategy",
]
