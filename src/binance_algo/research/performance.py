"""Common transparent return statistics for backtests and strategy portfolios."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from binance_algo.common.errors import ResearchError

HOURS_PER_YEAR = 24 * 365


@dataclass(frozen=True, slots=True)
class ReturnStatistics:
    periods: int
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    calmar: float
    positive_period_fraction: float


def calculate_return_statistics(
    returns: npt.ArrayLike,
    *,
    periods_per_year: int = HOURS_PER_YEAR,
) -> ReturnStatistics:
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ResearchError("return statistics require a non-empty one-dimensional series")
    if periods_per_year < 1 or np.any(~np.isfinite(values)):
        raise ResearchError("return statistics inputs must be finite and correctly annualized")
    if np.any(1.0 + values <= 0):
        raise ResearchError("return statistics require positive compounded equity")
    equity = np.cumprod(1.0 + values)
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1.0
    mean = float(np.mean(values))
    standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    annualized_volatility = standard_deviation * math.sqrt(periods_per_year)
    sharpe = mean / standard_deviation * math.sqrt(periods_per_year) if standard_deviation else 0.0
    annualized_return = float(math.expm1(np.mean(np.log1p(values)) * periods_per_year))
    max_drawdown = float(np.min(drawdowns))
    calmar = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    return ReturnStatistics(
        periods=len(values),
        total_return=float(equity[-1] - 1.0),
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        calmar=calmar,
        positive_period_fraction=float(np.mean(values > 0)),
    )


__all__ = ["HOURS_PER_YEAR", "ReturnStatistics", "calculate_return_statistics"]
