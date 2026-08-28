"""Deflated Sharpe Ratio and conditional CSCV/PBO calculations.

The DSR implementation follows Bailey and López de Prado's probabilistic Sharpe adjustment. It
estimates the expected maximum Sharpe under repeated trials, then evaluates the observed Sharpe
with the finite-sample skewness/kurtosis correction. Inputs and outputs use annualized Sharpe;
the formula is evaluated at the observation frequency after explicit de-annualization.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from enum import StrEnum
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

from binance_algo.common.errors import ResearchError

EULER_MASCHERONI = 0.5772156649015329
MINIMUM_DSR_OBSERVATIONS = 30
MINIMUM_PBO_STRATEGIES = 8
MINIMUM_PBO_SEGMENTS = 8


class StatisticalStatus(StrEnum):
    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, slots=True)
class ReturnMoments:
    sample_size: int
    skewness: float
    kurtosis: float


@dataclass(frozen=True, slots=True)
class DeflatedSharpeResult:
    status: StatisticalStatus
    probability: float
    observed_sharpe: float
    benchmark_sharpe: float
    sample_size: int
    skewness: float
    kurtosis: float
    number_of_trials: int
    periods_per_year: int


@dataclass(frozen=True, slots=True)
class PBOResult:
    status: StatisticalStatus
    probability: float | None
    reason: str
    combinations: int
    strategy_count: int
    observation_count: int
    segments: int


def return_moments(returns: npt.ArrayLike) -> ReturnMoments:
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1 or values.size < MINIMUM_DSR_OBSERVATIONS:
        raise ResearchError(
            f"DSR requires at least {MINIMUM_DSR_OBSERVATIONS} one-dimensional observations"
        )
    if np.any(~np.isfinite(values)):
        raise ResearchError("DSR returns contain NaN or infinity")
    centered = values - float(np.mean(values))
    variance = float(np.mean(centered**2))
    if variance <= 1e-24:
        raise ResearchError("DSR requires non-zero return variance")
    standard_deviation = math.sqrt(variance)
    skewness = float(np.mean((centered / standard_deviation) ** 3))
    kurtosis = float(np.mean((centered / standard_deviation) ** 4))
    if not math.isfinite(skewness) or not math.isfinite(kurtosis):
        raise ResearchError("DSR return moments are not finite")
    return ReturnMoments(
        sample_size=int(values.size),
        skewness=skewness,
        kurtosis=kurtosis,
    )


def _expected_maximum_sharpe(
    *,
    trial_sharpe_standard_deviation: float,
    number_of_trials: int,
) -> float:
    if number_of_trials <= 1 or trial_sharpe_standard_deviation <= 1e-18:
        return 0.0
    normal = NormalDist()
    first = normal.inv_cdf(1.0 - 1.0 / number_of_trials)
    second = normal.inv_cdf(1.0 - 1.0 / (number_of_trials * math.e))
    return trial_sharpe_standard_deviation * (
        (1.0 - EULER_MASCHERONI) * first + EULER_MASCHERONI * second
    )


def deflated_sharpe_ratio(
    *,
    observed_sharpe: float,
    sample_size: int,
    skewness: float,
    kurtosis: float,
    number_of_trials: int,
    trial_sharpes: npt.ArrayLike,
    periods_per_year: int,
) -> DeflatedSharpeResult:
    """Calculate the DSR probability after explicit repeated-trial deflation."""

    values = np.asarray(trial_sharpes, dtype=np.float64)
    scalar_inputs = (observed_sharpe, skewness, kurtosis)
    if any(not math.isfinite(value) for value in scalar_inputs) or np.any(~np.isfinite(values)):
        raise ResearchError("DSR inputs must be finite")
    if sample_size < MINIMUM_DSR_OBSERVATIONS:
        raise ResearchError(
            f"DSR requires at least {MINIMUM_DSR_OBSERVATIONS} observations; got {sample_size}"
        )
    if number_of_trials < 1 or values.ndim != 1 or values.size != number_of_trials:
        raise ResearchError("DSR trial count must equal the one-dimensional Sharpe sample")
    if periods_per_year < 1:
        raise ResearchError("DSR periods_per_year must be positive")
    annualization = math.sqrt(periods_per_year)
    periodic_trials = values / annualization
    trial_standard_deviation = (
        float(np.std(periodic_trials, ddof=1)) if number_of_trials > 1 else 0.0
    )
    benchmark_periodic = _expected_maximum_sharpe(
        trial_sharpe_standard_deviation=trial_standard_deviation,
        number_of_trials=number_of_trials,
    )
    observed_periodic = observed_sharpe / annualization
    denominator_squared = (
        1.0 - skewness * observed_periodic + ((kurtosis - 1.0) / 4.0) * observed_periodic**2
    )
    if denominator_squared <= 0 or not math.isfinite(denominator_squared):
        raise ResearchError("DSR finite-sample correction is undefined for these inputs")
    statistic = (
        (observed_periodic - benchmark_periodic)
        * math.sqrt(sample_size - 1)
        / math.sqrt(denominator_squared)
    )
    probability = NormalDist().cdf(statistic)
    return DeflatedSharpeResult(
        status=StatisticalStatus.APPLICABLE,
        probability=probability,
        observed_sharpe=observed_sharpe,
        benchmark_sharpe=benchmark_periodic * annualization,
        sample_size=sample_size,
        skewness=skewness,
        kurtosis=kurtosis,
        number_of_trials=number_of_trials,
        periods_per_year=periods_per_year,
    )


def _sharpe_like(values: npt.NDArray[np.float64]) -> float:
    standard_deviation = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    mean = float(np.mean(values))
    if standard_deviation <= 1e-18:
        return mean / 1e-18
    return mean / standard_deviation


def probability_of_backtest_overfitting(
    returns: npt.ArrayLike,
    *,
    segments: int = MINIMUM_PBO_SEGMENTS,
) -> PBOResult:
    """Estimate PBO with CSCV only when trial/segment structure is adequate."""

    matrix = np.asarray(returns, dtype=np.float64)
    if matrix.ndim != 2:
        raise ResearchError("PBO returns must be a two-dimensional observation-by-trial matrix")
    observation_count, strategy_count = matrix.shape
    if np.any(~np.isfinite(matrix)):
        raise ResearchError("PBO returns contain NaN or infinity")
    reasons = []
    if strategy_count < MINIMUM_PBO_STRATEGIES:
        reasons.append(
            f"requires at least {MINIMUM_PBO_STRATEGIES} comparable trials; got {strategy_count}"
        )
    if segments < MINIMUM_PBO_SEGMENTS or segments % 2:
        reasons.append(
            f"requires an even segment count of at least {MINIMUM_PBO_SEGMENTS}; got {segments}"
        )
    if observation_count < segments * 2:
        reasons.append(
            f"requires at least two observations per segment; got {observation_count}/{segments}"
        )
    if reasons:
        return PBOResult(
            status=StatisticalStatus.NOT_APPLICABLE,
            probability=None,
            reason="; ".join(reasons),
            combinations=0,
            strategy_count=strategy_count,
            observation_count=observation_count,
            segments=segments,
        )
    segment_indices = tuple(
        np.asarray(part, dtype=np.int64)
        for part in np.array_split(np.arange(observation_count), segments)
    )
    logits: list[float] = []
    half = segments // 2
    for in_segments in itertools.combinations(range(segments), half):
        if 0 not in in_segments:
            continue
        in_set = set(in_segments)
        out_segments = tuple(index for index in range(segments) if index not in in_set)
        in_index = np.concatenate(tuple(segment_indices[index] for index in in_segments))
        out_index = np.concatenate(tuple(segment_indices[index] for index in out_segments))
        in_scores = np.asarray(
            [_sharpe_like(matrix[in_index, column]) for column in range(strategy_count)]
        )
        selected = int(np.argmax(in_scores))
        out_scores = np.asarray(
            [_sharpe_like(matrix[out_index, column]) for column in range(strategy_count)]
        )
        order = np.lexsort((np.arange(strategy_count), -out_scores))
        rank_index = int(np.flatnonzero(order == selected)[0])
        percentile = (strategy_count - rank_index) / (strategy_count + 1.0)
        logits.append(math.log(percentile / (1.0 - percentile)))
    if not logits:
        return PBOResult(
            status=StatisticalStatus.NOT_APPLICABLE,
            probability=None,
            reason="CSCV produced no unique train/test combinations",
            combinations=0,
            strategy_count=strategy_count,
            observation_count=observation_count,
            segments=segments,
        )
    probability = sum(value <= 0 for value in logits) / len(logits)
    return PBOResult(
        status=StatisticalStatus.APPLICABLE,
        probability=probability,
        reason="CSCV estimate over unique complementary segment splits",
        combinations=len(logits),
        strategy_count=strategy_count,
        observation_count=observation_count,
        segments=segments,
    )


__all__ = [
    "DeflatedSharpeResult",
    "PBOResult",
    "ReturnMoments",
    "StatisticalStatus",
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "return_moments",
]
