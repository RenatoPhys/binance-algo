from __future__ import annotations

import numpy as np
import pytest

from binance_algo.common.errors import ResearchError
from binance_algo.research.validation.multiple_testing import (
    StatisticalStatus,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    return_moments,
)


def test_deflated_sharpe_matches_independent_numeric_fixture() -> None:
    result = deflated_sharpe_ratio(
        observed_sharpe=1.2,
        sample_size=500,
        skewness=-0.2,
        kurtosis=3.5,
        number_of_trials=5,
        trial_sharpes=np.asarray([0.1, 0.3, 0.6, 0.9, 1.2]),
        periods_per_year=252,
    )

    assert result.status is StatisticalStatus.APPLICABLE
    assert result.benchmark_sharpe == pytest.approx(0.5293290554064866)
    assert result.probability == pytest.approx(0.8251207512775627)
    fewer_trials = deflated_sharpe_ratio(
        observed_sharpe=1.2,
        sample_size=500,
        skewness=-0.2,
        kurtosis=3.5,
        number_of_trials=2,
        trial_sharpes=np.asarray([0.9, 1.2]),
        periods_per_year=252,
    )
    assert fewer_trials.probability > result.probability


def test_dsr_fails_clearly_for_insufficient_or_invalid_samples() -> None:
    with pytest.raises(ResearchError, match="at least 30"):
        return_moments(np.arange(20, dtype=np.float64))
    with pytest.raises(ResearchError, match="non-zero"):
        return_moments(np.ones(30, dtype=np.float64))
    with pytest.raises(ResearchError, match="trial count"):
        deflated_sharpe_ratio(
            observed_sharpe=1.0,
            sample_size=100,
            skewness=0.0,
            kurtosis=3.0,
            number_of_trials=3,
            trial_sharpes=np.asarray([0.5, 1.0]),
            periods_per_year=252,
        )


def test_pbo_is_conditional_and_reports_trial_count() -> None:
    inadequate = probability_of_backtest_overfitting(np.ones((100, 3)))
    assert inadequate.status is StatisticalStatus.NOT_APPLICABLE
    assert inadequate.probability is None
    assert inadequate.strategy_count == 3
    assert "at least 8 comparable trials" in inadequate.reason

    time = np.arange(160, dtype=np.float64)
    comparable = np.column_stack(
        [np.sin(time / (3 + index)) * 0.01 + (index - 4) * 0.00001 for index in range(8)]
    )
    result = probability_of_backtest_overfitting(comparable)
    assert result.status is StatisticalStatus.APPLICABLE
    assert result.strategy_count == 8
    assert result.segments == 8
    assert result.combinations == 35
    assert result.probability is not None and 0 <= result.probability <= 1
