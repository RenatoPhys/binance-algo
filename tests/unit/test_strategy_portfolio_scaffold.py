from __future__ import annotations

import pytest

from binance_algo.common.errors import ResearchError
from binance_algo.research.strategy_portfolio.models import PortfolioFile
from binance_algo.research.strategy_portfolio.scaffold import scaffold_payload


def test_scaffold_contains_only_explicit_ids_with_equal_weights() -> None:
    payload = scaffold_payload(("experiment-b", "experiment-a"))
    declared = PortfolioFile.model_validate(payload)

    portfolio = declared.portfolios[0]
    assert portfolio.weighting.value == "equal_weight"
    assert [component.experiment_id for component in portfolio.components] == [
        "experiment-b",
        "experiment-a",
    ]
    assert all(component.capital_weight is None for component in portfolio.components)
    assert portfolio.resolved_weights()[0] == portfolio.resolved_weights()[1]


@pytest.mark.parametrize(
    ("experiment_ids", "message"),
    [
        ((), "at least one"),
        (("same", "same"), "unique"),
        (("valid", " "), "non-empty"),
    ],
)
def test_scaffold_rejects_implicit_or_ambiguous_components(
    experiment_ids: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ResearchError, match=message):
        scaffold_payload(experiment_ids)
