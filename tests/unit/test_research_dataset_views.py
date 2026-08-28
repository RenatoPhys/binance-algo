from __future__ import annotations

from dataclasses import FrozenInstanceError

import polars as pl
import pytest

from binance_algo.common.errors import ResearchError
from binance_algo.research.datasets.schemas import RESEARCH_DATASET_SCHEMA_V2, ColumnRole
from binance_algo.research.datasets.views import build_feature_view, build_target_view
from binance_algo.research.features.base import FeatureDefinition
from binance_algo.research.features.registry import (
    PHASE3_FEATURE_REGISTRY,
    FeatureRegistry,
    FeatureSetSpec,
)
from binance_algo.research.labels.forward_returns import GROSS_FORWARD_RETURN_1H


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "decision_time_ms": [1, 1],
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "rolling_beta": [0.8, 1.1],
            "future_return_1h": [0.01, -0.02],
            "outcome_quote_volume_1h": [10.0, 20.0],
        }
    )


def _masquerading_outcome_registry() -> FeatureRegistry:
    definition = FeatureDefinition(
        feature_id="outcome_quote_volume_1h:v1",
        name="outcome_quote_volume_1h",
        version="v1",
        description="A deliberately invalid scoring dependency used by the contract test.",
        dtype="Float64",
        lookback="future 1h",
        timestamp_semantics="after decision_time_ms",
        required_datasets=("klines",),
        required_columns=("quote_volume",),
        implementation_path="tests.invalid",
        parameters={},
    )
    return FeatureRegistry((definition,))


def test_feature_view_exposes_only_registered_feature_columns() -> None:
    view = build_feature_view(_frame(), required_features=("rolling_beta",))

    assert view.columns == ["decision_time_ms", "symbol", "rolling_beta"]
    assert "future_return_1h" not in view.columns
    assert "outcome_quote_volume_1h" not in view.columns


def test_schema_role_blocks_outcome_even_if_it_masquerades_as_a_feature() -> None:
    with pytest.raises(ResearchError, match="not a FEATURE"):
        build_feature_view(
            _frame(),
            required_features=("outcome_quote_volume_1h",),
            feature_registry=_masquerading_outcome_registry(),
        )


@pytest.mark.parametrize("column", ["future_return_1h", "future_residual_return_1h"])
def test_registered_targets_cannot_be_requested_as_scoring_features(column: str) -> None:
    with pytest.raises(ResearchError, match="not registered"):
        build_feature_view(_frame(), required_features=(column,))


def test_target_view_requires_a_registered_label_and_is_separate() -> None:
    target = build_target_view(_frame(), label_id=GROSS_FORWARD_RETURN_1H.label_id)

    assert target.columns == ["decision_time_ms", "symbol", "future_return_1h"]
    assert RESEARCH_DATASET_SCHEMA_V2.role_for("future_return_1h") is ColumnRole.TARGET
    assert RESEARCH_DATASET_SCHEMA_V2.role_for("outcome_quote_volume_1h") is ColumnRole.OUTCOME


def test_feature_set_identity_canonicalizes_member_order_but_reports_declared_order() -> None:
    beta = PHASE3_FEATURE_REGISTRY.resolve_name("rolling_beta").feature_id
    volatility = PHASE3_FEATURE_REGISTRY.resolve_name("realized_volatility_24h").feature_id
    first = FeatureSetSpec(
        feature_set_id="test:v1",
        feature_ids=(beta, volatility),
        per_feature_parameters={beta: {"window": 168}},
        version="v1",
        description="test set",
    )
    reversed_set = FeatureSetSpec(
        feature_set_id="test:v1",
        feature_ids=(volatility, beta),
        per_feature_parameters={beta: {"window": 168}},
        version="v1",
        description="test set",
    )

    assert first.canonical_checksum == reversed_set.canonical_checksum
    assert first.to_manifest()["declared_feature_order"] == (beta, volatility)


def test_feature_definitions_are_immutable() -> None:
    definition = PHASE3_FEATURE_REGISTRY.resolve_name("rolling_beta")

    with pytest.raises(FrozenInstanceError):
        definition.name = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        definition.parameters["window"] = 1  # type: ignore[index]
