from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from binance_algo.research.datasets.references import DatasetReference
from binance_algo.research.experiments.ids import experiment_id, result_digest
from binance_algo.research.experiments.models import (
    CodeFingerprint,
    DatasetIdentity,
    ExperimentSpec,
    FeatureSetIdentity,
    LabelIdentity,
    ParameterizedComponent,
    ProvenanceQuality,
    VersionedComponent,
)


def _code(*, commit: str = "a" * 40, dirty_diff: str | None = None) -> CodeFingerprint:
    return CodeFingerprint(
        git_commit=commit,
        git_dirty=dirty_diff is not None,
        git_diff_sha256=dirty_diff,
        source_tree_sha256=None,
        provenance_quality=(
            ProvenanceQuality.GIT_DIRTY if dirty_diff is not None else ProvenanceQuality.GIT_CLEAN
        ),
    )


def _dataset(*, dataset_id: str = "dataset-a") -> DatasetIdentity:
    return DatasetIdentity(
        dataset_id=dataset_id,
        dataset_schema_version=2,
        feature_set_id="features:v1",
        label_id="gross_forward_return_1h:v1",
        universe_version="universe-v1",
        start_time_ms=10,
        end_time_ms=20,
        row_count=3,
        content_checksum="c" * 64,
        fingerprint_method="lineage_v2",
    )


def _spec(
    *,
    hypothesis_id: str = "HYP-TEST-0001",
    dataset: DatasetIdentity | None = None,
    strategy_parameters: dict[str, Any] | None = None,
    cost_parameters: dict[str, Any] | None = None,
    split_parameters: dict[str, Any] | None = None,
    code: CodeFingerprint | None = None,
) -> ExperimentSpec:
    return ExperimentSpec(
        hypothesis_id=hypothesis_id,
        campaign_id=None,
        dataset_reference=dataset or _dataset(),
        feature_set=FeatureSetIdentity(
            feature_set_id="features:v1",
            canonical_checksum="f" * 64,
        ),
        label=LabelIdentity(
            label_id="gross_forward_return_1h:v1",
            version="v1",
            target_column="future_return_1h",
        ),
        strategy=VersionedComponent(component_id="residual_momentum", version="v1"),
        strategy_parameters=strategy_parameters or {"weights": [0.2, 0.3, 0.5]},
        portfolio_policy=VersionedComponent(component_id="neutral_long_short", version="v1"),
        portfolio_parameters={"gross_exposure": Decimal("0.5")},
        execution_model=ParameterizedComponent(
            component_id="next_open",
            version="v1",
            parameters={"lag_bars": 1},
        ),
        cost_model=ParameterizedComponent(
            component_id="phase3_costs",
            version="v1",
            parameters=cost_parameters or {"spread_bps": Decimal("1.0")},
        ),
        split_plan=ParameterizedComponent(
            component_id="walk_forward",
            version="v1",
            parameters=split_parameters or {"train_days": 30, "test_days": 14},
        ),
        validation_plan=ParameterizedComponent(
            component_id="phase3_validation",
            version="v1",
            parameters={"embargo_bars": 1},
        ),
        random_seed=42,
        code_fingerprint=code or _code(),
        artifact_policy="summary",
    )


@given(
    st.dictionaries(
        st.text(alphabet=st.characters(categories=("L", "N")), min_size=1),
        st.integers(min_value=-(2**63), max_value=2**63 - 1),
        max_size=12,
    )
)
def test_parameter_key_order_does_not_change_experiment_id(parameters: dict[str, int]) -> None:
    reversed_parameters = dict(reversed(tuple(parameters.items())))

    assert experiment_id(_spec(strategy_parameters=parameters)) == experiment_id(
        _spec(strategy_parameters=reversed_parameters)
    )


def test_absolute_manifest_path_is_excluded_from_dataset_and_experiment_identity() -> None:
    common = {
        "dataset_id": "dataset-a",
        "dataset_schema_version": 2,
        "feature_set_id": "features:v1",
        "label_id": "gross_forward_return_1h:v1",
        "universe_version": "universe-v1",
        "start_time_ms": 10,
        "end_time_ms": 20,
        "row_count": 3,
        "content_checksum": "c" * 64,
        "fingerprint_method": "lineage_v2",
    }
    first = DatasetReference(manifest_path="C:\\first\\dataset.json", **common)
    second = DatasetReference(manifest_path="D:\\other\\dataset.json", **common)

    first_spec = _spec(dataset=DatasetIdentity.from_reference(first))
    second_spec = _spec(dataset=DatasetIdentity.from_reference(second))
    assert experiment_id(first_spec) == experiment_id(second_spec)


def test_material_inputs_change_experiment_identity() -> None:
    baseline = experiment_id(_spec())

    assert experiment_id(_spec(dataset=_dataset(dataset_id="dataset-b"))) != baseline
    assert experiment_id(_spec(strategy_parameters={"weights": [0.1, 0.3, 0.6]})) != baseline
    assert experiment_id(_spec(cost_parameters={"spread_bps": Decimal("2.0")})) != baseline
    assert experiment_id(_spec(split_parameters={"train_days": 60, "test_days": 14})) != baseline
    assert experiment_id(_spec(code=_code(commit="b" * 40))) != baseline
    assert experiment_id(_spec(code=_code(dirty_diff="d" * 64))) != baseline


def test_decimal_normalization_and_result_digest_are_canonical() -> None:
    first = _spec(cost_parameters={"spread_bps": Decimal("1.000")})
    second = _spec(cost_parameters={"spread_bps": Decimal("1")})
    assert experiment_id(first) == experiment_id(second)

    assert result_digest(
        metrics={"sharpe": 1.25, "return": Decimal("0.10")},
        artifact_checksums={"curve": "a", "report": "b"},
    ) == result_digest(
        metrics={"return": Decimal("0.1"), "sharpe": 1.25},
        artifact_checksums={"report": "b", "curve": "a"},
    )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_identity_values_are_rejected(invalid: float) -> None:
    with pytest.raises(ValidationError, match="NaN or infinity"):
        _spec(strategy_parameters={"invalid": invalid})


def test_absolute_path_in_arbitrary_parameters_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="absolute path"):
        _spec(strategy_parameters={"path": str(tmp_path.resolve())})

    with pytest.raises(ValidationError, match="absolute path"):
        _spec(hypothesis_id=str(tmp_path.resolve()))


def test_absolute_path_used_as_parameter_key_is_rejected() -> None:
    with pytest.raises(ValidationError, match="absolute path key"):
        _spec(strategy_parameters={"C:\\local\\secret": 1})
