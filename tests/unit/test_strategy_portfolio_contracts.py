from __future__ import annotations

from decimal import Decimal
from typing import Any, cast

import pytest
from pydantic import ValidationError

from binance_algo.common.errors import ResearchError
from binance_algo.research.experiments.models import RunStatus
from binance_algo.research.experiments.store import ExperimentRunRecord, ResearchStore
from binance_algo.research.strategy_portfolio import loader
from binance_algo.research.strategy_portfolio.models import (
    PortfolioFile,
    StrategyPortfolioComponent,
    StrategyPortfolioSpec,
    WeightingMode,
)


def _run(
    run_id: str,
    experiment_id: str,
    status: RunStatus,
    *,
    attempt: int,
) -> ExperimentRunRecord:
    return ExperimentRunRecord(
        run_id=run_id,
        experiment_id=experiment_id,
        attempt=attempt,
        status=status,
        worker_id=None,
        host_name=None,
        process_id=None,
        started_at_ms=None,
        heartbeat_at_ms=None,
        finished_at_ms=None,
        runtime_seconds=None,
        result_digest="d" * 64 if status is RunStatus.SUCCEEDED else None,
        error_type=None,
        error_message=None,
        traceback_path=None,
        created_at_ms=attempt,
    )


class _FakeStore:
    def __init__(self, runs: tuple[ExperimentRunRecord, ...]) -> None:
        self.runs = runs

    def get_experiment(self, experiment_id: str) -> object:
        return {"experiment_id": experiment_id}

    def get_run(self, run_id: str) -> ExperimentRunRecord | None:
        return next((run for run in self.runs if run.run_id == run_id), None)

    def list_successful_runs(self, experiment_id: str) -> tuple[ExperimentRunRecord, ...]:
        return tuple(
            run
            for run in self.runs
            if run.experiment_id == experiment_id and run.status is RunStatus.SUCCEEDED
        )


def test_schema_is_strict_and_fixed_weights_are_exact() -> None:
    component = {"experiment_id": "one", "label": "One", "capital_weight": "1.0"}
    declared = PortfolioFile.model_validate(
        {
            "schema_version": 1,
            "portfolios": [
                {
                    "portfolio_id": "one_only",
                    "title": "One",
                    "description": "Synthetic",
                    "components": [component],
                }
            ],
        }
    )
    assert declared.portfolios[0].resolved_weights() == (Decimal("1.0"),)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        StrategyPortfolioSpec.model_validate(
            {
                "portfolio_id": "unknown_field",
                "title": "Unknown",
                "description": "Synthetic",
                "components": [component],
                "optimizer": "max_sharpe",
            }
        )
    with pytest.raises(ValidationError, match="sum exactly to 1"):
        StrategyPortfolioSpec.model_validate(
            {
                "portfolio_id": "bad_sum",
                "title": "Bad",
                "description": "Synthetic",
                "components": [{**component, "capital_weight": "0.9"}],
            }
        )
    with pytest.raises(ValidationError, match="unique experiment_id"):
        StrategyPortfolioSpec.model_validate(
            {
                "portfolio_id": "duplicate",
                "title": "Duplicate",
                "description": "Synthetic",
                "components": [
                    {**component, "capital_weight": "0.5"},
                    {**component, "label": "Again", "capital_weight": "0.5"},
                ],
            }
        )


def test_equal_weight_rejects_manual_and_negative_weights() -> None:
    with pytest.raises(ValidationError, match="must not declare"):
        StrategyPortfolioSpec(
            portfolio_id="equal",
            title="Equal",
            description="Synthetic",
            weighting=WeightingMode.EQUAL_WEIGHT,
            components=(
                StrategyPortfolioComponent(
                    experiment_id="one",
                    label="One",
                    capital_weight=Decimal(1),
                ),
            ),
        )
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        StrategyPortfolioComponent(
            experiment_id="one",
            label="One",
            capital_weight=Decimal("-0.1"),
        )


def test_run_resolution_rejects_wrong_owner_and_non_success() -> None:
    wrong_owner = _run("wrong", "other", RunStatus.SUCCEEDED, attempt=1)
    pending = _run("pending", "expected", RunStatus.PENDING, attempt=2)
    store = cast(ResearchStore, _FakeStore((wrong_owner, pending)))
    declaration = StrategyPortfolioComponent(
        experiment_id="expected",
        run_id="wrong",
        label="Expected",
        capital_weight=Decimal(1),
    )
    with pytest.raises(ResearchError, match="belongs to experiment"):
        loader.resolve_component_run(
            store=store,
            data_root=cast(Any, None),
            declaration=declaration,
        )
    with pytest.raises(ResearchError, match="not SUCCEEDED"):
        loader.resolve_component_run(
            store=store,
            data_root=cast(Any, None),
            declaration=declaration.model_copy(update={"run_id": "pending"}),
        )


def test_implicit_resolution_uses_newest_verified_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    older = _run("older", "expected", RunStatus.SUCCEEDED, attempt=1)
    corrupt_newer = _run("newer", "expected", RunStatus.SUCCEEDED, attempt=2)
    store = cast(ResearchStore, _FakeStore((older, corrupt_newer)))

    def verify(**kwargs: Any) -> tuple[()]:
        run = cast(ExperimentRunRecord, kwargs["run"])
        if run.run_id == "newer":
            raise ResearchError("checksum mismatch")
        return ()

    monkeypatch.setattr(loader, "_verified_artifacts", verify)
    _, resolved, artifacts = loader.resolve_component_run(
        store=store,
        data_root=cast(Any, None),
        declaration=StrategyPortfolioComponent(
            experiment_id="expected",
            label="Expected",
            capital_weight=Decimal(1),
        ),
    )
    assert resolved.run_id == "older"
    assert artifacts == ()
