from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import orjson
import polars as pl

from binance_algo.config import load_settings
from binance_algo.data.storage import LocalFilesystemStorage
from binance_algo.research.backtest import run_walk_forward
from binance_algo.research.baseline import (
    build_phase3_baseline_components,
    run_and_persist_phase3_baseline,
    run_phase3_walk_forward,
)
from binance_algo.research.contracts import FoldContext, StrategyScores, TrainingDataset
from binance_algo.research.strategies.base import FittedStrategy, Strategy
from binance_algo.research.visualization import render_pnl_svg

from ..research_fixtures import research_frame

PROJECT_ROOT = Path(__file__).parents[2]
BASE_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
GOLDEN_BASELINE = PROJECT_ROOT / "tests" / "golden" / "research_phase3_synthetic.json"


def _canonicalize_golden_value(value: Any) -> Any:
    """Remove sub-ULP CPU differences while preserving material golden changes."""

    if isinstance(value, float):
        formatted = f"{value:.12f}"
        return "0.000000000000" if formatted == "-0.000000000000" else formatted
    if isinstance(value, str) and value[:1] in {"{", "["}:
        try:
            return _canonicalize_golden_value(json.loads(value))
        except json.JSONDecodeError:
            return value
    if isinstance(value, dict):
        return {key: _canonicalize_golden_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonicalize_golden_value(item) for item in value]
    return value


class _SpyFittedStrategy:
    def __init__(
        self,
        delegate: FittedStrategy,
        score_observations: list[tuple[tuple[str, ...], int, int]],
    ) -> None:
        self._delegate = delegate
        self._score_observations = score_observations

    def score(self, features: pl.DataFrame, *, context: FoldContext) -> StrategyScores:
        self._score_observations.append(
            (
                tuple(features.columns),
                int(features["decision_time_ms"].min()),
                int(features["decision_time_ms"].max()),
            )
        )
        return self._delegate.score(features, context=context)


class _SpyStrategy:
    def __init__(self, delegate: Strategy) -> None:
        self._delegate = delegate
        self.strategy_id = "spy_residual_momentum"
        self.strategy_version = "1"
        self.fit_observations: list[tuple[tuple[str, ...], int, int, bool]] = []
        self.score_observations: list[tuple[tuple[str, ...], int, int]] = []

    def required_features(self) -> tuple[str, ...]:
        return self._delegate.required_features()

    def target_column(self) -> str | None:
        return self._delegate.target_column()

    def fit(self, train: TrainingDataset, *, context: FoldContext) -> FittedStrategy:
        self.fit_observations.append(
            (
                tuple(train.features.columns),
                int(train.features["decision_time_ms"].min()),
                int(train.features["decision_time_ms"].max()),
                train.target is None,
            )
        )
        fitted = self._delegate.fit(train, context=context)
        return _SpyFittedStrategy(fitted, self.score_observations)


def _research_frame(days: int = 11) -> pl.DataFrame:
    return research_frame(days)


def test_walk_forward_is_temporal_costed_and_accounting_balances() -> None:
    settings = load_settings(BASE_CONFIG)
    config = settings.research.model_copy(
        update={"walk_forward_train_days": 7, "walk_forward_test_days": 1}
    )
    frame = _research_frame()
    baseline = run_phase3_walk_forward(frame, config=config)
    expensive = run_phase3_walk_forward(frame, config=config, cost_multiplier=2.0)
    delayed = run_phase3_walk_forward(frame, config=config, signal_delay_bars=1)

    assert baseline.folds
    assert all(fold.train_end_ms < fold.test_start_ms for fold in baseline.folds)
    assert all(fold.embargo_bars == 1 for fold in baseline.folds)
    assert baseline.metrics.accounting_error_max <= 1e-15
    assert baseline.metrics.trading_fees > 0
    assert baseline.metrics.turnover > 0
    assert baseline.curve["gross_exposure"].max() <= 0.5 + 1e-12
    assert baseline.curve["net_exposure"].abs().max() <= 1e-12
    assert expensive.metrics.total_return < baseline.metrics.total_return
    assert not delayed.curve["net_return"].equals(baseline.curve["net_return"])

    svg = render_pnl_svg(baseline.curve)
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "Equity OOS" in svg
    assert "Drawdown líquido" in svg
    assert "Decomposição acumulada" in svg
    assert svg.count("<polyline") == 7
    assert all(f"fold {number}" in svg for number in range(1, len(baseline.folds) + 1))


def test_generic_engine_fits_only_train_and_scores_only_declared_test_features() -> None:
    settings = load_settings(BASE_CONFIG)
    config = settings.research.model_copy(
        update={"walk_forward_train_days": 7, "walk_forward_test_days": 1}
    )
    components = build_phase3_baseline_components(config)
    strategy = _SpyStrategy(components.strategy)

    result = run_walk_forward(
        _research_frame(),
        config=config,
        strategy=strategy,
        portfolio_policy=components.portfolio_policy,
    )

    expected_columns = (
        "decision_time_ms",
        "symbol",
        *components.strategy.required_features(),
    )
    assert len(strategy.fit_observations) == len(result.folds)
    assert len(strategy.score_observations) == len(result.folds)
    for fold, fit_observation, score_observation in zip(
        result.folds,
        strategy.fit_observations,
        strategy.score_observations,
        strict=True,
    ):
        fit_columns, fit_min, fit_max, target_is_none = fit_observation
        score_columns, score_min, score_max = score_observation
        assert fit_columns == expected_columns
        assert score_columns == expected_columns
        assert target_is_none
        assert fit_min == fold.train_start_ms
        assert fit_max == fold.train_end_ms
        assert score_min == fold.test_start_ms
        assert score_max == fold.test_end_ms
        assert fit_max < score_min


def test_synthetic_phase3_baseline_matches_golden_snapshot() -> None:
    expected = json.loads(GOLDEN_BASELINE.read_text(encoding="utf-8"))
    settings = load_settings(BASE_CONFIG)
    config = settings.research.model_copy(
        update={"walk_forward_train_days": 7, "walk_forward_test_days": 1}
    )

    result = run_phase3_walk_forward(_research_frame(), config=config)
    rows = result.curve.to_dicts()
    canonical_rows = _canonicalize_golden_value(rows)
    curve_digest = hashlib.sha256(
        orjson.dumps(canonical_rows, option=orjson.OPT_SORT_KEYS)
    ).hexdigest()

    assert [asdict(fold) for fold in result.folds] == expected["folds"]
    assert _canonicalize_golden_value(asdict(result.metrics)) == _canonicalize_golden_value(
        expected["metrics"]
    )
    assert result.curve.height == expected["curve"]["row_count"]
    assert result.curve.columns == expected["curve"]["columns"]
    assert _canonicalize_golden_value(rows[0]) == _canonicalize_golden_value(
        expected["curve"]["first_row"]
    )
    assert _canonicalize_golden_value(rows[-1]) == _canonicalize_golden_value(
        expected["curve"]["last_row"]
    )
    assert curve_digest == expected["curve"]["canonical_rows_sha256"]


def test_persisted_pnl_chart_is_opt_in(tmp_path: Path) -> None:
    settings = load_settings(BASE_CONFIG)
    config = settings.research.model_copy(
        update={"walk_forward_train_days": 7, "walk_forward_test_days": 1}
    )
    dataset_path = tmp_path / "dataset.parquet"
    _research_frame().write_parquet(dataset_path)
    storage = LocalFilesystemStorage(tmp_path / "data")
    reports_root = tmp_path / "reports"

    default_result = run_and_persist_phase3_baseline(
        dataset_path=dataset_path,
        storage=storage,
        reports_root=reports_root,
        compression="zstd",
        config=config,
    )
    report_before = Path(default_result.report_json_path).read_bytes()

    assert default_result.report_chart_path is None
    assert not list(reports_root.glob("*_pnl.svg"))

    charted_result = run_and_persist_phase3_baseline(
        dataset_path=dataset_path,
        storage=storage,
        reports_root=reports_root,
        compression="zstd",
        config=config,
        generate_chart=True,
    )

    assert charted_result.run_version == default_result.run_version
    assert charted_result.curve_path == default_result.curve_path
    assert Path(charted_result.report_json_path).read_bytes() == report_before
    assert charted_result.report_chart_path is not None
    assert Path(charted_result.report_chart_path).is_file()
    assert len(list(reports_root.glob("*_pnl.svg"))) == 1
