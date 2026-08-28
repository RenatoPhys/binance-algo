from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from binance_algo.common.errors import ResearchError
from binance_algo.research.panel import PanelData, WorkerDatasetCache


def _complete_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "decision_time_ms": [2, 1, 2, 1],
            "symbol": ["ETHUSDT", "BTCUSDT", "BTCUSDT", "ETHUSDT"],
            "feature_a": [4.0, 1.0, 3.0, 2.0],
            "outcome_a": [0.4, 0.1, 0.3, 0.2],
            "execution_time_ms": [3, 2, 3, 2],
            "unused": [99, 99, 99, 99],
        }
    )


def test_panel_data_is_stable_separated_and_read_only() -> None:
    panel = PanelData.from_frame(
        _complete_frame(),
        feature_columns=("feature_a",),
        outcome_columns=("outcome_a",),
        metadata_columns=("execution_time_ms",),
    )

    assert panel.shape == (2, 2)
    assert panel.times.tolist() == [1, 2]
    assert panel.symbols == ("BTCUSDT", "ETHUSDT")
    assert dict(panel.symbol_to_index) == {"BTCUSDT": 0, "ETHUSDT": 1}
    assert panel.features["feature_a"].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert panel.outcomes["outcome_a"].tolist() == [[0.1, 0.2], [0.3, 0.4]]
    assert "outcome_a" not in panel.features
    assert panel.availability.all()
    assert panel.metadata["listing_time_ms"].tolist() == [[-1, -1], [-1, -1]]
    with pytest.raises(ValueError, match="read-only"):
        panel.features["feature_a"][0, 0] = 7.0


def test_partial_panel_preserves_time_and_records_explicit_unavailability() -> None:
    partial = _complete_frame().filter(
        ~((pl.col("decision_time_ms") == 2) & (pl.col("symbol") == "ETHUSDT"))
    )

    panel = PanelData.from_frame(
        partial,
        feature_columns=("feature_a",),
        outcome_columns=("outcome_a",),
    )

    assert panel.shape == (2, 2)
    assert panel.availability.tolist() == [[True, True], [True, False]]
    assert panel.metadata["exclusion_reason"][1, 1] == "missing_panel_row"
    with pytest.raises(ResearchError, match="1 unavailable"):
        panel.require_complete(role="test")


def test_panel_rejects_non_finite_required_value_when_available() -> None:
    frame = _complete_frame().with_columns(
        pl.when((pl.col("decision_time_ms") == 1) & (pl.col("symbol") == "BTCUSDT"))
        .then(float("nan"))
        .otherwise(pl.col("feature_a"))
        .alias("feature_a")
    )

    with pytest.raises(ResearchError, match="feature_a is non-finite"):
        PanelData.from_frame(frame, feature_columns=("feature_a",))


@given(
    time_count=st.integers(min_value=1, max_value=8),
    symbol_count=st.integers(min_value=1, max_value=6),
)
def test_panel_shapes_hold_for_complete_grids(time_count: int, symbol_count: int) -> None:
    times = np.arange(time_count, dtype=np.int64)
    symbols = [f"S{index:02d}" for index in range(symbol_count)]
    frame = pl.DataFrame(
        {
            "decision_time_ms": np.repeat(times, symbol_count),
            "symbol": np.tile(symbols, time_count),
            "feature_a": np.arange(time_count * symbol_count, dtype=np.float64),
        }
    )

    panel = PanelData.from_frame(frame, feature_columns=("feature_a",))

    assert panel.shape == (time_count, symbol_count)
    assert panel.features["feature_a"].shape == panel.shape
    assert panel.availability.shape == panel.shape
    assert panel.availability.all()


def test_worker_cache_projects_columns_and_reuses_loaded_panel(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.parquet"
    _complete_frame().write_parquet(dataset_path)
    cache = WorkerDatasetCache(max_entries=1)

    first = cache.load(
        dataset_path,
        feature_columns=("feature_a",),
        outcome_columns=("outcome_a",),
        metadata_columns=("execution_time_ms",),
    )
    second = cache.load(
        dataset_path,
        feature_columns=("feature_a",),
        outcome_columns=("outcome_a",),
        metadata_columns=("execution_time_ms",),
    )

    assert first is second
    assert "unused" not in first.frame.columns
    assert first.projected_columns == (
        "decision_time_ms",
        "symbol",
        "feature_a",
        "outcome_a",
        "execution_time_ms",
    )
    assert first.load_seconds >= 0
    assert cache.info().entries == 1
    assert cache.info().hits == 1
    assert cache.info().misses == 1
