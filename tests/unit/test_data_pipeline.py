from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from binance_algo.common.errors import StorageError, UniverseError
from binance_algo.config import UniverseConfig
from binance_algo.data.metadata import MetadataSnapshotService
from binance_algo.data.storage import LocalFilesystemStorage
from binance_algo.data.universe import (
    build_seed_universe,
    find_metadata_snapshot,
)

FIXTURE = Path(__file__).parents[1] / "fixtures" / "exchange_info.json"


class FixtureSource:
    async def exchange_info(self) -> dict[str, Any]:
        payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        return payload

    async def server_time(self) -> int:
        return 1_787_873_856_926


def test_storage_is_idempotent_and_confined(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(tmp_path / "data")
    target = storage.path("bronze", "sample.parquet")
    frame = pl.DataFrame({"value": [1, 2]})

    assert storage.write_parquet_atomic(target, frame, compression="zstd") == target
    assert storage.write_parquet_atomic(target, frame, compression="zstd") == target
    with pytest.raises(StorageError, match="different data"):
        storage.write_parquet_atomic(target, pl.DataFrame({"value": [3]}), compression="zstd")
    with pytest.raises(StorageError, match="escapes"):
        storage.path("..", "outside.parquet")


async def test_snapshot_then_point_in_time_universe_is_deterministic(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(tmp_path / "data")
    snapshot = await MetadataSnapshotService(
        source=FixtureSource(), storage=storage, compression="zstd"
    ).snapshot()
    metadata_path = Path(snapshot.parquet_path)
    assert snapshot.row_count == 3
    assert Path(snapshot.raw_path).exists()
    assert metadata_path.exists()

    as_of_ms = snapshot.valid_from_ms + 1_000
    selected_path = find_metadata_snapshot(storage.root, as_of_ms=as_of_ms)
    config = UniverseConfig()
    first = build_seed_universe(
        metadata_path=selected_path,
        as_of_ms=as_of_ms,
        config=config,
        storage=storage,
        compression="zstd",
    )
    second = build_seed_universe(
        metadata_path=selected_path,
        as_of_ms=as_of_ms,
        config=config,
        storage=storage,
        compression="zstd",
    )

    assert first.version == second.version
    assert first.included_symbols == ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    assert pl.read_parquet(first.parquet_path)["included"].all()
    manifest = json.loads(Path(first.manifest_path).read_text(encoding="utf-8"))
    assert manifest["included_symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


async def test_snapshot_after_cutoff_is_not_visible(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(tmp_path / "data")
    snapshot = await MetadataSnapshotService(
        source=FixtureSource(), storage=storage, compression="zstd"
    ).snapshot()

    with pytest.raises(UniverseError, match="no metadata snapshot"):
        find_metadata_snapshot(storage.root, as_of_ms=snapshot.valid_from_ms - 1)
