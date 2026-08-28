from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from binance_algo.config import load_settings
from binance_algo.data.storage import LocalFilesystemStorage
from binance_algo.research.dataset import DatasetAudit, persist_research_dataset
from binance_algo.research.datasets.fingerprints import (
    FINGERPRINT_METHOD,
    InputFileFingerprint,
    logical_content_checksum,
)
from binance_algo.research.datasets.references import load_dataset_reference
from binance_algo.research.features.registry import phase3_feature_set

PROJECT_ROOT = Path(__file__).parents[2]
BASE_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def _input_file() -> InputFileFingerprint:
    return InputFileFingerprint(
        file_id="canonical-kline-file",
        logical_dataset="klines",
        symbol="BTCUSDT",
        interval="1m",
        start_time_ms=1,
        end_time_ms=2,
        row_count=2,
        schema_version=1,
        checksum="a" * 64,
    )


def test_persisted_v2_manifest_has_portable_reference_and_separate_checksums(
    tmp_path: Path,
) -> None:
    settings = load_settings(BASE_CONFIG)
    config = settings.research.model_copy(update={"beta_window_hours": 24})
    frame = pl.DataFrame(
        {
            "decision_time_ms": [10, 10, 10],
            "symbol": list(SYMBOLS),
            "rolling_beta": [0.9, 1.1, 1.0],
            "dataset_schema_version": [2, 2, 2],
        }
    )
    audit = DatasetAudit(
        row_count=3,
        decision_count=1,
        symbols=SYMBOLS,
        min_decision_time_ms=10,
        max_decision_time_ms=10,
        feature_after_cutoff_count=0,
        execution_not_after_decision_count=0,
        label_not_after_execution_count=0,
        duplicate_key_count=0,
        null_feature_count=0,
        passed=True,
    )
    universe_version = "test-universe-v1"

    result = persist_research_dataset(
        frame=frame,
        audit=audit,
        feature_set=phase3_feature_set(config),
        universe_version=universe_version,
        input_files=(_input_file(),),
        source_start_time_ms=1,
        source_end_time_ms=2,
        storage=LocalFilesystemStorage(tmp_path),
        compression="zstd",
        symbols=SYMBOLS,
        config=config,
    )
    reference = load_dataset_reference(Path(result.manifest_path))
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))

    assert reference.dataset_id == result.dataset_id
    assert reference.fingerprint_method == FINGERPRINT_METHOD
    assert reference.content_checksum == logical_content_checksum(frame)
    assert manifest["content_checksum"] == result.content_checksum
    assert manifest["parquet_checksum"] == result.parquet_checksum
    assert manifest["content_checksum"] != manifest["parquet_checksum"]
    assert "manifest_path" not in reference.identity_payload()
    assert "path" not in json.dumps(manifest["fingerprint_payload"])


def test_legacy_manifest_remains_readable_and_is_marked_legacy(tmp_path: Path) -> None:
    parquet_path = tmp_path / "dataset.parquet"
    pl.DataFrame({"value": [1]}).write_parquet(parquet_path)
    manifest_path = tmp_path / "dataset.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_version": "legacy-id",
                "dataset_schema_version": 1,
                "feature_version": "legacy-features",
                "universe_version": "legacy-universe",
                "audit": {
                    "min_decision_time_ms": 10,
                    "max_decision_time_ms": 20,
                    "row_count": 1,
                },
            }
        ),
        encoding="utf-8",
    )

    reference = load_dataset_reference(manifest_path)

    assert reference.dataset_id == "legacy-id"
    assert reference.dataset_schema_version == 1
    assert reference.feature_set_id == "legacy-features"
    assert reference.fingerprint_method == "legacy_content_hash"
    assert len(reference.content_checksum) == 64
