"""Reproducible point-in-time seed universe construction."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import orjson
import polars as pl

from binance_algo.common.errors import UniverseError
from binance_algo.config import UniverseConfig
from binance_algo.data.storage import LocalFilesystemStorage

MILLISECONDS_PER_DAY = 86_400_000


@dataclass(frozen=True, slots=True)
class UniverseBuildResult:
    parquet_path: str
    manifest_path: str
    version: str
    included_symbols: tuple[str, ...]
    metadata_path: str
    as_of_ms: int


def parse_as_of(value: str | None) -> int:
    """Interpret YYYY-MM-DD as inclusive end-of-day UTC; default to the current instant."""

    if value is None:
        return int(datetime.now(tz=UTC).timestamp() * 1_000)
    try:
        parsed_date = date.fromisoformat(value)
    except ValueError as exc:
        raise UniverseError(f"--as-of must use YYYY-MM-DD, received {value!r}") from exc
    inclusive_end = datetime.combine(parsed_date, time.max, tzinfo=UTC)
    return int(inclusive_end.timestamp() * 1_000)


def find_metadata_snapshot(data_root: Path, *, as_of_ms: int) -> Path:
    metadata_root = data_root / "bronze" / "binance" / "usdm" / "instrument_metadata"
    candidates: list[tuple[int, Path]] = []
    for path in metadata_root.rglob("*.parquet") if metadata_root.exists() else []:
        try:
            values = pl.read_parquet(path, columns=["valid_from_ms"])["valid_from_ms"].unique()
        except pl.exceptions.PolarsError as exc:
            raise UniverseError(f"cannot inspect metadata snapshot {path}: {exc}") from exc
        if len(values) != 1:
            raise UniverseError(f"metadata snapshot must have one valid_from_ms value: {path}")
        valid_from_ms = int(values.item())
        if valid_from_ms <= as_of_ms:
            candidates.append((valid_from_ms, path))
    if not candidates:
        raise UniverseError(
            f"no metadata snapshot at or before {as_of_ms}; "
            "run `binance-algo exchange-info snapshot`"
        )
    return max(candidates, key=lambda item: item[0])[1]


def metadata_valid_from(path: Path) -> int:
    values = pl.read_parquet(path, columns=["valid_from_ms"])["valid_from_ms"].unique()
    if len(values) != 1:
        raise UniverseError(f"metadata snapshot must have one valid_from_ms value: {path}")
    return int(values.item())


def build_seed_universe(
    *,
    metadata_path: Path,
    as_of_ms: int,
    config: UniverseConfig,
    storage: LocalFilesystemStorage,
    compression: str,
) -> UniverseBuildResult:
    frame = pl.read_parquet(metadata_path)
    rows_by_symbol: dict[str, dict[str, Any]] = {
        str(row["symbol"]): row for row in frame.iter_rows(named=True)
    }
    entries: list[dict[str, Any]] = []
    included: list[str] = []

    for symbol in config.seed_symbols:
        metadata = rows_by_symbol.get(symbol)
        reason = "included"
        is_included = True
        if metadata is None:
            reason, is_included = "missing_from_metadata", False
        elif metadata["contract_type"] != config.contract_type:
            reason, is_included = "wrong_contract_type", False
        elif metadata["quote_asset"] != config.quote_asset:
            reason, is_included = "wrong_quote_asset", False
        elif metadata["status"] != config.require_status:
            reason, is_included = "not_trading", False
        elif as_of_ms - int(metadata["onboard_date_ms"]) < (
            config.minimum_listing_days * MILLISECONDS_PER_DAY
        ):
            reason, is_included = "insufficient_listing_history", False
        elif len(included) >= config.maximum_symbols:
            reason, is_included = "maximum_symbols_reached", False

        if is_included:
            included.append(symbol)
        entries.append(
            {
                "symbol": symbol,
                "included": is_included,
                "reason": reason,
                "as_of_ms": as_of_ms,
                "metadata_valid_from_ms": (
                    int(metadata["valid_from_ms"]) if metadata is not None else None
                ),
            }
        )

    if not included:
        raise UniverseError("universe selection produced zero included symbols")

    version_input = {
        "as_of_ms": as_of_ms,
        "metadata_path": metadata_path.name,
        "filters": config.model_dump(mode="json"),
        "entries": entries,
    }
    version = hashlib.sha256(orjson.dumps(version_input, option=orjson.OPT_SORT_KEYS)).hexdigest()[
        :16
    ]
    for entry in entries:
        entry["universe_version"] = version

    as_of_date = datetime.fromtimestamp(as_of_ms / 1_000, tz=UTC).date().isoformat()
    parquet_path = storage.path(
        "gold",
        "binance",
        "usdm",
        "universe",
        f"version={version}",
        f"as_of={as_of_date}",
        "universe.parquet",
    )
    manifest_path = parquet_path.with_suffix(".json")
    output = pl.DataFrame(entries)
    storage.write_parquet_atomic(parquet_path, output, compression=compression)
    storage.write_json_atomic(
        manifest_path,
        {
            **version_input,
            "universe_version": version,
            "included_symbols": included,
            "metadata_path": str(metadata_path),
        },
    )
    return UniverseBuildResult(
        parquet_path=str(parquet_path),
        manifest_path=str(manifest_path),
        version=version,
        included_symbols=tuple(included),
        metadata_path=str(metadata_path),
        as_of_ms=as_of_ms,
    )
