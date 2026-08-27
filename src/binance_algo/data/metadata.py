"""Immutable raw exchangeInfo and canonical Parquet metadata snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import polars as pl

from binance_algo.data.storage import LocalFilesystemStorage
from binance_algo.exchange.binance_usdm.models import parse_instruments


class ExchangeInfoSource(Protocol):
    async def exchange_info(self) -> dict[str, Any]: ...

    async def server_time(self) -> int: ...


@dataclass(frozen=True, slots=True)
class MetadataSnapshotResult:
    raw_path: str
    parquet_path: str
    row_count: int
    valid_from_ms: int


class MetadataSnapshotService:
    def __init__(
        self,
        *,
        source: ExchangeInfoSource,
        storage: LocalFilesystemStorage,
        compression: str,
    ) -> None:
        self._source = source
        self._storage = storage
        self._compression = compression

    async def snapshot(self) -> MetadataSnapshotResult:
        payload = await self._source.exchange_info()
        # Binance documents exchangeInfo.serverTime as ignorable for current time.
        # Use the dedicated endpoint so point-in-time lineage is not backdated by cached metadata.
        valid_from_ms = await self._source.server_time()
        instruments = parse_instruments(payload, valid_from_ms=valid_from_ms)
        snapshot_date = datetime.fromtimestamp(valid_from_ms / 1_000, tz=UTC).date().isoformat()

        raw_path = self._storage.path(
            "raw",
            "binance",
            "usdm",
            "exchange_info",
            f"date={snapshot_date}",
            f"exchange_info_{valid_from_ms}.json",
        )
        parquet_path = self._storage.path(
            "bronze",
            "binance",
            "usdm",
            "instrument_metadata",
            f"date={snapshot_date}",
            f"instrument_metadata_{valid_from_ms}.parquet",
        )

        records = [instrument.model_dump(mode="json") for instrument in instruments]
        frame = pl.DataFrame(records).sort("symbol")
        self._storage.write_json_atomic(raw_path, payload)
        self._storage.write_parquet_atomic(parquet_path, frame, compression=self._compression)
        return MetadataSnapshotResult(
            raw_path=str(raw_path),
            parquet_path=str(parquet_path),
            row_count=frame.height,
            valid_from_ms=valid_from_ms,
        )
