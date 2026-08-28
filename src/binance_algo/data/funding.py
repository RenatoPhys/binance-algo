"""Public funding-history ingestion with immutable raw and canonical artifacts."""

from __future__ import annotations

import hashlib
import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import duckdb
import orjson
import polars as pl

from binance_algo.common.errors import DataQualityError
from binance_algo.data.manifest import (
    DataFileRecord,
    DataFileStatus,
    deterministic_file_id,
    now_ms,
)
from binance_algo.data.state_store import StateStore
from binance_algo.data.storage import LocalFilesystemStorage
from binance_algo.exchange.binance_usdm.models import FundingRatePayload

FUNDING_SCHEMA_VERSION = 1
FUNDING_SCHEMA = {
    "logical_key": ["symbol", "funding_time_ms", "rate_type"],
    "schema_version": FUNDING_SCHEMA_VERSION,
    "timestamp_semantics": "funding_time_ms is the venue funding event time",
}


class FundingHistorySource(Protocol):
    async def funding_rate_history(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1_000,
    ) -> list[FundingRatePayload]: ...


@dataclass(frozen=True, slots=True)
class FundingSyncResult:
    symbol: str
    row_count: int
    start_time_ms: int
    end_time_ms: int
    raw_path: str
    parquet_path: str
    checksum: str
    skipped: bool


@dataclass(frozen=True, slots=True)
class FundingCatalogResult:
    database_path: str
    file_count: int
    row_count: int
    min_funding_time_ms: int | None
    max_funding_time_ms: int | None


async def _fetch_all(
    source: FundingHistorySource,
    *,
    symbol: str,
    start_time_ms: int,
    end_time_ms: int,
) -> list[FundingRatePayload]:
    cursor = start_time_ms
    events: list[FundingRatePayload] = []
    while cursor <= end_time_ms:
        page = await source.funding_rate_history(
            symbol=symbol,
            start_time_ms=cursor,
            end_time_ms=end_time_ms,
            limit=1_000,
        )
        if not page:
            break
        events.extend(page)
        last_time = max(item.funding_time_ms for item in page)
        if last_time < cursor:
            raise DataQualityError("funding pagination did not advance")
        if len(page) < 1_000 or last_time >= end_time_ms:
            break
        cursor = last_time + 1
    return events


def _canonicalize_events(
    events: list[FundingRatePayload],
    *,
    symbol: str,
    start_time_ms: int,
    end_time_ms: int,
) -> list[FundingRatePayload]:
    keyed: dict[tuple[str, int, str], FundingRatePayload] = {}
    for event in events:
        if event.symbol != symbol:
            raise DataQualityError(
                f"funding response symbol mismatch: expected {symbol}, received {event.symbol}"
            )
        if not start_time_ms <= event.funding_time_ms <= end_time_ms:
            raise DataQualityError(
                f"funding event outside requested range: {event.funding_time_ms}"
            )
        try:
            rate = float(event.funding_rate)
            mark = float(event.mark_price) if event.mark_price is not None else None
        except ValueError as exc:
            raise DataQualityError("funding rate or mark price is not numeric") from exc
        if not math.isfinite(rate) or abs(rate) > 1:
            raise DataQualityError(f"invalid funding rate for {symbol}: {event.funding_rate}")
        if mark is not None and (not math.isfinite(mark) or mark <= 0):
            raise DataQualityError(f"invalid funding mark price for {symbol}: {event.mark_price}")
        key = (event.symbol, event.funding_time_ms, event.rate_type)
        previous = keyed.get(key)
        if previous is not None and previous != event:
            raise DataQualityError(f"conflicting duplicate funding event: {key}")
        keyed[key] = event
    return [keyed[key] for key in sorted(keyed, key=lambda item: (item[1], item[2]))]


async def sync_funding_history(
    *,
    source: FundingHistorySource,
    storage: LocalFilesystemStorage,
    state_store: StateStore,
    symbols: tuple[str, ...],
    start_time_ms: int,
    end_time_ms: int,
    compression: str,
) -> list[FundingSyncResult]:
    """Fetch and persist a bounded public funding range for each configured symbol."""

    if end_time_ms < start_time_ms:
        raise DataQualityError("funding end precedes start")
    state_store.register_schema_version(
        "funding_rates",
        FUNDING_SCHEMA_VERSION,
        orjson.dumps(FUNDING_SCHEMA, option=orjson.OPT_SORT_KEYS).decode("utf-8"),
    )
    run_id = uuid.uuid4().hex
    results: list[FundingSyncResult] = []
    for symbol in symbols:
        events = _canonicalize_events(
            await _fetch_all(
                source,
                symbol=symbol,
                start_time_ms=start_time_ms,
                end_time_ms=end_time_ms,
            ),
            symbol=symbol,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
        )
        if not events:
            raise DataQualityError(f"no funding events returned for {symbol}")
        raw_rows = [event.model_dump(mode="json", by_alias=True) for event in events]
        raw_payload = orjson.dumps(raw_rows, option=orjson.OPT_SORT_KEYS) + b"\n"
        checksum = hashlib.sha256(raw_payload).hexdigest()
        identity = f"{start_time_ms}-{end_time_ms}-{checksum[:16]}"
        raw_path = storage.path(
            "raw",
            "binance",
            "usdm",
            "funding_rates",
            f"symbol={symbol}",
            f"funding-{identity}.json",
        )
        parquet_path = storage.path(
            "bronze",
            "binance",
            "usdm",
            "funding_rates",
            f"symbol={symbol}",
            f"funding-{identity}.parquet",
        )
        manifest_path = parquet_path.with_suffix(".json")
        skipped = raw_path.exists() and parquet_path.exists() and manifest_path.exists()
        if not raw_path.exists():
            storage.write_bytes_atomic(raw_path, raw_payload)
        if not parquet_path.exists():
            ingested_at_ns = time.time_ns()
            frame = pl.DataFrame(
                {
                    "symbol": [event.symbol for event in events],
                    "funding_rate_str": [event.funding_rate for event in events],
                    "funding_rate": [float(event.funding_rate) for event in events],
                    "funding_time_ms": [event.funding_time_ms for event in events],
                    "mark_price_str": [event.mark_price for event in events],
                    "mark_price": [
                        float(event.mark_price) if event.mark_price is not None else None
                        for event in events
                    ],
                    "rate_type": [event.rate_type for event in events],
                    "ingested_at_ns": [ingested_at_ns] * len(events),
                    "source": ["binance_public_rest"] * len(events),
                    "schema_version": [FUNDING_SCHEMA_VERSION] * len(events),
                }
            )
            storage.write_parquet_atomic(parquet_path, frame, compression=compression)
        if not manifest_path.exists():
            storage.write_json_atomic(
                manifest_path,
                {
                    "schema": FUNDING_SCHEMA,
                    "symbol": symbol,
                    "requested_start_time_ms": start_time_ms,
                    "requested_end_time_ms": end_time_ms,
                    "row_count": len(events),
                    "payload_sha256": checksum,
                    "raw_path": str(raw_path),
                    "source_endpoint": "/fapi/v1/fundingRate",
                },
            )
        timestamp = now_ms()
        file_id = deterministic_file_id(
            "binance",
            "usdm_futures",
            "funding_rates",
            symbol,
            str(start_time_ms),
            str(end_time_ms),
            checksum,
        )
        state_store.register_data_file(
            DataFileRecord(
                file_id=file_id,
                logical_dataset="funding_rates",
                layer="bronze",
                source="binance_public_rest",
                symbol=symbol,
                interval="event",
                start_time_ms=events[0].funding_time_ms,
                end_time_ms=events[-1].funding_time_ms,
                row_count=len(events),
                schema_version=FUNDING_SCHEMA_VERSION,
                checksum=checksum,
                path=str(parquet_path),
                status=DataFileStatus.NORMALIZED,
                created_at_ms=timestamp,
                updated_at_ms=timestamp,
                ingestion_run_id=run_id,
            )
        )
        results.append(
            FundingSyncResult(
                symbol=symbol,
                row_count=len(events),
                start_time_ms=events[0].funding_time_ms,
                end_time_ms=events[-1].funding_time_ms,
                raw_path=str(raw_path),
                parquet_path=str(parquet_path),
                checksum=checksum,
                skipped=skipped,
            )
        )
    return results


def rebuild_funding_catalog(
    *, database_path: Path, records: list[DataFileRecord]
) -> FundingCatalogResult:
    paths = sorted(
        {
            str(Path(record.path).resolve()).replace("\\", "/")
            for record in records
            if Path(record.path).exists()
        }
    )
    if not paths:
        raise DataQualityError("cannot catalog funding without normalized files")
    quoted = ", ".join("'" + path.replace("'", "''") + "'" for path in paths)
    database_path = database_path.resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with duckdb.connect(str(database_path)) as connection:
            connection.execute(
                "CREATE OR REPLACE VIEW funding_rates AS "
                "SELECT * EXCLUDE (revision_rank) FROM ("
                "SELECT *, ROW_NUMBER() OVER ("
                "PARTITION BY symbol, funding_time_ms, rate_type "
                "ORDER BY ingested_at_ns DESC"
                ") AS revision_rank "
                f"FROM read_parquet([{quoted}], union_by_name = true)"
                ") WHERE revision_rank = 1"
            )
            row = connection.execute(
                "SELECT COUNT(*)::BIGINT, MIN(funding_time_ms)::BIGINT, "
                "MAX(funding_time_ms)::BIGINT FROM funding_rates"
            ).fetchone()
    except duckdb.Error as exc:
        raise DataQualityError(f"cannot rebuild funding catalog: {exc}") from exc
    if row is None:
        raise DataQualityError("funding catalog verification returned no result")
    return FundingCatalogResult(
        database_path=str(database_path),
        file_count=len(paths),
        row_count=int(row[0]),
        min_funding_time_ms=int(row[1]) if row[1] is not None else None,
        max_funding_time_ms=int(row[2]) if row[2] is not None else None,
    )
