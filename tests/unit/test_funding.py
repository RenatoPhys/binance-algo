from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from binance_algo.data.funding import rebuild_funding_catalog, sync_funding_history
from binance_algo.data.manifest import DataFileStatus
from binance_algo.data.state_store import StateStore
from binance_algo.data.storage import LocalFilesystemStorage
from binance_algo.exchange.binance_usdm.models import FundingRatePayload


class FixtureFundingSource:
    async def funding_rate_history(
        self,
        *,
        symbol: str,
        start_time_ms: int,
        end_time_ms: int,
        limit: int = 1_000,
    ) -> list[FundingRatePayload]:
        assert limit == 1_000
        return [
            FundingRatePayload(
                symbol=symbol,
                fundingRate="0.00010000",
                fundingTime=start_time_ms + 10,
                markPrice="",
                rateType="Regular",
            ),
            FundingRatePayload(
                symbol=symbol,
                fundingRate="-0.00005000",
                fundingTime=end_time_ms - 10,
                markPrice="51000.0",
                rateType="Regular",
            ),
        ]


async def test_funding_sync_is_immutable_idempotent_and_cataloged(tmp_path: Path) -> None:
    storage = LocalFilesystemStorage(tmp_path / "data")
    state_store = StateStore(tmp_path / "state.sqlite3")
    state_store.initialize()
    arguments = {
        "source": FixtureFundingSource(),
        "storage": storage,
        "state_store": state_store,
        "symbols": ("BTCUSDT",),
        "start_time_ms": 100,
        "end_time_ms": 1_000,
        "compression": "zstd",
    }
    first = await sync_funding_history(**arguments)
    second = await sync_funding_history(**arguments)

    assert first[0].row_count == 2
    assert first[0].skipped is False
    assert second[0].skipped is True
    assert first[0].checksum == second[0].checksum
    records = state_store.list_data_files(
        logical_dataset="funding_rates",
        statuses={DataFileStatus.NORMALIZED},
    )
    catalog = rebuild_funding_catalog(database_path=tmp_path / "catalog.duckdb", records=records)
    assert catalog.row_count == 2
    with duckdb.connect(catalog.database_path, read_only=True) as connection:
        total = connection.execute("SELECT SUM(funding_rate) FROM funding_rates").fetchone()
        legacy = connection.execute(
            "SELECT mark_price_str, mark_price FROM funding_rates ORDER BY funding_time_ms LIMIT 1"
        ).fetchone()
        assert total is not None
        assert total[0] == pytest.approx(0.00005)
        assert legacy == ("", None)
    assert '"markPrice":""' in Path(first[0].raw_path).read_text(encoding="utf-8")
