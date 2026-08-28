from datetime import date
from pathlib import Path

import pytest

from binance_algo.config import load_settings
from binance_algo.data.archive_client import ArchiveDownloader, ArchiveTarget, DownloadOutcome
from binance_algo.data.state_store import StateStore
from binance_algo.data.storage import LocalFilesystemStorage


@pytest.mark.network
async def test_official_daily_kline_archive(tmp_path: Path) -> None:
    settings = load_settings()
    state_store = StateStore(tmp_path / "ingestion.sqlite3")
    state_store.initialize()
    downloader = ArchiveDownloader(
        base_url=settings.archives.base_url,
        storage=LocalFilesystemStorage(tmp_path / "data"),
        state_store=state_store,
        request_timeout_seconds=settings.archives.request_timeout_seconds,
        max_concurrency=1,
        max_attempts=settings.archives.max_attempts,
        retry_base_seconds=settings.archives.retry_base_seconds,
        max_archive_bytes=settings.archives.max_archive_bytes,
        max_uncompressed_bytes=settings.archives.max_uncompressed_bytes,
        chunk_bytes=settings.archives.chunk_bytes,
    )
    result = (
        await downloader.download_many(
            [ArchiveTarget("BTCUSDT", "1m", date(2026, 8, 25))],
            ingestion_run_id="network-contract",
        )
    )[0]

    assert result.outcome is DownloadOutcome.DOWNLOADED
    assert result.row_count == 1_440
    assert result.checksum == "1651da32387a1342bdba15b28504dc4d55caee905a58fec04f52c280b1d69f7f"
