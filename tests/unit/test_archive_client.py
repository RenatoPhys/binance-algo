from __future__ import annotations

import hashlib
import io
import json
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from aiohttp import web

from binance_algo.common.errors import ArchiveError
from binance_algo.data.archive_client import (
    KLINE_HEADER,
    ArchiveDownloader,
    ArchiveDownloadResult,
    ArchiveTarget,
    DownloadOutcome,
    parse_checksum,
    validate_and_extract_kline_archive,
)
from binance_algo.data.backfill import (
    build_daily_kline_targets,
    new_backfill_job,
    write_download_report,
)
from binance_algo.data.state_store import StateStore
from binance_algo.data.storage import LocalFilesystemStorage


def kline_csv(*, has_header: bool = True) -> bytes:
    header = ",".join(KLINE_HEADER)
    rows = [
        "1787616000000,78952.90,78970.10,78906.20,78931.50,113.083,"
        "1787616059999,8927550.04670,3983,64.951,5127679.55880,0",
        "1787616060000,78931.50,78940.00,78910.00,78920.00,10.000,"
        "1787616119999,789200.00000,200,5.000,394600.00000,0",
    ]
    prefix = f"{header}\n" if has_header else ""
    return (prefix + "\n".join(rows) + "\n").encode()


def zip_bytes(member_name: str = "BTCUSDT-1m-2026-08-25.csv", *, has_header: bool = True) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, kline_csv(has_header=has_header))
    return output.getvalue()


@asynccontextmanager
async def archive_server(payload: bytes, stats: dict[str, Any]) -> AsyncIterator[str]:
    checksum = hashlib.sha256(payload).hexdigest()

    async def handler(request: web.Request) -> web.Response:
        if request.path.endswith(".CHECKSUM"):
            stats["checksum_hits"] = stats.get("checksum_hits", 0) + 1
            return web.Response(body=f"{checksum}  BTCUSDT-1m-2026-08-25.zip\n".encode())
        stats["archive_hits"] = stats.get("archive_hits", 0) + 1
        range_header = request.headers.get("Range")
        if range_header:
            offset = int(range_header.removeprefix("bytes=").removesuffix("-"))
            stats["range"] = range_header
            return web.Response(
                body=payload[offset:],
                status=206,
                headers={"Content-Range": f"bytes {offset}-{len(payload) - 1}/{len(payload)}"},
            )
        return web.Response(body=payload)

    app = web.Application()
    app.router.add_get("/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}/data"
    finally:
        await runner.cleanup()


def downloader(
    *, base_url: str, tmp_path: Path
) -> tuple[ArchiveDownloader, LocalFilesystemStorage, StateStore]:
    storage = LocalFilesystemStorage(tmp_path / "data")
    state_store = StateStore(tmp_path / "state.sqlite3")
    state_store.initialize()
    client = ArchiveDownloader(
        base_url=base_url,
        storage=storage,
        state_store=state_store,
        request_timeout_seconds=5,
        max_concurrency=2,
        max_attempts=2,
        retry_base_seconds=0,
        max_archive_bytes=1_000_000,
        max_uncompressed_bytes=1_000_000,
        chunk_bytes=65_536,
        jitter=lambda _ceiling: 0,
    )
    return client, storage, state_store


def target() -> ArchiveTarget:
    return ArchiveTarget(symbol="BTCUSDT", interval="1m", day=date(2026, 8, 25))


async def test_download_validates_extracts_and_skips_idempotently(tmp_path: Path) -> None:
    payload = zip_bytes()
    stats: dict[str, Any] = {}
    async with archive_server(payload, stats) as base_url:
        client, storage, state_store = downloader(base_url=base_url, tmp_path=tmp_path)
        first = (await client.download_many([target()], ingestion_run_id="run-1"))[0]
        hits_after_first = dict(stats)
        second = (await client.download_many([target()], ingestion_run_id="run-2"))[0]

    assert first.outcome is DownloadOutcome.DOWNLOADED
    assert first.row_count == 2
    assert first.checksum == hashlib.sha256(payload).hexdigest()
    assert Path(first.archive_path).exists()
    assert first.extracted_path is not None and Path(first.extracted_path).exists()
    assert second.outcome is DownloadOutcome.SKIPPED
    assert stats == hits_after_first
    assert state_store.data_file_counts() == {"VALIDATED": 1}
    assert storage.path("raw_archives").exists()


async def test_resumes_partial_archive_with_http_range(tmp_path: Path) -> None:
    payload = zip_bytes()
    stats: dict[str, Any] = {}
    async with archive_server(payload, stats) as base_url:
        client, storage, _state_store = downloader(base_url=base_url, tmp_path=tmp_path)
        archive_path = storage.path(
            "raw_archives",
            "binance",
            "usdm",
            "klines",
            "BTCUSDT",
            "1m",
            target().filename,
        )
        archive_path.parent.mkdir(parents=True)
        part_path = archive_path.with_name(f"{archive_path.name}.part")
        part_path.write_bytes(payload[:20])

        result = (await client.download_many([target()], ingestion_run_id="resume"))[0]

    assert result.outcome is DownloadOutcome.DOWNLOADED
    assert result.resumed is True
    assert stats["range"] == "bytes=20-"
    assert archive_path.read_bytes() == payload


async def test_checksum_mismatch_fails_and_is_manifested(tmp_path: Path) -> None:
    payload = zip_bytes()

    async def handler(request: web.Request) -> web.Response:
        if request.path.endswith(".CHECKSUM"):
            return web.Response(body=f"{'0' * 64}  {target().filename}\n".encode())
        return web.Response(body=payload)

    app = web.Application()
    app.router.add_get("/{tail:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    assert site._server is not None
    port = site._server.sockets[0].getsockname()[1]
    try:
        client, _storage, state_store = downloader(
            base_url=f"http://127.0.0.1:{port}/data", tmp_path=tmp_path
        )
        result = (await client.download_many([target()], ingestion_run_id="bad"))[0]
    finally:
        await runner.cleanup()

    assert result.outcome is DownloadOutcome.FAILED
    assert "checksum mismatch" in (result.error or "")
    assert state_store.data_file_counts() == {"FAILED": 1}


def test_rejects_zip_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    archive_path.write_bytes(zip_bytes("../BTCUSDT-1m-2026-08-25.csv"))

    with pytest.raises(ArchiveError, match="unsafe"):
        validate_and_extract_kline_archive(
            archive_path,
            tmp_path / "extracted.csv",
            expected_csv_filename="BTCUSDT-1m-2026-08-25.csv",
            max_uncompressed_bytes=1_000_000,
            chunk_bytes=65_536,
        )
    assert not (tmp_path.parent / "BTCUSDT-1m-2026-08-25.csv").exists()


def test_accepts_legacy_headerless_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / target().filename
    archive_path.write_bytes(zip_bytes(has_header=False))
    extracted_path = tmp_path / target().csv_filename

    rows = validate_and_extract_kline_archive(
        archive_path,
        extracted_path,
        expected_csv_filename=target().csv_filename,
        max_uncompressed_bytes=1_000_000,
        chunk_bytes=65_536,
    )

    assert rows == 2
    assert extracted_path.read_bytes() == kline_csv(has_header=False)


def test_checksum_and_target_planning_are_strict() -> None:
    checksum = "a" * 64
    assert (
        parse_checksum(
            f"{checksum}  BTCUSDT-1m-2026-08-25.zip\n",
            expected_filename="BTCUSDT-1m-2026-08-25.zip",
        )
        == checksum
    )
    with pytest.raises(ArchiveError, match="filename mismatch"):
        parse_checksum(
            f"{checksum}  ETHUSDT-1m-2026-08-25.zip",
            expected_filename="BTCUSDT-1m-2026-08-25.zip",
        )
    targets = build_daily_kline_targets(
        symbols=("BTCUSDT", "ETHUSDT"),
        interval="1m",
        start=date(2026, 8, 24),
        end=date(2026, 8, 25),
        publication_lag_days=1,
        today=date(2026, 8, 27),
    )
    assert len(targets) == 4
    with pytest.raises(ArchiveError, match="publication cutoff"):
        build_daily_kline_targets(
            symbols=("BTCUSDT",),
            interval="1m",
            start=date(2026, 8, 26),
            end=date(2026, 8, 27),
            publication_lag_days=1,
            today=date(2026, 8, 27),
        )


def test_writes_json_and_markdown_report(tmp_path: Path) -> None:
    job = new_backfill_job(
        symbols=("BTCUSDT",),
        interval="1m",
        start=date(2026, 8, 25),
        end=date(2026, 8, 25),
        total_files=1,
    )
    result = ArchiveDownloadResult(
        target=target(),
        outcome=DownloadOutcome.SKIPPED,
        archive_path="archive.zip",
        extracted_path="archive.csv",
        checksum="a" * 64,
        row_count=1_440,
        bytes_downloaded=0,
        duration_ms=1.5,
        resumed=False,
    )
    json_path, markdown_path = write_download_report(
        results=[result], job=job, reports_root=tmp_path / "reports"
    )

    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["summary"]["skipped"] == 1
    assert "| BTCUSDT | 2026-08-25 | skipped |" in markdown_path.read_text(encoding="utf-8")
