"""Backfill target planning, job records, and durable reports."""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import orjson

from binance_algo.common.errors import ArchiveError
from binance_algo.data.archive_client import (
    ArchiveDownloadResult,
    ArchiveTarget,
    DownloadOutcome,
)
from binance_algo.data.manifest import BackfillJobRecord, BackfillJobStatus, now_ms
from binance_algo.data.storage import LocalFilesystemStorage


def parse_symbols(value: str) -> tuple[str, ...]:
    symbols = tuple(part.strip().upper() for part in value.split(",") if part.strip())
    if not symbols:
        raise ArchiveError("--symbols must contain at least one symbol")
    if len(symbols) != len(set(symbols)):
        raise ArchiveError("--symbols must not contain duplicates")
    for symbol in symbols:
        ArchiveTarget(symbol=symbol, interval="1m", day=date(2000, 1, 1))
    return symbols


def parse_date(value: str, *, option: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ArchiveError(f"{option} must use YYYY-MM-DD, received {value!r}") from exc


def build_daily_kline_targets(
    *,
    symbols: tuple[str, ...],
    interval: str,
    start: date,
    end: date,
    publication_lag_days: int,
    today: date | None = None,
) -> list[ArchiveTarget]:
    if interval != "1m":
        raise ArchiveError("the current archive milestone supports only interval 1m")
    if end < start:
        raise ArchiveError("--end must be on or after --start")
    current_day = today or datetime.now(tz=UTC).date()
    latest_available_day = current_day - timedelta(days=publication_lag_days)
    if end > latest_available_day:
        raise ArchiveError(
            f"--end {end} is newer than the safe publication cutoff {latest_available_day}"
        )
    days = (end - start).days + 1
    return [
        ArchiveTarget(symbol=symbol, interval=interval, day=start + timedelta(days=offset))
        for symbol in symbols
        for offset in range(days)
    ]


def new_backfill_job(
    *,
    symbols: tuple[str, ...],
    interval: str,
    start: date,
    end: date,
    total_files: int,
) -> BackfillJobRecord:
    timestamp = now_ms()
    return BackfillJobRecord(
        job_id=uuid.uuid4().hex,
        dataset="klines",
        symbols_json=orjson.dumps(symbols).decode("utf-8"),
        interval=interval,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        status=BackfillJobStatus.PENDING,
        total_files=total_files,
        completed_files=0,
        failed_files=0,
        created_at_ms=timestamp,
        updated_at_ms=timestamp,
    )


def write_download_report(
    *,
    results: list[ArchiveDownloadResult],
    job: BackfillJobRecord,
    reports_root: Path,
) -> tuple[Path, Path]:
    storage = LocalFilesystemStorage(reports_root)
    base_name = f"downloads_{job.created_at_ms}_{job.job_id[:8]}"
    json_path = storage.path(f"{base_name}.json")
    markdown_path = storage.path(f"{base_name}.md")
    downloaded = sum(result.outcome is DownloadOutcome.DOWNLOADED for result in results)
    skipped = sum(result.outcome is DownloadOutcome.SKIPPED for result in results)
    failed = sum(result.outcome is DownloadOutcome.FAILED for result in results)
    total_bytes = sum(result.bytes_downloaded for result in results)
    payload = {
        "job": asdict(job),
        "summary": {
            "total": len(results),
            "downloaded": downloaded,
            "skipped": skipped,
            "failed": failed,
            "bytes_downloaded": total_bytes,
        },
        "files": [
            {
                **asdict(result),
                "target": {
                    "symbol": result.target.symbol,
                    "interval": result.target.interval,
                    "day": result.target.day.isoformat(),
                    "dataset": result.target.dataset,
                },
            }
            for result in results
        ],
    }
    storage.write_json_atomic(json_path, payload)

    lines = [
        "# Download report",
        "",
        f"- Job: `{job.job_id}`",
        f"- Range: `{job.start_date}` to `{job.end_date}` (inclusive UTC days)",
        f"- Total: {len(results)}",
        f"- Downloaded: {downloaded}",
        f"- Skipped: {skipped}",
        f"- Failed: {failed}",
        f"- Bytes downloaded: {total_bytes}",
        "",
        "| Symbol | Day | Outcome | Rows | Bytes | Resumed | Error |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for result in results:
        error = (result.error or "").replace("|", "\\|")
        lines.append(
            f"| {result.target.symbol} | {result.target.day} | {result.outcome.value} | "
            f"{result.row_count or ''} | {result.bytes_downloaded} | "
            f"{str(result.resumed).lower()} | {error} |"
        )
    storage.write_bytes_atomic(markdown_path, ("\n".join(lines) + "\n").encode("utf-8"))
    return json_path, markdown_path
