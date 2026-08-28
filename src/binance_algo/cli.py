"""Operational command-line interface for the first project milestone."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from binance_algo.common.errors import BinanceAlgoError, DataQualityError
from binance_algo.config import Settings, load_settings
from binance_algo.data.archive_client import (
    ArchiveDownloader,
    ArchiveDownloadResult,
    ArchiveTarget,
    DownloadOutcome,
)
from binance_algo.data.backfill import (
    build_daily_kline_targets,
    new_backfill_job,
    parse_date,
    parse_symbols,
    write_download_report,
)
from binance_algo.data.catalog import rebuild_kline_catalog
from binance_algo.data.manifest import (
    BackfillJobRecord,
    BackfillJobStatus,
    DataFileStatus,
    now_ms,
)
from binance_algo.data.metadata import MetadataSnapshotService
from binance_algo.data.normalize import (
    KlineNormalizer,
    NormalizeOutcome,
    write_normalization_report,
)
from binance_algo.data.quality import audit_kline_files, write_quality_report
from binance_algo.data.state_store import StateStore
from binance_algo.data.storage import LocalFilesystemStorage
from binance_algo.data.universe import (
    build_seed_universe,
    find_metadata_snapshot,
    metadata_valid_from,
    parse_as_of,
)
from binance_algo.doctor import run_doctor
from binance_algo.exchange.binance_usdm.rest import BinanceUSDMRestClient
from binance_algo.logging import configure_logging, get_logger

app = typer.Typer(help="Auditable Binance USD-M Futures public-data foundation.")
exchange_info_app = typer.Typer(help="Snapshot and inspect exchange metadata.")
universe_app = typer.Typer(help="Build point-in-time research universes.")
backfill_app = typer.Typer(help="Download official historical public archives.")
data_app = typer.Typer(help="Normalize, catalog, and audit canonical market data.")
app.add_typer(exchange_info_app, name="exchange-info")
app.add_typer(universe_app, name="universe")
app.add_typer(backfill_app, name="backfill")
app.add_typer(data_app, name="data")
console = Console()


@dataclass(frozen=True, slots=True)
class CLIContext:
    config_path: Path


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Base or overlay YAML configuration.",
            exists=True,
            dir_okay=False,
        ),
    ] = Path("configs/base.yaml"),
) -> None:
    ctx.obj = CLIContext(config_path=config)


def _settings(ctx: typer.Context) -> Settings:
    cli_context = ctx.ensure_object(CLIContext)
    return load_settings(cli_context.config_path)


def _configure(settings: Settings) -> None:
    configure_logging(
        level=settings.app.log_level,
        service="binance-algo",
        environment=settings.app.environment,
    )


def _fail(exc: BinanceAlgoError) -> NoReturn:
    console.print(f"[red]error:[/red] {exc}")
    raise typer.Exit(code=1) from exc


@app.command()
def doctor(ctx: typer.Context) -> None:
    """Validate runtime, storage, safety, DNS, REST, and clock offset."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        results = asyncio.run(run_doctor(settings))
    except BinanceAlgoError as exc:
        _fail(exc)

    table = Table(title="binance-algo doctor")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for result in results:
        status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
        table.add_row(result.name, status, result.detail)
    console.print(table)
    if not all(result.passed for result in results):
        raise typer.Exit(code=1)


async def _snapshot(settings: Settings) -> None:
    async with BinanceUSDMRestClient(
        base_url=settings.binance.rest_base_url,
        timeout_seconds=settings.binance.request_timeout_seconds,
        max_attempts=settings.binance.retry_max_attempts,
        retry_base_seconds=settings.binance.retry_base_seconds,
    ) as client:
        service = MetadataSnapshotService(
            source=client,
            storage=LocalFilesystemStorage(settings.data_root),
            compression=settings.storage.parquet_compression,
        )
        result = await service.snapshot()
        get_logger(operation="exchange_info_snapshot").info(
            "metadata_snapshot_complete",
            rows=result.row_count,
            valid_from_ms=result.valid_from_ms,
            parquet_path=result.parquet_path,
            rate_limits=dict(client.last_rate_limits),
        )
        console.print(
            f"Snapshot complete: {result.row_count} instruments\n"
            f"Raw: {result.raw_path}\nParquet: {result.parquet_path}"
        )


@exchange_info_app.command("snapshot")
def exchange_info_snapshot(ctx: typer.Context) -> None:
    """Persist raw exchangeInfo and a canonical immutable Parquet snapshot."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        asyncio.run(_snapshot(settings))
    except BinanceAlgoError as exc:
        _fail(exc)


@universe_app.command("build")
def universe_build(
    ctx: typer.Context,
    as_of: Annotated[
        str | None,
        typer.Option(help="Point-in-time cutoff as YYYY-MM-DD (inclusive UTC day)."),
    ] = None,
) -> None:
    """Build the configured seed universe from the latest valid metadata snapshot."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        search_cutoff_ms = parse_as_of(as_of)
        metadata_path = find_metadata_snapshot(settings.data_root, as_of_ms=search_cutoff_ms)
        as_of_ms = search_cutoff_ms if as_of is not None else metadata_valid_from(metadata_path)
        result = build_seed_universe(
            metadata_path=metadata_path,
            as_of_ms=as_of_ms,
            config=settings.universe,
            storage=LocalFilesystemStorage(settings.data_root),
            compression=settings.storage.parquet_compression,
        )
    except BinanceAlgoError as exc:
        _fail(exc)

    get_logger(operation="universe_build").info(
        "universe_build_complete",
        universe_version=result.version,
        symbols=list(result.included_symbols),
        as_of_ms=result.as_of_ms,
        parquet_path=result.parquet_path,
    )
    console.print(
        f"Universe {result.version}: {', '.join(result.included_symbols)}\n"
        f"Parquet: {result.parquet_path}\nManifest: {result.manifest_path}"
    )


async def _run_kline_backfill(
    settings: Settings,
    *,
    targets: list[ArchiveTarget],
    job: BackfillJobRecord,
    max_concurrency: int,
) -> tuple[list[ArchiveDownloadResult], BackfillJobRecord, Path, Path]:
    state_store = StateStore(settings.state_db_path)
    state_store.initialize()
    state_store.create_backfill_job(job)
    state_store.transition_backfill_job(job.job_id, BackfillJobStatus.RUNNING)
    downloader = ArchiveDownloader(
        base_url=settings.archives.base_url,
        storage=LocalFilesystemStorage(settings.data_root),
        state_store=state_store,
        request_timeout_seconds=settings.archives.request_timeout_seconds,
        max_concurrency=max_concurrency,
        max_attempts=settings.archives.max_attempts,
        retry_base_seconds=settings.archives.retry_base_seconds,
        max_archive_bytes=settings.archives.max_archive_bytes,
        max_uncompressed_bytes=settings.archives.max_uncompressed_bytes,
        chunk_bytes=settings.archives.chunk_bytes,
    )
    results = await downloader.download_many(targets, ingestion_run_id=job.job_id)
    failed = sum(result.outcome is DownloadOutcome.FAILED for result in results)
    completed = len(results) - failed
    final_status = BackfillJobStatus.FAILED if failed else BackfillJobStatus.COMPLETED
    error = f"{failed} archive(s) failed; inspect report" if failed else None
    state_store.transition_backfill_job(
        job.job_id,
        final_status,
        completed_files=completed,
        failed_files=failed,
        last_error=error,
    )
    final_job = replace(
        job,
        status=final_status,
        completed_files=completed,
        failed_files=failed,
        updated_at_ms=now_ms(),
        last_error=error,
    )
    json_report, markdown_report = write_download_report(
        results=results, job=final_job, reports_root=settings.reports_root
    )
    return results, final_job, json_report, markdown_report


@backfill_app.command("klines")
def backfill_klines(
    ctx: typer.Context,
    start: Annotated[str, typer.Option(help="Inclusive UTC start date as YYYY-MM-DD.")],
    end: Annotated[str, typer.Option(help="Inclusive UTC end date as YYYY-MM-DD.")],
    symbols: Annotated[
        str,
        typer.Option(help="Comma-separated USD-M symbols."),
    ] = "BTCUSDT,ETHUSDT,SOLUSDT",
    interval: Annotated[str, typer.Option(help="Kline interval; currently only 1m.")] = "1m",
    max_concurrency: Annotated[
        int | None,
        typer.Option(help="Override bounded download concurrency for this run."),
    ] = None,
) -> None:
    """Download, verify, safely extract, and manifest official daily kline archives."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        parsed_symbols = parse_symbols(symbols)
        start_date = parse_date(start, option="--start")
        end_date = parse_date(end, option="--end")
        targets = build_daily_kline_targets(
            symbols=parsed_symbols,
            interval=interval,
            start=start_date,
            end=end_date,
            publication_lag_days=settings.archives.publication_lag_days,
        )
        concurrency = (
            settings.archives.max_concurrency if max_concurrency is None else max_concurrency
        )
        if not 1 <= concurrency <= 32:
            raise typer.BadParameter("max concurrency must be between 1 and 32")
        job = new_backfill_job(
            symbols=parsed_symbols,
            interval=interval,
            start=start_date,
            end=end_date,
            total_files=len(targets),
        )
        results, final_job, json_report, markdown_report = asyncio.run(
            _run_kline_backfill(settings, targets=targets, job=job, max_concurrency=concurrency)
        )
    except BinanceAlgoError as exc:
        _fail(exc)

    table = Table(title=f"kline backfill {final_job.job_id[:8]}")
    table.add_column("symbol")
    table.add_column("day")
    table.add_column("outcome")
    table.add_column("rows", justify="right")
    table.add_column("bytes", justify="right")
    table.add_column("resumed")
    for result in results:
        table.add_row(
            result.target.symbol,
            result.target.day.isoformat(),
            result.outcome.value,
            str(result.row_count or ""),
            str(result.bytes_downloaded),
            str(result.resumed).lower(),
        )
    console.print(table)
    console.print(f"Reports: {json_report}\n         {markdown_report}")
    get_logger(operation="backfill_klines").info(
        "backfill_complete",
        job_id=final_job.job_id,
        status=final_job.status.value,
        completed_files=final_job.completed_files,
        failed_files=final_job.failed_files,
        json_report=str(json_report),
    )
    if final_job.status is BackfillJobStatus.FAILED:
        raise typer.Exit(code=1)


def _date_range_ms(start: str, end: str) -> tuple[int, int]:
    start_date = parse_date(start, option="--start")
    end_date = parse_date(end, option="--end")
    if end_date < start_date:
        raise typer.BadParameter("--end must be on or after --start")
    start_ms = int(
        datetime.combine(start_date, datetime.min.time(), tzinfo=UTC).timestamp() * 1_000
    )
    end_ms = (
        int(
            (
                datetime.combine(end_date, datetime.min.time(), tzinfo=UTC) + timedelta(days=1)
            ).timestamp()
            * 1_000
        )
        - 1
    )
    return start_ms, end_ms


@data_app.command("normalize")
def data_normalize(
    ctx: typer.Context,
    start: Annotated[str, typer.Option(help="Inclusive UTC start date as YYYY-MM-DD.")],
    end: Annotated[str, typer.Option(help="Inclusive UTC end date as YYYY-MM-DD.")],
    dataset: Annotated[
        str, typer.Option(help="Logical dataset; currently only klines.")
    ] = "klines",
    symbols: Annotated[str, typer.Option(help="Comma-separated USD-M symbols.")] = (
        "BTCUSDT,ETHUSDT,SOLUSDT"
    ),
    interval: Annotated[str, typer.Option(help="Kline interval; currently only 1m.")] = "1m",
) -> None:
    """Normalize validated archives to canonical immutable Parquet and refresh DuckDB."""

    try:
        if dataset != "klines":
            raise typer.BadParameter("the current normalization supports only dataset klines")
        settings = _settings(ctx)
        _configure(settings)
        parsed_symbols = parse_symbols(symbols)
        start_ms, end_ms = _date_range_ms(start, end)
        state_store = StateStore(settings.state_db_path)
        state_store.initialize()
        raw_records = state_store.list_data_files(
            logical_dataset="klines",
            layer="raw_archives",
            statuses={DataFileStatus.VALIDATED, DataFileStatus.NORMALIZED},
            symbols=parsed_symbols,
            interval=interval,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
        )
        if not raw_records:
            raise DataQualityError("no validated raw archives match the requested range")
        normalizer = KlineNormalizer(
            storage=LocalFilesystemStorage(settings.data_root),
            state_store=state_store,
            compression=settings.storage.parquet_compression,
            max_uncompressed_bytes=settings.archives.max_uncompressed_bytes,
            chunk_bytes=settings.archives.chunk_bytes,
        )
        results = normalizer.normalize_many(raw_records)
        json_report, markdown_report = write_normalization_report(
            results=results, reports_root=settings.reports_root
        )
        catalog_records = state_store.list_data_files(
            logical_dataset="klines",
            layer="bronze",
            statuses={DataFileStatus.NORMALIZED},
        )
        catalog = rebuild_kline_catalog(database_path=settings.duckdb_path, records=catalog_records)
    except BinanceAlgoError as exc:
        _fail(exc)

    failed = sum(result.outcome is NormalizeOutcome.FAILED for result in results)
    normalized = sum(result.outcome is NormalizeOutcome.NORMALIZED for result in results)
    skipped = sum(result.outcome is NormalizeOutcome.SKIPPED for result in results)
    console.print(
        f"Normalization: {normalized} normalized, {skipped} skipped, {failed} failed\n"
        f"DuckDB: {catalog.database_path} ({catalog.row_count} rows in {catalog.view_name})\n"
        f"Reports: {json_report}\n         {markdown_report}"
    )
    if failed:
        raise typer.Exit(code=1)


@data_app.command("audit")
def data_audit(
    ctx: typer.Context,
    start: Annotated[str, typer.Option(help="Inclusive UTC start date as YYYY-MM-DD.")],
    end: Annotated[str, typer.Option(help="Inclusive UTC end date as YYYY-MM-DD.")],
    dataset: Annotated[
        str, typer.Option(help="Logical dataset; currently only klines.")
    ] = "klines",
    symbols: Annotated[str, typer.Option(help="Comma-separated USD-M symbols.")] = (
        "BTCUSDT,ETHUSDT,SOLUSDT"
    ),
    interval: Annotated[str, typer.Option(help="Kline interval; currently only 1m.")] = "1m",
) -> None:
    """Audit schema, checksums, uniqueness, order, continuity, and market invariants."""

    try:
        if dataset != "klines":
            raise typer.BadParameter("the current audit supports only dataset klines")
        settings = _settings(ctx)
        _configure(settings)
        parsed_symbols = parse_symbols(symbols)
        start_ms, end_ms = _date_range_ms(start, end)
        state_store = StateStore(settings.state_db_path)
        state_store.initialize()
        records = state_store.list_data_files(
            logical_dataset="klines",
            layer="bronze",
            statuses={DataFileStatus.NORMALIZED},
            symbols=parsed_symbols,
            interval=interval,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
        )
        report = audit_kline_files(
            records,
            state_store=state_store,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            expected_symbols=parsed_symbols,
        )
        json_report, markdown_report = write_quality_report(
            report=report, reports_root=settings.reports_root
        )
    except BinanceAlgoError as exc:
        _fail(exc)

    table = Table(title=f"kline quality gate: {'PASS' if report.passed else 'FAIL'}")
    table.add_column("symbol")
    table.add_column("rows", justify="right")
    table.add_column("duplicates", justify="right")
    table.add_column("gaps", justify="right")
    table.add_column("invalid OHLC", justify="right")
    table.add_column("gate")
    for item in report.aggregate:
        table.add_row(
            item.symbol,
            str(item.row_count),
            str(item.duplicate_key_count),
            str(item.gap_count),
            str(item.invalid_ohlc_count),
            "PASS" if item.passed else "FAIL",
        )
    console.print(table)
    console.print(f"Reports: {json_report}\n         {markdown_report}")
    if not report.passed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
