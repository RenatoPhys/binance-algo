"""Operational command-line interface for the first project milestone."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, NoReturn

import orjson
import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from binance_algo.clock import ExchangeClock
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
from binance_algo.data.funding import (
    FundingCatalogResult,
    FundingSyncResult,
    rebuild_funding_catalog,
    sync_funding_history,
)
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
from binance_algo.data.recorder import RecorderRunResult, run_recorder
from binance_algo.data.replay import (
    RealReplayClock,
    ReplayEngine,
    VirtualReplayClock,
    parse_replay_datasets,
    select_replay_files,
)
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
from binance_algo.observability.metrics import RecorderMetrics
from binance_algo.research.baseline import run_and_persist_phase3_baseline
from binance_algo.research.dataset import build_and_persist_research_dataset

app = typer.Typer(help="Auditable Binance USD-M Futures public-data foundation.")
exchange_info_app = typer.Typer(help="Snapshot and inspect exchange metadata.")
universe_app = typer.Typer(help="Build point-in-time research universes.")
backfill_app = typer.Typer(help="Download official historical public archives.")
data_app = typer.Typer(help="Normalize, catalog, and audit canonical market data.")
recorder_app = typer.Typer(help="Record resilient public market streams to Parquet.")
funding_app = typer.Typer(help="Ingest public historical funding events.")
research_app = typer.Typer(help="Build causal datasets and run the Phase 3 baseline.")
app.add_typer(exchange_info_app, name="exchange-info")
app.add_typer(universe_app, name="universe")
app.add_typer(backfill_app, name="backfill")
app.add_typer(data_app, name="data")
app.add_typer(recorder_app, name="recorder")
app.add_typer(funding_app, name="funding")
app.add_typer(research_app, name="research")
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


async def _sync_funding_range(
    settings: Settings,
    *,
    symbols: tuple[str, ...],
    start_time_ms: int,
    end_time_ms: int,
) -> tuple[list[FundingSyncResult], FundingCatalogResult]:
    state_store = StateStore(settings.state_db_path)
    state_store.initialize()
    async with BinanceUSDMRestClient(
        base_url=settings.binance.market_data_rest_base_url,
        timeout_seconds=settings.binance.request_timeout_seconds,
        max_attempts=settings.binance.retry_max_attempts,
        retry_base_seconds=settings.binance.retry_base_seconds,
    ) as client:
        results = await sync_funding_history(
            source=client,
            storage=LocalFilesystemStorage(settings.data_root),
            state_store=state_store,
            symbols=symbols,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            compression=settings.storage.parquet_compression,
        )
    records = state_store.list_data_files(
        logical_dataset="funding_rates",
        layer="bronze",
        statuses={DataFileStatus.NORMALIZED},
        symbols=symbols,
    )
    catalog = rebuild_funding_catalog(database_path=settings.duckdb_path, records=records)
    return results, catalog


@funding_app.command("sync")
def funding_sync(
    ctx: typer.Context,
    start: Annotated[str, typer.Option(help="Inclusive UTC start date as YYYY-MM-DD.")],
    end: Annotated[str, typer.Option(help="Inclusive UTC end date as YYYY-MM-DD.")],
    symbols: Annotated[str, typer.Option(help="Comma-separated USD-M symbols.")] = (
        "BTCUSDT,ETHUSDT,SOLUSDT"
    ),
) -> None:
    """Persist public funding history and rebuild its deduplicated DuckDB view."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        parsed_symbols = parse_symbols(symbols)
        start_ms, end_ms = _date_range_ms(start, end)
        results, catalog = asyncio.run(
            _sync_funding_range(
                settings,
                symbols=parsed_symbols,
                start_time_ms=start_ms,
                end_time_ms=end_ms,
            )
        )
    except BinanceAlgoError as exc:
        _fail(exc)
    table = Table(title="funding history: PASS")
    table.add_column("symbol")
    table.add_column("rows", justify="right")
    table.add_column("outcome")
    for result in results:
        table.add_row(
            result.symbol,
            str(result.row_count),
            "skipped" if result.skipped else "persisted",
        )
    console.print(table)
    console.print(f"DuckDB: {catalog.database_path} ({catalog.row_count} deduplicated events)")


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


@research_app.command("build")
def research_build(
    ctx: typer.Context,
    start: Annotated[str, typer.Option(help="Inclusive UTC input start as YYYY-MM-DD.")],
    end: Annotated[str, typer.Option(help="Inclusive UTC input end as YYYY-MM-DD.")],
    symbols: Annotated[str, typer.Option(help="Comma-separated fixed seed symbols.")] = (
        "BTCUSDT,ETHUSDT,SOLUSDT"
    ),
) -> None:
    """Build and audit the causal point-in-time research dataset."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        parsed_symbols = parse_symbols(symbols)
        start_ms, end_ms = _date_range_ms(start, end)
        result = build_and_persist_research_dataset(
            database_path=settings.duckdb_path,
            state_db_path=settings.state_db_path,
            storage=LocalFilesystemStorage(settings.data_root),
            symbols=parsed_symbols,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            config=settings.research,
            compression=settings.storage.parquet_compression,
        )
    except BinanceAlgoError as exc:
        _fail(exc)
    audit = result.audit
    console.print(
        f"Point-in-time dataset PASS: {audit.row_count} rows, "
        f"{audit.decision_count} decisions\n"
        f"Dataset version: {result.dataset_version}\n"
        f"Universe / features: {result.universe_version} / {result.feature_version}\n"
        f"Parquet: {result.parquet_path}\nManifest: {result.manifest_path}"
    )


def _latest_research_dataset(data_root: Path) -> Path:
    candidates = sorted(
        data_root.glob("gold/binance/usdm/research_dataset/version=*/dataset.parquet"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not candidates:
        raise DataQualityError(
            "no research dataset exists; run `binance-algo research build` first"
        )
    return candidates[-1]


@research_app.command("backtest")
def research_backtest(
    ctx: typer.Context,
    dataset: Annotated[
        Path | None,
        typer.Option(help="Point-in-time dataset Parquet; defaults to the latest local version."),
    ] = None,
    chart: Annotated[
        bool,
        typer.Option("--chart", help="Generate the P&L SVG for this run (disabled by default)."),
    ] = False,
) -> None:
    """Run expanding walk-forward research and all Phase 3 stability scenarios."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        dataset_path = (
            dataset.resolve()
            if dataset is not None
            else _latest_research_dataset(settings.data_root)
        )
        if not dataset_path.is_file():
            raise DataQualityError(f"research dataset does not exist: {dataset_path}")
        result = run_and_persist_phase3_baseline(
            dataset_path=dataset_path,
            storage=LocalFilesystemStorage(settings.data_root),
            reports_root=settings.reports_root,
            compression=settings.storage.parquet_compression,
            config=settings.research,
            generate_chart=chart,
        )
    except (BinanceAlgoError, OSError, pl.exceptions.PolarsError) as exc:
        if isinstance(exc, BinanceAlgoError):
            _fail(exc)
        _fail(DataQualityError(str(exc)))
    metrics = result.metrics
    table = Table(title="Phase 3 walk-forward baseline")
    table.add_column("metric")
    table.add_column("out-of-sample", justify="right")
    for name, value in (
        ("folds", str(result.fold_count)),
        ("periods", str(metrics.periods)),
        ("total return", f"{metrics.total_return:.4%}"),
        ("Sharpe", f"{metrics.sharpe:.3f}"),
        ("max drawdown", f"{metrics.max_drawdown:.4%}"),
        ("turnover", f"{metrics.turnover:.3f}"),
        ("funding P&L", f"{metrics.funding_pnl:.6f}"),
        ("accounting error", f"{metrics.accounting_error_max:.3e}"),
    ):
        table.add_row(name, value)
    console.print(table)
    output = (
        "Baseline only; no claim of edge.\n"
        f"Run version: {result.run_version}\n"
        f"Curve: {result.curve_path}\n"
        f"Reports: {result.report_json_path}\n         {result.report_markdown_path}"
    )
    if result.report_chart_path is not None:
        output += f"\nChart: {result.report_chart_path}"
    console.print(output)


async def _run_recorder_with_clock(
    settings: Settings,
    *,
    symbols: tuple[str, ...],
    duration_seconds: float,
    metrics_port: int | None,
) -> RecorderRunResult:
    metrics = RecorderMetrics(queue_capacity=settings.recorder.queue_capacity)
    async with BinanceUSDMRestClient(
        base_url=settings.binance.rest_base_url,
        timeout_seconds=settings.binance.request_timeout_seconds,
        max_attempts=settings.binance.retry_max_attempts,
        retry_base_seconds=settings.binance.retry_base_seconds,
        metrics=metrics,
    ) as client:
        clock = ExchangeClock()
        sync = await clock.synchronize(client)
    if abs(sync.offset_ms) > settings.binance.clock_max_offset_ms:
        raise DataQualityError(
            f"clock offset {sync.offset_ms}ms exceeds {settings.binance.clock_max_offset_ms}ms"
        )
    return await run_recorder(
        settings,
        symbols=symbols,
        duration_seconds=duration_seconds,
        clock_offset_ms=sync.offset_ms,
        metrics_port=metrics_port,
        metrics=metrics,
    )


@recorder_app.command("start")
def recorder_start(
    ctx: typer.Context,
    symbols: Annotated[str, typer.Option(help="Comma-separated USD-M symbols.")] = (
        "BTCUSDT,ETHUSDT,SOLUSDT"
    ),
    duration_seconds: Annotated[
        float,
        typer.Option(help="Recording duration; use 3600 for the Phase 2 acceptance run."),
    ] = 3_600.0,
    metrics_port: Annotated[
        int | None,
        typer.Option(help="Override the local metrics/health port; 0 selects an ephemeral port."),
    ] = None,
) -> None:
    """Record configured public streams with lossless bounded buffering."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        parsed_symbols = parse_symbols(symbols)
        if duration_seconds <= 0:
            raise typer.BadParameter("duration must be positive")
        if metrics_port is not None and not 0 <= metrics_port <= 65_535:
            raise typer.BadParameter("metrics port must be between 0 and 65535")
        result = asyncio.run(
            _run_recorder_with_clock(
                settings,
                symbols=parsed_symbols,
                duration_seconds=duration_seconds,
                metrics_port=metrics_port,
            )
        )
    except BinanceAlgoError as exc:
        _fail(exc)

    table = Table(title=f"recorder quality gate: {'PASS' if result.report.passed else 'FAIL'}")
    table.add_column("stream")
    table.add_column("messages", justify="right")
    table.add_column("rows", justify="right")
    table.add_column("duplicates", justify="right")
    table.add_column("gaps", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("gate")
    for dataset in result.report.datasets:
        table.add_row(
            dataset.event_type,
            str(dataset.messages_received),
            str(dataset.rows_persisted),
            str(dataset.duplicate_event_count),
            str(dataset.sequence_gap_count),
            "" if dataset.latency_p95_ms is None else f"{dataset.latency_p95_ms:.2f}",
            "PASS" if dataset.passed else "FAIL",
        )
    console.print(table)
    console.print(
        f"Run: {result.report.run_id}\n"
        f"Dropped: {result.report.dropped_events_total}; "
        f"reconnects: {result.report.reconnects_total}\n"
        f"DuckDB: {settings.duckdb_path}\n"
        f"Reports: {result.json_report}\n         {result.markdown_report}"
    )


@recorder_app.command("status")
def recorder_status(ctx: typer.Context) -> None:
    """Show the latest durable recorder report and checkpoint count."""

    try:
        settings = _settings(ctx)
        candidates = sorted(
            settings.reports_root.glob("recorder_*.json"), key=lambda path: path.stat().st_mtime_ns
        )
        if not candidates:
            raise DataQualityError("no recorder report exists yet")
        latest = candidates[-1]
        payload = orjson.loads(latest.read_bytes())
        if not isinstance(payload, dict):
            raise DataQualityError(f"invalid recorder report: {latest}")
        state_store = StateStore(settings.state_db_path)
        state_store.initialize()
        checkpoints = state_store.list_stream_checkpoints()
    except (BinanceAlgoError, OSError, orjson.JSONDecodeError) as exc:
        if isinstance(exc, BinanceAlgoError):
            _fail(exc)
        _fail(DataQualityError(str(exc)))

    console.print(
        f"Latest report: {latest}\n"
        f"Run: {payload.get('run_id')}\n"
        f"Gate: {'PASS' if payload.get('passed') else 'FAIL'}\n"
        f"Messages/rows: {payload.get('messages_received_total')} / "
        f"{payload.get('rows_persisted_total')}\n"
        f"Dropped/reconnects: {payload.get('dropped_events_total')} / "
        f"{payload.get('reconnects_total')}\n"
        f"Checkpoints: {len(checkpoints)}"
    )


def _parse_replay_time(value: str, *, option: str) -> int:
    if value.isdigit():
        return int(value)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DataQualityError(
            f"{option} must be epoch milliseconds or ISO-8601 with timezone"
        ) from exc
    if parsed.tzinfo is None:
        raise DataQualityError(f"{option} must include a timezone, preferably Z")
    return int(parsed.timestamp() * 1_000)


@app.command("replay")
def replay(
    ctx: typer.Context,
    start: Annotated[str, typer.Option(help="Inclusive epoch milliseconds or ISO-8601 timestamp.")],
    end: Annotated[str, typer.Option(help="Inclusive epoch milliseconds or ISO-8601 timestamp.")],
    dataset: Annotated[
        str,
        typer.Option(help="One or more recorder datasets, comma-separated, or all."),
    ] = "all",
    symbols: Annotated[str, typer.Option(help="Comma-separated USD-M symbols.")] = (
        "BTCUSDT,ETHUSDT,SOLUSDT"
    ),
    speed: Annotated[float, typer.Option(help="Temporal replay speed multiplier.")] = 100.0,
    wall_clock: Annotated[
        bool,
        typer.Option(help="Actually wait between events; default uses an injected virtual clock."),
    ] = False,
) -> None:
    """Replay recorded events in deterministic temporal order without network access."""

    try:
        settings = _settings(ctx)
        datasets = parse_replay_datasets(dataset)
        parsed_symbols = parse_symbols(symbols)
        start_ms = _parse_replay_time(start, option="--start")
        end_ms = _parse_replay_time(end, option="--end")
        state_store = StateStore(settings.state_db_path)
        state_store.initialize()
        records = select_replay_files(
            state_store,
            datasets=datasets,
            symbols=parsed_symbols,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
        )
        engine = ReplayEngine(records=records, start_time_ms=start_ms, end_time_ms=end_ms)
        deterministic = engine.verify_determinism()
        clock = RealReplayClock() if wall_clock else VirtualReplayClock()
        replayed = asyncio.run(engine.run(speed=speed, clock=clock))
        if replayed.digest != deterministic.digest:
            raise DataQualityError("scheduled replay digest diverged from deterministic scan")
    except BinanceAlgoError as exc:
        _fail(exc)

    virtual_elapsed = clock.elapsed_seconds if isinstance(clock, VirtualReplayClock) else None
    console.print(
        f"Replay PASS: {replayed.event_count} events\n"
        f"Digest: {replayed.digest}\n"
        f"Range: {replayed.first_event_time_ms}..{replayed.last_event_time_ms}\n"
        f"Speed: {speed}x; clock: {'wall' if wall_clock else 'virtual'}"
        + (f"; virtual elapsed: {virtual_elapsed:.3f}s" if virtual_elapsed is not None else "")
    )


if __name__ == "__main__":
    app()
