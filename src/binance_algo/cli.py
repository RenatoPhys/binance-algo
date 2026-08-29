"""Operational command-line interface for the first project milestone."""

from __future__ import annotations

import asyncio
import webbrowser
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, NoReturn

import orjson
import polars as pl
import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from binance_algo.clock import ExchangeClock
from binance_algo.common.errors import BinanceAlgoError, DataQualityError, ResearchError
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
from binance_algo.research.alpha_reboot_report import build_alpha_reboot_wave1_report
from binance_algo.research.dashboard import build_research_dashboard
from binance_algo.research.dataset import build_and_persist_research_dataset
from binance_algo.research.datasets.references import load_dataset_reference
from binance_algo.research.experiments.ablation import AblationRunner
from binance_algo.research.experiments.campaign import (
    CampaignPlan,
    campaign_spec_from_stored_payload,
    load_campaign_spec,
    plan_campaign,
)
from binance_algo.research.experiments.campaign_runner import CampaignRunner
from binance_algo.research.experiments.compare import write_campaign_comparison
from binance_algo.research.experiments.ledger import (
    write_feature_history_report,
    write_hypothesis_history_report,
)
from binance_algo.research.experiments.models import FeatureEvaluationSpec, HypothesisSpec
from binance_algo.research.experiments.promotion import CandidateAssessment, PromotionManager
from binance_algo.research.experiments.provenance import build_code_fingerprint
from binance_algo.research.experiments.registry import sync_builtin_registry
from binance_algo.research.experiments.runner import (
    ExperimentRunner,
    build_phase3_experiment_spec,
    phase3_baseline_hypothesis,
)
from binance_algo.research.experiments.store import ResearchStore
from binance_algo.research.strategy_portfolio.inventory import (
    write_strategy_portfolio_inventory,
)
from binance_algo.research.strategy_portfolio.scaffold import write_scaffold
from binance_algo.research.strategy_portfolio.validation import (
    validate_portfolio_declarations,
)
from binance_algo.research.validation.robustness import build_campaign_robustness

app = typer.Typer(help="Auditable Binance USD-M Futures public-data foundation.")
exchange_info_app = typer.Typer(help="Snapshot and inspect exchange metadata.")
universe_app = typer.Typer(help="Build point-in-time research universes.")
backfill_app = typer.Typer(help="Download official historical public archives.")
data_app = typer.Typer(help="Normalize, catalog, and audit canonical market data.")
recorder_app = typer.Typer(help="Record resilient public market streams to Parquet.")
funding_app = typer.Typer(help="Ingest public historical funding events.")
research_app = typer.Typer(help="Build causal datasets and manage reproducible research.")
research_registry_app = typer.Typer(help="Initialize and inspect the research registry.")
research_hypothesis_app = typer.Typer(help="Register and inspect research hypotheses.")
research_feature_app = typer.Typer(help="Inspect persisted feature definitions.")
research_experiment_app = typer.Typer(help="Run and verify immutable research experiments.")
research_campaign_app = typer.Typer(help="Plan, run, resume, and compare research campaigns.")
research_ablation_app = typer.Typer(help="Evaluate contextual feature ablations.")
research_promote_app = typer.Typer(help="Apply auditable research promotion gates.")
research_candidate_app = typer.Typer(help="Generate candidate assessments in campaign context.")
research_dashboard_app = typer.Typer(help="Build the offline research dashboard.")
research_portfolio_app = typer.Typer(help="Validate declared portfolios of research strategies.")
app.add_typer(exchange_info_app, name="exchange-info")
app.add_typer(universe_app, name="universe")
app.add_typer(backfill_app, name="backfill")
app.add_typer(data_app, name="data")
app.add_typer(recorder_app, name="recorder")
app.add_typer(funding_app, name="funding")
app.add_typer(research_app, name="research")
research_app.add_typer(research_registry_app, name="registry")
research_app.add_typer(research_hypothesis_app, name="hypothesis")
research_app.add_typer(research_feature_app, name="feature")
research_app.add_typer(research_experiment_app, name="experiment")
research_app.add_typer(research_campaign_app, name="campaign")
research_app.add_typer(research_ablation_app, name="ablation")
research_app.add_typer(research_promote_app, name="promote")
research_app.add_typer(research_candidate_app, name="candidate")
research_app.add_typer(research_dashboard_app, name="dashboard")
research_app.add_typer(research_portfolio_app, name="portfolio")
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


def _hypothesis_file(path: Path) -> HypothesisSpec:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ResearchError("hypothesis file root must be a mapping")
        return HypothesisSpec.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ResearchError(f"cannot load hypothesis file {path}: {exc}") from exc


def _experiment_runner(settings: Settings, store: ResearchStore) -> ExperimentRunner:
    return ExperimentRunner(
        store=store,
        data_root=settings.data_root,
        research_config=settings.research,
        compression=settings.storage.parquet_compression,
        heartbeat_seconds=settings.research_platform.heartbeat_seconds,
    )


def _campaign_runner(settings: Settings, store: ResearchStore) -> CampaignRunner:
    return CampaignRunner(
        store=store,
        data_root=settings.data_root,
        reports_root=settings.reports_root,
        research_config=settings.research,
        compression=settings.storage.parquet_compression,
        heartbeat_seconds=settings.research_platform.heartbeat_seconds,
    )


def _promotion_manager(settings: Settings, store: ResearchStore) -> PromotionManager:
    return PromotionManager(
        store=store,
        experiment_runner=_experiment_runner(settings, store),
        data_root=settings.data_root,
        reports_root=settings.reports_root,
        platform=settings.research_platform,
        current_code_fingerprint=build_code_fingerprint(settings.project_root),
    )


@research_dashboard_app.command("build")
def research_dashboard_build(
    ctx: typer.Context,
    open_browser: Annotated[
        bool,
        typer.Option("--open", help="Open the generated dashboard in the default browser."),
    ] = False,
    portfolio_file: Annotated[
        Path | None,
        typer.Option(
            "--portfolio-file",
            help="Optional strict YAML declaration of strategy portfolios.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
) -> None:
    """Build deterministic JSON and a self-contained offline HTML dashboard."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        result = build_research_dashboard(
            store=ResearchStore(settings.research_db_path),
            reports_root=settings.reports_root,
            data_root=settings.data_root,
            portfolio_file=portfolio_file,
        )
    except BinanceAlgoError as exc:
        _fail(exc)
    if open_browser:
        webbrowser.open(result.index_path.resolve().as_uri())
    console.print(
        f"Research dashboard built:\nHTML: {result.index_path}\nSnapshot: {result.snapshot_path}"
    )


@research_portfolio_app.command("validate")
def research_portfolio_validate(
    ctx: typer.Context,
    portfolio_file: Annotated[
        Path,
        typer.Option(
            "--file",
            help="Strict YAML declaration of strategy portfolios.",
            exists=True,
            dir_okay=False,
        ),
    ],
) -> None:
    """Validate portfolio schema, runs, artifacts, alignment, and compatibility."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        rows = validate_portfolio_declarations(
            store=ResearchStore(settings.research_db_path),
            data_root=settings.data_root,
            portfolio_file=portfolio_file,
        )
    except BinanceAlgoError as exc:
        _fail(exc)
    table = Table(title="research strategy portfolio validation")
    for column in (
        "portfolio",
        "component",
        "experiment / run",
        "artifacts",
        "compatibility group",
        "window",
        "weight",
        "accounting",
        "error / warning",
    ):
        table.add_column(column)
    for row in rows:
        table.add_row(
            row.portfolio_id,
            row.component_label,
            f"{row.experiment_id}\n{row.run_id or '—'}",
            row.artifact_status,
            row.compatibility_group or "—",
            row.window,
            str(row.capital_weight),
            row.accounting_mode,
            row.message or "—",
        )
    console.print(table)
    if not rows or any(not row.valid for row in rows):
        raise typer.Exit(code=1)


@research_portfolio_app.command("scaffold")
def research_portfolio_scaffold(
    ctx: typer.Context,
    experiment_ids: Annotated[
        list[str],
        typer.Option(
            "--experiment-id",
            help="Exact registry experiment ID; repeat for each declared component.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Destination YAML file.", dir_okay=False),
    ] = Path("var/config/research_strategy_portfolios.yaml"),
) -> None:
    """Write equal weights for only the experiment IDs explicitly supplied."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        identifiers = tuple(experiment_ids)
        store = ResearchStore(settings.research_db_path)
        unknown = [
            identifier for identifier in identifiers if store.get_experiment(identifier) is None
        ]
        if unknown:
            raise ResearchError(f"unknown experiment IDs: {unknown}")
        path = write_scaffold(output, identifiers)
    except BinanceAlgoError as exc:
        _fail(exc)
    console.print(f"Strategy portfolio scaffold written: {path}")


@research_portfolio_app.command("inventory")
def research_portfolio_inventory(ctx: typer.Context) -> None:
    """Audit local successful runs and write the non-versioned portfolio inventory."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        json_path, markdown_path, inventory = write_strategy_portfolio_inventory(
            store=ResearchStore(settings.research_db_path),
            data_root=settings.data_root,
            reports_root=settings.reports_root,
        )
    except BinanceAlgoError as exc:
        _fail(exc)
    console.print(
        "Research strategy portfolio inventory written:\n"
        f"JSON: {json_path}\nMarkdown: {markdown_path}\n"
        f"Verified: {inventory['verified_experiments']} / "
        f"{inventory['successful_experiments']}"
    )


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


@research_registry_app.command("init")
def research_registry_init(ctx: typer.Context) -> None:
    """Create the research database and register built-in feature definitions."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        store = ResearchStore(settings.research_db_path)
        schema_version = store.initialize()
        sync = sync_builtin_registry(store, research_config=settings.research)
    except BinanceAlgoError as exc:
        _fail(exc)
    console.print(
        f"Research registry initialized: schema {schema_version}, "
        f"{sync.feature_count} features, feature set {sync.feature_set_id}\n"
        f"SQLite: {settings.research_db_path}"
    )


@research_registry_app.command("migrate")
def research_registry_migrate(ctx: typer.Context) -> None:
    """Apply pending versioned research migrations transactionally."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        schema_version = ResearchStore(settings.research_db_path).initialize()
    except BinanceAlgoError as exc:
        _fail(exc)
    console.print(f"Research registry migrated to schema {schema_version}")


@research_registry_app.command("status")
def research_registry_status(ctx: typer.Context) -> None:
    """Show schema, WAL/foreign-key state, and durable registry counts."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        if not settings.research_db_path.exists():
            raise ResearchError(
                "research registry is not initialized; run `binance-algo research registry init`"
            )
        status = ResearchStore(settings.research_db_path).status()
    except BinanceAlgoError as exc:
        _fail(exc)
    table = Table(title="research registry")
    table.add_column("entity")
    table.add_column("count", justify="right")
    for entity, count in status.counts.items():
        table.add_row(entity.removeprefix("research_"), str(count))
    console.print(table)
    console.print(
        f"Schema: {status.schema_version}/{status.latest_schema_version}; "
        f"journal={status.journal_mode}; foreign_keys={'on' if status.foreign_keys else 'off'}\n"
        f"SQLite: {status.database_path}"
    )


@research_hypothesis_app.command("create")
def research_hypothesis_create(
    ctx: typer.Context,
    file: Annotated[
        Path,
        typer.Option("--file", help="Strict hypothesis YAML file.", exists=True, dir_okay=False),
    ],
) -> None:
    """Register one immutable hypothesis definition idempotently."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        store = ResearchStore(settings.research_db_path)
        store.initialize()
        registered = store.register_hypothesis(_hypothesis_file(file))
    except BinanceAlgoError as exc:
        _fail(exc)
    console.print(f"Hypothesis registered: {registered.hypothesis_id} [{registered.status.value}]")


@research_hypothesis_app.command("list")
def research_hypothesis_list(ctx: typer.Context) -> None:
    """List hypotheses without hiding rejected or inconclusive definitions."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        hypotheses = ResearchStore(settings.research_db_path).list_hypotheses()
    except BinanceAlgoError as exc:
        _fail(exc)
    table = Table(title="research hypotheses")
    table.add_column("id")
    table.add_column("status")
    table.add_column("title")
    for hypothesis in hypotheses:
        table.add_row(hypothesis.hypothesis_id, hypothesis.status.value, hypothesis.title)
    console.print(table)


@research_hypothesis_app.command("show")
def research_hypothesis_show(ctx: typer.Context, hypothesis_id: str) -> None:
    """Show one durable hypothesis definition."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        hypothesis = ResearchStore(settings.research_db_path).get_hypothesis(hypothesis_id)
        if hypothesis is None:
            raise ResearchError(f"unknown hypothesis: {hypothesis_id}")
    except BinanceAlgoError as exc:
        _fail(exc)
    console.print_json(json=orjson.dumps(hypothesis.model_dump(mode="json")).decode())


@research_hypothesis_app.command("history")
def research_hypothesis_history(ctx: typer.Context, hypothesis_id: str) -> None:
    """Write the contextual feature-evaluation history for one hypothesis."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        result = write_hypothesis_history_report(
            store=ResearchStore(settings.research_db_path),
            hypothesis_id=hypothesis_id,
            reports_root=settings.reports_root,
        )
    except BinanceAlgoError as exc:
        _fail(exc)
    console.print(
        f"Hypothesis history: evaluations={result.evaluation_count}\n"
        f"JSON: {result.report_json_path}\nMarkdown: {result.report_markdown_path}"
    )


@research_feature_app.command("list")
def research_feature_list(ctx: typer.Context) -> None:
    """List persisted feature IDs, versions, and lifecycle status."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        features = ResearchStore(settings.research_db_path).list_feature_definitions()
    except BinanceAlgoError as exc:
        _fail(exc)
    table = Table(title="research features")
    table.add_column("feature id")
    table.add_column("status")
    table.add_column("lookback")
    for feature in features:
        table.add_row(
            str(feature["feature_id"]),
            str(feature["status"]),
            str(feature["lookback"]),
        )
    console.print(table)


@research_feature_app.command("show")
def research_feature_show(ctx: typer.Context, feature_id: str) -> None:
    """Show one persisted feature contract."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        feature = ResearchStore(settings.research_db_path).get_feature_definition(feature_id)
        if feature is None:
            raise ResearchError(f"unknown feature: {feature_id}")
    except BinanceAlgoError as exc:
        _fail(exc)
    console.print_json(json=orjson.dumps(feature).decode())


@research_feature_app.command("history")
def research_feature_history(ctx: typer.Context, feature_id: str) -> None:
    """Show every contextual decision and write a derived ledger report."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        store = ResearchStore(settings.research_db_path)
        evaluations = store.list_feature_evaluations(feature_id=feature_id)
        result = write_feature_history_report(
            store=store,
            feature_id=feature_id,
            reports_root=settings.reports_root,
        )
    except BinanceAlgoError as exc:
        _fail(exc)
    table = Table(title=f"feature history — {feature_id}")
    table.add_column("type")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_column("decision")
    table.add_column("run")
    for evaluation in evaluations:
        value = "-" if evaluation.metric_value is None else f"{evaluation.metric_value:.8g}"
        table.add_row(
            evaluation.evaluation_type.value,
            evaluation.metric_name,
            value,
            evaluation.decision.value,
            evaluation.run_id[:16],
        )
    console.print(table)
    console.print(
        f"Evaluations: {result.evaluation_count}\n"
        f"JSON: {result.report_json_path}\nMarkdown: {result.report_markdown_path}"
    )


@research_feature_app.command("evaluate")
def research_feature_evaluate(
    ctx: typer.Context,
    run_id: Annotated[str, typer.Option("--run", help="Succeeded experiment run ID.")],
    feature_id: Annotated[str, typer.Option("--feature", help="Registered feature ID.")],
    file: Annotated[
        Path,
        typer.Option("--file", help="Strict contextual evaluation YAML.", exists=True),
    ],
) -> None:
    """Record one manual/contextual evaluation without mutating the feature definition."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        payload = yaml.safe_load(file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ResearchError("feature evaluation YAML root must be a mapping")
        spec = FeatureEvaluationSpec.model_validate(
            {**payload, "run_id": run_id, "feature_id": feature_id}
        )
        store = ResearchStore(settings.research_db_path)
        store.initialize()
        record = store.record_feature_evaluation(spec)
    except (OSError, yaml.YAMLError, ValidationError, BinanceAlgoError) as exc:
        _fail(exc if isinstance(exc, BinanceAlgoError) else ResearchError(str(exc)))
    console.print(
        f"Feature evaluation registered: {record.evaluation_id}\n"
        f"Decision: {record.decision.value}; metric={record.metric_name}"
    )


@research_ablation_app.command("evaluate")
def research_ablation_evaluate(ctx: typer.Context, campaign_id_or_name: str) -> None:
    """Evaluate every preregistered ablation pair in a completed campaign."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        store = ResearchStore(settings.research_db_path)
        store.initialize()
        campaign = store.get_campaign(campaign_id_or_name)
        if campaign is None:
            raise ResearchError(f"unknown campaign: {campaign_id_or_name}")
        source = campaign_spec_from_stored_payload(campaign.spec_json)
        result = AblationRunner(
            store=store,
            data_root=settings.data_root,
            reports_root=settings.reports_root,
        ).evaluate_campaign(campaign, source.ablation)
    except (BinanceAlgoError, OSError, pl.exceptions.PolarsError) as exc:
        _fail(exc if isinstance(exc, BinanceAlgoError) else ResearchError(str(exc)))
    table = Table(title=f"feature ablations — {campaign.name}")
    table.add_column("feature")
    table.add_column("change")
    table.add_column("delta Sharpe", justify="right")
    table.add_column("delta return", justify="right")
    table.add_column("decision")
    for evaluation in result.evaluations:
        table.add_row(
            evaluation.feature_id,
            evaluation.change.value,
            f"{evaluation.metrics.delta_sharpe:.6f}",
            f"{evaluation.metrics.delta_total_return:.6%}",
            evaluation.decision.value,
        )
    console.print(table)
    console.print(
        f"Registry evaluations: {sum(len(item.evaluations) for item in result.evaluations)}\n"
        f"JSON: {result.report_json_path}\nMarkdown: {result.report_markdown_path}"
    )


@research_experiment_app.command("list")
def research_experiment_list(ctx: typer.Context) -> None:
    """List immutable experiment definitions and their latest run status."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        store = ResearchStore(settings.research_db_path)
        identifiers = store.list_experiment_ids()
    except BinanceAlgoError as exc:
        _fail(exc)
    table = Table(title="research experiments")
    table.add_column("experiment id")
    table.add_column("latest status")
    table.add_column("attempt", justify="right")
    for identifier in identifiers:
        runs = store.list_runs(experiment_id_value=identifier)
        latest = runs[-1] if runs else None
        table.add_row(
            identifier,
            latest.status.value if latest is not None else "NOT_RUN",
            str(latest.attempt) if latest is not None else "-",
        )
    console.print(table)


@research_experiment_app.command("show")
def research_experiment_show(ctx: typer.Context, experiment_id: str) -> None:
    """Show one immutable experiment specification and all attempts."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        store = ResearchStore(settings.research_db_path)
        spec = store.get_experiment(experiment_id)
        if spec is None:
            raise ResearchError(f"unknown experiment: {experiment_id}")
        runs = store.list_runs(experiment_id_value=experiment_id)
    except BinanceAlgoError as exc:
        _fail(exc)
    console.print_json(
        json=orjson.dumps(
            {
                "experiment_id": experiment_id,
                "spec": spec.model_dump(mode="json"),
                "runs": [
                    {
                        "run_id": run.run_id,
                        "attempt": run.attempt,
                        "status": run.status.value,
                        "result_digest": run.result_digest,
                        "error_type": run.error_type,
                        "error_message": run.error_message,
                    }
                    for run in runs
                ],
            }
        ).decode()
    )


@research_experiment_app.command("rerun")
def research_experiment_rerun(
    ctx: typer.Context,
    experiment_id: str,
    chart: Annotated[
        bool,
        typer.Option("--chart", help="Generate an optional P&L SVG outside the result digest."),
    ] = False,
) -> None:
    """Create a new immutable attempt and require the prior result digest to match."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        store = ResearchStore(settings.research_db_path)
        result = _experiment_runner(settings, store).run(
            experiment_id,
            generate_chart=chart,
        )
    except (BinanceAlgoError, OSError, pl.exceptions.PolarsError) as exc:
        _fail(exc if isinstance(exc, BinanceAlgoError) else ResearchError(str(exc)))
    console.print(
        f"Experiment rerun succeeded: {result.experiment_id}\n"
        f"Run: {result.run.run_id} attempt={result.run.attempt}\n"
        f"Result digest: {result.run.result_digest}\n"
        f"Determinism: {'CONFIRMED' if result.deterministic_with_previous else 'FIRST_RUN'}\n"
        f"Artifacts: {result.artifact_directory}"
    )


@research_experiment_app.command("verify")
def research_experiment_verify(ctx: typer.Context, experiment_id: str) -> None:
    """Verify every registered checksum, size, row count, and the result digest."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        store = ResearchStore(settings.research_db_path)
        verification = _experiment_runner(settings, store).verify_experiment(experiment_id)
        if not verification.valid:
            raise ResearchError("; ".join(verification.issues))
    except BinanceAlgoError as exc:
        _fail(exc)
    console.print(
        f"Experiment artifacts PASS: run={verification.run_id}, files={verification.checked_files}"
    )


def _print_campaign_plan(plan: CampaignPlan) -> None:
    table = Table(title=f"campaign plan — {plan.source.campaign.name}")
    table.add_column("ordinal", justify="right")
    table.add_column("experiment id")
    table.add_column("strategy parameters")
    table.add_column("portfolio parameters")
    for trial in plan.trials[:5]:
        table.add_row(
            str(trial.ordinal),
            trial.experiment_id[:24],
            orjson.dumps(trial.tags["strategy_parameters"], option=orjson.OPT_SORT_KEYS).decode(),
            orjson.dumps(trial.tags["portfolio_parameters"], option=orjson.OPT_SORT_KEYS).decode(),
        )
    console.print(table)
    console.print(
        f"Campaign ID: {plan.campaign_id}\n"
        f"Combinations: possible={plan.possible_combinations}, "
        f"valid={plan.valid_combinations}, "
        f"rejected_by_constraints={plan.rejected_by_constraints}\n"
        "Dry plan only: no campaign, experiment, or run was persisted."
    )


@research_campaign_app.command("plan")
def research_campaign_plan(
    ctx: typer.Context,
    file: Annotated[
        Path,
        typer.Option("--file", help="Strict campaign YAML file.", exists=True, dir_okay=False),
    ],
    allow_large_campaign: Annotated[
        bool,
        typer.Option("--allow-large-campaign", help="Override the campaign max-trials guard."),
    ] = False,
) -> None:
    """Expand and validate a campaign without mutating the registry."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        plan = plan_campaign(
            load_campaign_spec(file),
            project_root=settings.project_root,
            data_root=settings.data_root,
            research_config=settings.research,
            allow_large_campaign=allow_large_campaign,
        )
    except BinanceAlgoError as exc:
        _fail(exc)
    _print_campaign_plan(plan)


@research_campaign_app.command("run")
def research_campaign_run(
    ctx: typer.Context,
    file: Annotated[
        Path,
        typer.Option("--file", help="Strict campaign YAML file.", exists=True, dir_okay=False),
    ],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Plan only; do not persist campaign state or runs."),
    ] = False,
    allow_large_campaign: Annotated[
        bool,
        typer.Option("--allow-large-campaign", help="Override the campaign max-trials guard."),
    ] = False,
) -> None:
    """Execute every valid trial with isolated failures and resumable state."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        plan = plan_campaign(
            load_campaign_spec(file),
            project_root=settings.project_root,
            data_root=settings.data_root,
            research_config=settings.research,
            allow_large_campaign=allow_large_campaign,
        )
        if dry_run:
            _print_campaign_plan(plan)
            return
        store = ResearchStore(settings.research_db_path)
        store.initialize()
        sync_builtin_registry(store, research_config=settings.research)
        result = _campaign_runner(settings, store).run(plan)
    except (BinanceAlgoError, OSError, pl.exceptions.PolarsError) as exc:
        _fail(exc if isinstance(exc, BinanceAlgoError) else ResearchError(str(exc)))
    console.print(
        f"Campaign {result.campaign.status.value}: {result.campaign.name}\n"
        f"Campaign ID: {result.campaign.campaign_id}\n"
        f"Trials: planned={result.planned_count}, executed={result.executed_count}, "
        f"cache_hits={result.cache_hit_count}, succeeded={result.succeeded_count}, "
        f"failed={result.failed_count}\n"
        f"Comparison: {result.comparison.comparison_path}\n"
        f"Report: {result.comparison.report_markdown_path}"
    )


@research_campaign_app.command("resume")
def research_campaign_resume(ctx: typer.Context, campaign_id_or_name: str) -> None:
    """Resume a partial campaign without duplicating definitions or successful runs."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        store = ResearchStore(settings.research_db_path)
        campaign = store.get_campaign(campaign_id_or_name)
        if campaign is None:
            raise ResearchError(f"unknown campaign: {campaign_id_or_name}")
        source = campaign_spec_from_stored_payload(campaign.spec_json)
        plan = plan_campaign(
            source,
            project_root=settings.project_root,
            data_root=settings.data_root,
            research_config=settings.research,
        )
        if plan.campaign_id != campaign.campaign_id:
            raise ResearchError(
                "campaign code/data fingerprint changed; resume requires the original state"
            )
        result = _campaign_runner(settings, store).run(plan)
    except (BinanceAlgoError, OSError, pl.exceptions.PolarsError) as exc:
        _fail(exc if isinstance(exc, BinanceAlgoError) else ResearchError(str(exc)))
    console.print(
        f"Campaign resume {result.campaign.status.value}: executed={result.executed_count}, "
        f"cache_hits={result.cache_hit_count}, failed={result.failed_count}\n"
        f"Comparison: {result.comparison.comparison_path}"
    )


@research_campaign_app.command("status")
def research_campaign_status(ctx: typer.Context, campaign_id_or_name: str) -> None:
    """Show durable campaign and trial status without hiding failures."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        store = ResearchStore(settings.research_db_path)
        campaign = store.get_campaign(campaign_id_or_name)
        if campaign is None:
            raise ResearchError(f"unknown campaign: {campaign_id_or_name}")
        experiments = store.list_campaign_experiments(campaign.campaign_id)
    except BinanceAlgoError as exc:
        _fail(exc)
    counts: dict[str, int] = {}
    for _, identifier, _ in experiments:
        runs = store.list_runs(experiment_id_value=identifier)
        status = runs[-1].status.value if runs else "NOT_RUN"
        counts[status] = counts.get(status, 0) + 1
    table = Table(title=f"campaign status — {campaign.name}")
    table.add_column("status")
    table.add_column("trials", justify="right")
    for status, count in sorted(counts.items()):
        table.add_row(status, str(count))
    console.print(table)
    console.print(
        f"Campaign ID: {campaign.campaign_id}\n"
        f"State: {campaign.status.value}; trials={campaign.trial_count}\n"
        f"Last error: {campaign.last_error or '-'}"
    )


@research_campaign_app.command("compare")
def research_campaign_compare(ctx: typer.Context, campaign_id_or_name: str) -> None:
    """Regenerate the aggregate report with every persisted trial."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        store = ResearchStore(settings.research_db_path)
        campaign = store.get_campaign(campaign_id_or_name)
        if campaign is None:
            raise ResearchError(f"unknown campaign: {campaign_id_or_name}")
        result = write_campaign_comparison(
            store=store,
            campaign=campaign,
            reports_root=settings.reports_root,
            compression=settings.storage.parquet_compression,
        )
    except (BinanceAlgoError, OSError, pl.exceptions.PolarsError) as exc:
        _fail(exc if isinstance(exc, BinanceAlgoError) else ResearchError(str(exc)))
    console.print(
        f"Campaign comparison: trials={result.trial_count}, "
        f"succeeded={result.succeeded_count}, failed={result.failed_count}\n"
        f"Parquet: {result.comparison_path}\n"
        f"Reports: {result.report_json_path}\n         {result.report_markdown_path}"
    )


@research_campaign_app.command("robustness")
def research_campaign_robustness(ctx: typer.Context, campaign_id_or_name: str) -> None:
    """Verify segmented artifacts and report neighborhood, DSR, PBO, and lockbox state."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        store = ResearchStore(settings.research_db_path)
        store.initialize()
        campaign = store.get_campaign(campaign_id_or_name)
        if campaign is None:
            raise ResearchError(f"unknown campaign: {campaign_id_or_name}")
        result = build_campaign_robustness(
            store=store,
            campaign=campaign,
            data_root=settings.data_root,
            reports_root=settings.reports_root,
            platform=settings.research_platform,
        )
    except (BinanceAlgoError, OSError, pl.exceptions.PolarsError) as exc:
        _fail(exc if isinstance(exc, BinanceAlgoError) else ResearchError(str(exc)))
    console.print(
        f"Campaign robustness: {result.campaign_name}\n"
        f"Trials: planned={result.planned_trials}, successful={result.successful_trials}, "
        f"distinct={result.distinct_strategies}, "
        f"independent_approx={result.approximate_independent_strategies:.3f}\n"
        f"Best (selected, not independent OOS): {result.best_experiment_id}\n"
        f"DSR: {result.dsr.probability:.6f} over {result.dsr.number_of_trials} trials\n"
        f"PBO: {result.pbo.status.value} — {result.pbo.reason}\n"
        f"Lockbox: {result.lockbox.status.value} — {result.lockbox.reason}\n"
        f"JSON: {result.report_json_path}\nMarkdown: {result.report_markdown_path}"
    )


def _print_candidate_assessment(result: CandidateAssessment) -> None:
    table = Table(title=f"candidate gates — {result.experiment_id[:24]}")
    table.add_column("gate")
    table.add_column("result")
    table.add_column("detail")
    for gate in result.gates:
        table.add_row(gate.name, "PASS" if gate.passed else "FAIL", gate.detail)
    console.print(table)
    console.print(
        f"Candidate assessment: {'PASS' if result.passed else 'BLOCKED'}\n"
        f"Campaign trials: {result.robustness.planned_trials}\n"
        f"JSON: {result.report_json_path}\nMarkdown: {result.report_markdown_path}"
    )


@research_candidate_app.command("report")
def research_candidate_report(ctx: typer.Context, experiment_id: str) -> None:
    """Generate a gate report without creating a promotion event."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        store = ResearchStore(settings.research_db_path)
        store.initialize()
        result = _promotion_manager(settings, store).assess_candidate(experiment_id)
    except (BinanceAlgoError, OSError, pl.exceptions.PolarsError) as exc:
        _fail(exc if isinstance(exc, BinanceAlgoError) else ResearchError(str(exc)))
    _print_candidate_assessment(result)


@research_promote_app.command("candidate")
def research_promote_candidate(
    ctx: typer.Context,
    experiment_id: str,
    reason: Annotated[str, typer.Option("--reason", help="Mandatory promotion rationale.")],
) -> None:
    """Promote from discovery only when every preregistered candidate gate passes."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        store = ResearchStore(settings.research_db_path)
        store.initialize()
        result = _promotion_manager(settings, store).promote_candidate(experiment_id, reason=reason)
    except (BinanceAlgoError, OSError, pl.exceptions.PolarsError) as exc:
        _fail(exc if isinstance(exc, BinanceAlgoError) else ResearchError(str(exc)))
    _print_candidate_assessment(result.assessment)
    console.print(f"Promotion event: {result.event.promotion_id} [{result.event.decision.value}]")
    if not result.assessment.passed:
        raise typer.Exit(code=1)


@research_promote_app.command("phase4")
def research_promote_phase4(
    ctx: typer.Context,
    experiment_id: str,
    reason: Annotated[str, typer.Option("--reason", help="Mandatory promotion rationale.")],
) -> None:
    """Require an approved lockbox event before creating a Phase 4 candidate."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        store = ResearchStore(settings.research_db_path)
        store.initialize()
        event = _promotion_manager(settings, store).promote_phase4(experiment_id, reason=reason)
    except (BinanceAlgoError, OSError, pl.exceptions.PolarsError) as exc:
        _fail(exc if isinstance(exc, BinanceAlgoError) else ResearchError(str(exc)))
    console.print(f"Phase 4 promotion event: {event.promotion_id} [{event.decision.value}]")
    if event.decision.value != "APPROVED":
        raise typer.Exit(code=1)


@research_promote_app.command("history")
def research_promotion_history(ctx: typer.Context, experiment_id: str) -> None:
    """Show every immutable promotion, block, rejection, and invalidation event."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        events = ResearchStore(settings.research_db_path).list_promotions(experiment_id)
    except BinanceAlgoError as exc:
        _fail(exc)
    table = Table(title=f"promotion history — {experiment_id[:24]}")
    table.add_column("from")
    table.add_column("to")
    table.add_column("decision")
    table.add_column("reason")
    for event in events:
        table.add_row(
            event.from_stage.value,
            event.to_stage.value,
            event.decision.value,
            event.reason,
        )
    console.print(table)


@research_app.command("reject")
def research_reject(
    ctx: typer.Context,
    experiment_id: str,
    reason: Annotated[str, typer.Option("--reason", help="Mandatory rejection rationale.")],
) -> None:
    """Create an explicit immutable rejection event without deleting any result."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        store = ResearchStore(settings.research_db_path)
        store.initialize()
        event = _promotion_manager(settings, store).reject(experiment_id, reason=reason)
    except BinanceAlgoError as exc:
        _fail(exc)
    console.print(f"Rejection event: {event.promotion_id} [{event.to_stage.value}]")


@research_app.command("wave1-report")
def research_wave1_report(
    ctx: typer.Context,
    file: Annotated[
        Path,
        typer.Option(
            "--file",
            help="Strict Alpha Reboot Wave 1 YAML file.",
            exists=True,
            dir_okay=False,
        ),
    ] = Path("configs/alpha_reboot_wave1.yaml"),
) -> None:
    """Build the strict 18-trial Alpha Reboot Wave 1 report."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        store = ResearchStore(settings.research_db_path)
        store.initialize()
        result = build_alpha_reboot_wave1_report(
            store=store,
            config_path=file,
            project_root=settings.project_root,
            data_root=settings.data_root,
            reports_root=settings.reports_root,
            compression=settings.storage.parquet_compression,
        )
    except (BinanceAlgoError, OSError, pl.exceptions.PolarsError) as exc:
        _fail(exc if isinstance(exc, BinanceAlgoError) else ResearchError(str(exc)))
    console.print(
        f"Alpha Reboot Wave 1: trials={result.trial_count}\n"
        f"Champion run: {result.champion_run_id}\n"
        f"Report: {result.report_markdown_path}\n"
        f"Candidates: {result.candidates_path}\n"
        f"Correlations: {result.daily_return_correlation_path}"
    )


@research_app.command("build")
def research_build(
    ctx: typer.Context,
    start: Annotated[str, typer.Option(help="Inclusive UTC input start as YYYY-MM-DD.")],
    end: Annotated[str, typer.Option(help="Inclusive UTC input end as YYYY-MM-DD.")],
    symbols: Annotated[str, typer.Option(help="Comma-separated fixed seed symbols.")] = (
        "BTCUSDT,ETHUSDT,SOLUSDT"
    ),
    feature_set: Annotated[
        str,
        typer.Option(help="Registered feature-set identity, for example alpha_reboot_features:v1."),
    ] = "phase3_baseline_features:v1",
) -> None:
    """Build and audit the causal point-in-time research dataset."""

    try:
        settings = _settings(ctx)
        _configure(settings)
        parsed_symbols = parse_symbols(symbols)
        try:
            feature_set_name, feature_set_version = feature_set.rsplit(":", 1)
        except ValueError as exc:
            raise typer.BadParameter("--feature-set must be NAME:VERSION") from exc
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
            feature_set_name=feature_set_name,
            feature_set_version=feature_set_version,
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
        manifest_path = dataset_path.with_suffix(".json")
        reference = load_dataset_reference(manifest_path)
        store = ResearchStore(settings.research_db_path)
        store.initialize()
        sync_builtin_registry(store, research_config=settings.research)
        store.register_hypothesis(phase3_baseline_hypothesis())
        spec = build_phase3_experiment_spec(
            dataset_reference=reference,
            config=settings.research,
            project_root=settings.project_root,
        )
        experiment_id = store.register_experiment(spec)
        result = _experiment_runner(settings, store).run(
            experiment_id,
            generate_chart=chart,
        )
    except (BinanceAlgoError, OSError, pl.exceptions.PolarsError) as exc:
        if isinstance(exc, BinanceAlgoError):
            _fail(exc)
        _fail(DataQualityError(str(exc)))
    metric_values = {
        metric.metric_name: metric.metric_value
        for metric in store.list_metrics(result.run.run_id)
        if metric.scope.value == "TEST" and metric.fold is None and metric.regime is None
    }
    table = Table(title="Phase 3 walk-forward baseline")
    table.add_column("metric")
    table.add_column("out-of-sample", justify="right")
    for name, value in (
        ("periods", str(int(metric_values["periods"]))),
        ("total return", f"{metric_values['total_return']:.4%}"),
        ("Sharpe", f"{metric_values['sharpe']:.3f}"),
        ("max drawdown", f"{metric_values['max_drawdown']:.4%}"),
        ("turnover", f"{metric_values['turnover']:.3f}"),
        ("funding P&L", f"{metric_values['funding_pnl']:.6f}"),
        ("accounting error", f"{metric_values['accounting_error_max']:.3e}"),
    ):
        table.add_row(name, value)
    console.print(table)
    output = (
        "Baseline only; no claim of edge.\n"
        f"Experiment: {result.experiment_id}\n"
        f"Run: {result.run.run_id}\n"
        f"Result digest: {result.run.result_digest}\n"
        f"Artifacts: {result.artifact_directory}"
    )
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
