"""Operational command-line interface for the first project milestone."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from binance_algo.common.errors import BinanceAlgoError
from binance_algo.config import Settings, load_settings
from binance_algo.data.metadata import MetadataSnapshotService
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
app.add_typer(exchange_info_app, name="exchange-info")
app.add_typer(universe_app, name="universe")
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


if __name__ == "__main__":
    app()
