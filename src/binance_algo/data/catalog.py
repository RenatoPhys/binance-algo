"""DuckDB catalog over manifest-selected immutable canonical Parquet files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb

from binance_algo.common.errors import DataQualityError
from binance_algo.data.manifest import DataFileRecord


@dataclass(frozen=True, slots=True)
class CatalogResult:
    database_path: str
    view_name: str
    file_count: int
    row_count: int
    min_open_time_ms: int | None
    max_open_time_ms: int | None


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def rebuild_kline_catalog(*, database_path: Path, records: list[DataFileRecord]) -> CatalogResult:
    if not records:
        raise DataQualityError("cannot build DuckDB catalog without normalized kline files")
    paths = [str(Path(record.path).resolve()).replace("\\", "/") for record in records]
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        raise DataQualityError(f"cannot catalog missing Parquet: {missing[0]}")
    parquet_list = ", ".join(_sql_string(path) for path in paths)
    database_path = database_path.resolve()
    database_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with duckdb.connect(str(database_path)) as connection:
            connection.execute(
                "CREATE OR REPLACE VIEW klines AS "
                f"SELECT * FROM read_parquet([{parquet_list}], union_by_name = true)"
            )
            row = connection.execute(
                """
                SELECT COUNT(*)::BIGINT, MIN(open_time_ms)::BIGINT, MAX(open_time_ms)::BIGINT
                FROM klines
                """
            ).fetchone()
    except duckdb.Error as exc:
        raise DataQualityError(f"cannot rebuild DuckDB kline catalog: {exc}") from exc
    if row is None:
        raise DataQualityError("DuckDB catalog verification returned no result")
    return CatalogResult(
        database_path=str(database_path),
        view_name="klines",
        file_count=len(paths),
        row_count=int(row[0]),
        min_open_time_ms=int(row[1]) if row[1] is not None else None,
        max_open_time_ms=int(row[2]) if row[2] is not None else None,
    )
