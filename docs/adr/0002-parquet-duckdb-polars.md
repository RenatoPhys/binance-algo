# ADR 0002 — Parquet, Polars e DuckDB

Status: aceito em 2026-08-27.

Parquet é o formato analítico, Polars transforma/valida e DuckDB consulta localmente. SQLite será
reservado ao pequeno estado operacional no próximo marco.

Consequência: numeric strings da exchange são preservadas em bronze e conversões analíticas serão
versionadas em silver/gold.
