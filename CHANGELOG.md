# Changelog

## 0.2.0 — 2026-08-27

- State store SQLite com WAL, migrations, rollback e state machines explícitas.
- Manifest de `data_files`, `backfill_jobs`, checkpoints, qualidade e schemas.
- Downloader concorrente e idempotente de arquivos oficiais daily klines 1m.
- SHA-256, retomada HTTP Range, limites de tamanho, extração segura e quarantine.
- CLI `backfill klines` e relatórios JSON/Markdown por job.
- Contract test real e testes de corrupção, path traversal, retry e rerun.

## 0.1.0 — 2026-08-27

- Bootstrap Python/uv com lint, typing, testes, CI e container.
- Configuração Pydantic/YAML com defaults Demo e travas contra ordens.
- Logging JSON com redaction de segredos.
- Cliente REST público para ping, server time e exchangeInfo.
- Snapshot raw + Parquet de metadados e universo seed point-in-time.
- CLI operacional e diagnóstico de ambiente.
