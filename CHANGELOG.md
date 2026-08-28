# Changelog

## 0.5.0 — 2026-08-28

- Funding histórico público com raw imutável, Parquet canônico, manifesto e view DuckDB.
- Dataset point-in-time com features interpretáveis, as-of funding e labels no próximo open.
- Baseline cross-sectional de momentum residual com limites de exposição e no-trade band.
- Backtest vetorizado com fees, spread, slippage, funding, turnover e invariantes contábeis.
- Walk-forward, regimes, custos 1,5×/2×, atraso, perturbações e bootstrap em blocos.
- Relatório real da janela de 90 dias preservando o resultado negativo sem tuning retrospectivo.

## 0.4.0 — 2026-08-28

- Recorder WebSocket público resiliente em rotas `public` e `market`.
- Buffer limitado, micro-batches Parquet atômicos, recovery, métricas e health checks.
- Gate real de 60 minutos e replay 1×/100× com digest idêntico.

## 0.3.0 — 2026-08-27

- Normalização canônica de klines, auditoria de qualidade e catálogo DuckDB.
- Backfill real de 90 dias para BTCUSDT, ETHUSDT e SOLUSDT com rerun idempotente.

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
