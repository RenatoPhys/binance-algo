# Progress

## Current milestone

Fase 1 — state store e archive downloader de klines 1m.

## Completed

- [x] Estrutura do pacote e gestão com uv
- [x] Configuração tipada e travas de segurança
- [x] Logging JSON e mascaramento de segredos
- [x] Clock e cliente REST público
- [x] Parser de instrumentos e filtros
- [x] Snapshot raw/Parquet de exchangeInfo
- [x] Universo seed point-in-time
- [x] Testes, CI, Docker e documentação operacional
- [x] State store SQLite em WAL mode e migration 1
- [x] Manifest e state machines de arquivos/jobs
- [x] Downloader daily de klines 1m com concorrência limitada
- [x] SHA-256 oficial, retomada HTTP Range e extração segura
- [x] Relatórios JSON/Markdown e idempotência em rerun

## Pending

- [ ] Normalização e auditoria das klines
- [ ] Backfill completo de 90 dias para os três símbolos
- [ ] Recorder WebSocket e replay

## Blockers

- Nenhum bloqueio de implementação.
- Docker e GNU Make não estão instalados no host atual; os equivalentes `uv` são a validação
  primária.

## Validation

- `uv sync`: passed; lockfile com 59 packages
- `ruff format --check .`: passed
- `ruff check .`: passed
- `mypy src`: passed; 22 source files
- `pytest -m "not network"`: 28 passed, 2 deselected
- `pytest -m network`: 2 passed, 28 deselected
- `binance-algo doctor`: todos os checks passaram; SQLite `journal_mode=wal`
- `exchange-info snapshot`: 733 instrumentos persistidos
- DuckDB: 733 linhas, 733 símbolos distintos, zero filtros tick/step ausentes
- `universe build`: BTCUSDT, ETHUSDT e SOLUSDT; rerun manteve a versão
  `3d3e2f79aef8de4e`
- archive smoke: BTCUSDT, ETHUSDT e SOLUSDT em 2026-08-25, 3 × 1.440 linhas,
  187.382 bytes, zero falhas
- archive rerun: 3 `skipped`, zero bytes baixados, três manifests `VALIDATED`
- checksum contract BTCUSDT: `1651da32387a1342bdba15b28504dc4d55caee905a58fec04f52c280b1d69f7f`

## Risks and mitigations

- O `serverTime` embutido em `exchangeInfo` veio três dias defasado no teste real. Conforme a
  documentação oficial, o lineage usa `/fapi/v1/time`; os artefatos da primeira tentativa foram
  movidos para `var/data/quarantine/invalid_exchange_info_servertime`.
- Contract tests externos continuam opcionais porque disponibilidade da Binance não deve bloquear
  toda contribuição.
- A Binance pode substituir archives históricos. O rerun normal preserva idempotência e não
  consulta o checksum remoto; detecção periódica de replacements ainda precisa de uma política
  versionada para não sobrescrever raw silenciosamente.

## Known limitations

- Somente Binance USDⓈ-M Futures e dados públicos
- Universo inicial limitado a BTCUSDT, ETHUSDT e SOLUSDT
- Downloader atual cobre apenas arquivos daily de klines 1m; monthly e outros datasets virão em
  incrementos posteriores
- O smoke real cobriu um dia; a janela de aceite de 90 dias permanece pendente
- Sem normalização Parquet, auditoria de gaps ou views DuckDB para klines
- Sem WebSocket, backtest, estratégia, autenticação ou envio de ordens
