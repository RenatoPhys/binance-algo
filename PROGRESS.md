# Progress

## Current milestone

Fase 2 — recorder WebSocket e replay determinístico de market data.

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
- [x] Normalização canônica CSV para Parquet com schema versionado e lineage
- [x] Deduplicação determinística e ordenação pela chave canônica
- [x] Auditoria por partição e range de gaps, schema, checksum e invariantes
- [x] Catálogo DuckDB persistente com view `klines`
- [x] Backfill de 90 dias para BTCUSDT, ETHUSDT e SOLUSDT
- [x] Gate histórico da Fase 1 aprovado e rerun integral idempotente

## Pending

- [ ] Recorder WebSocket e replay

## Blockers

- Nenhum bloqueio de implementação.
- Docker e GNU Make não estão instalados no host atual; os equivalentes `uv` são a validação
  primária.

## Validation

- `uv sync`: passed; lockfile com 59 packages
- `ruff format --check .`: passed
- `ruff check .`: passed
- `mypy src`: passed; 25 source files
- `pytest -m "not network"`: 33 passed, 2 deselected
- `pytest -m network`: 2 passed, 33 deselected
- `binance-algo doctor`: todos os checks passaram; SQLite `journal_mode=wal`
- `exchange-info snapshot`: 733 instrumentos persistidos
- DuckDB: 733 linhas, 733 símbolos distintos, zero filtros tick/step ausentes
- `universe build`: BTCUSDT, ETHUSDT e SOLUSDT; rerun manteve a versão
  `3d3e2f79aef8de4e`
- archive 90d: 2026-05-28 a 2026-08-25, 270 arquivos, 388.800 linhas; 267 downloads
  novos, 3 skips, 15.878.389 bytes e zero falhas
- normalização: 270 Parquets e 388.800 linhas; zero falhas
- quality gate: 129.600 linhas por símbolo, zero gaps, duplicatas, desordem, nulos,
  preços/quantidades negativos ou OHLC inválido; schema e checksum passaram
- DuckDB: view `klines` com 388.800 linhas
- rerun integral: 270 archives e 270 Parquets `skipped`, zero bytes baixados e row count estável
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
- O histórico validado cobre 90 dias; ranges maiores e arquivos monthly ainda não foram testados
- A comparação cruzada com uma segunda fonte permanece `NOT_RUN`
- Sem WebSocket, replay, backtest, estratégia, autenticação ou envio de ordens
