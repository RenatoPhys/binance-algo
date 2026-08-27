# Progress

## Current milestone

Primeiro PR — bootstrap, REST público e metadados.

## Completed

- [x] Estrutura do pacote e gestão com uv
- [x] Configuração tipada e travas de segurança
- [x] Logging JSON e mascaramento de segredos
- [x] Clock e cliente REST público
- [x] Parser de instrumentos e filtros
- [x] Snapshot raw/Parquet de exchangeInfo
- [x] Universo seed point-in-time
- [x] Testes, CI, Docker e documentação operacional

## Pending

- [ ] State store SQLite WAL e manifest
- [ ] Archive downloader idempotente
- [ ] Normalização e auditoria das klines
- [ ] Recorder WebSocket e replay

## Blockers

- Nenhum bloqueio de implementação.
- Docker e GNU Make não estão instalados no host atual; os equivalentes `uv` são a validação
  primária.

## Validation

- `uv sync`: passed; lockfile com 59 packages
- `ruff format --check .`: passed
- `ruff check .`: passed
- `mypy src`: passed; 18 source files
- `pytest -m "not network"`: 17 passed, 1 deselected
- `pytest -m network`: 1 passed, 17 deselected
- `binance-algo doctor`: todos os checks passaram; clock offset observado -11 ms
- `exchange-info snapshot`: 733 instrumentos persistidos
- DuckDB: 733 linhas, 733 símbolos distintos, zero filtros tick/step ausentes
- `universe build`: BTCUSDT, ETHUSDT e SOLUSDT; rerun manteve a versão
  `3d3e2f79aef8de4e`

## Risks and mitigations

- O `serverTime` embutido em `exchangeInfo` veio três dias defasado no teste real. Conforme a
  documentação oficial, o lineage usa `/fapi/v1/time`; os artefatos da primeira tentativa foram
  movidos para `var/data/quarantine/invalid_exchange_info_servertime`.
- Contract tests externos continuam opcionais porque disponibilidade da Binance não deve bloquear
  toda contribuição.

## Known limitations

- Somente Binance USDⓈ-M Futures e dados públicos
- Universo inicial limitado a BTCUSDT, ETHUSDT e SOLUSDT
- Sem state store, manifest global ou archive downloader
- Sem WebSocket, backtest, estratégia, autenticação ou envio de ordens
