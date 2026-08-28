# Progress

## Current milestone

Fase 4 — simulador orientado a eventos (planejamento). A Fase 3 foi concluída como baseline de
validação; o resultado negativo não foi promovido a alpha.

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
- [x] WebSocket resiliente nas rotas oficiais `public` e `market`
- [x] `bookTicker`, `aggTrade`, `markPrice@1s`/funding e `kline_1m` para 3 símbolos
- [x] Fila assíncrona limitada, staleness, backoff com jitter e rotação preventiva
- [x] Parquet raw atômico por micro-batch, manifesto e checkpoint transacional
- [x] Recovery de arquivos em voo/órfãos e quarentena de temporários/inválidos
- [x] Métricas Prometheus e health checks live/ready
- [x] Catálogo DuckDB e gate de qualidade do recorder
- [x] Replay temporal 1×/acelerado, clock injetável e determinismo por digest
- [x] Gate real de 60 minutos da Fase 2 aprovado
- [x] Funding histórico público com raw/Parquet/checksum e catálogo deduplicado
- [x] Dataset point-in-time horário, features versionadas e labels no próximo open
- [x] Auditorias contra look-ahead, backward-fill, painel incompleto e timestamps inválidos
- [x] Baseline cross-sectional de momentum residual com limites de exposição
- [x] Backtest vetorizado com fees, spread, slippage, funding e turnover
- [x] Walk-forward temporal, embargo, regimes, estresses e bootstrap em blocos
- [x] Relatório de performance/estabilidade da Fase 3 e rerun determinístico

## Pending

- [ ] Especificar eventos e invariantes contábeis do simulador da Fase 4
- [ ] Implementar marketable limit, IOC, post-only e fills parciais
- [ ] Modelar latência, adverse selection e markout no replay
- [ ] Reconciliar posições, fees e funding antes de qualquer Demo Trading

## Blockers

- Nenhum bloqueio de implementação.
- Docker e GNU Make não estão instalados no host atual; os equivalentes `uv` são a validação
  primária.

## Validation

- `uv sync`: passed; lockfile com 59 packages
- `uv sync --frozen`: passed; `binance-algo==0.4.0`, lockfile com 59 packages
- `ruff format --check .`: passed; 72 files
- `ruff check .`: passed
- `mypy src`: passed; 37 source files
- `pytest -m "not network"`: 55 passed, 2 deselected
- `pytest -m network`: 2 passed, 55 deselected
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
- recorder real 60m (Demo): run `b44ed33456b147179435f7cd34733e4e`, 3.601,663s,
  99.847 mensagens/linhas, 1.440 arquivos, 22.038.204 bytes, queue peak 130/100.000,
  zero drops, duplicatas, gaps, regressões ou valores inválidos
- streams 60m: book ticker 54.605, aggregate trades 21.922, mark price 10.798 e kline
  12.522; checksums e schemas passaram em todos os 1.440 Parquets
- latência p95 observada: book 519,466ms, trades 512,946ms, mark 585,191ms e kline
  517,422ms; staleness máxima por símbolo reportada entre 2,626s e 27,854s
- replay 1×/100×: 99.847 eventos, digest idêntico
  `42920f3830e70d5d0541eec179be514cd5219b201da355c93568813178d1e647`, clocks
  virtuais 3.599,727s e 35,997s
- reconnect forçado em servidor fake: passed, zero loss; cancelamento gracioso, recovery e
  quarentena também cobertos por testes
- funding 90d: 270 eventos por símbolo, 810 na view deduplicada, zero duplicatas ou valores
  inválidos; rerun integral retornou `skipped`
- dataset Fase 3: versão `50f51aa076fe5426`, 5.832 linhas, 1.944 decisões horárias, painel
  completo e zero violações temporais, duplicatas ou features nulas
- backtest Fase 3: versão `974ab2c8643ace95`, 3 folds e 1.008 horas OOS; erro contábil zero,
  exposição líquida média próxima de zero e participação máxima de volume de 0,0426%
- baseline OOS: retorno de preço somado +1,527%, funding -0,046%, custos explícitos 17,348%,
  retorno composto líquido -14,688%, max drawdown -14,711% e turnover 266,888
- estabilidade: custos 1,5×/2× produziram -21,781%/-28,285%; atraso de uma barra -12,760%;
  bootstrap em 500 amostras teve percentis 5/50/95 de -18,278%/-14,663%/-11,022%

## Risks and mitigations

- O `serverTime` embutido em `exchangeInfo` veio três dias defasado no teste real. Conforme a
  documentação oficial, o lineage usa `/fapi/v1/time`; os artefatos da primeira tentativa foram
  movidos para `var/data/quarantine/invalid_exchange_info_servertime`.
- Contract tests externos continuam opcionais porque disponibilidade da Binance não deve bloquear
  toda contribuição.
- A Binance pode substituir archives históricos. O rerun normal preserva idempotência e não
  consulta o checksum remoto; detecção periódica de replacements ainda precisa de uma política
  versionada para não sobrescrever raw silenciosamente.
- A execução de 60 minutos não sofreu disconnect espontâneo; reconnect, resubscribe e continuidade
  foram provados deterministicamente com servidor fake.
- A granularidade de 30 segundos produziu 1.440 arquivos na hora. Compactação é o primeiro cuidado
  operacional antes de ampliar duração/universo.
- O baseline apresentou rank IC médio de -0,0045 e perda em todos os regimes/estresses. Ele prova
  o pipeline e rejeita a configuração atual; não é evidência de edge nem deve ser otimizado na
  mesma janela.
- O fee schedule é uma hipótese configurável com tier `unknown`, não uma consulta da conta. Todo
  uso financeiro exige confirmar fee/tier e manter nova vigência versionada.

## Known limitations

- Somente Binance USDⓈ-M Futures e dados públicos
- Universo inicial limitado a BTCUSDT, ETHUSDT e SOLUSDT
- Downloader atual cobre apenas arquivos daily de klines 1m; monthly e outros datasets virão em
  incrementos posteriores
- O histórico validado cobre 90 dias; ranges maiores e arquivos monthly ainda não foram testados
- A comparação cruzada com uma segunda fonte permanece `NOT_RUN`
- Um único recorder deve operar por raiz de storage/state DB; não há coordenação distribuída
- `bookTicker` não é depth nem sustenta um order book local
- O universo dinâmico continua bloqueado sem snapshots históricos de status/liquidez; o baseline
  usa apenas o seed fixado ex ante
- O backtest é bar/next-open e não modela fila, fill parcial, latência ou adverse selection
- Sem estratégia promovida, simulador orientado a eventos, autenticação ou envio de ordens
