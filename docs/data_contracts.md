# Contratos de dados

## instrument_metadata — schema version 1

Chave lógica: `symbol + valid_from_ms`.

| Campo | Tipo Parquet | Semântica |
|---|---:|---|
| exchange | String | `binance` |
| market | String | `usdm_futures` |
| symbol, pair | String | Identificadores do contrato |
| contract_type, status | String | Estado observado no snapshot |
| base_asset, quote_asset, margin_asset | String | Ativos reportados pela exchange |
| onboard_date_ms, delivery_date_ms | Int64 | Epoch UTC da exchange |
| price_precision, quantity_precision | Int64 | Precisões informativas; não servem para quantização |
| tick_size, step_size | String | Valores decimais exatos dos filtros |
| min_qty, max_qty, min_notional | String | Limites exatos dos filtros |
| raw_filters_json | String | Todos os filtros preservados em JSON canônico |
| valid_from_ms | Int64 | Hora do endpoint dedicado `/fapi/v1/time` no snapshot |
| ingested_at_ns | Int64 | Epoch UTC local de ingestão |
| schema_version | Int64 | Versão do contrato canônico |

O `serverTime` presente em `exchangeInfo` é preservado no raw, mas não define `valid_from_ms`,
pois a própria documentação manda ignorá-lo para hora atual. O payload completo é preservado
como JSON em raw. Campos numéricos usados futuramente para
execução permanecem strings; nenhuma decisão usa `round()` ou `pricePrecision` no lugar dos
filtros.

## universe — schema version implícita 1

Cada símbolo seed gera uma linha com `included`, `reason`, `as_of_ms`,
`metadata_valid_from_ms` e `universe_version`. O hash inclui cutoff, snapshot, filtros e razões,
permitindo reprodução e detecção de mudanças.

## data_files — SQLite migration 1

Chave: `file_id`, um SHA-256 determinístico da fonte, mercado, frequência, dataset, símbolo,
intervalo e dia. `path` também é único.

Campos de lineage incluem dataset, layer, source, janela temporal, row count, schema version,
checksum, status, ingestion run, pais e último erro. Estados válidos:

```text
DOWNLOADING -> DOWNLOADED -> VALIDATED -> NORMALIZED -> COMPACTED
       |             |            |             |
       +-----------> FAILED / QUARANTINED <-----+
```

`FAILED` e `QUARANTINED` podem voltar para `DOWNLOADING` por retomada/reparo explícito.

## backfill_jobs — SQLite migration 1

Um job registra símbolos, intervalo, range inclusivo, total, concluídos, falhas e último erro.
Estados: `PENDING -> RUNNING -> COMPLETED|FAILED`.

## Arquivo oficial de kline 1m

O ZIP e seu `.CHECKSUM` são preservados em `raw_archives`. O ZIP deve conter exatamente um CSV
flat, sem symlink ou path traversal, com o cabeçalho oficial de 12 colunas. O row count persistido
é o número de linhas de dados; não se impõe 1.440 para que gaps e dias parciais permaneçam
observáveis na auditoria seguinte.

## klines — schema version 1

Chave lógica: `symbol + interval + open_time_ms`.

| Campo | Tipo Parquet | Semântica |
|---|---:|---|
| symbol, interval | String | Série canônica |
| open_time_ms, close_time_ms | Int64 | Limites do candle em epoch UTC |
| open, high, low, close | Float64 | OHLC observado |
| base_volume, quote_volume | Float64 | Volumes totais |
| trade_count | Int64 | Número de trades |
| taker_buy_base_volume, taker_buy_quote_volume | Float64 | Volumes taker buy |
| is_closed | Boolean | `true` para archive histórico fechado |
| ingested_at_ns | Int64 | Timestamp determinístico herdado da ingestão raw |
| source | String | `binance_public_data` |
| schema_version | Int32 | `1` |

A normalização converte tipos estritamente, deduplica pela chave conservando a primeira linha e
ordena pela chave. O Parquet bronze referencia o `file_id` raw em `parent_file_ids_json`; seu nome
incorpora o checksum da origem.

O gate de qualidade registra row count, timestamps mínimo/máximo, nulos, duplicatas, desordem,
gaps e duração, preços/quantidades negativos, OHLC inválido, close time inválido, timestamps
desalinhados, candles abertos, checksum, schema e status de comparação com outra fonte. Gaps são
contados também entre partições diárias.

## Streams em tempo real — schema version 1

Todos os datasets carregam `event_id`, `symbol`, `event_time_ms`, `received_time_ns`,
`processed_time_ns`, `connection_id`, `ingestion_run_id`, `source`, `schema_version` e
`payload_json`. `event_id` é estável entre runs para o mesmo stream/payload; os timestamps locais
e IDs operacionais preservam cada observação. Strings decimais do venue são mantidas junto das
representações `Float64`, e valores não numéricos ou não finitos são rejeitados no parser.

| Dataset | Chave/ordem observada | Campos específicos principais |
|---|---|---|
| `book_ticker` | `symbol + update_id` | bid/ask price e quantity, transaction time |
| `aggregate_trades` | `symbol + aggregate_trade_id` | price, quantity, first/last trade ID, maker flag |
| `mark_price` | `symbol + event_time_ms` | mark/index/settle price, funding rate e próxima funding |
| `kline_1m` | atualizações de `symbol + open_time_ms` | OHLCV, trades, taker volumes e `is_closed` |

Os arquivos ficam em `raw/binance/usdm/<dataset>/date=.../hour=.../symbol=...`. Cada micro-batch
é ordenado por evento/recebimento/ID, recebe checksum SHA-256 e passa por
`DOWNLOADING -> DOWNLOADED -> VALIDATED`. O checkpoint `dataset:symbol` só muda depois desse
commit durável.

O gate considera gap de `aggregate_trade_id`, regressões de trade/update ID, invariantes de
preço/volume, schema, checksum, duplicatas globais e igualdade mensagens/linhas. `bookTicker` não
é tratado como diff-depth e não sustenta um order book local.

## funding_rates — schema version 1

Chave lógica: `symbol + funding_time_ms + rate_type`.

| Campo | Semântica |
|---|---|
| `funding_rate_str`, `funding_rate` | valor exato recebido e representação analítica |
| `funding_time_ms` | timestamp efetivo do evento no venue |
| `mark_price_str`, `mark_price` | mark associado, quando retornado |
| `rate_type` | `Regular`, `Special` ou novo valor preservado do upstream |
| `ingested_at_ns` | observação local da primeira persistência daquele conteúdo |
| `source`, `schema_version` | `binance_public_rest`, `1` |

O raw JSON canônico é imutável e nomeado pelo checksum. Ranges sobrepostos podem criar revisões;
a view DuckDB seleciona a observação mais recente por chave. Eventos simultâneos de tipos
distintos são somados na contabilidade.

## research_dataset — schema version 2

Chave: `decision_time_ms + symbol`. O painel contém os três símbolos em todo timestamp aceito.
Campos `feature_*` e features nomeadas usam somente fontes até a decisão. `execution_time_ms` é o
próximo open e `label_end_time_ms` ocorre 60 minutos depois. `future_*` e `outcome_*` são labels ou
resultados e nunca entram no score. O schema declara cada coluna como `KEY`, `FEATURE`, `TARGET`,
`OUTCOME` ou `METADATA`; scoring aceita somente features ativas no registry com role `FEATURE`.
`feature_version`, `universe_version` e `dataset_id` identificam contratos e lineage.

O manifesto inclui definições versionadas, feature set e checksum canônico, label selecionado,
roles, configuração e auditoria de duplicatas, nulos e desigualdades temporais. `lineage_v2`
combina checksums/schemas dos arquivos de entrada, universo, range, feature set, label, parâmetros
e versão do builder. `content_checksum` lógico e `parquet_checksum` físico são campos separados.
Manifests v1 continuam legíveis via `DatasetReference` e são marcados `legacy_content_hash`.
Consulte `docs/research_protocol.md` para o protocolo integral.

## Research component contracts — Fase 3.5 contract version 1

A feature view tem chave lógica `decision_time_ms + symbol`. Ela contém somente essas duas chaves
e a lista explícita retornada por `Strategy.required_features()`. Nomes vazios, repetidos, chaves
declaradas como features e prefixos `future_`, `outcome_` ou `label_` são inválidos. A projeção
falha diante de coluna ausente, chave nula ou duplicada.

`TrainingDataset.features` obedece ao mesmo contrato. O dataframe opcional `target` é separado,
possui ao menos uma coluna de target além das chaves e deve ter exatamente a mesma sequência de
chaves das features.

`StrategyScores.frame` tem no mínimo:

| Campo | Semântica |
|---|---|
| `decision_time_ms` | decisão causal que originou o score |
| `symbol` | contrato avaliado |
| `score` | valor numérico, finito e não nulo |

A chave é única. Campos opcionais futuros podem incluir `score_raw`, `score_rank`,
`score_confidence` e `strategy_state_json`, sem transformar outcomes em inputs.

`PortfolioPolicy.target_weights` recebe scores e uma market-state view separada, alinhadas
exatamente pela chave. A implementação neutral long/short v1 exige `rolling_beta` e
`realized_volatility_24h` e produz:

| Campo | Semântica |
|---|---|
| `decision_time_ms` | decisão correspondente ao score |
| `symbol` | contrato do peso-alvo |
| `target_weight` | fração do capital, finita, antes de custos e outcomes |

O painel de pesos precisa cobrir as mesmas decisões/símbolos do teste. O engine pode atrasar o
painel para stress de sinal, mas não permite que a política leia `future_*` ou `outcome_*`.

## Research registry — schema version 2

O banco `research.sqlite3` possui migrations próprias e 12 tabelas de domínio:
`hypotheses`, `campaigns`, `feature_definitions`, `feature_sets`, `feature_set_members`,
`experiments`, `campaign_experiments`, `experiment_runs`, `metrics`, `artifacts`,
`feature_evaluations` e `promotions`. Foreign keys são obrigatórias e migrations falham de forma
atômica.

`experiment_id` é o SHA-256 hexadecimal completo do JSON canônico de `ExperimentSpec`. A
identidade inclui `DatasetIdentity`, feature set, label, componentes/params, execução, custos,
splits, validação, seed, code fingerprint e política de artefatos. Não inclui path de manifesto,
timestamps operacionais, tentativa, métricas ou resultados.

A chave lógica de run é `experiment_id + attempt`; a tentativa é alocada transacionalmente.
Estados válidos:

```text
PENDING -> QUEUED -> RUNNING -> SUCCEEDED
             |          |  \-> FAILED
             |          \---> STALE -> QUEUED
             \--------------> CANCELLED
```

Um run `SUCCEEDED` exige `result_digest`, calculado canonicamente sobre métricas e checksums dos
artefatos. Métricas são identificadas por run, nome, scope, fold, regime e stress; artefatos por
run, tipo e path lógico. Rerun idêntico é idempotente, enquanto conteúdo divergente sob a mesma
chave é conflito.
