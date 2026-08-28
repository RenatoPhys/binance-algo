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
