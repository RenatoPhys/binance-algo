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
