# Operações

## Ordem de execução

1. `uv sync`
2. `uv run binance-algo doctor`
3. `uv run binance-algo exchange-info snapshot`
4. `uv run binance-algo universe build`

O `doctor` retorna status não zero se Python, storage, DNS, REST, clock ou travas falharem. A
ausência de credenciais é um sucesso esperado para dados públicos.

## Falhas conhecidas e resposta

- DNS/rede indisponível: o cliente encerra após retries limitados e informa o endpoint.
- HTTP 429/5xx: GET público é repetido com backoff exponencial e jitter; `Retry-After` é respeitado
  até o teto local de 60 segundos.
- HTTP 418 ou outro 4xx: falha imediata; não se agrava ban temporário com retries automáticos.
- Schema incompatível: o snapshot não é promovido a Parquet.
- Arquivo imutável divergente: falha explícita em vez de sobrescrever.
- Snapshot ausente no cutoff: o universe builder informa o comando necessário.

Arquivos `.tmp` pertencem somente a uma tentativa de escrita e são removidos quando a própria
tentativa falha. Recovery sistemático e quarantine entram com o state store/manifest.
