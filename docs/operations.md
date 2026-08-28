# Operações

## Ordem de execução

1. `uv sync`
2. `uv run binance-algo doctor`
3. `uv run binance-algo exchange-info snapshot`
4. `uv run binance-algo universe build`
5. Execute o backfill desejado:

   ```bash
   uv run binance-algo backfill klines --symbols BTCUSDT,ETHUSDT,SOLUSDT \
     --interval 1m --start 2026-08-25 --end 2026-08-25
   ```

O `doctor` retorna status não zero se Python, storage, SQLite WAL, DNS, REST, clock ou travas
falharem. A ausência de credenciais é um sucesso esperado para dados públicos.

O range do backfill é inclusivo em UTC e termina no máximo no cutoff seguro de publicação. Cada
execução gera um job e relatórios JSON/Markdown. Rerun de arquivo já validado retorna `skipped`,
zero bytes e não consulta a rede.

## Falhas conhecidas e resposta

- DNS/rede indisponível: o cliente encerra após retries limitados e informa o endpoint.
- HTTP 429/5xx: GET público é repetido com backoff exponencial e jitter; `Retry-After` é respeitado
  até o teto local de 60 segundos.
- HTTP 418 ou outro 4xx: falha imediata; não se agrava ban temporário com retries automáticos.
- Schema incompatível: o snapshot não é promovido a Parquet.
- Arquivo imutável divergente: falha explícita em vez de sobrescrever.
- Snapshot ausente no cutoff: o universe builder informa o comando necessário.
- Interrupção de archive: o `.part` é retomado com `Range`; se o servidor não suportar, o arquivo
  é reiniciado sem anexar bytes incorretos.
- Checksum/ZIP/schema divergente: arquivo nunca chega a `VALIDATED` e a evidência é quarantined.
- SQLite ocupado: `busy_timeout` é aplicado; a transação faz rollback em qualquer exceção.

Arquivos `.tmp` pertencem somente a uma tentativa atômica. `.part` pertence a um download
retomável. Não remova manualmente os arquivos nem edite o SQLite durante uma ingestão.
