# Operações

## Ordem de execução

1. `uv sync`
2. `uv run binance-algo doctor`
3. `uv run binance-algo exchange-info snapshot`
4. `uv run binance-algo universe build`
5. Execute o backfill desejado:

   ```bash
   uv run binance-algo backfill klines --symbols BTCUSDT,ETHUSDT,SOLUSDT \
     --interval 1m --start 2026-05-28 --end 2026-08-25
   uv run binance-algo data normalize --dataset klines \
     --symbols BTCUSDT,ETHUSDT,SOLUSDT --interval 1m \
     --start 2026-05-28 --end 2026-08-25
   uv run binance-algo data audit --dataset klines \
     --symbols BTCUSDT,ETHUSDT,SOLUSDT --interval 1m \
     --start 2026-05-28 --end 2026-08-25
   ```

O `doctor` retorna status não zero se Python, storage, SQLite WAL, DNS, REST, clock ou travas
falharem. A ausência de credenciais é um sucesso esperado para dados públicos.

O range do backfill é inclusivo em UTC e termina no máximo no cutoff seguro de publicação. Cada
execução gera um job e relatórios JSON/Markdown. Rerun de arquivo já validado retorna `skipped`,
zero bytes e não consulta a rede.

A normalização seleciona somente archives `VALIDATED` ou já `NORMALIZED`, registra a versão do
schema e o parent file, remove chaves repetidas conservando a primeira observação, ordena e grava
Parquet imutável. No rerun, cada resultado deve ser `skipped`. O comando também recria a view
`klines` em `var/state/market_data.duckdb` a partir do manifesto.

A auditoria cobre cada partição e o range agregado por símbolo, inclusive gaps entre dias. JSON e
Markdown são gravados em `var/reports`; qualquer gate reprovado encerra o comando com status 1.

## Recorder e replay

Execute o aceite com os defaults seguros do Demo:

```bash
uv run binance-algo recorder start \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT --duration-seconds 3600 --metrics-port 9108
uv run binance-algo recorder status
```

Antes de abrir o WebSocket, o comando sincroniza o relógio via `/fapi/v1/time` e rejeita offset
acima do limite. Durante a captura, `/health/live` indica processo ativo, `/health/ready` exige
eventos recentes nas duas rotas, writer saudável e fila não saturada, e `/metrics` expõe REST,
WebSocket, queue, flush, qualidade e offset. `Ctrl+C` inicia shutdown gracioso; aguarde o relatório
antes de reiniciar.

O gate exige igualdade entre mensagens recebidas e linhas persistidas, zero drop, duplicata, gap,
regressão e valor de mercado inválido, além de checksum e schema exatos. O relatório inclui
p50/p95/p99 da latência ajustada pelo clock, staleness, conexões, reconnects, bytes e recuperação.

Depois, copie `min_event_time_ms` e `max_event_time_ms` do JSON para reproduzir o range:

```bash
uv run binance-algo replay --dataset all --start <epoch-ms> --end <epoch-ms> --speed 1
uv run binance-algo replay --dataset all --start <epoch-ms> --end <epoch-ms> --speed 100
```

O modo virtual não acessa rede nem espera tempo real. Use `--wall-clock` de modo intencional: em
1×, um range de 60 minutos leva aproximadamente 60 minutos para ser entregue ao consumidor.

## Dataset e backtest de pesquisa

Depois dos gates histórico e do recorder, execute:

```bash
uv run binance-algo --config configs/research.yaml funding sync \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT --start 2026-05-28 --end 2026-08-25
uv run binance-algo --config configs/research.yaml research build \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT --start 2026-05-28 --end 2026-08-25
uv run binance-algo --config configs/research.yaml research backtest
```

O primeiro comando acessa apenas `/fapi/v1/fundingRate`, valida símbolo/range/tipos e recria a
view `funding_rates`. O segundo falha diante de candle aberto, grid divergente, falta de funding,
timestamp causal inválido, painel incompleto, duplicata ou feature nula. O terceiro usa o dataset
mais recente por padrão; passe `--dataset <path>` para fixar uma versão explicitamente. Figuras
ficam desativadas por padrão; use `research backtest --chart` apenas nas execuções selecionadas.

Antes de interpretar o relatório, confira `accounting_error_max=0`, decomposição de custos,
turnover e estresses. Resultado líquido positivo não promove estratégia; resultado negativo não
autoriza tuning na mesma janela. Preserve o JSON e avance qualquer nova hipótese como versão
distinta de feature/configuração.

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
- Parquet ausente ou alterado: checksum falha e o gate não é promovido.
- Gap, duplicata, desordem ou OHLC inválido: o relatório preserva contagens e o CLI retorna 1.
- WebSocket stale/fechado: registra motivo, aplica backoff com jitter e reassina a mesma lista.
- Fila saturada: falha explicitamente; não continua com perda silenciosa.
- Crash depois do rename e antes do manifesto: o restart valida e promove o arquivo em voo.
- `.tmp` residual ou Parquet órfão inválido: move a evidência para `recorder_recovery`.
- Gate do recorder falhou: preserve JSON/Markdown e não avance para research/estratégia.
- Funding ausente/divergente: preserve o raw e não substitua por zero ou backward-fill.
- Dataset point-in-time falhou: não execute o backtest sobre uma versão parcial.
- Fee schedule fora da validade: atualize a hipótese versionada com fonte, não prolongue datas
  silenciosamente.

Arquivos `.tmp` pertencem somente a uma tentativa atômica. `.part` pertence a um download
retomável. Não remova manualmente os arquivos nem edite o SQLite durante uma ingestão.
