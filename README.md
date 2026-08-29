# binance-algo

Infraestrutura auditável para dados públicos da Binance USDⓈ-M Futures. O data plane atual
implementa configuração, diagnóstico, REST público, snapshots de `exchangeInfo`, universo seed,
state store SQLite, histórico idempotente e um recorder WebSocket resiliente com Parquet
atômico, auditoria DuckDB e replay temporal determinístico. A Fase 3 adicionou funding histórico,
dataset causal e um baseline vetorizado walk-forward. A Fase 3.5 agora evolui essa referência para
uma plataforma reproduzível de pesquisa: o golden baseline está preservado e estratégia,
portfólio e engine usam contratos separados; feature/label registries, dataset views com roles e
fingerprint de lineage v2 já estão ativos. O `ResearchStore` persiste hipóteses, componentes,
experimentos imutáveis, tentativas, métricas e artefatos por identidade canônica. O artifact
pipeline e o experiment runner promovem bundles atômicos e verificáveis; campanhas reutilizam um
`PanelData` imutável carregado com projeção de colunas uma vez por worker. A Fase 3.5 está
concluída. Não há autenticação, simulador de execução, Demo Trading ou envio de ordens.

## Segurança por padrão

- Os endpoints Demo são o padrão.
- Dados públicos funcionam sem API key.
- `LIVE_TRADING=false`, `ALLOW_ORDER_SUBMISSION=false` e notional máximo zero são invariantes.
- O processo falha ao iniciar se qualquer trava acima for relaxada.
- Não existe código de ordem, saque ou transferência neste marco.

Não coloque credenciais em YAML, argumentos, commits ou logs. Se precisar preparar o ambiente
para uma fase futura, copie `.env.example` para `.env`; o arquivo local é ignorado pelo Git.

## Requisitos

- Python 3.12 ou superior (o `uv` pode selecionar/baixar um runtime compatível)
- [`uv`](https://docs.astral.sh/uv/)
- Rede liberada para `https://demo-fapi.binance.com`, `https://fapi.binance.com` e
  `https://data.binance.vision`
- Rede liberada para `wss://demo-fstream.binance.com` durante gravações em tempo real

Docker e GNU Make são opcionais. Todos os comandos do Makefile têm um equivalente `uv` abaixo.

## Bootstrap reproduzível

```bash
uv sync
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -m "not network"
```

Se `uv` foi instalado como módulo Python e ainda não entrou no `PATH`, use `python -m uv` no
lugar de `uv` até abrir um novo terminal.

## Fluxo operacional

Execute os comandos na raiz do repositório:

```bash
uv run binance-algo doctor
uv run binance-algo exchange-info snapshot
uv run binance-algo universe build
uv run binance-algo backfill klines --symbols BTCUSDT,ETHUSDT,SOLUSDT \
  --interval 1m --start 2026-05-28 --end 2026-08-25
uv run binance-algo data normalize --dataset klines \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT --interval 1m \
  --start 2026-05-28 --end 2026-08-25
uv run binance-algo data audit --dataset klines \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT --interval 1m \
  --start 2026-05-28 --end 2026-08-25
uv run binance-algo recorder start --duration-seconds 3600 --metrics-port 9108
uv run binance-algo recorder status
uv run binance-algo --config configs/research.yaml funding sync \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT --start 2026-05-28 --end 2026-08-25
uv run binance-algo --config configs/research.yaml research build \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT --start 2026-05-28 --end 2026-08-25
uv run binance-algo --config configs/research.yaml research backtest
uv run binance-algo --config configs/research.yaml research experiment list
uv run binance-algo --config configs/research.yaml research experiment verify <experiment_id>
uv run binance-algo --config configs/research.yaml research experiment rerun <experiment_id>
uv run binance-algo --config configs/research.yaml research hypothesis create \
  --file configs/hypotheses/residual_momentum_slow.yaml
uv run binance-algo --config configs/research.yaml research campaign plan \
  --file configs/experiments/smoke_residual_momentum.yaml
uv run binance-algo --config configs/research.yaml research campaign run \
  --file configs/experiments/smoke_residual_momentum.yaml
uv run binance-algo --config configs/research.yaml research campaign status \
  smoke_residual_momentum
uv run binance-algo --config configs/research.yaml research campaign compare \
  smoke_residual_momentum
uv run binance-algo --config configs/research.yaml research campaign run \
  --file configs/experiments/residual_momentum_ablation.yaml
uv run binance-algo --config configs/research.yaml research ablation evaluate \
  residual_momentum_remove_1h
uv run binance-algo --config configs/research.yaml research feature history \
  residual_momentum_1h:v1
uv run binance-algo --config configs/research.yaml research campaign robustness \
  residual_momentum_remove_1h
uv run binance-algo --config configs/research.yaml research candidate report <experiment_id>
uv run binance-algo --config configs/research.yaml research hypothesis create \
  --file configs/hypotheses/funding_carry_v1.yaml
uv run binance-algo --config configs/research.yaml research hypothesis create \
  --file configs/hypotheses/residual_mean_reversion_v1.yaml
uv run binance-algo --config configs/research.yaml research campaign run \
  --file configs/experiments/funding_carry_discovery.yaml
uv run binance-algo --config configs/research.yaml research campaign run \
  --file configs/experiments/residual_mean_reversion_discovery.yaml
uv run binance-algo --config configs/research.yaml research dashboard build --open
uv run binance-algo --config configs/research.yaml research portfolio inventory
uv run binance-algo --config configs/research.yaml research portfolio scaffold \
  --experiment-id <experiment_id_1> --experiment-id <experiment_id_2> \
  --output var/config/research_strategy_portfolios.yaml
uv run binance-algo --config configs/research.yaml research portfolio validate \
  --file var/config/research_strategy_portfolios.yaml
uv run binance-algo --config configs/research.yaml research dashboard build \
  --portfolio-file var/config/research_strategy_portfolios.yaml --open
uv run python scripts/benchmark_campaign.py --trials 20 --workers 4
```

### Dashboard de portfólios de estratégias

O dashboard v2 combina apenas sleeves declaradas por `experiment_id`; não seleciona estratégias
nem otimiza pesos. O inventário local em `var/reports` verifica os bundles do registry e expõe
compatibilidade, janela, perfil, estágio, métricas e disponibilidade de artifacts. O scaffold
aceita somente IDs fornecidos pelo pesquisador. A configuração operacional deve permanecer em
`var/config/research_strategy_portfolios.yaml`, que é ignorada pelo Git; o exemplo versionado em
`configs/examples/research_strategy_portfolios.example.yaml` contém placeholders documentais.

`strict` exige dataset, label, execução, custos, split, frequência, símbolos, folds e timestamps
compatíveis. `intersection` é opt-in exploratório e informa toda perda de cobertura. O painel
mostra contabilidade `sleeve` e `netted`, atribuição, drawdown, correlação de retornos, similaridade
de posições e operações bar-based. Esses trade legs são reconstruções simuladas, não ordens ou
fills. O período final atual não é lockbox independente e nenhum resultado do painel promove
alpha ou autoriza trading.

### Estratégias carry diversificadas

As families `carry_multi_horizon:v1` e `carry_dual_trend:v1` combinam três carteiras neutras
construídas separadamente e rebalanceadas a cada 48 horas. A primeira usa carry mais força
relativa de 7/14 dias; a segunda usa carry, força relativa de 7 dias e tendência SMA 12/168h.
Ambas usam `buffered_three_sleeve_neutral:v1`, que exige pesos convexos somando exatamente um.

As campanhas de desenvolvimento preservam seis pesos pré-registrados, e os arquivos
`*_oos_2y_confirmation.yaml` congelam somente o vencedor antes dos 728 dias finais. Os resultados
continuam sendo development OOS: não são lockbox, recomendação financeira nem autorização para
ordens. A multi-horizon confirmou +12,24% líquido e Sharpe 1,027; a dual-trend confirmou +8,93% e
Sharpe 0,774. Ambas permaneceram positivas com custos 2×, mas os candidate gates seguem
bloqueados por proveniência/lockbox e, no caso multi-horizon, concentração por símbolo.

### Trend following long/flat

As families `multi_horizon_trend:v1` e `market_regime_trend:v1` usam somente retornos horários
causais e a policy `buffered_long_flat:v1`, que transforma sinais negativos em caixa. O segundo
mecanismo exige tendência positiva de 168h/720h no ativo e retorno médio de 720h positivo no
universo fixo, mantendo as posições por 72 horas entre rebalanceamentos.

Três hipóteses foram preservadas sem esconder falhas. O voto multi-horizon teve um vencedor de
desenvolvimento com +18,46%, mas não alcançou o Sharpe mínimo pré-registrado. O Donchian long/flat
fez +34,17% no desenvolvimento e falhou com -5,60% na confirmação. O filtro de regime teve 4/4
variantes positivas no desenvolvimento e retornou +6,54% no trecho final, Sharpe 0,318, drawdown
máximo de -21,63%, +5,06% sob custo 1,5×, +3,59% sob custo 2× e +7,70% com atraso de uma barra.
Seu bootstrap foi positivo em 62,8% e o DSR final foi 0,673. Como os 728 dias finais já foram
consultados por outras hipóteses, isso é evidência exploratória, não lockbox, recomendação
financeira ou autorização para operar.

### Rodada com requisito de Sharpe acima de 1

Uma rodada posterior fixou Sharpe líquido mínimo de 1,00 antes de cada campanha. O
`carry_multi_regime:v1` adicionou uma sleeve long/flat quase ortogonal ao carry e obteve Sharpe
1,196 no desenvolvimento, mas caiu para 0,952 nos 728 dias finais. O consenso de força relativa
rápida/lenta chegou a 0,987 e foi interrompido no desenvolvimento. Buffers de 72/96/168h tiveram
Sharpe máximo 0,821. O model average fixo dos seis pesos originais atingiu 1,068 no desenvolvimento
e 0,928 no período final.

Assim, o único resultado que permanece acima de um no período final é o
`carry_multi_horizon:v1` 60/30/10: +12,24% líquido, Sharpe 1,027, drawdown máximo -6,14% e retorno
positivo sob custo 2×. Os quase-acertos não foram arredondados nem retestados com novos pesos no
mesmo período. Isso limita mineração do holdout, mas não torna o resultado restante uma lockbox
independente.

Durante o recorder, consulte `http://127.0.0.1:9108/health/live`, `/health/ready` e `/metrics`.
Use porta `0` para selecionar uma porta efêmera em testes. A Binance separa `bookTicker` na rota
`public` e `aggTrade`, `markPrice` e `kline_1m` na rota `market`; o adapter constrói essas
assinaturas de forma determinística.

Para replay offline, use o range UTC ou epoch-ms mostrado no relatório:

```bash
uv run binance-algo replay --dataset all \
  --start 2026-08-28T02:00:00Z --end 2026-08-28T03:00:00Z --speed 1
uv run binance-algo replay --dataset all \
  --start 2026-08-28T02:00:00Z --end 2026-08-28T03:00:00Z --speed 100
```

O clock virtual é o default: ele respeita os deltas temporais sem esperar tempo de parede. Passe
`--wall-clock` somente quando um consumidor realmente precisar aguardar entre eventos. Cada
execução verifica duas leituras completas e falha se contagem ou digest divergir.

Para um cutoff explícito:

```bash
uv run binance-algo universe build --as-of 2026-08-27
```

`--as-of` representa o fim inclusivo daquele dia em UTC. A seleção usa apenas o snapshot de
metadados mais recente cujo `valid_from_ms` não ultrapasse o cutoff.

Saídas:

```text
var/data/raw/binance/usdm/exchange_info/date=YYYY-MM-DD/*.json
var/data/bronze/binance/usdm/instrument_metadata/date=YYYY-MM-DD/*.parquet
var/data/gold/binance/usdm/universe/version=<hash>/as_of=YYYY-MM-DD/*
var/data/raw_archives/binance/usdm/klines/<symbol>/1m/*.zip
var/data/raw_archives/binance/usdm/klines/<symbol>/1m/*.CHECKSUM
var/data/raw_archives/binance/usdm/klines/<symbol>/1m/extracted/*.csv
var/data/bronze/binance/usdm/klines/date=YYYY-MM-DD/symbol=<symbol>/*.parquet
var/data/raw/binance/usdm/<stream>/date=YYYY-MM-DD/hour=HH/symbol=<symbol>/*.parquet
var/data/raw/binance/usdm/funding_rates/symbol=<symbol>/*.json
var/data/bronze/binance/usdm/funding_rates/symbol=<symbol>/*.parquet
var/data/gold/binance/usdm/research_dataset/version=<hash>/*
var/data/gold/binance/usdm/research_backtest/version=<hash>/*
var/data/gold/binance/usdm/research_experiments/experiment_id=<hash>/run_id=<hash>/*
var/data/tmp/research/<run_id>/*
var/data/quarantine/research/<run_id>-*/*
var/data/quarantine/recorder_recovery/<timestamp>/*
var/state/ingestion.sqlite3
var/state/research.sqlite3
var/state/market_data.duckdb
var/reports/downloads_*.{json,md}
var/reports/normalization_*.{json,md}
var/reports/data_quality_*.{json,md}
var/reports/recorder_*.{json,md}
var/reports/research_phase3_*.{json,md}
```

Raw e bronze são imutáveis. A escrita passa por arquivo temporário, `fsync`, leitura de validação
e promoção atômica. Archives exigem SHA-256 oficial, ZIP seguro, CRC, cabeçalho e largura de linha
válidos antes do estado `VALIDATED`. Downloads parciais usam `.part` e HTTP Range na retomada;
reruns de arquivos válidos não acessam a rede. O Parquet usa schema versionado e lineage até o
ZIP de origem; a chave `symbol + interval + open_time_ms` é deduplicada de forma determinística.

As datas de backfill são dias UTC inclusivos. Como os arquivos diários são publicados no dia
seguinte, o CLI rejeita datas dentro da janela de publicação configurada.

`data audit` falha com status não zero quando encontra checksum ou schema divergente, nulos,
desordem, duplicatas, gaps, timestamps desalinhados, preços/quantidades negativos, candle aberto
ou OHLC inválido. A view persistente `klines` do DuckDB aponta somente para arquivos
`NORMALIZED` presentes no manifesto.

O recorder usa filas limitadas e nunca descarta silenciosamente. Saturação encerra a captura com
erro e incrementa `recorder_dropped_events_total`; o valor aceito é zero. Flush ocorre por linhas,
tempo ou shutdown e só atualiza o checkpoint depois da promoção atômica e do manifesto
`VALIDATED`. No restart, arquivos finais ainda em `DOWNLOADING`/`DOWNLOADED` são validados e
promovidos, órfãos válidos são manifestados e `.tmp` ou Parquet inválido vai para quarentena.
As views DuckDB `realtime_book_ticker`, `realtime_aggregate_trades`, `realtime_mark_price` e
`realtime_kline_1m` apontam exclusivamente para arquivos validados. O raw preserva
`payload_json` para reprocessamento.

Execute apenas um recorder por raiz de storage/state DB. Processos concorrentes para os mesmos
streams produziriam observações duplicadas e disputariam o checkpoint; coordenação distribuída
fica fora deste marco.

## Research baseline

O comando `funding sync` usa somente o endpoint público de market data e preserva raw JSON,
Parquet canônico, checksum e manifesto. A view DuckDB `funding_rates` deduplica revisões pela
chave `symbol + funding_time_ms + rate_type`. O rerun de conteúdo idêntico é `skipped`.

`research build` aceita somente klines fechadas, exige grid 1m idêntico entre os três símbolos e
produz uma linha por símbolo/hora. Features terminam no cutoff; entrada e labels começam no
próximo open. `research backtest` usa somente splits temporais, fecha cada fold flat e gera os
estresses descritos em [docs/research_protocol.md](docs/research_protocol.md). Passe `--chart`
para gerar, sob demanda, `research_phase3_<version>_pnl.svg` com equity gross/net, drawdown,
custos acumulados e os limites dos folds. O padrão não gera figuras, evitando I/O e artefatos
desnecessários durante sweeps com muitos backtests.

O baseline é um teste end-to-end, não uma recomendação. Na janela validada de 90 dias ele foi
negativo depois dos custos, e essa evidência foi preservada sem otimização retrospectiva.

## Performance local reproduzível

O experiment runner usa `scan_parquet().select(...)`, mantém um cache LRU por processo e
compartilha arrays read-only entre baseline, estresses e trials do mesmo worker. O benchmark
observacional padrão cobre 30 símbolos, dois anos horários, 20 features e 100 trials simples:

```bash
uv run python scripts/benchmark_panel.py
```

O relatório atômico em `var/reports/panel_benchmark.json` registra carga, tempo médio por trial,
memória aproximada do painel e tamanhos do dataset/artifact. Ele não define SLA nem roda no gate
comum. Os campos de disponibilidade preparam painéis parciais, mas o universo atual continua sendo
o seed fixo de três símbolos; não há metadata histórica suficiente para backtest de universo
dinâmico sem survivorship bias.

## Configuração

`configs/base.yaml` contém todos os defaults. Os overlays `demo.yaml`, `research.yaml` e
`universe.yaml` são mesclados sobre a base quando passados com `--config`:

```bash
uv run binance-algo --config configs/demo.yaml doctor
```

Chaves desconhecidas, URL REST sem HTTPS, símbolos duplicados e qualquer tentativa de habilitar
ordens falham com mensagem explícita.

## Teste de rede opcional

```bash
uv run pytest -m network
```

O job de qualidade exclui esse teste para não transformar indisponibilidade externa em falha de
contribuição. O comando `doctor`, por outro lado, verifica DNS, ping, server time, clock offset e
SQLite em WAL mode.

## Docker

```bash
docker compose build
docker compose run --rm binance-algo doctor
```

O volume `./var` preserva datasets locais. A imagem fixa as duas flags de segurança em `false`.

## Próximo marco

A Fase 3.5 — plataforma de experimentos e pesquisa em escala — está concluída. Golden regression,
contratos, registries/views, `ResearchStore`, artifacts atômicos, campaigns com resume/cache,
ledger contextual, DSR/PBO condicional, promotion gates e o painel reutilizável estão cobertos por
testes e documentação; consulte [docs/research_platform.md](docs/research_platform.md). A Fase 4
permanece pendente e condicionada a uma hipótese candidata e lockbox independente. Demo Trading,
alpha promovido e live permanecem fora de escopo; `LIVE_TRADING` e envio de ordens continuam
impossíveis por configuração.
