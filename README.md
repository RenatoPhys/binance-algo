# binance-algo

Infraestrutura auditável para dados públicos da Binance USDⓈ-M Futures. O data plane atual
implementa configuração, diagnóstico, REST público, snapshots de `exchangeInfo`, universo seed,
state store SQLite e backfill idempotente dos arquivos oficiais de klines 1m. Não há estratégia,
WebSocket, autenticação, backtest ou envio de ordens.

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
- Rede liberada para `https://demo-fapi.binance.com` e `https://data.binance.vision`

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
  --interval 1m --start 2026-08-25 --end 2026-08-25
```

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
var/state/ingestion.sqlite3
var/reports/downloads_*.{json,md}
```

Raw e bronze são imutáveis. A escrita passa por arquivo temporário, `fsync`, leitura de validação
e promoção atômica. Archives exigem SHA-256 oficial, ZIP seguro, CRC, cabeçalho e largura de linha
válidos antes do estado `VALIDATED`. Downloads parciais usam `.part` e HTTP Range na retomada;
reruns de arquivos válidos não acessam a rede.

As datas de backfill são dias UTC inclusivos. Como os arquivos diários são publicados no dia
seguinte, o CLI rejeita datas dentro da janela de publicação configurada.

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

Normalizar os CSVs de klines para Parquet canônico, deduplicar, auditar gaps/invariantes e criar
views DuckDB e relatórios de qualidade. WebSocket, alpha e execução permanecem fora de escopo até
os respectivos critérios de qualidade de dados serem satisfeitos.
