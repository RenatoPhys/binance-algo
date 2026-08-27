# binance-algo

Infraestrutura auditável para dados públicos da Binance USDⓈ-M Futures. Este primeiro marco
implementa somente bootstrap, configuração, diagnóstico, REST público, snapshots de
`exchangeInfo` e construção reproduzível do universo seed. Não há estratégia, WebSocket,
autenticação, backtest ou envio de ordens.

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
- Rede liberada para `https://demo-fapi.binance.com` nos comandos públicos

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
```

Raw e bronze são imutáveis. A escrita passa por arquivo temporário, `fsync`, leitura de
validação e promoção atômica. O universo grava Parquet e um manifesto JSON com razões de
inclusão/exclusão.

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
contribuição. O comando `doctor`, por outro lado, verifica DNS, ping, server time e clock offset.

## Docker

```bash
docker compose build
docker compose run --rm binance-algo doctor
```

O volume `./var` preserva datasets locais. A imagem fixa as duas flags de segurança em `false`.

## Próximo marco

State store SQLite em WAL mode, manifest de arquivos/jobs e downloader idempotente dos arquivos
oficiais de klines 1m. WebSocket, alpha e execução permanecem fora de escopo até os respectivos
critérios de qualidade de dados serem satisfeitos.
