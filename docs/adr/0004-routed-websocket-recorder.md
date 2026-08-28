# ADR 0004 — Recorder WebSocket por rotas, manifesto e replay offline

- Status: accepted
- Date: 2026-08-27

## Context

A Binance USDⓈ-M passou a separar streams públicos entre as rotas `public` e `market`, e limita o
ciclo de uma conexão. O recorder precisa tolerar disconnects e restart sem perder silenciosamente
eventos nem transformar arquivos parciais em dados válidos. O mesmo raw deve sustentar auditoria e
replay reproduzível sem rede.

## Decision

Manter duas conexões combinadas: `bookTicker` em `public`; `aggTrade`, `markPrice@1s` e
`kline_1m` em `market`. Cada conexão é descartável, identificada por UUID, rotacionada antes de 24
horas e refeita com backoff exponencial, jitter e resubscribe determinístico.

Os parsers produzem contratos Polars estritos e preservam o JSON canônico. Eventos entram numa
`asyncio.Queue` limitada; timeout ao enfileirar encerra a captura e incrementa a métrica de drop.
O writer particiona por tipo, hora e símbolo, promove Parquet por rename atômico, registra o
manifesto e só depois atualiza o checkpoint.

No restart, `.tmp` é quarantined; Parquet final em estado incompleto é revalidado e promovido;
órfão válido é manifestado; órfão inválido é quarantined. O catálogo DuckDB usa somente os paths
validados no SQLite. O replay ordena todos os tipos por uma chave total estável e recebe um clock
real ou virtual por injeção.

## Consequences

- Falha de backpressure é explícita e pode interromper disponibilidade para preservar integridade.
- A separação de rotas adiciona uma conexão, mas acompanha o contrato oficial vigente.
- Microarquivos continuam possíveis em execuções curtas; compactação permanece um marco futuro.
- `bookTicker` oferece melhor bid/ask, não profundidade nem posição em fila.
- Replay reproduz a sequência observada, mas não modela fills, latência de ordem ou mercado.
