# Changelog

## Unreleased — Fase 3.5

- Golden snapshot sintético e registro dos checksums do baseline real da Fase 3.
- Contratos tipados para strategy/fitted strategy, portfolio policy e contexto walk-forward.
- Feature view por allowlist, separação de targets e bloqueio de `future_*`/`outcome_*`/`label_*`.
- ADRs da plataforma/identidade e da separação entre estratégia, features e portfólio.
- Residual momentum e neutral long/short extraídos para módulos versionados e testáveis.
- Walk-forward genérico executando `fit` no treino, `score` no teste e política de portfólio
  separada.
- CLI legada migrada para o adaptador do novo engine, com golden sintético e artifacts reais
  byte a byte idênticos.
- Feature/label registries, feature sets canônicos, dataset roles e lineage v2 sem paths.
- `ResearchStore` SQLite isolado, migrations atômicas, WAL, foreign keys e state machines de
  hipóteses e runs.
- Identidade imutável de experimento por JSON canônico, code fingerprint clean/dirty/fallback e
  result digest separado.
- Registro idempotente de hipóteses, features, feature sets, experimentos, métricas e artefatos,
  com CLI de inicialização, status e inspeção.
- Artifact bundles atômicos com temp/promote/quarantine, policies `summary`/`full`, checksums e
  scores/positions long-form em Parquet.
- Experiment runner offline com factories explícitas, conclusão transacional, verify/rerun e
  confirmação determinística de `result_digest`; gráficos continuam opt-in.
- Campaign YAML estrito com grid/constraints determinísticos, guard de trials, dry-run,
  multiprocessing local, falhas isoladas, resume/cache e comparison completo.
- Feature ledger contextual e imutável, histórico por feature/hipótese e preservação de decisões
  negativas sem alterar o lifecycle global da feature.
- Runner de ablação com oito deltas orientados como `with feature - without feature`, regra
  registrada, validação de artifacts e relatórios JSON/Markdown derivados.
- Robustez por fold/regime/mês/símbolo, distribuições da campanha, vizinhança de parâmetros e
  contagem aproximada de estratégias independentes.
- DSR com skewness/kurtosis e trial count, PBO/CSCV condicional e lockbox indisponível explícita.
- Candidate reports e eventos imutáveis de promoção, bloqueio e rejeição; Git dirty não promove.

## 0.5.0 — 2026-08-28

- Funding histórico público com raw imutável, Parquet canônico, manifesto e view DuckDB.
- Dataset point-in-time com features interpretáveis, as-of funding e labels no próximo open.
- Baseline cross-sectional de momentum residual com limites de exposição e no-trade band.
- Backtest vetorizado com fees, spread, slippage, funding, turnover e invariantes contábeis.
- Walk-forward, regimes, custos 1,5×/2×, atraso, perturbações e bootstrap em blocos.
- Relatório real da janela de 90 dias preservando o resultado negativo sem tuning retrospectivo.
- Curva SVG opt-in com equity, drawdown, decomposição de custos e folds walk-forward.

## 0.4.0 — 2026-08-28

- Recorder WebSocket público resiliente em rotas `public` e `market`.
- Buffer limitado, micro-batches Parquet atômicos, recovery, métricas e health checks.
- Gate real de 60 minutos e replay 1×/100× com digest idêntico.

## 0.3.0 — 2026-08-27

- Normalização canônica de klines, auditoria de qualidade e catálogo DuckDB.
- Backfill real de 90 dias para BTCUSDT, ETHUSDT e SOLUSDT com rerun idempotente.

## 0.2.0 — 2026-08-27

- State store SQLite com WAL, migrations, rollback e state machines explícitas.
- Manifest de `data_files`, `backfill_jobs`, checkpoints, qualidade e schemas.
- Downloader concorrente e idempotente de arquivos oficiais daily klines 1m.
- SHA-256, retomada HTTP Range, limites de tamanho, extração segura e quarantine.
- CLI `backfill klines` e relatórios JSON/Markdown por job.
- Contract test real e testes de corrupção, path traversal, retry e rerun.

## 0.1.0 — 2026-08-27

- Bootstrap Python/uv com lint, typing, testes, CI e container.
- Configuração Pydantic/YAML com defaults Demo e travas contra ordens.
- Logging JSON com redaction de segredos.
- Cliente REST público para ping, server time e exchangeInfo.
- Snapshot raw + Parquet de metadados e universo seed point-in-time.
- CLI operacional e diagnóstico de ambiente.
