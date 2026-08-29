# Changelog

## Alpha Research Reboot — Wave 1

- Added the causal `alpha_reboot_features:v1` set, four preregistered strategy families and the
  hedge-preserving pair-spread portfolio policy.
- Added summary-level trade events, trade distributions, daily positions, and pair fit/P&L
  diagnostics without changing legacy baseline artifacts.
- Added a strict 18-trial consolidated report with explicit champion `run_id`, DSR, standalone and
  diversification gates, immutable negative results, and `DEVELOPMENT_SEEN` banners.
- Wave 1 found no standalone or diversifier pass; further technical-signal ensembles remain
  blocked in favor of premium/aggTrades data and structural relative-value research.

## Unreleased — MVP de exploração de estratégias

- Dashboard schema v2 para portfólios de estratégias declarados, preservando catálogo, falhas e
  resultados negativos quando nenhum portfolio file é fornecido.
- Schema YAML estrito, loader registry-only com fallback para o último sucesso verificado,
  compatibilidade strict/intersection e provenance por run/digest/checksum.
- Contabilidade comparável `sleeve`/`netted` com o mesmo helper de custos explícitos do backtest,
  fechamento por fold, reconciliações e métricas de economia de netting.
- Analytics e HTML executivo para equity/drawdown/rolling, attribution, correlação/similaridade,
  concentração, turnover, operações simuladas e estabilidade por mês/fold/regime.
- CLI `research portfolio inventory|scaffold|validate`, configuração operacional ignorada em
  `var/config`, schema/exemplo versionados e ADR 0015.
- Dashboard HTML offline e snapshot JSON determinístico, com filtros, ordenação, links relativos,
  falhas e resultados negativos preservados.
- Perfil de campanha `discovery` restrito a baseline, custo `1.5x`, atraso de uma barra e artifacts
  `summary`; bootstrap, DSR/PBO, artifacts completos e promoção ficam bloqueados.
- Heartbeat periódico durante trials e benchmark observacional pelo caminho real de campanhas.
- Feature bundles explícitos para returns/momentum, volatilidade, volume, microestrutura e funding,
  dirigidos por `configs/feature_sets/phase3_baseline.yaml` sem alterar o golden baseline.
- Strategies fixas `funding_carry:v1` e `residual_mean_reversion:v1`, com schemas estritos,
  hipóteses e campanhas pequenas de descoberta sem promoção automática.
- Strategies `carry_multi_horizon:v1` e `carry_dual_trend:v1`, com carry e dois sleeves de
  tendência causal construídos separadamente.
- Policy `buffered_three_sleeve_neutral:v1`, pesos convexos estritos e combinação de target
  weights depois da neutralização/risk scaling de cada sleeve.
- Duas grades pré-registradas de seis pesos, confirmações congeladas nos 728 dias finais e
  validação full: +12,24%/Sharpe 1,027 e +8,93%/Sharpe 0,774, ambas positivas a custo 2×.
- Relatórios de robustez com DSR 0,962/0,966; PBO explicitamente não aplicável com seis trials e
  candidate gates preservados como bloqueados, sem promoção ou lockbox artificial.
- Strategies de trend following `multi_horizon_trend:v1` e `market_regime_trend:v1`, além da
  policy long/flat `buffered_long_flat:v1`, sem ponta vendida e com rebalanceamento amortecido.
- Três screens pré-registrados de tendência: o voto multi-horizon não atingiu o Sharpe mínimo; o
  Donchian long/flat venceu no desenvolvimento e perdeu 5,60% no trecho final; o filtro de regime
  confirmou +6,54%, Sharpe 0,318 e permaneceu positivo com custos 2× (+3,59%).
- Validação full do filtro de regime com bootstrap positivo em 62,8%, DSR final 0,673 e quatro de
  quatro vizinhos de desenvolvimento positivos; resultado preservado como exploratório e não
  promovido porque o período final já foi reutilizado e não existe lockbox independente.
- Nova rodada com requisito pré-registrado de Sharpe 1,00: `carry_multi_regime:v1`,
  `carry_consensus_strength:v1`, buffers lentos e model average dos seis pesos originais.
- O overlay de regime atingiu Sharpe 1,196 no desenvolvimento, mas 0,952 no período final; o model
  average atingiu 1,068 e depois 0,928. Consenso (0,987) e buffers lentos (0,821) falharam antes da
  avaliação final. Somente o `carry_multi_horizon` 60/30/10 original permanece confirmado acima
  de um, com Sharpe 1,027.
- Sleeves de consenso totalmente zeradas agora mapeiam deterministicamente para caixa em vez de
  abortar a policy de duas sleeves; a tentativa técnica falha anterior permanece no registry.

## 0.6.0 — 2026-08-28 — Fase 3.5

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
- `PanelData` imutável com namespaces de features/outcomes/metadata, availability explícita e
  suporte estrutural a painéis parciais sem fabricar histórico de universo.
- `scan_parquet` com projeção de colunas, cache LRU por worker e reutilização do painel nos
  cenários de validação e trials de campanha.
- Conversões long/wide, z-scores e materialização long-form vetorizadas; loop stateful do
  no-trade band preservado intencionalmente.
- Benchmark não bloqueante de 30 símbolos, dois anos horários, 20 features e 100 trials, com
  runtime, memória aproximada e tamanho de artifacts.
- Documentação e aceite final da Fase 3.5 concluídos; Fase 4 e qualquer envio de ordens permanecem
  fora de escopo.

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
