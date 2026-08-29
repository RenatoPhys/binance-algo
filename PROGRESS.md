# Progress

## Current milestone

Fase 3.5 — plataforma de experimentos e pesquisa em escala — concluída. A Fase 3 permanece como
golden baseline auditável; seu resultado negativo não foi promovido a alpha. A Fase 4 não foi
iniciada.

## Completed

- [x] Estrutura do pacote e gestão com uv
- [x] Configuração tipada e travas de segurança
- [x] Logging JSON e mascaramento de segredos
- [x] Clock e cliente REST público
- [x] Parser de instrumentos e filtros
- [x] Snapshot raw/Parquet de exchangeInfo
- [x] Universo seed point-in-time
- [x] Testes, CI, Docker e documentação operacional
- [x] State store SQLite em WAL mode e migration 1
- [x] Manifest e state machines de arquivos/jobs
- [x] Downloader daily de klines 1m com concorrência limitada
- [x] SHA-256 oficial, retomada HTTP Range e extração segura
- [x] Relatórios JSON/Markdown e idempotência em rerun
- [x] Normalização canônica CSV para Parquet com schema versionado e lineage
- [x] Deduplicação determinística e ordenação pela chave canônica
- [x] Auditoria por partição e range de gaps, schema, checksum e invariantes
- [x] Catálogo DuckDB persistente com view `klines`
- [x] Backfill de 90 dias para BTCUSDT, ETHUSDT e SOLUSDT
- [x] Gate histórico da Fase 1 aprovado e rerun integral idempotente
- [x] WebSocket resiliente nas rotas oficiais `public` e `market`
- [x] `bookTicker`, `aggTrade`, `markPrice@1s`/funding e `kline_1m` para 3 símbolos
- [x] Fila assíncrona limitada, staleness, backoff com jitter e rotação preventiva
- [x] Parquet raw atômico por micro-batch, manifesto e checkpoint transacional
- [x] Recovery de arquivos em voo/órfãos e quarentena de temporários/inválidos
- [x] Métricas Prometheus e health checks live/ready
- [x] Catálogo DuckDB e gate de qualidade do recorder
- [x] Replay temporal 1×/acelerado, clock injetável e determinismo por digest
- [x] Gate real de 60 minutos da Fase 2 aprovado
- [x] Funding histórico público com raw/Parquet/checksum e catálogo deduplicado
- [x] Dataset point-in-time horário, features versionadas e labels no próximo open
- [x] Auditorias contra look-ahead, backward-fill, painel incompleto e timestamps inválidos
- [x] Baseline cross-sectional de momentum residual com limites de exposição
- [x] Backtest vetorizado com fees, spread, slippage, funding e turnover
- [x] Walk-forward temporal, embargo, regimes, estresses e bootstrap em blocos
- [x] Relatório de performance/estabilidade da Fase 3 e rerun determinístico
- [x] Curva SVG opt-in de equity OOS, drawdown e decomposição do P&L por fold
- [x] Fase 3.5 PR 1: snapshot golden sintético e checksums do baseline real
- [x] Contratos `Strategy`, `FittedStrategy`, `PortfolioPolicy` e `FoldContext`
- [x] `TrainingDataset` com target separado e feature view por allowlist
- [x] Bloqueio contratual de `future_*`, `outcome_*` e `label_*` no scoring
- [x] ADRs 0006/0007 e documentação inicial da research platform
- [x] Fase 3.5 PR 2: residual momentum extraído como strategy fixa versionada
- [x] Neutral long/short extraído como portfolio policy independente
- [x] Engine recebe `Strategy` + `PortfolioPolicy` e executa treino/teste por fold
- [x] CLI `research backtest` usa o adaptador do engine genérico, sem rota paralela
- [x] Regressão sintética e real byte a byte equivalente ao baseline da Fase 3
- [x] Fase 3.5 PR 3: feature/label registries e feature sets canônicos
- [x] Dataset views com roles `KEY`/`FEATURE`/`TARGET`/`OUTCOME`/`METADATA`
- [x] Features atuais extraídas em módulos sem alteração dos valores financeiros
- [x] `DatasetReference` compatível com manifests v1 e sem paths na identidade
- [x] Fingerprint `lineage_v2` por inputs manifestados, schema, universo, features, label e builder
- [x] Checksums lógico do conteúdo e físico do Parquet persistidos separadamente
- [x] Fase 3.5 PR 4: `ResearchStore` SQLite isolado, migrations 1–2 e schema versionado
- [x] Hipóteses, feature registry, feature sets, experimentos, runs/tentativas, métricas e artefatos
- [x] `experiment_id` canônico sem paths, code fingerprint e `result_digest` independente
- [x] State machines transacionais, recuperação de runs stale e registro concorrente idempotente
- [x] CLI `research registry`, `research hypothesis` e `research feature`
- [x] Fase 3.5 PR 5: artifact pipeline temp/promote/quarantine e bundles imutáveis
- [x] Policies `summary`/`full`, checksums, long-form scores/positions e chart opt-in
- [x] Experiment runner pelo registry, conclusão SQLite atômica e falha explícita
- [x] CLI `research experiment list/show/rerun/verify` e baseline legado na rota registrada
- [x] Rerun com confirmação determinística de `result_digest` e detecção de corrupção
- [x] Fase 3.5 PR 6: campaign YAML estrito, expansão cartesiana e constraints determinísticas
- [x] Guard `max_trials`, dry-run sem persistência e IDs independentes de formatação/path
- [x] Process workers locais, falha isolada, estados de campanha, resume e cache verificado
- [x] CLI `research campaign plan/run/resume/status/compare` e report de todos os trials
- [x] Smoke campaign com 9 combinações possíveis, 3 válidas e 6 rejeitadas pela constraint
- [x] Fase 3.5 PR 7: feature evaluations imutáveis e decisões contextuais no registry
- [x] Histórico por feature/hipótese, campanhas relacionadas e rejeições preservadas
- [x] Ablation runner com oito deltas, artifacts verificados e regra registrada no contexto
- [x] CLI `research ablation evaluate`, `feature history` e `hypothesis history`
- [x] Fase 3.5 PR 8: relatórios por fold/regime/mês/símbolo e vizinhança de parâmetros
- [x] DSR com skew/kurtosis/trial count e PBO condicional com `NOT_APPLICABLE` explícito
- [x] Candidate report com distribuição integral da campanha e melhor trial contextualizado
- [x] Eventos auditáveis de promoção/bloqueio/rejeição e state machine de estágios
- [x] Git dirty bloqueado, lockbox ausente reportada e promoção para Fase 4 impedida
- [x] Fase 3.5 PR 9: `PanelData` imutável com features/outcomes/metadata e availability explícita
- [x] Lazy Parquet scan com column projection e cache LRU por process worker
- [x] Painel reutilizado entre trials, folds e cenários; conversões long/wide sem `iter_rows`
- [x] Modelo interno preparado para exclusão localizada sem alegar universo histórico inexistente
- [x] Benchmark slow de 30 símbolos, 2 anos horários, 20 features e 100 trials
- [x] README, architecture, operations, data contracts, research protocol, changelog e ADR 0014
- [x] Bump coerente para 0.6.0 e aceite integral da Fase 3.5
- [x] MVP exploratório: dashboard HTML/JSON offline, determinístico e sem dependências externas
- [x] Perfis `discovery`/`full`, heartbeat por trial e benchmark real de campanhas
- [x] Feature bundles explícitos dirigidos por YAML com equivalência exata do baseline
- [x] Strategies `funding_carry:v1` e `residual_mean_reversion:v1` com campanhas discovery smoke
- [x] Strategy `carry_multi_horizon:v1` e policy de três sleeves com pesos convexos estritos
- [x] Strategy `carry_dual_trend:v1` combinando carry, força relativa e tendência SMA causal
- [x] Doze variantes de desenvolvimento lucrativas, duas confirmações congeladas e validação full
- [x] DSR 0,962/0,966 nas grades completas; PBO não aplicável e candidate gates bloqueados
- [x] Policy `buffered_long_flat:v1` com pesos não negativos, inverse-vol scaling e buffer temporal
- [x] Strategies `multi_horizon_trend:v1` e `market_regime_trend:v1` com retornos causais
- [x] Screens long/flat multi-horizon, Donchian e filtro de regime preservando sucessos e falhas
- [x] Filtro de regime 168h/720h: +6,54% final, Sharpe 0,318 e +3,59% sob custos 2×
- [x] Bootstrap 62,8% e DSR final 0,673; resultado exploratório bloqueado sem nova lockbox
- [x] Strategy `carry_multi_regime:v1` e policy convexa de carry neutro + trend long/flat
- [x] Strategy `carry_consensus_strength:v1` com desacordo rápido/lento mapeado para caixa
- [x] Rodada Sharpe 1,00: overlay 1,196→0,952 e model average 1,068→0,928
- [x] Consenso 0,987 e buffers lentos 0,821 rejeitados ainda no desenvolvimento
- [x] Única confirmação acima de um permanece `carry_multi_horizon` 60/30/10, Sharpe 1,027
- [x] Dashboard v2 de portfólios declarados, offline, determinístico e sem nova stack web
- [x] Contrato YAML/schema v1 com pesos fixed/equal, resolução registry-only e artifacts verificados
- [x] Compatibilidade strict/intersection com identidade, grid, símbolos e cobertura explícitos
- [x] Contabilidade sleeve/netted reutilizando custos do backtest e reconciliação em `1e-10`
- [x] Analytics de performance, drawdown, attribution, concentração, correlação e operações simuladas
- [x] Inventário/validate/scaffold na CLI e isolamento visual de portfólios inválidos
- [x] Três configurações locais conscientes: champion, equal comparison e diversified manual
- [x] ADR 0015, exemplo documental, testes sintéticos e dashboard real sobre artifacts existentes

## Pending

- Fase 4 permanece deliberadamente não iniciada; depende de candidato robusto e lockbox
  independente.

## Blockers

- Nenhum bloqueio de implementação.
- Docker e GNU Make não estão instalados no host atual; os equivalentes `uv` são a validação
  primária.
- `uv` 0.12.7 existe em `C:/Users/User/AppData/Roaming/Python/Python39/Scripts/uv.exe`, mas esse
  diretório não está no `PATH`; os gates usam o path explícito.

## Differences observed at Phase 3.5 start

- O checkout inicial estava na branch limpa `fix/optional-pnl-chart`; `main` local estava atrás do
  remoto. Após `git fetch`, `main` avançou somente por fast-forward até o SHA inicial registrado.
- `PROGRESS.md` e `README.md` apontavam a Fase 4 como próximo marco; a especificação da Fase 3.5
  substitui esse estado e mantém a Fase 4 não iniciada.
- O dataset real da Fase 3 está disponível localmente, portanto o baseline real foi reexecutado;
  não foi necessário limitar o snapshot à fixture sintética.
- O executável `uv` não está no `PATH`. A instalação existente foi usada por path absoluto, sem
  alterar o ambiente global.

## Validation

- SHA inicial da Fase 3.5: `59dd134f13dd788bfcbd03d6f5bc4e3ac9685ab7` (`main` limpo)
- Versão inicial do pacote: `0.5.0`; runtime resolvido pelo lockfile: Python 3.14.3
- `uv sync`: passed; lockfile com 59 packages
- `uv sync --frozen`: passed; 59 packages verificados
- `ruff format --check .`: passed; 86 files
- `ruff check .`: passed
- `mypy src`: passed; 46 source files
- `pytest -m "not network"`: 78 passed, 2 deselected
- gates PR 3: `ruff format --check` em 105 arquivos, `ruff check`, mypy estrito em 62 módulos e
  `pytest -m "not network"` com 87 passed/2 deselected
- gates PR 4: `ruff format --check` em 117 arquivos, `ruff check`, mypy estrito em 70 módulos e
  `pytest -m "not network"` com 105 passed/2 deselected
- gates PR 5: `ruff format --check` em 123 arquivos, `ruff check`, mypy estrito em 74 módulos e
  `pytest -m "not network"` com 110 passed/2 deselected
- gates PR 6: `ruff format --check` em 131 arquivos, `ruff check`, mypy estrito em 77 módulos e
  `pytest -m "not network"` com 115 passed/2 deselected
- gates PR 7: `ruff format --check` em 135 arquivos, `ruff check`, mypy estrito em 79 módulos e
  `pytest -m "not network"` com 119 passed/2 deselected
- gates PR 8: `ruff format --check` em 141 arquivos, `ruff check`, mypy estrito em 83 módulos e
  `pytest -m "not network"` com 123 passed/2 deselected
- benchmark PR 9: 525.600 linhas, 30 símbolos, 17.520 horas, 20 features e 100 trials; números
  observados: carga 0,271s, total de trials 0,328s, média 3,284ms/trial, painel 99.478.560 bytes,
  Parquet 36.808.339 bytes e artifact 2.655 bytes; cache 1 hit/1 miss. O relatório fica em
  `var/reports/panel_benchmark.json` e não constitui SLA
- gates PR 9: `ruff format --check` em 145 arquivos, `ruff check`, mypy estrito em 84 módulos e
  `pytest -m "not network"` com 128 passed/2 deselected
- gates do incremento de estratégias: `ruff format --check` em 174 arquivos, `ruff check`, mypy
  estrito em 103 módulos e `pytest -m "not network"` com 164 passed/2 deselected
- gates da rodada Sharpe 1,00: `ruff format --check` em 178 arquivos, `ruff check`, mypy estrito
  em 106 módulos e `pytest -m "not network"` com 167 passed/2 deselected
- registry real: schema 4 em `research.sqlite3`, WAL e foreign keys ativos, 18 features e o
  feature set `phase3_baseline_features:v1`; registro repetido da hipótese
  `HYP-RESMOM-0002` permaneceu idempotente
- testes `network`: não reexecutados neste incremento offline; referência anterior de 2 passed
- `binance-algo doctor`: todos os checks passaram; SQLite `journal_mode=wal`
- golden sintético: `tests/golden/research_phase3_synthetic.json`, SHA-256
  `6186c36fc56e8cd40a6f66478f92fbc0ef882debed9fddd3a30a937ed6f51488`; 3 folds, 72
  períodos, retorno `0.022538662281505806`, Sharpe `112.98691088898067`, turnover `7.5` e
  digest canônico portátil da curva
  `7440e3bb9367d49175b0932621c2bc1e9506991d4258305c39bfc53dec164fe0`
- `exchange-info snapshot`: 733 instrumentos persistidos
- DuckDB: 733 linhas, 733 símbolos distintos, zero filtros tick/step ausentes
- `universe build`: BTCUSDT, ETHUSDT e SOLUSDT; rerun manteve a versão
  `3d3e2f79aef8de4e`
- archive 90d: 2026-05-28 a 2026-08-25, 270 arquivos, 388.800 linhas; 267 downloads
  novos, 3 skips, 15.878.389 bytes e zero falhas
- normalização: 270 Parquets e 388.800 linhas; zero falhas
- quality gate: 129.600 linhas por símbolo, zero gaps, duplicatas, desordem, nulos,
  preços/quantidades negativos ou OHLC inválido; schema e checksum passaram
- DuckDB: view `klines` com 388.800 linhas
- rerun integral: 270 archives e 270 Parquets `skipped`, zero bytes baixados e row count estável
- checksum contract BTCUSDT: `1651da32387a1342bdba15b28504dc4d55caee905a58fec04f52c280b1d69f7f`
- recorder real 60m (Demo): run `b44ed33456b147179435f7cd34733e4e`, 3.601,663s,
  99.847 mensagens/linhas, 1.440 arquivos, 22.038.204 bytes, queue peak 130/100.000,
  zero drops, duplicatas, gaps, regressões ou valores inválidos
- streams 60m: book ticker 54.605, aggregate trades 21.922, mark price 10.798 e kline
  12.522; checksums e schemas passaram em todos os 1.440 Parquets
- latência p95 observada: book 519,466ms, trades 512,946ms, mark 585,191ms e kline
  517,422ms; staleness máxima por símbolo reportada entre 2,626s e 27,854s
- replay 1×/100×: 99.847 eventos, digest idêntico
  `42920f3830e70d5d0541eec179be514cd5219b201da355c93568813178d1e647`, clocks
  virtuais 3.599,727s e 35,997s
- reconnect forçado em servidor fake: passed, zero loss; cancelamento gracioso, recovery e
  quarentena também cobertos por testes
- funding 90d: 270 eventos por símbolo, 810 na view deduplicada, zero duplicatas ou valores
  inválidos; rerun integral retornou `skipped`
- dataset Fase 3: versão `50f51aa076fe5426`, 5.832 linhas, 1.944 decisões horárias, painel
  completo e zero violações temporais, duplicatas ou features nulas
- backtest Fase 3: versão `974ab2c8643ace95`, 3 folds e 1.008 horas OOS; erro contábil zero,
  exposição líquida média próxima de zero e participação máxima de volume de 0,0426%
- artifacts do baseline real `974ab2c8643ace95`:
  - `var/data/gold/binance/usdm/research_backtest/version=974ab2c8643ace95/oos_curve.parquet` —
    SHA-256 `9ab038223d469e8b8563eda2468bb9c3f56cdc1fe484fe5e3bb4a566612e2d9a`
  - `var/reports/research_phase3_974ab2c8643ace95.json` — SHA-256
    `d2c3e87a9d6db5aa84daea98ac29716ece70773224d00e5614b9f603d2d7f079`
  - `var/reports/research_phase3_974ab2c8643ace95.md` — SHA-256
    `71b6c6bd546a9086b72660c59e9889edfc65df9c152b40a86a2dd473229139c7`
  - `var/reports/research_phase3_974ab2c8643ace95_pnl.svg` — SHA-256
    `c8a7a110b3b973dc13ebf76495e37f51baff5908efbfa5079a97c30d31a06a8d`
- regressão PR 2: CLI retornou novamente `974ab2c8643ace95`; os quatro SHA-256 acima permaneceram
  idênticos após a extração de strategy/policy e a ativação de `fit` no treino
- dataset schema v2: `dataset_id`
  `0a0d252ea0690ca59be2dd037dcb1087da3af33cecdb7588709705bc5579ee61`, versão curta
  `0a0d252ea0690ca5`, método `lineage_v2`, 273 inputs manifestados, 5.832 linhas e 1.944 decisões
- checksum lógico do dataset v2:
  `d509760e21b73b4bc2d7f74053d0d1cd4927068d232f686b43b45fdd6b160af2`; checksum físico do
  Parquet: `d444a60920f621bf3a0804551cc08a83b6f074d88291a0d67d85e9c7daabb8b0`
- regressão PR 3: todas as colunas/valores do dataset v1 e v2 são idênticos, exceto
  `feature_version` e `dataset_schema_version`; a curva OOS v2 (1.008 × 22) é exatamente igual à
  curva `974ab2c8643ace95` e mantém retorno -14,6882%, Sharpe -25,050 e erro contábil zero
- o run do backtest v2 é `bb685672ab78d8b5`; essa identidade pertence ao adaptador legado e ainda
  inclui o path. A identidade canônica do PR 4 usa `DatasetReference.identity_payload()` sem
  paths; o CLI legado passará a executá-la pelo novo runner no PR 5
- baseline OOS: retorno de preço somado +1,527%, funding -0,046%, custos explícitos 17,348%,
  retorno composto líquido -14,688%, max drawdown -14,711% e turnover 266,888
- estabilidade: custos 1,5×/2× produziram -21,781%/-28,285%; atraso de uma barra -12,760%;
  bootstrap em 500 amostras teve percentis 5/50/95 de -18,278%/-14,663%/-11,022%

## Risks and mitigations

- O `serverTime` embutido em `exchangeInfo` veio três dias defasado no teste real. Conforme a
  documentação oficial, o lineage usa `/fapi/v1/time`; os artefatos da primeira tentativa foram
  movidos para `var/data/quarantine/invalid_exchange_info_servertime`.
- Contract tests externos continuam opcionais porque disponibilidade da Binance não deve bloquear
  toda contribuição.
- A Binance pode substituir archives históricos. O rerun normal preserva idempotência e não
  consulta o checksum remoto; detecção periódica de replacements ainda precisa de uma política
  versionada para não sobrescrever raw silenciosamente.
- A execução de 60 minutos não sofreu disconnect espontâneo; reconnect, resubscribe e continuidade
  foram provados deterministicamente com servidor fake.
- A granularidade de 30 segundos produziu 1.440 arquivos na hora. Compactação é o primeiro cuidado
  operacional antes de ampliar duração/universo.
- O baseline apresentou rank IC médio de -0,0045 e perda em todos os regimes/estresses. Ele prova
  o pipeline e rejeita a configuração atual; não é evidência de edge nem deve ser otimizado na
  mesma janela.
- Os parâmetros específicos do baseline ainda existem em `ResearchConfig` para compatibilidade.
  Somente `baseline.py` os traduz em strategy/policy; a migração para specs próprias permanece
  pendente e não deve criar uma segunda rota de execução.
- O fee schedule é uma hipótese configurável com tier `unknown`, não uma consulta da conta. Todo
  uso financeiro exige confirmar fee/tier e manter nova vigência versionada.

## Known limitations

- Somente Binance USDⓈ-M Futures e dados públicos
- Universo inicial limitado a BTCUSDT, ETHUSDT e SOLUSDT
- Downloader atual cobre apenas arquivos daily de klines 1m; monthly e outros datasets virão em
  incrementos posteriores
- O histórico validado cobre 90 dias; ranges maiores e arquivos monthly ainda não foram testados
- A comparação cruzada com uma segunda fonte permanece `NOT_RUN`
- Um único recorder deve operar por raiz de storage/state DB; não há coordenação distribuída
- `bookTicker` não é depth nem sustenta um order book local
- O universo dinâmico continua bloqueado sem snapshots históricos de status/liquidez; o baseline
  usa apenas o seed fixado ex ante
- O backtest é bar/next-open e não modela fila, fill parcial, latência ou adverse selection
- Sem estratégia promovida, simulador orientado a eventos, autenticação ou envio de ordens
