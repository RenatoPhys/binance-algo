# Plataforma de pesquisa — Fase 3.5

## Estado do incremento

Os PRs 1 a 9 estão implementados e a Fase 3.5 está concluída. O golden baseline permanece como referência e o motor agora
recebe `Strategy` e `PortfolioPolicy`, usando treino e teste reais em cada fold. Residual momentum
e neutral long/short são as primeiras implementações dos contratos. Feature/label registries,
roles de schema, dataset views e fingerprint `lineage_v2` estão ativos. A CLI legada usa essa
rota por meio de um adaptador explícito; não existe segundo motor. `ResearchStore`, experiment
registry, code fingerprint, artifact pipeline, experiment/campaign runners, feature ledger,
multiple-testing adjustment, promotion gates, `PanelData` reutilizável, column projection e cache
por worker estão ativos. A Fase 4 não foi iniciada.

## Fronteiras arquiteturais

O fluxo alvo separa estas decisões:

```text
dataset point-in-time
        |
        +-> feature view -> Strategy.fit(train) -> FittedStrategy.score(test)
        |                                             |
        |                                             v
        +-> market state --------------------> PortfolioPolicy
                                                      |
                                                      v
                              engine -> costs/accounting -> validation
                                                      |
                                                      v
                                      registry + immutable artifacts
```

`Strategy` não escolhe fees, slippage, leverage ou regras contábeis. `PortfolioPolicy` não recebe
labels. O engine recorta treino/teste, aplica embargo, cria `FoldContext` e entrega apenas as views
permitidas. Cada fold chama `fit` exclusivamente no treino; o objeto congelado pontua o teste e a
política converte esses scores em pesos antes da contabilidade.

## Contratos disponíveis

- `FoldContext`: limites temporais ordenados, fold, embargo e seed.
- `TrainingDataset`: feature frame causal e target opcional separado, com chaves alinhadas.
- `StrategyScores`: `decision_time_ms + symbol + score`, sem duplicatas ou valores não finitos.
- `Strategy`/`FittedStrategy`: interface para estratégias fixas ou treináveis.
- `PortfolioPolicy`: interface independente para converter scores em pesos-alvo.
- `FeatureDefinition`/`FeatureSetSpec`: identidade versionada, dependências, parâmetros, status e
  checksum canônico; a ordem declarada é reportada, mas não altera identidade.
- `LabelDefinition`: semântica explícita de horizonte, lag de execução e coluna target. Retorno
  futuro bruto e residual são labels distintos.
- `DatasetSchema`: roles `KEY`, `FEATURE`, `TARGET`, `OUTCOME` e `METADATA`.
- `build_feature_view`: resolve somente features ativas no registry e exige role `FEATURE`.
- `build_target_view`: seleciona um único label registrado em dataframe separado.
- `DatasetReference`: referência portátil por conteúdo/lineage, sem path em sua identidade.
- `ExperimentSpec`: composição imutável de hipótese, dataset, features, label, strategy,
  portfólio, execução, custos, splits, validação, seed, código e política de artefatos.
- `CodeFingerprint`: commit Git limpo, commit + SHA-256 do diff sujo ou hash determinístico da
  árvore de fontes quando Git não está disponível.
- `ResearchStore`: registry SQLite transacional de hipóteses, componentes, experimentos,
  tentativas, métricas e artefatos.
- `ResidualMomentumStrategy`: implementação fixa versionada, com pesos imutáveis e sem calibração
  OOS.
- `NeutralLongShortPolicy`: seleção de caudas, no-trade band, neutralização net/beta, volatility
  target, limite por símbolo e trava contra alavancagem econômica.
- `PanelData`: arrays read-only separados em features, outcomes e metadata, mapa estável de
  símbolos e availability explícita para reutilização entre trials.
- `WorkerDatasetCache`: LRU process-local que usa lazy Parquet scan e projeção exata das colunas
  requeridas pelo spec e pela contabilidade.

As únicas chaves entregues ao scoring são `decision_time_ms` e `symbol`. Features precisam ser
declaradas explicitamente, existir no registry e ter role `FEATURE`. A checagem de prefixos
permanece como defesa adicional; targets e outcomes falham por role mesmo se tentarem se
apresentar como features.

## Dataset identity e compatibilidade

O schema v2 mantém as colunas e fórmulas financeiras do dataset anterior, adicionando roles e
metadados de registry. Seu `dataset_id` usa `lineage_v2`: checksums e schemas dos arquivos
canônicos de entrada, universo, range consultado, feature set, label, parâmetros e versão do
builder. O checksum lógico do conteúdo e o checksum físico do Parquet são persistidos
separadamente. O cálculo é incremental e não materializa todo o dataset com `to_dicts()`.

`load_dataset_reference` continua lendo manifests do schema v1. Eles recebem o marcador
`legacy_content_hash` e um checksum do Parquet vizinho, sem reescrita do artefato legado.

## Golden baseline

`tests/golden/research_phase3_synthetic.json` preserva folds, métricas, schema da curva, primeira e
última observações e um SHA-256 portátil do conteúdo canônico das 72 linhas OOS. A
canonicalização preserva 12 casas decimais para eliminar apenas ruído numérico entre CPUs;
mudanças materiais em funding, custos, exposures, turnover ou accounting continuam falhando.

O baseline real local continua identificado por `974ab2c8643ace95`; seus artefatos e checksums
estão registrados em `PROGRESS.md`. Esses resultados são evidência de regressão e rejeição do
baseline, não uma hipótese promovida.

Após a migração do PR 2, a fixture sintética, a `run_version` real e os SHA-256 do Parquet, JSON,
Markdown e SVG permaneceram idênticos. No PR 3, o dataset schema v2 preservou todos os valores do
schema v1 exceto os campos de versão, e a curva OOS real permaneceu exatamente igual. A
`run_version` mudou porque a identidade legada do backtest ainda inclui o novo path do dataset.
A identidade canônica criada no PR 4 remove essa dependência usando o payload portátil do
`DatasetReference`. O PR 5 conectou a CLI `research backtest` ao `ExperimentRunner`; ela registra
o baseline de compatibilidade e chama o mesmo engine genérico usado pelos experimentos.

## Identidade e ResearchStore

`experiment_id` é o SHA-256 do JSON canônico do `ExperimentSpec`. Chaves são ordenadas, enums e
timestamps são normalizados, decimais recebem representação estável e valores não finitos ou
paths absolutos são rejeitados. Trocar dataset, feature set, label, parâmetros, custos, splits,
seed, commit ou diff cria outra identidade. Métricas e checksums de artefatos não fazem parte
dela: após sucesso, eles formam um `result_digest` separado.

`var/state/research.sqlite3` é independente do manifesto de ingestão. As migrations 1–4 criam 12
tabelas de domínio e seus índices, com WAL, foreign keys, `busy_timeout` e rollback atômico. Um
experimento é imutável; rerun aloca uma nova tentativa. Runs seguem
`PENDING -> QUEUED -> RUNNING -> SUCCEEDED|FAILED|STALE`, com cancelamento antes da execução e
retomada explícita de `STALE`. Sucesso exige métricas, artifacts e `result_digest` inseridos na
mesma transação final.

O registry pode ser inicializado ou migrado repetidamente. Features embutidas, feature sets e
hipóteses são idempotentes quando o conteúdo imutável coincide e falham diante de conflito.

## Artifact pipeline e execução

`ExperimentRunner` resolve somente strategy/policy registradas, reconstrói custos e splits a
partir do spec e localiza o dataset pelo conteúdo. O motor continua único. Cada tentativa escreve
em `tmp/research/<run_id>`, valida o bundle e promove o diretório inteiro para:

```text
gold/binance/usdm/research_experiments/
  experiment_id=<prefixo-24>/run_id=<prefixo-24>/
    manifest.json
    experiment_spec.json
    metrics.json
    report.md
    oos_curve.parquet
    fold_metrics.parquet
    regime_metrics.parquet
    monthly_metrics.parquet
    symbol_metrics.parquet
    positions.parquet  # full
    scores.parquet     # full
    pnl.svg            # opt-in
```

O registry retém os IDs completos. `summary` não materializa scores/positions; `full` usa schemas
long-form. Falhas antes da conclusão vão para quarantine e ficam `FAILED`. `experiment verify`
recalcula checksums, sizes, row counts e `result_digest`; `experiment rerun` cria novo attempt e
exige digest idêntico ao sucesso anterior. O SVG opt-in e o manifest operacional são verificados,
mas não alteram o digest científico.

## Campaign planner e runner

O schema YAML é estrito e aceita grid cartesiano determinístico para strategy/portfolio,
parâmetros fixos, constraints de soma, `max_trials`, policy de artifacts e controle local de
workers. O planner resolve o manifest para `DatasetIdentity`; o path fica apenas no payload
operacional armazenado para resume. Formatação/comentários YAML e relocação de conteúdo idêntico
não alteram IDs.

`campaign plan` e `campaign run --dry-run` não persistem estado. `campaign run` registra primeiro
todos os trials válidos e então usa processos locais com conexões SQLite independentes. Cache hit
exige run `SUCCEEDED` e verificação integral dos artifacts. Falhas ficam isoladas quando
`fail_fast=false`; uma campanha `PARTIAL` pode ser retomada sem repetir sucessos. O compare agrega
todos os trials em Parquet/JSON/Markdown e mantém resultados negativos visíveis.

O smoke `configs/experiments/smoke_residual_momentum.yaml` possui nove combinações possíveis,
três válidas e seis removidas pela constraint de soma dos pesos. Gráficos permanecem desligados e
os artifacts usam policy `summary`.

## Feature ledger e ablações

`research_feature_evaluations` preserva decisões imutáveis por run, feature, métrica e contexto.
Um registro exige run `SUCCEEDED`, feature existente, valor finito quando presente e motivo não
vazio. Seu ID é derivado do conteúdo canônico; repetir a mesma avaliação é idempotente. Decisões
`SUPPORTED`, `REJECTED` ou outras nunca alteram automaticamente o status global da definição.

Campanhas podem declarar pares de ablação por seletores de parâmetros. O runner exige baseline e
candidate únicos, verifica que os specs diferem apenas nos parâmetros da strategy e valida o
checksum de `monthly_metrics.parquet`. Os oito deltas usam uma convenção única: **com feature
menos sem feature**, inclusive em `REMOVED`. A regra, runs, tags, métricas de origem, folds e
stress de custos ficam em `context_json`.

`research ablation evaluate` grava as avaliações no registry e gera uma visão JSON/Markdown.
`research feature history` e `research hypothesis history` regeneram o histórico completo,
incluindo rejeições. O exemplo `residual_momentum_ablation.yaml` testa remoção do componente 1h
com redistribuição declarada do peso; ele registra evidência contextual, não promoção de alpha.

## Robustez, múltiplos testes e promoção

O relatório `campaign robustness` verifica checksums e agrega os artifacts por fold, regime, mês e
símbolo. Ele mostra retorno gross/net, pior fold/regime, concentração, custos/atraso, distribuições
da campanha e a vizinhança normalizada do trial de maior Sharpe. Esse trial é sempre rotulado como
selecionado; nunca é apresentado como evidência OOS independente.

O DSR desanualiza os Sharpes, usa o número explícito de trials, a dispersão observada entre eles e
a correção por skewness/kurtosis. Amostras insuficientes falham. O PBO/CSCV exige ao menos oito
trials comparáveis, oito segmentos pares e duas observações por segmento; abaixo disso o relatório
usa `NOT_APPLICABLE` com motivo. O número aproximado de estratégias independentes vem do espectro
da matriz de correlação dos retornos.

Estágios são `DISCOVERY`, `CANDIDATE`, `LOCKBOX_EVALUATED`, `PHASE4_CANDIDATE`, `REJECTED` e
`INVALIDATED`. Uma tentativa cria evento `APPROVED` ou `BLOCKED`; rejeição explícita cria evento
imutável. Gates de candidate verificam run/artifacts, Git limpo, hipótese pré-registrada,
performance líquida, folds, concentrações, estresses, vizinhança, DSR/PBO e campanha completa.

`lockbox_manifest` permanece `null`: os 90 dias atuais participaram do desenvolvimento e não são
uma lockbox legítima. O estado é `NOT_AVAILABLE`, e promoção para Fase 4 fica bloqueada até existir
dataset/período independente e evento `LOCKBOX_EVALUATED` aprovado.

## Próximos incrementos

1. pré-registrar screenings pequenos de momentum lento, funding carry e mean reversion residual;
2. ampliar histórico e capturar metadata point-in-time para um universo dinâmico legítimo;
3. reservar uma lockbox independente antes de qualquer avaliação para Fase 4.

Campanhas extensas continuam condicionadas aos guards, protocolo de pesquisa e gates de promoção;
o ledger não transforma screening repetido em evidência independente. A Fase 4 permanece pendente;
não há strategy promovida, simulator, autenticação ou envio de ordens.
