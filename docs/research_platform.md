# Plataforma de pesquisa — Fase 3.5

## Estado do incremento

Os PRs 1 a 3 estão implementados. O golden baseline permanece como referência e o motor agora
recebe `Strategy` e `PortfolioPolicy`, usando treino e teste reais em cada fold. Residual momentum
e neutral long/short são as primeiras implementações dos contratos. Feature/label registries,
roles de schema, dataset views e fingerprint `lineage_v2` estão ativos. A CLI legada usa essa
rota por meio de um adaptador explícito; não existe segundo motor. ResearchStore, experiment
registry, campaign runner, ledger, multiple-testing adjustment e promotion gates ainda não estão
implementados e não devem ser simulados por scripts ad hoc.

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
- `ResidualMomentumStrategy`: implementação fixa versionada, com pesos imutáveis e sem calibração
  OOS.
- `NeutralLongShortPolicy`: seleção de caudas, no-trade band, neutralização net/beta, volatility
  target, limite por símbolo e trava contra alavancagem econômica.

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
última observações e um SHA-256 do conteúdo canônico das 72 linhas OOS. O teste falha diante de
qualquer alteração numérica, inclusive em funding, custos, exposures, turnover ou accounting.

O baseline real local continua identificado por `974ab2c8643ace95`; seus artefatos e checksums
estão registrados em `PROGRESS.md`. Esses resultados são evidência de regressão e rejeição do
baseline, não uma hipótese promovida.

Após a migração do PR 2, a fixture sintética, a `run_version` real e os SHA-256 do Parquet, JSON,
Markdown e SVG permaneceram idênticos. No PR 3, o dataset schema v2 preservou todos os valores do
schema v1 exceto os campos de versão, e a curva OOS real permaneceu exatamente igual. A
`run_version` mudou porque a identidade legada do backtest ainda inclui o novo path do dataset;
essa limitação é removida pelo PR 4. A CLI `research backtest` importa o adaptador
`run_and_persist_phase3_baseline`, que constrói componentes explícitos e chama o engine genérico.

## Próximos incrementos

1. criar identidade imutável, ResearchStore e migrations;
2. criar pipeline atômico de artifacts e experiment runner;
3. criar campaign planner/runner com resume e cache;
4. adicionar ledger, ablações, robustez, multiple testing e promotion gates;
5. otimizar o painel e concluir documentação/aceite.

Nenhuma campanha extensa deve ser executada antes de existirem separação, registry, artifacts e
runner (PRs 1 a 6).
