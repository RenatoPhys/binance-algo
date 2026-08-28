# Plataforma de pesquisa — Fase 3.5

## Estado do incremento

Os PRs 1 e 2 estão implementados. O golden baseline permanece como referência e o motor agora
recebe `Strategy` e `PortfolioPolicy`, usando treino e teste reais em cada fold. Residual momentum
e neutral long/short são as primeiras implementações dos contratos. A CLI legada usa essa rota
por meio de um adaptador explícito; não existe segundo motor. Registry, campaign runner, ledger,
multiple-testing adjustment e promotion gates ainda não estão implementados e não devem ser
simulados por scripts ad hoc.

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
- `select_feature_view`: allowlist de features; outcomes e labels são excluídos por projeção.
- `ResidualMomentumStrategy`: implementação fixa versionada, com pesos imutáveis e sem calibração
  OOS.
- `NeutralLongShortPolicy`: seleção de caudas, no-trade band, neutralização net/beta, volatility
  target, limite por símbolo e trava contra alavancagem econômica.

As únicas chaves entregues ao scoring neste incremento são `decision_time_ms` e `symbol`. Features
precisam ser declaradas explicitamente. Prefixos `future_`, `outcome_` e `label_` falham antes da
execução da estratégia.

## Golden baseline

`tests/golden/research_phase3_synthetic.json` preserva folds, métricas, schema da curva, primeira e
última observações e um SHA-256 do conteúdo canônico das 72 linhas OOS. O teste falha diante de
qualquer alteração numérica, inclusive em funding, custos, exposures, turnover ou accounting.

O baseline real local continua identificado por `974ab2c8643ace95`; seus artefatos e checksums
estão registrados em `PROGRESS.md`. Esses resultados são evidência de regressão e rejeição do
baseline, não uma hipótese promovida.

Após a migração do PR 2, a fixture sintética, a `run_version` real e os SHA-256 do Parquet, JSON,
Markdown e SVG permaneceram idênticos. A CLI `research backtest` importa o adaptador
`run_and_persist_phase3_baseline`, que constrói componentes explícitos e chama o engine genérico.

## Próximos incrementos

1. criar feature/label registries, dataset views e fingerprint de lineage;
2. criar identidade imutável, ResearchStore e migrations;
3. criar pipeline atômico de artifacts e experiment runner;
4. criar campaign planner/runner com resume e cache;
5. adicionar ledger, ablações, robustez, multiple testing e promotion gates;
6. otimizar o painel e concluir documentação/aceite.

Nenhuma campanha extensa deve ser executada antes de existirem separação, registry, artifacts e
runner (PRs 1 a 6).
