# ADR 0007 — Separação entre estratégia, features e portfólio

- Status: accepted
- Date: 2026-08-28
- Implemented through: Phase 3.5 PR 2

## Context

O baseline residual momentum atual calcula scores e pesos dentro do motor. O treino definido nos
folds não é entregue a uma interface geral de `fit`, e uma estratégia nova receberia facilmente
labels ou outcomes se o dataframe completo fosse reutilizado. Estratégia, construção de posição
e contabilidade precisam evoluir sem alterar a semântica financeira já validada.

## Decision

Adotar contratos estruturais, tipados e sem dependência de adapters Binance:

- `Strategy` declara ID, versão, features e target, e recebe um `TrainingDataset` no `fit`;
- `FittedStrategy` recebe somente uma feature view projetada e produz `StrategyScores` long-form;
- `PortfolioPolicy` transforma scores e market state em pesos-alvo, sem conhecer o treinamento;
- `FoldContext` transporta limites imutáveis de treino/teste, embargo e seed;
- targets de treino permanecem em dataframe separado das features;
- `select_feature_view` projeta apenas `decision_time_ms`, `symbol` e features declaradas;
- nomes `future_*`, `outcome_*` e `label_*` são proibidos como features.

O contrato de score exige chave completa, ausência de duplicatas e `score` finito. O PR 2 conecta
esses contratos ao motor, extrai residual momentum e neutral long/short e cria `baseline.py` como
adaptador explícito da configuração legada. A rota CLI continua com o mesmo comando, mas executa
o ciclo genérico `fit -> score -> target_weights`. O aceite exige regressão numérica e checksums
idênticos nos artefatos históricos.

## Consequences

- Vazamento por colunas auxiliares deixa de depender apenas de convenção de chamada.
- Estratégias fixas e treináveis compartilham o mesmo ciclo explícito `fit -> score`.
- Política de portfólio pode ser testada independentemente de sinal, custos e contabilidade.
- O engine não conhece nomes de features nem pesos de residual momentum.
- A configuração global específica da estratégia permanece somente no adaptador de compatibilidade
  e será migrada para specs estritas em incremento posterior.
- O PR 3 complementa a allowlist com registries e roles persistidos de schema; prefixos continuam
  como defesa adicional.
