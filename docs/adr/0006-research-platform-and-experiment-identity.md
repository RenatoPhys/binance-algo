# ADR 0006 — Plataforma de pesquisa e identidade de experimentos

- Status: accepted
- Date: 2026-08-28

## Context

A Fase 3 provou causalidade temporal, contabilidade e reprodução de um baseline específico, mas
o identificador atual é calculado junto dos resultados e inclui o path absoluto do dataset. O
motor também mistura score, política de portfólio, custos e validação. Essa estrutura não permite
comparar muitas hipóteses sem aumentar o risco de data snooping, perder resultados negativos ou
confundir uma tentativa operacional com a definição científica do experimento.

## Decision

Inserir a Fase 3.5 antes da Fase 4 e evoluir incrementalmente para um research plane offline. Uma
definição imutável de experimento terá identidade determinística calculada somente a partir de
inputs materiais canônicos: dataset por conteúdo/lineage, universo, features, label, estratégia,
portfólio, execução, custos, splits, validação, seed e proveniência de código. Paths absolutos,
timestamps de execução, métricas e checksums de resultados não participarão dessa identidade.

Outputs terão `result_digest` próprio. Hypotheses, campaigns, definitions, attempts, metrics,
artifacts, feature evaluations e promotions serão preservados em SQLite WAL e Parquet por
incrementos posteriores. Resultados negativos serão imutáveis e consultáveis no mesmo nível dos
positivos. O desenvolvimento seguirá os nove incrementos registrados em `PROGRESS.md`; campanhas
extensas permanecem proibidas antes da conclusão do runner e do registry.

O primeiro incremento cria apenas o golden baseline e os contratos compartilhados. Não altera o
motor financeiro e não antecipa schema de banco, campaign runner ou promotion gate.

## Consequences

- O baseline existente continua sendo a referência financeira até a regressão do novo motor.
- Uma reexecução poderá ser distinguida de uma nova definição de experimento.
- O melhor trial não poderá ser apresentado sem a população de trials e seu contexto.
- O custo inicial é uma migração em etapas, com compatibilidade temporária da CLI atual.
- Fase 4, Demo Trading, autenticação e envio de ordens continuam fora de escopo.
