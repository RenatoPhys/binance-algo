# ADR 0009 — ResearchStore e identidade de experimento

- Status: accepted
- Date: 2026-08-28
- Implemented through: Phase 3.5 PR 4

## Context

O baseline produz versões determinísticas, mas não oferece um registry durável para hipóteses,
especificações imutáveis, tentativas, métricas ou artefatos. Sua identidade legada também inclui
o path de materialização do dataset, impedindo que o mesmo conteúdo tenha a mesma identidade em
outro checkout.

## Decision

- Persistir pesquisa em `var/state/research.sqlite3`, separado de ingestion e market data.
- Aplicar migrations atômicas e abrir cada conexão com WAL, foreign keys e `busy_timeout`.
- Definir `experiment_id = SHA-256(canonical_json(ExperimentSpec))`.
- Incluir na especificação hipótese, dataset portátil, features, label, componentes e parâmetros,
  execução, custos, splits, validação, seed, code fingerprint e política de artefatos.
- Rejeitar paths absolutos, números não finitos e valores sem serialização canônica.
- Representar código por commit limpo, commit + checksum do diff sujo ou checksum da árvore de
  fontes quando Git não estiver disponível.
- Manter resultados fora da identidade e calcular `result_digest` a partir de métricas e
  checksums de artefatos após execução bem-sucedida.
- Tratar experimento como imutável e rerun como nova tentativa, com transições explícitas de run.

## Consequences

- O mesmo experimento recebe a mesma identidade em roots diferentes quando conteúdo e código são
  iguais; qualquer alteração causal da especificação produz outra identidade.
- Registro repetido com o mesmo conteúdo é idempotente; colisão de ID com conteúdo diferente
  falha em vez de sobrescrever evidência.
- Definições e resultados podem ser auditados antes do campaign runner existir.
- O adaptador legado continua com `run_version` própria até ser conectado ao experiment runner no
  PR 5; não há duas implementações financeiras.
