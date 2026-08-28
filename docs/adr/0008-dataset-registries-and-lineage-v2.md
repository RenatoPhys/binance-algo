# ADR 0008 — Dataset registries, roles e lineage v2

- Status: accepted
- Date: 2026-08-28
- Implemented through: Phase 3.5 PR 3

## Context

O dataset da Fase 3 mantém features, labels, outcomes e metadados no mesmo Parquet. A projeção do
PR 2 bloqueia prefixos perigosos, mas não registra semântica, dependências ou roles. Sua identidade
também serializa todo o dataframe com `to_dicts()`, não expressa lineage das entradas e mistura a
noção de conteúdo lógico com o arquivo Parquet físico.

## Decision

- Registrar features em `FeatureDefinition` imutável, com ID contendo nome/versão, dtype,
  lookback, timestamp semantics, datasets/colunas requeridos, implementação, parâmetros e status.
- Identificar `FeatureSetSpec` por checksum canônico independente da ordem declarada.
- Registrar retorno futuro bruto e residual como `LabelDefinition` distintos.
- Declarar roles `KEY`, `FEATURE`, `TARGET`, `OUTCOME` e `METADATA` no schema v2.
- Permitir no scoring somente colunas ativas no feature registry com role `FEATURE`; targets são
  selecionados separadamente pelo label registry.
- Calcular `dataset_id` com `lineage_v2` sobre checksums/schemas dos inputs manifestados, universo,
  range, feature set, label, parâmetros e versão do builder, sempre sem paths absolutos.
- Persistir `content_checksum` lógico incremental e `parquet_checksum` físico separadamente.
- Adaptar manifests v1 como `legacy_content_hash`, sem modificar os artefatos históricos.

## Consequences

- Uma coluna não pode entrar no score apenas por ter um nome inofensivo; sua role e registro são
  obrigatórios.
- Identidade deixa de depender da materialização integral do dataframe ou da compressão Parquet.
- Alterações de fonte, schema, universo, label, feature set ou builder produzem novo dataset.
- O adaptador legado ainda calcula sua própria `run_version`, mas a identidade canônica do PR 4
  usa `DatasetReference.identity_payload()` sem paths; consulte o ADR 0009.
- Fórmulas e valores financeiros do baseline permanecem iguais; somente metadados de versão do
  dataset mudam no schema v2.
