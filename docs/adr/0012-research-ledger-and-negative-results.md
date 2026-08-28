# ADR 0012 — Research ledger and contextual negative results

## Status

Accepted for Phase 3.5 PR 7.

The phase specification suggested number 0008 for this decision. That number already belongs to
the accepted dataset-registry/lineage decision, so the ledger ADR uses the next available number
without rewriting history.

## Context

A feature is not globally good or bad. Its effect depends on hypothesis, dataset, strategy,
portfolio, costs, horizon and validation context. A Markdown note cannot enforce referential
integrity, preserve every negative result or prove which successful runs were compared.

Ablation direction is another source of ambiguity: candidate minus baseline reverses meaning for
remove-one tests. Decisions made from inconsistently oriented deltas cannot be compared.

## Decision

The SQLite registry is the source of truth for feature evaluations. Every record references a
successful run and a registered feature, has a deterministic content ID, finite optional metric,
mandatory reason, decision enum and canonical context. Records are immutable and idempotent. A
feature definition remains active when it is rejected in one context.

An ablation declaration selects exactly one baseline and one candidate within a registered
campaign. The runner rejects comparisons that differ outside strategy parameters, verifies the
monthly artifact checksum and calculates return, Sharpe, drawdown, rank IC, turnover, explicit
cost, capacity and monthly-concentration deltas. Deltas always mean `with feature - without
feature`, independently of whether the declared operation is `ADDED` or `REMOVED`.

The automatic rule is recorded in context and may suggest `SUPPORTED`, `REJECTED` or
`INCONCLUSIVE`; an explicit override requires a reason. JSON and Markdown histories are derived
reports and can be regenerated from the registry.

## Consequences

- The same feature may be supported and rejected in different contexts without contradiction.
- Negative and inconclusive results remain queryable alongside favorable ones.
- Evaluation fails for missing/non-successful runs, unknown features, ambiguous selectors,
  corrupt monthly artifacts or non-comparable experiment contexts.
- Remove-one results have one stable sign convention.
- The ledger does not promote a candidate or mutate the feature lifecycle.
