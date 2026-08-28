# ADR 0011 — Deterministic local campaigns, cache, and resume

## Status

Accepted for Phase 3.5 PR 6.

## Context

Running parameter variants manually loses the number of trials attempted, makes negative results
easy to omit, and cannot distinguish a completed cached experiment from an interrupted attempt.
Campaign identity must also remain independent of YAML formatting and absolute dataset paths.

## Decision

Campaign YAML is parsed by a strict Pydantic schema. The planner resolves the dataset manifest to
`DatasetIdentity`, canonicalizes fixed/grid parameters and constraints, and fingerprints the code
before calculating `campaign_id`. Operational worker settings and the manifest locator are stored
for resume but excluded from scientific identity. Cartesian axes and expanded trials are sorted;
constraints run before experiment registration, and `max_trials` guards the valid count.

The coordinator registers every valid immutable experiment and its ordinal/tags before execution.
Workers receive serializable specs and open independent SQLite connections. A failed trial is
recorded independently and does not cancel peers when `fail_fast=false`. A successful run is a
cache hit only after all artifacts and its result digest verify. `PARTIAL` campaigns can resume;
successful trials are not duplicated and failed/missing trials receive new attempts.

Campaign reports contain every valid trial, including failed and poor results. They are mutable
aggregates over immutable runs and are atomically replaced after execution/resume. Ranking is a
view, never a deletion or promotion decision.

## Consequences

- Comments and YAML key order do not change campaign or experiment IDs.
- Relocating an identical dataset does not change identity.
- Changing code, data, strategy, portfolio, costs, validation, or constraints changes identity.
- Local multiprocessing is bounded and uses no distributed framework.
- The smoke campaign has nine possible combinations, three valid trials, and six constraint
  rejections; rerun must yield three cache hits.
- No Demo/live order path, credential, or trading authentication is introduced.
