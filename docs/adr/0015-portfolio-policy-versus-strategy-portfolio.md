# ADR 0015 — PortfolioPolicy versus declared strategy portfolio

## Status

Accepted for the research portfolio dashboard increment.

## Context

`PortfolioPolicy` already converts one strategy's cross-sectional scores into symbol target
weights inside each walk-forward fold. A dashboard that combines completed strategies needs a
different abstraction: declared capital allocation among verified strategy sleeves. Conflating
the two would blur experiment identity, invite a second backtest route, and make cross-strategy
netting hard to audit.

## Decision

Keep `PortfolioPolicy` unchanged and introduce a strict, schema-versioned strategy-portfolio
declaration. Each component references an `experiment_id` and optionally a `run_id`; paths are
resolved only from `ResearchStore`. Missing run IDs select the newest successful attempt whose
registered artifacts pass checksum, size and row-count verification. Capital weights are either
fixed and sum exactly to one, or deterministic equal weights. No optimizer or automatic strategy
selection exists.

Build both accounting views from the same verified OOS grid. `sleeve` allocates capital among
already-net backtests. `netted` aggregates symbol target weights before applying the same explicit
cost-rate function used by the existing backtest. Strict alignment is the default;
`intersection` is an explicit exploratory opt-in with discarded coverage reported. All outputs
remain derived, local reports outside experiment digests.

## Consequences

- Strategy experiments, runs and immutable artifact bundles remain the source of truth.
- The existing backtest engine, safety configuration and promotion state machine are unchanged.
- Netted savings are visible without being misrepresented as independently validated alpha.
- Portfolio results are reproducible but remain exploratory until evaluated on a genuinely
  independent lockbox.
