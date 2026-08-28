# ADR 0013 — Multiple testing, lockbox, and promotion events

## Status

Accepted for Phase 3.5 PR 8.

## Context

Ranking a campaign selects an extreme from repeated trials. Reporting that trial as if it were an
independent test overstates evidence. A promotion also needs durable reasons and must not mutate
the experiment definition. The current 90-day history was used for development and cannot be
renamed a lockbox after results are visible.

## Decision

Campaign robustness reports retain all trials and distributions, verify fold/regime/month/symbol
artifacts, inspect the selected point's numeric parameter neighborhood and estimate the effective
strategy count from the return-correlation spectrum.

The Deflated Sharpe Ratio is implemented in an isolated module. Annualized Sharpes are converted
to the observation frequency, the expected maximum across the explicit trial count is calculated,
and the probabilistic Sharpe statistic includes observed skewness and Pearson kurtosis. Fewer than
30 observations, non-finite inputs or zero variance fail explicitly.

PBO uses combinatorially symmetric cross-validation only with at least eight comparable trials,
an even minimum of eight segments and at least two observations per segment. Otherwise its state
is `NOT_APPLICABLE` with a reason; no placeholder probability is emitted.

Promotion is an append-only event. Candidate gates cover successful/integral artifacts, clean Git
provenance, preregistration, net performance, fold/month/symbol stability, cost and delay stress,
parameter neighborhood, DSR and full campaign context. A failed request is recorded as `BLOCKED`
without changing stage. Explicit rejection is also an immutable event.

`lockbox_manifest: null` means `NOT_AVAILABLE`. Without an independent dataset or period and a
prior `LOCKBOX_EVALUATED` stage, Phase 4 promotion is blocked.

## Consequences

- The best trial is always shown with trial counts and distributions.
- Dirty experiments and dirty promotion code cannot advance to candidate/lockbox stages.
- Small campaigns get DSR but an explicit non-applicable PBO state.
- Current development data cannot be silently accessed or relabeled as lockbox.
- Promotion/rejection history is auditable and experiment specs remain immutable.
