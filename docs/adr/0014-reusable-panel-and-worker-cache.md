# ADR 0014 — Reusable PanelData and worker-local dataset cache

## Status

Accepted for Phase 3.5 PR 9.

## Context

Campaign trials repeatedly read the same point-in-time Parquet and rebuilt dense arrays with
Python row iteration. Validation then repeated the same transformation for baseline, cost and
delay scenarios. This was correct for three symbols, but multiplied parsing and allocation work
as the local campaign grew.

The current dataset represents a fixed, ex-ante seed. It does not contain trustworthy historical
listing, delisting, liquidity or quality membership for a dynamic universe. Preparing the engine
for partial availability must not fabricate that history.

## Decision

Introduce `PanelData`, an immutable dense representation with sorted times and symbols, separate
feature/outcome/metadata mappings and an explicit availability mask. Construction uses vectorized
key-to-index mappings and rejects duplicate keys, invalid shapes and non-finite required values in
available cells. The internal universe fields are always present. Unknown listing/delisting times
use `-1`; current quality/liquidity eligibility is derived only from inclusion in the already
filtered fixed-seed dataset and is not claimed as historical metadata.

Parquet is loaded through a bounded, process-local LRU. The lazy scan projects only the strategy,
portfolio, accounting and available universe columns. Its key includes resolved path, size,
modification time and projection. A process worker therefore loads a dataset once and reuses both
the projected frame and immutable panel across its assigned trials. Validation scenarios and
walk-forward folds receive array views from that same panel.

Keep the public `Strategy` and `PortfolioPolicy` DataFrame contracts stable in this increment.
Their long-to-wide adapters are vectorized; the accounting path reuses `PanelData`. The small
stateful time loop in the no-trade-band portfolio remains intentional.

## Consequences

- Dataset parsing and point-in-time array construction are separated from parameter execution.
- Identical features are loaded once per worker and shared without mutation between trials.
- Full analytical artifacts remain opt-in; the legacy JSON curve columns remain only for golden
  compatibility while long-form Parquet is the analytical contract.
- The benchmark is observational and has no ordinary-CI SLA.
- Dynamic historical universe selection remains blocked until genuine point-in-time metadata is
  ingested; `PanelData` support does not remove that research limitation.
