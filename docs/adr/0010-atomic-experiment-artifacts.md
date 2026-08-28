# ADR 0010 — Atomic experiment artifacts and registry-backed execution

## Status

Accepted for Phase 3.5 PR 5.

## Context

An experiment definition already had a stable identity and a run state machine, but execution
still wrote the Phase 3 report directly to legacy paths. Registering files one at a time before
their validation could leave a scientifically ambiguous partial result, and marking a run as
successful before durable artifacts existed would violate the research protocol.

## Decision

`ExperimentRunner` is the only registry-backed entry point for an immutable experiment. It
resolves strategies and portfolio policies through explicit allowlisted factories, reconstructs
the cost/split/validation configuration from `ExperimentSpec`, locates the local dataset by its
portable `DatasetIdentity`, and calls the existing walk-forward engine.

Each run writes first below `tmp/research/<run_id>`. JSON and Parquet files are read back, row
counts and checksums are validated, and the complete directory is renamed into the immutable
experiment/run layout. Any pre-completion failure moves the bundle to `quarantine/research`.
Only after filesystem promotion does one SQLite transaction insert metrics, insert artifact
records, persist the `result_digest`, and move the run from `RUNNING` to `SUCCEEDED`.

The digest covers canonical metrics and deterministic scientific artifacts. `manifest.json` and
the opt-in `pnl.svg` are integrity checked and registered, but excluded from the digest so adding
a visualization does not change a scientific result. A rerun of the same experiment must match
the prior successful digest; a mismatch is quarantined and recorded as a failed attempt.

The analytical source is long-form Parquet. `summary` preserves the curve and segmented metrics;
`full` additionally preserves scores and positions. Legacy `weights_json` and `scores_json` stay
in the golden curve temporarily for compatibility, not as the primary analytical representation.

## Consequences

- A partial filesystem or database write cannot appear as a successful run.
- Artifact corruption is discoverable by checksum, size, row count, and digest verification.
- Paths use 24-character display prefixes on disk to remain safe on Windows; manifests and the
  registry retain the complete 64-character IDs.
- Dataset relocation does not alter experiment identity, but rerun requires the matching local
  manifest and Parquet to be available.
- No Binance client, credential, Demo Trading, live execution, or order path is introduced.
