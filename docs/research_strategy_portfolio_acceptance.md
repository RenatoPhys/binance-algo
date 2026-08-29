# Strategy portfolio dashboard — local acceptance

## Scope and provenance

Acceptance was executed on 2026-08-29 against the existing local `ResearchStore` and immutable
registered artifacts. The operational declaration remains intentionally unversioned at
`var/config/research_strategy_portfolios.yaml`; exact local experiment/run IDs are traceable in
that file and in the generated snapshot, not promoted to a repository-wide reference.

The inventory found 179 experiments with successful attempts and verified 179/179 selected
artifact bundles. The three declared portfolios passed strict compatibility and artifact
validation. Components from different code fingerprints retain an explicit provenance warning.

## Declarations and observed results

The allocations below were declared before inspecting the combined portfolio output and were not
readjusted afterward:

| Portfolio | Declaration | Total return | Sharpe | Max drawdown |
|---|---|---:|---:|---:|
| `champion_only` | confirmed/full carry multi-horizon 60/30/10 at 100% | 12.2442% | 1.0267 | -6.1402% |
| `equal_weight_comparison` | three comparable sleeves at deterministic equal weights | 10.0694% | 0.8398 | -8.7322% |
| `manual_diversified_research` | fixed 60% / 25% / 15% research allocation | 11.1066% | 1.0011 | -7.0644% |

These are exploratory observations on an already evaluated final period. They are not independent
lockbox evidence, promoted alpha, financial advice, or authorization to submit orders.

## Determinism and structural checks

Two consecutive real builds were byte-identical:

- HTML SHA-256: `5A7ACB42AEDCB4AB9EA267046B2CC0399D106E6E1D573E823287D06BAE5C6A24`
- snapshot SHA-256: `7E22EB6EC756B2C52700255A1CF340B754F5884549093B40600FB859891DD314`

The generated HTML contained 27 inline SVG charts, all with labeled X/Y axes and ticks, three
valid portfolios, no absolute filesystem paths, no external HTTP assets, no unsafe links, and no
duplicate HTML IDs. The snapshot parsed with finite JSON values.

The final local quality gates passed Ruff formatting/lint, mypy on `src`, and the complete
non-network test suite. Safety configuration and the existing synthetic backtest golden remained
unchanged.
