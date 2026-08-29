# Alpha Research Reboot — Wave 1

## Scope

Wave 1 adds four preregistered mechanisms without modifying the numerical baseline feature set or
its golden artifacts:

- quarter-hour opening-flow continuation;
- aggressive-flow absorption reversal;
- volatility-compression breakout;
- fold-frozen ETH/BTC and SOL/BTC spread reversion.

The shared implementation includes causal rolling operators, fold-frozen training quantiles, a
reusable sparse-signal state machine, explicit pair hedges, and trade-level artifacts for summary
runs. Every Wave 1 campaign uses `alpha_reboot_features:v1`, next-open execution, configured taker
costs, cost stresses at 1.5x and 2.0x, and a one-bar signal-delay stress.

## Trial budget and provenance

The closed grids contain 6 + 4 + 4 + 4 = 18 economic variants. The first quarter-hour campaign
attempt failed before economic evaluation because the dataset-view contract did not yet admit the
new registered schema-v3 features. The corrected campaign is named
`quarter_hour_flow_v1_development_seen_r1`; the six infrastructure failures remain visible in the
registry but are not additional DSR trials.

All candidate history is explicitly `development_seen`. The report banner is emitted in every
candidate campaign's Markdown, JSON and HTML output and in all three consolidated report formats.

## Champion resolution

`configs/alpha_reboot_wave1.yaml` pins the documented final 728-day benchmark run by exact
`run_id`. The resolver verifies the requested strategy, policy version and 60/30/10 sleeve weights.
If the selector is removed while more than one successful run has that economic identity, report
generation fails instead of selecting the highest Sharpe.

The final champion window begins after the Wave 1 development window ends. Consequently there are
no common dates on which to estimate return correlation, position correlation or fixed blends.
Those gates fail closed and the generated aligned-return artifact is intentionally empty; a
development proxy champion is not silently substituted.

## Generate the report

```bash
uv run binance-algo research wave1-report --file configs/alpha_reboot_wave1.yaml
```

The command validates the 18-trial budget and four completed campaigns before writing:

- `var/reports/alpha_reboot_wave1/report.md`, `report.json` and `report.html`;
- `candidates.parquet` with standalone/diversifier gates and DSR over all 18 trials;
- daily return and position correlation tables;
- hourly returns, daily returns, daily positions and explicitly aligned daily returns.
- strategy diagnostics for flow deciles/sign/agreement, breakout regimes, and pair beta, half-life,
  disabled folds, episode drawdown and two-leg P&L including funding.

## Result

No variant passed the standalone or diversifier gates. Quarter-hour flow produced positive gross
P&L in all six variants, but the measured edge per turnover was below explicit taker costs and all
net/stress results were negative. It remains useful only as a mechanism to validate later with
`aggTrades` and premium data. Flow absorption was not stable across its closed grid,
compression-conditioned breakout was negative in three of four variants before costs, and every
pair-spread variant was negative before and after costs.

Wave 1 therefore stops without an ensemble or automatic winner. The next authorized research work
is the separately specified premium/aggTrades data plane and structural relative value
(cash-and-carry, then calendar basis), not further technical-signal combinations.
