from __future__ import annotations

import math
from pathlib import Path

import polars as pl

from binance_algo.config import load_settings
from binance_algo.research.backtest import run_walk_forward
from binance_algo.research.visualization import render_pnl_svg

PROJECT_ROOT = Path(__file__).parents[2]
BASE_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
START_MS = 1_767_225_600_000


def _research_frame(days: int = 11) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for hour in range(days * 24):
        decision = START_MS + hour * 3_600_000 + 3_599_999
        common_volatility = 0.012 + 0.002 * math.sin(hour / 24)
        for symbol_index, symbol in enumerate(SYMBOLS):
            phase = math.sin(hour / 9 + symbol_index)
            residual_1h = 0.002 * phase
            rows.append(
                {
                    "decision_time_ms": decision,
                    "execution_time_ms": decision + 1,
                    "label_end_time_ms": decision + 3_600_001,
                    "symbol": symbol,
                    "residual_momentum_1h": residual_1h,
                    "residual_momentum_4h": residual_1h * 2 + symbol_index * 0.0001,
                    "residual_momentum_24h": residual_1h * 4 - symbol_index * 0.0001,
                    "realized_volatility_24h": common_volatility * (1 + symbol_index * 0.1),
                    "rolling_beta": 0.8 + symbol_index * 0.25,
                    "future_return_1h": 0.0015 * phase - 0.0002 * symbol_index,
                    "future_residual_return_1h": 0.0012 * phase,
                    "outcome_funding_rate_1h": (
                        0.0001 * (symbol_index + 1) if hour % 8 == 7 else 0.0
                    ),
                    "outcome_quote_volume_1h": 100_000_000.0,
                    "market_volatility_regime": common_volatility * math.sqrt(365),
                }
            )
    return pl.DataFrame(rows)


def test_walk_forward_is_temporal_costed_and_accounting_balances() -> None:
    settings = load_settings(BASE_CONFIG)
    config = settings.research.model_copy(
        update={"walk_forward_train_days": 7, "walk_forward_test_days": 1}
    )
    frame = _research_frame()
    baseline = run_walk_forward(frame, config=config)
    expensive = run_walk_forward(frame, config=config, cost_multiplier=2.0)
    delayed = run_walk_forward(frame, config=config, signal_delay_bars=1)

    assert baseline.folds
    assert all(fold.train_end_ms < fold.test_start_ms for fold in baseline.folds)
    assert all(fold.embargo_bars == 1 for fold in baseline.folds)
    assert baseline.metrics.accounting_error_max <= 1e-15
    assert baseline.metrics.trading_fees > 0
    assert baseline.metrics.turnover > 0
    assert baseline.curve["gross_exposure"].max() <= 0.5 + 1e-12
    assert baseline.curve["net_exposure"].abs().max() <= 1e-12
    assert expensive.metrics.total_return < baseline.metrics.total_return
    assert not delayed.curve["net_return"].equals(baseline.curve["net_return"])

    svg = render_pnl_svg(baseline.curve)
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "Equity OOS" in svg
    assert "Drawdown líquido" in svg
    assert "Decomposição acumulada" in svg
    assert svg.count("<polyline") == 7
    assert all(f"fold {number}" in svg for number in range(1, len(baseline.folds) + 1))
