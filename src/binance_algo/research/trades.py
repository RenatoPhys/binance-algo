"""Deterministic reconstruction and diagnostics for completed position episodes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError

HOUR_MS = 3_600_000


@dataclass(slots=True)
class _OpenTrade:
    fold: int
    symbol: str
    entry_time_ms: int
    side: str
    entry_weight: float
    maximum_weight: float
    entry_reason: str
    price_return: float = 0.0
    funding_return: float = 0.0
    explicit_cost: float = 0.0
    turnover: float = 0.0


def _sign(value: float) -> int:
    return 1 if value > 1e-15 else (-1 if value < -1e-15 else 0)


def reconstruct_trade_events(
    positions: pl.DataFrame,
    *,
    strategy_id: str,
) -> pl.DataFrame:
    """Reconstruct non-overlapping per-symbol episodes and allocate every explicit cost."""

    required = {
        "fold",
        "symbol",
        "execution_time_ms",
        "label_end_time_ms",
        "previous_weight",
        "target_weight",
        "trade_weight",
        "price_pnl",
        "funding_pnl",
        "allocated_fee",
        "allocated_spread_cost",
        "allocated_slippage_cost",
    }
    missing = sorted(required.difference(positions.columns))
    if missing:
        raise ResearchError(f"trade reconstruction is missing position fields: {missing}")
    events: list[dict[str, object]] = []
    for key in positions.select("fold", "symbol").unique().sort("fold", "symbol").iter_rows():
        fold, symbol = int(key[0]), str(key[1])
        rows = positions.filter((pl.col("fold") == fold) & (pl.col("symbol") == symbol)).sort(
            "execution_time_ms"
        )
        active: _OpenTrade | None = None
        for index, row in enumerate(rows.iter_rows(named=True)):
            previous = float(row["previous_weight"])
            target = float(row["target_weight"])
            previous_sign = _sign(previous)
            target_sign = _sign(target)
            explicit = (
                float(row["allocated_fee"])
                + float(row["allocated_spread_cost"])
                + float(row["allocated_slippage_cost"])
            )
            trade_weight = float(row["trade_weight"])
            rate = explicit / trade_weight if trade_weight > 1e-15 else 0.0
            execution_time = int(row["execution_time_ms"])
            label_end_time = int(row["label_end_time_ms"])

            if active is not None and previous_sign != target_sign:
                exit_turnover = abs(previous)
                active.explicit_cost += exit_turnover * rate
                active.turnover += exit_turnover
                events.append(
                    _closed_event(
                        active,
                        exit_time_ms=execution_time,
                        exit_reason="direction_flip" if target_sign else "signal_exit",
                    )
                )
                active = None

            if target_sign and active is None:
                entry_turnover = abs(target)
                active = _OpenTrade(
                    fold=fold,
                    symbol=symbol,
                    entry_time_ms=execution_time,
                    side="LONG" if target_sign > 0 else "SHORT",
                    entry_weight=target,
                    maximum_weight=abs(target),
                    entry_reason=f"{strategy_id}:signal",
                    explicit_cost=entry_turnover * rate,
                    turnover=entry_turnover,
                )
            elif active is not None and previous_sign == target_sign:
                rebalance_turnover = abs(target - previous)
                active.explicit_cost += rebalance_turnover * rate
                active.turnover += rebalance_turnover
                active.maximum_weight = max(active.maximum_weight, abs(target))

            if active is not None and target_sign:
                active.price_return += float(row["price_pnl"])
                active.funding_return += float(row["funding_pnl"])

            if index == rows.height - 1 and active is not None:
                forced_turnover = abs(target)
                active.explicit_cost += forced_turnover * rate
                active.turnover += forced_turnover
                events.append(
                    _closed_event(
                        active,
                        exit_time_ms=label_end_time,
                        exit_reason="fold_close",
                    )
                )
                active = None
        if active is not None:
            raise ResearchError("trade reconstruction left an open episode at fold end")
    schema = {
        "strategy_id": pl.String,
        "fold": pl.Int64,
        "symbol": pl.String,
        "entry_time_ms": pl.Int64,
        "exit_time_ms": pl.Int64,
        "side": pl.String,
        "entry_weight": pl.Float64,
        "maximum_weight": pl.Float64,
        "holding_hours": pl.Float64,
        "entry_reason": pl.String,
        "exit_reason": pl.String,
        "price_return": pl.Float64,
        "gross_return": pl.Float64,
        "funding_return": pl.Float64,
        "explicit_cost": pl.Float64,
        "net_return": pl.Float64,
        "turnover": pl.Float64,
    }
    return pl.DataFrame(events, schema=schema).sort("fold", "entry_time_ms", "symbol")


def _closed_event(
    trade: _OpenTrade,
    *,
    exit_time_ms: int,
    exit_reason: str,
) -> dict[str, object]:
    gross = trade.price_return + trade.funding_return
    return {
        "strategy_id": trade.entry_reason.split(":signal", 1)[0],
        "fold": trade.fold,
        "symbol": trade.symbol,
        "entry_time_ms": trade.entry_time_ms,
        "exit_time_ms": exit_time_ms,
        "side": trade.side,
        "entry_weight": trade.entry_weight,
        "maximum_weight": trade.maximum_weight,
        "holding_hours": (exit_time_ms - trade.entry_time_ms) / HOUR_MS,
        "entry_reason": trade.entry_reason,
        "exit_reason": exit_reason,
        "price_return": trade.price_return,
        "gross_return": gross,
        "funding_return": trade.funding_return,
        "explicit_cost": trade.explicit_cost,
        "net_return": gross - trade.explicit_cost,
        "turnover": trade.turnover,
    }


def _metric_row(
    events: pl.DataFrame,
    *,
    scope: str,
    group: str,
    direction_flips: int,
) -> dict[str, object]:
    net = np.asarray(events["net_return"].to_numpy(), dtype=np.float64)
    holding = np.asarray(events["holding_hours"].to_numpy(), dtype=np.float64)
    turnover = float(events["turnover"].sum())
    gross = float(events["gross_return"].sum())
    explicit = float(events["explicit_cost"].sum())
    positive = float(np.sum(net[net > 0]))
    negative = float(-np.sum(net[net < 0]))
    return {
        "scope": scope,
        "group": group,
        "completed_trades": events.height,
        "entries": events.height,
        "exits": events.height,
        "direction_flips": direction_flips,
        "holding_hours_mean": float(np.mean(holding)),
        "holding_hours_median": float(np.median(holding)),
        "holding_hours_p10": float(np.quantile(holding, 0.10)),
        "holding_hours_p90": float(np.quantile(holding, 0.90)),
        "win_rate": float(np.mean(net > 0)),
        "payoff_mean": float(np.mean(net)),
        "payoff_median": float(np.median(net)),
        "profit_factor": positive / max(negative, 1e-15),
        "price_pnl": float(events["price_return"].sum()),
        "funding_pnl": float(events["funding_return"].sum()),
        "gross_pnl": gross,
        "explicit_cost": explicit,
        "net_pnl": float(events["net_return"].sum()),
        "turnover": turnover,
        "gross_edge_bps_per_turnover": 10_000.0 * gross / max(turnover, 1e-15),
        "net_edge_bps_per_turnover": 10_000.0
        * float(events["net_return"].sum())
        / max(turnover, 1e-15),
        "explicit_cost_bps_per_turnover": 10_000.0 * explicit / max(turnover, 1e-15),
    }


def _groups(events: pl.DataFrame, column: str) -> Iterable[tuple[str, pl.DataFrame]]:
    for value in sorted(str(item) for item in events[column].unique().to_list()):
        yield value, events.filter(pl.col(column).cast(pl.String) == value)


def build_trade_metrics(events: pl.DataFrame, positions: pl.DataFrame) -> pl.DataFrame:
    """Build overall and segmented trade diagnostics in one stable wide schema."""

    if events.is_empty():
        return pl.DataFrame(
            [
                {
                    "scope": "overall",
                    "group": "all",
                    "completed_trades": 0,
                    "entries": 0,
                    "exits": 0,
                    "direction_flips": 0,
                    "holding_hours_mean": 0.0,
                    "holding_hours_median": 0.0,
                    "holding_hours_p10": 0.0,
                    "holding_hours_p90": 0.0,
                    "win_rate": 0.0,
                    "payoff_mean": 0.0,
                    "payoff_median": 0.0,
                    "profit_factor": 0.0,
                    "price_pnl": 0.0,
                    "funding_pnl": 0.0,
                    "gross_pnl": 0.0,
                    "explicit_cost": 0.0,
                    "net_pnl": 0.0,
                    "turnover": 0.0,
                    "gross_edge_bps_per_turnover": 0.0,
                    "net_edge_bps_per_turnover": 0.0,
                    "explicit_cost_bps_per_turnover": 0.0,
                }
            ]
        )
    flips = 0
    for key in positions.select("fold", "symbol").unique().iter_rows():
        weights = positions.filter(
            (pl.col("fold") == int(key[0])) & (pl.col("symbol") == str(key[1]))
        ).sort("execution_time_ms")["target_weight"]
        signs = np.sign(np.asarray(weights.to_numpy(), dtype=np.float64))
        flips += int(np.sum(signs[1:] * signs[:-1] < 0))
    enriched = events.with_columns(
        pl.from_epoch("entry_time_ms", time_unit="ms").dt.strftime("%Y-%m").alias("month"),
        pl.from_epoch("entry_time_ms", time_unit="ms").dt.hour().alias("hour_utc"),
    )
    rows = [_metric_row(enriched, scope="overall", group="all", direction_flips=flips)]
    for dimension in ("symbol", "side", "month", "hour_utc", "entry_reason"):
        rows.extend(
            _metric_row(selected, scope=dimension, group=value, direction_flips=0)
            for value, selected in _groups(enriched, dimension)
        )
    return pl.DataFrame(rows).sort("scope", "group")


def daily_positions(positions: pl.DataFrame) -> pl.DataFrame:
    """Persist daily mean positions for deterministic position-correlation diagnostics."""

    return (
        positions.with_columns(
            pl.from_epoch("decision_time_ms", time_unit="ms")
            .dt.strftime("%Y-%m-%d")
            .alias("date_utc")
        )
        .group_by("date_utc", "symbol")
        .agg(pl.col("target_weight").mean().alias("mean_target_weight"))
        .sort("date_utc", "symbol")
    )


def pair_diagnostics(positions: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return fold-fit and approximate sleeve P&L diagnostics when pair columns exist."""

    target_columns = sorted(
        column
        for column in positions.columns
        if column.startswith("pair_") and column.endswith("_target_weight")
    )
    if not target_columns:
        return pl.DataFrame(), pl.DataFrame()
    fit_rows: list[dict[str, object]] = []
    pnl_rows: list[dict[str, object]] = []
    for target_column in target_columns:
        pair_id = target_column.removesuffix("_target_weight")
        beta_column = f"score_diagnostic_{pair_id}_beta"
        half_life_column = f"score_diagnostic_{pair_id}_half_life_hours"
        eligible_column = f"score_diagnostic_{pair_id}_eligible"
        for fold in sorted(int(value) for value in positions["fold"].unique().to_list()):
            frame = positions.filter(pl.col("fold") == fold).sort("decision_time_ms", "symbol")
            fit_rows.append(
                {
                    "pair_id": pair_id,
                    "fold": fold,
                    "beta": float(frame[beta_column][0]),
                    "half_life_hours": float(frame[half_life_column][0]),
                    "eligible": bool(frame[eligible_column][0]),
                }
            )
            pair_weight = np.asarray(frame[target_column].to_numpy(), dtype=np.float64)
            prior = np.zeros_like(pair_weight)
            symbol_count = frame["symbol"].n_unique()
            if frame.height > symbol_count:
                prior[symbol_count:] = pair_weight[:-symbol_count]
            turnover = np.abs(pair_weight - prior)
            final_indices = np.arange(frame.height - symbol_count, frame.height)
            turnover[final_indices] += np.abs(pair_weight[final_indices])
            aggregate_turnover = np.asarray(frame["trade_weight"].to_numpy(), dtype=np.float64)
            aggregate_cost = np.asarray(
                (
                    frame["allocated_fee"]
                    + frame["allocated_spread_cost"]
                    + frame["allocated_slippage_cost"]
                ).to_numpy(),
                dtype=np.float64,
            )
            rate = np.divide(
                aggregate_cost,
                aggregate_turnover,
                out=np.zeros_like(aggregate_cost),
                where=aggregate_turnover > 1e-15,
            )
            future_return = np.asarray(frame["future_return"].to_numpy(), dtype=np.float64)
            funding_rate = np.asarray(frame["funding_rate"].to_numpy(), dtype=np.float64)
            price = float(np.sum(pair_weight * future_return))
            funding = float(np.sum(-pair_weight * funding_rate))
            cost = float(np.sum(turnover * rate))
            pnl_rows.append(
                {
                    "pair_id": pair_id,
                    "fold": fold,
                    "price_pnl": price,
                    "funding_pnl": funding,
                    "explicit_cost": cost,
                    "net_pnl": price + funding - cost,
                    "turnover": float(np.sum(turnover)),
                }
            )
    return pl.DataFrame(fit_rows).sort("pair_id", "fold"), pl.DataFrame(pnl_rows).sort(
        "pair_id", "fold"
    )


__all__ = [
    "build_trade_metrics",
    "daily_positions",
    "pair_diagnostics",
    "reconstruct_trade_events",
]
