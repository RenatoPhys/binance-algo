"""Dependency-free SVG summaries for persisted research results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import polars as pl

from binance_algo.common.errors import ResearchError

PNL_VISUALIZATION_VERSION = 1


@dataclass(frozen=True, slots=True)
class PlotSeries:
    label: str
    css_class: str
    values: np.ndarray[tuple[int], np.dtype[np.float64]]


def _points(
    values: np.ndarray[tuple[int], np.dtype[np.float64]],
    *,
    left: float,
    top: float,
    width: float,
    height: float,
    minimum: float,
    maximum: float,
) -> str:
    denominator = max(maximum - minimum, 1e-12)
    last_index = max(len(values) - 1, 1)

    def point(index: int, value: float) -> str:
        x = left + width * index / last_index
        y = top + height * (maximum - value) / denominator
        return f"{x:.2f},{y:.2f}"

    return " ".join(point(index, float(value)) for index, value in enumerate(values))


def _panel(
    *,
    title: str,
    y_label: str,
    top: float,
    series: tuple[PlotSeries, ...],
    times: np.ndarray[tuple[int], np.dtype[np.int64]],
    folds: np.ndarray[tuple[int], np.dtype[np.int64]],
    percent_axis: bool,
) -> list[str]:
    left, width, height = 96.0, 1_060.0, 170.0
    plot_top = top + 60
    all_values = np.concatenate([item.values for item in series])
    if percent_axis:
        all_values = np.append(all_values, 0.0)
    minimum, maximum = float(np.min(all_values)), float(np.max(all_values))
    padding = max((maximum - minimum) * 0.08, 0.005)
    minimum -= padding
    maximum += padding
    output = [f'<text class="panel-title" x="24" y="{top + 18:.0f}">{title}</text>']
    unique_folds = np.unique(folds)
    last_index = max(len(times) - 1, 1)
    for position, fold in enumerate(unique_folds):
        indices = np.flatnonzero(folds == fold)
        start_x = left + width * int(indices[0]) / last_index
        end_x = left + width * int(indices[-1]) / last_index
        output.append(
            f'<rect class="fold fold-{position % 2}" x="{start_x:.2f}" y="{plot_top:.2f}" '
            f'width="{max(0.0, end_x - start_x):.2f}" height="{height:.2f}"/>'
        )
        output.append(
            f'<text class="fold-label" x="{start_x + 7:.2f}" y="{plot_top + 16:.2f}">'
            f"fold {int(fold)}</text>"
        )
    for tick in range(5):
        ratio = tick / 4
        value = maximum - ratio * (maximum - minimum)
        y = plot_top + ratio * height
        label = f"{value:.1%}" if percent_axis else f"{value:.3f}"
        output.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + width}" y2="{y:.2f}"/>',
                f'<text class="tick" x="{left - 10}" y="{y + 4:.2f}" '
                f'text-anchor="end">{label}</text>',
            ]
        )
    for tick in range(6):
        index = round(tick * last_index / 5)
        x = left + width * index / last_index
        label = datetime.fromtimestamp(int(times[index]) / 1_000, tz=UTC).strftime("%d/%m")
        output.extend(
            [
                f'<line class="axis-tick" x1="{x:.2f}" y1="{plot_top + height}" '
                f'x2="{x:.2f}" y2="{plot_top + height + 5}"/>',
                f'<text class="tick" x="{x:.2f}" y="{plot_top + height + 20}" '
                f'text-anchor="middle">{label}</text>',
            ]
        )
    output.append(
        f'<rect class="frame" x="{left}" y="{plot_top}" width="{width}" height="{height}"/>'
    )
    legend_x = left
    for item in series:
        points = _points(
            item.values,
            left=left,
            top=plot_top,
            width=width,
            height=height,
            minimum=minimum,
            maximum=maximum,
        )
        output.extend(
            [
                f'<line class="legend-line {item.css_class}" x1="{legend_x}" '
                f'y1="{top + 40}" x2="{legend_x + 22}" y2="{top + 40}"/>',
                f'<text class="legend" x="{legend_x + 29}" y="{top + 44}">{item.label}</text>',
                f'<polyline class="series {item.css_class}" points="{points}"/>',
            ]
        )
        legend_x += 225
    output.extend(
        [
            f'<text class="axis-title" x="{left + width / 2}" '
            f'y="{plot_top + height + 40}" text-anchor="middle">'
            "Data de execução (UTC)</text>",
            '<text class="axis-title" '
            f'transform="translate(22 {plot_top + height / 2}) rotate(-90)" '
            f'text-anchor="middle">{y_label}</text>',
        ]
    )
    return output


def render_pnl_svg(curve: pl.DataFrame) -> str:
    """Render the full OOS equity, drawdown, and P&L decomposition as standalone SVG."""

    required = {
        "execution_time_ms",
        "fold",
        "price_pnl",
        "funding_pnl",
        "trading_fees",
        "spread_cost",
        "slippage_cost",
        "net_return",
    }
    missing = required.difference(curve.columns)
    if missing:
        raise ResearchError(f"curve missing visualization fields: {sorted(missing)}")
    if curve.is_empty():
        raise ResearchError("cannot visualize an empty research curve")
    ordered = curve.sort("execution_time_ms")
    times = np.asarray(ordered["execution_time_ms"].to_numpy(), dtype=np.int64)
    folds = np.asarray(ordered["fold"].to_numpy(), dtype=np.int64)
    price = np.asarray(ordered["price_pnl"].to_numpy(), dtype=np.float64)
    funding = np.asarray(ordered["funding_pnl"].to_numpy(), dtype=np.float64)
    costs = -np.asarray(
        ordered.select(
            pl.col("trading_fees") + pl.col("spread_cost") + pl.col("slippage_cost")
        ).to_series(),
        dtype=np.float64,
    )
    net_returns = np.asarray(ordered["net_return"].to_numpy(), dtype=np.float64)
    net_equity = np.cumprod(1 + net_returns)
    gross_equity = np.cumprod(1 + price + funding)
    drawdown = net_equity / np.maximum.accumulate(net_equity) - 1
    panels: list[str] = []
    panels.extend(
        _panel(
            title="Equity OOS: líquido versus antes dos custos explícitos",
            y_label="Equity (capital inicial = 1)",
            top=88,
            series=(
                PlotSeries("Líquido", "net", net_equity),
                PlotSeries("Preço + funding", "gross", gross_equity),
            ),
            times=times,
            folds=folds,
            percent_axis=False,
        )
    )
    panels.extend(
        _panel(
            title="Drawdown líquido",
            y_label="Drawdown",
            top=382,
            series=(PlotSeries("Drawdown", "drawdown", drawdown),),
            times=times,
            folds=folds,
            percent_axis=True,
        )
    )
    panels.extend(
        _panel(
            title="Decomposição acumulada do retorno",
            y_label="Retorno acumulado (soma)",
            top=676,
            series=(
                PlotSeries("Preço", "gross", np.cumsum(price)),
                PlotSeries("Funding", "funding", np.cumsum(funding)),
                PlotSeries("Custos", "costs", np.cumsum(costs)),
                PlotSeries("Líquido", "net", np.cumsum(net_returns)),
            ),
            times=times,
            folds=folds,
            percent_axis=True,
        )
    )
    if not all(math.isfinite(float(value)) for value in net_equity):
        raise ResearchError("non-finite equity cannot be visualized")
    return "\n".join(
        [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="980" '
            'viewBox="0 0 1200 980" role="img" aria-labelledby="title desc">',
            '<title id="title">Curva de P&amp;L fora da amostra — Fase 3</title>',
            '<desc id="desc">Equity líquida e antes de custos, drawdown e decomposição '
            "acumulada em três folds walk-forward.</desc>",
            "<style>",
            ".background{fill:#ffffff}.title,.panel-title,.legend,.tick,.axis-title,.fold-label{fill:#17202a;font-family:Arial,sans-serif}.title{font-size:24px;font-weight:600}.subtitle{fill:#5d6d7e;font-family:Arial,sans-serif;font-size:14px}.panel-title{font-size:17px;font-weight:600}.legend,.tick,.axis-title,.fold-label{font-size:12px}.frame,.grid,.axis-tick{fill:none;stroke:#d5d8dc;stroke-width:1}.grid{opacity:.65}.fold{fill:#85929e}.fold-0{opacity:.08}.fold-1{opacity:.03}.series,.legend-line{fill:none;stroke-width:2}.net{stroke:#2471a3}.gross{stroke:#ca6f1e}.drawdown{stroke:#239b56}.funding{stroke:#7d3c98}.costs{stroke:#c0397b}",
            "@media(prefers-color-scheme:dark){.background{fill:#151515}.title,.panel-title,.legend,.tick,.axis-title,.fold-label{fill:#f4f6f7}.subtitle{fill:#aeb6bf}.frame,.grid,.axis-tick{stroke:#424949}.net{stroke:#5dade2}.gross{stroke:#f5b041}.drawdown{stroke:#58d68d}.funding{stroke:#bb8fce}.costs{stroke:#ec87c0}}",
            "</style>",
            '<rect class="background" width="1200" height="980"/>',
            '<text class="title" x="24" y="34">Curva de P&amp;L — baseline da Fase 3</text>',
            '<text class="subtitle" x="24" y="58">Walk-forward OOS · 3 folds · execução '
            "no próximo open · fees, spread, slippage e funding</text>",
            *panels,
            "</svg>",
        ]
    )
