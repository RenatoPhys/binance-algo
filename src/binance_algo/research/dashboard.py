"""Deterministic, offline HTML dashboard for the research registry."""

# ruff: noqa: E501 -- keeping the embedded HTML/CSS/JavaScript readable is preferable here.

from __future__ import annotations

import html
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import orjson

from binance_algo.common.errors import ResearchError
from binance_algo.research.experiments.artifacts import verify_run_artifacts
from binance_algo.research.experiments.canonical import canonical_sha256
from binance_algo.research.experiments.models import (
    MetricScope,
    PromotionDecision,
    ResearchStage,
    RunStatus,
)
from binance_algo.research.experiments.store import (
    CampaignRecord,
    ExperimentRunRecord,
    ResearchArtifactRecord,
    ResearchMetricRecord,
    ResearchStore,
)
from binance_algo.research.strategy_portfolio.snapshot import build_portfolio_snapshots

DASHBOARD_SCHEMA_VERSION = 2
FAILED_RUN_STATUSES = frozenset({RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.STALE})


class DashboardBuildResult:
    """Paths and source snapshot produced by a dashboard build."""

    __slots__ = ("index_path", "snapshot", "snapshot_path")

    def __init__(
        self,
        *,
        index_path: Path,
        snapshot_path: Path,
        snapshot: Mapping[str, Any],
    ) -> None:
        self.index_path = index_path
        self.snapshot_path = snapshot_path
        self.snapshot = snapshot


def _current_stage(store: ResearchStore, experiment_id: str) -> ResearchStage:
    stage = ResearchStage.DISCOVERY
    for event in store.list_promotions(experiment_id):
        if event.decision is not PromotionDecision.BLOCKED:
            stage = event.to_stage
    return stage


def _metric_maps(
    metrics: Sequence[ResearchMetricRecord],
) -> tuple[dict[str, float], dict[tuple[str, str], float], list[float]]:
    summary: dict[str, float] = {}
    stress: dict[tuple[str, str], float] = {}
    fold_returns: list[float] = []
    for metric in metrics:
        if metric.scope is MetricScope.TEST and metric.fold is None and metric.regime is None:
            summary[metric.metric_name] = metric.metric_value
        elif metric.scope is MetricScope.STRESS and metric.regime is not None:
            stress[(metric.regime, metric.metric_name)] = metric.metric_value
        if (
            metric.scope is MetricScope.TEST
            and metric.fold is not None
            and metric.metric_name == "total_return"
        ):
            fold_returns.append(metric.metric_value)
    return summary, stress, fold_returns


def _relative_href(
    target: Path, *, dashboard_directory: Path, directory: bool = False
) -> str | None:
    try:
        relative = os.path.relpath(target.resolve(), dashboard_directory.resolve())
    except ValueError:
        return None
    href = quote(relative.replace("\\", "/"), safe="/:=._-")
    return f"{href}/" if directory and not href.endswith("/") else href


def _artifact_links(
    artifacts: Sequence[ResearchArtifactRecord],
    *,
    data_root: Path,
    dashboard_directory: Path,
) -> tuple[str | None, str | None]:
    report = next((artifact for artifact in artifacts if artifact.artifact_type == "report"), None)
    if report is None:
        return None, None
    report_path = data_root.resolve() / report.path
    return (
        _relative_href(report_path, dashboard_directory=dashboard_directory),
        _relative_href(
            report_path.parent,
            dashboard_directory=dashboard_directory,
            directory=True,
        ),
    )


def _campaign_report_href(
    campaign: CampaignRecord,
    *,
    reports_root: Path,
    dashboard_directory: Path,
) -> str | None:
    path = (
        reports_root.resolve()
        / "research_campaigns"
        / f"campaign_id={campaign.campaign_id[:24]}"
        / "report.md"
    )
    if not path.is_file():
        return None
    return _relative_href(path, dashboard_directory=dashboard_directory)


def build_dashboard_snapshot(
    *,
    store: ResearchStore,
    reports_root: Path,
    data_root: Path,
    portfolio_file: Path | None = None,
) -> dict[str, Any]:
    """Read a complete dashboard snapshot only through public store methods."""

    dashboard_directory = reports_root.resolve() / "research_dashboard"
    hypotheses = sorted(store.list_hypotheses(), key=lambda item: item.hypothesis_id)
    campaigns = sorted(store.list_campaigns(), key=lambda item: item.campaign_id)
    experiment_ids = sorted(store.list_experiment_ids())
    runs_by_experiment: dict[str, tuple[ExperimentRunRecord, ...]] = {
        identifier: store.list_runs(experiment_id_value=identifier) for identifier in experiment_ids
    }
    campaign_experiments: dict[str, tuple[tuple[int, str, Mapping[str, Any]], ...]] = {
        campaign.campaign_id: store.list_campaign_experiments(campaign.campaign_id)
        for campaign in campaigns
    }
    campaigns_by_experiment: dict[str, list[CampaignRecord]] = {
        identifier: [] for identifier in experiment_ids
    }
    for campaign in campaigns:
        for _, identifier, _ in campaign_experiments[campaign.campaign_id]:
            campaigns_by_experiment.setdefault(identifier, []).append(campaign)

    all_runs = [run for runs in runs_by_experiment.values() for run in runs]
    totals = {
        "hypotheses": len(hypotheses),
        "campaigns": len(campaigns),
        "experiments": len(experiment_ids),
        "successes": sum(run.status is RunStatus.SUCCEEDED for run in all_runs),
        "failures": sum(run.status in FAILED_RUN_STATUSES for run in all_runs),
        "portfolios": 0,
    }

    campaign_rows: list[dict[str, Any]] = []
    for campaign in campaigns:
        associations = campaign_experiments[campaign.campaign_id]
        latest_runs = [
            runs_by_experiment[identifier][-1]
            for _, identifier, _ in associations
            if runs_by_experiment.get(identifier)
        ]
        campaign_rows.append(
            {
                "campaign_id": campaign.campaign_id,
                "name": campaign.name,
                "description": campaign.description,
                "hypothesis_id": campaign.hypothesis_id,
                "status": campaign.status.value,
                "trial_count": campaign.trial_count,
                "successes": sum(run.status is RunStatus.SUCCEEDED for run in latest_runs),
                "failures": sum(run.status in FAILED_RUN_STATUSES for run in latest_runs),
                "created_at_ms": campaign.created_at_ms,
                "started_at_ms": campaign.started_at_ms,
                "finished_at_ms": campaign.finished_at_ms,
                "last_error": campaign.last_error,
                "report_href": _campaign_report_href(
                    campaign,
                    reports_root=reports_root,
                    dashboard_directory=dashboard_directory,
                ),
            }
        )

    experiment_rows: list[dict[str, Any]] = []
    for identifier in experiment_ids:
        spec = store.get_experiment(identifier)
        if spec is None:
            raise ResearchError(f"experiment disappeared while building dashboard: {identifier}")
        runs = runs_by_experiment[identifier]
        latest = runs[-1] if runs else None
        successful = next(
            (run for run in reversed(runs) if run.status is RunStatus.SUCCEEDED),
            None,
        )
        metrics = store.list_metrics(successful.run_id) if successful is not None else ()
        summary, stress, fold_returns = _metric_maps(metrics)
        artifacts = store.list_artifacts(successful.run_id) if successful is not None else ()
        verification = (
            verify_run_artifacts(
                data_root=data_root,
                run_id=successful.run_id,
                artifacts=artifacts,
            )
            if successful is not None
            else None
        )
        report_href, artifacts_href = _artifact_links(
            artifacts,
            data_root=data_root,
            dashboard_directory=dashboard_directory,
        )
        linked_campaigns = sorted(
            campaigns_by_experiment.get(identifier, []), key=lambda item: item.campaign_id
        )
        serialized_spec = spec.model_dump(mode="json")
        promotions = store.list_promotions(identifier)
        compatibility_group = canonical_sha256(
            {
                "dataset_reference": serialized_spec["dataset_reference"],
                "label": serialized_spec["label"],
                "execution_model": serialized_spec["execution_model"],
                "cost_model": serialized_spec["cost_model"],
                "split_plan": serialized_spec["split_plan"],
            }
        )
        annualized_return = summary.get("annualized_return")
        maximum_drawdown = summary.get("max_drawdown")
        calmar = (
            annualized_return / abs(maximum_drawdown)
            if annualized_return is not None
            and maximum_drawdown is not None
            and maximum_drawdown < 0
            else None
        )
        validation_profile = spec.validation_plan.parameters.get("profile")
        if not isinstance(validation_profile, str):
            validation_profile = "full" if ("cost_2_0x", "total_return") in stress else "discovery"
        experiment_rows.append(
            {
                "experiment_id": identifier,
                "strategy_id": spec.strategy.component_id,
                "strategy_version": spec.strategy.version,
                "strategy_parameters": serialized_spec["strategy_parameters"],
                "hypothesis_id": spec.hypothesis_id,
                "campaigns": [
                    {"campaign_id": campaign.campaign_id, "name": campaign.name}
                    for campaign in linked_campaigns
                ],
                "dataset_id": spec.dataset_reference.dataset_id,
                "latest_status": latest.status.value if latest is not None else "NOT_RUN",
                "attempts": len(runs),
                "runtime_seconds": latest.runtime_seconds if latest is not None else None,
                "total_return": summary.get("total_return"),
                "sharpe": summary.get("sharpe"),
                "max_drawdown": summary.get("max_drawdown"),
                "turnover": summary.get("turnover"),
                "worst_fold_return": min(fold_returns) if fold_returns else None,
                "cost_1_5x_return": stress.get(("cost_1_5x", "total_return")),
                "cost_2_0x_return": stress.get(("cost_2_0x", "total_return")),
                "signal_delay_1_bar_return": stress.get(("signal_delay_1_bar", "total_return")),
                "calmar": calmar,
                "validation_profile": validation_profile,
                "compatibility_group": compatibility_group,
                "artifact_verified": verification.valid if verification is not None else False,
                "artifact_issues": list(verification.issues) if verification is not None else [],
                "code_fingerprint": serialized_spec["code_fingerprint"],
                "promotion_gate": promotions[-1].decision.value if promotions else "NOT_ASSESSED",
                "lockbox_available": False,
                "dsr": None,
                "pbo": None,
                "symbol_concentration": None,
                "research_stage": _current_stage(store, identifier).value,
                "metrics_run_id": successful.run_id if successful is not None else None,
                "error_type": latest.error_type if latest is not None else None,
                "error_message": latest.error_message if latest is not None else None,
                "report_href": report_href,
                "artifacts_href": artifacts_href,
            }
        )

    portfolios = (
        build_portfolio_snapshots(
            store=store,
            data_root=data_root,
            portfolio_file=portfolio_file,
        )
        if portfolio_file is not None
        else []
    )
    totals["portfolios"] = len(portfolios)
    compatibility_groups: dict[str, list[str]] = {}
    for experiment in experiment_rows:
        compatibility_groups.setdefault(experiment["compatibility_group"], []).append(
            experiment["experiment_id"]
        )
    return {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "totals": totals,
        "hypotheses": [item.model_dump(mode="json") for item in hypotheses],
        "campaigns": campaign_rows,
        "experiments": experiment_rows,
        "strategy_shortlist": sorted(
            experiment_rows,
            key=lambda item: (
                item["sharpe"] is None,
                -(item["sharpe"] if item["sharpe"] is not None else 0.0),
                item["experiment_id"],
            ),
        ),
        "compatibility_groups": [
            {
                "compatibility_group": group,
                "experiment_ids": sorted(identifiers),
                "experiment_count": len(identifiers),
            }
            for group, identifiers in sorted(compatibility_groups.items())
        ],
        "portfolios": portfolios,
    }


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _format_timestamp(value: int | None) -> str:
    if value is None:
        return "—"
    return datetime.fromtimestamp(value / 1_000, tz=UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_number(value: float | int | None, *, digits: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _format_percent(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.4%}"


def _sort_value(value: object | None) -> str:
    return "" if value is None else _escape(value)


def _select_options(rows: Sequence[tuple[str, str]]) -> str:
    return "".join(
        f'<option value="{_escape(value)}">{_escape(label)}</option>'
        for value, label in sorted(set(rows), key=lambda item: (item[1], item[0]))
    )


def _link(href: str | None, label: str) -> str:
    if href is None:
        return "—"
    return f'<a href="{_escape(href)}">{_escape(label)}</a>'


def _render_campaign_rows(campaigns: Sequence[Mapping[str, Any]]) -> str:
    if not campaigns:
        return '<tr class="empty"><td colspan="11">Nenhuma campanha registrada.</td></tr>'
    rows = []
    for campaign in campaigns:
        rows.append(
            "".join(
                (
                    '<tr class="campaign-row">',
                    f'<td data-sort="{_escape(campaign["name"])}">'
                    f"<strong>{_escape(campaign['name'])}</strong><br>"
                    f"<small>{_escape(campaign['campaign_id'])}</small></td>",
                    f'<td data-sort="{_escape(campaign["status"])}">'
                    f'<span class="badge">{_escape(campaign["status"])}</span></td>',
                    f'<td data-sort="{_escape(campaign["hypothesis_id"])}">'
                    f"{_escape(campaign['hypothesis_id'])}</td>",
                    f'<td data-sort="{_sort_value(campaign["trial_count"])}">'
                    f"{_escape(campaign['trial_count'])}</td>",
                    f'<td data-sort="{_sort_value(campaign["successes"])}">'
                    f"{_escape(campaign['successes'])}</td>",
                    f'<td data-sort="{_sort_value(campaign["failures"])}">'
                    f"{_escape(campaign['failures'])}</td>",
                    f'<td data-sort="{_sort_value(campaign["created_at_ms"])}">'
                    f"{_format_timestamp(campaign['created_at_ms'])}</td>",
                    f'<td data-sort="{_sort_value(campaign["started_at_ms"])}">'
                    f"{_format_timestamp(campaign['started_at_ms'])}</td>",
                    f'<td data-sort="{_sort_value(campaign["finished_at_ms"])}">'
                    f"{_format_timestamp(campaign['finished_at_ms'])}</td>",
                    f'<td data-sort="{_sort_value(campaign["last_error"])}">'
                    f"{_escape(campaign['last_error'] or '—')}</td>",
                    f"<td>{_link(campaign['report_href'], 'relatório')}</td>",
                    "</tr>",
                )
            )
        )
    return "".join(rows)


def _render_experiment_rows(experiments: Sequence[Mapping[str, Any]]) -> str:
    if not experiments:
        return '<tr class="empty"><td colspan="27">Nenhum experimento registrado.</td></tr>'
    rows = []
    for experiment in experiments:
        campaigns = experiment["campaigns"]
        campaign_names = ", ".join(item["name"] for item in campaigns) or "—"
        campaign_ids = "|".join(item["campaign_id"] for item in campaigns)
        strategy = f"{experiment['strategy_id']}:{experiment['strategy_version']}"
        parameters = orjson.dumps(
            experiment["strategy_parameters"], option=orjson.OPT_SORT_KEYS
        ).decode()
        error = " — ".join(
            str(value) for value in (experiment["error_type"], experiment["error_message"]) if value
        )
        links = (
            " / ".join(
                item
                for item in (
                    _link(experiment["report_href"], "relatório")
                    if experiment["report_href"]
                    else "",
                    _link(experiment["artifacts_href"], "artifacts")
                    if experiment["artifacts_href"]
                    else "",
                )
                if item
            )
            or "—"
        )
        rows.append(
            "".join(
                (
                    '<tr class="experiment-row" '
                    f'data-strategy="{_escape(strategy)}" '
                    f'data-campaigns="{_escape(campaign_ids)}" '
                    f'data-hypothesis="{_escape(experiment["hypothesis_id"])}" '
                    f'data-status="{_escape(experiment["latest_status"])}" '
                    f'data-stage="{_escape(experiment["research_stage"])}" '
                    f'data-validation="{_escape(experiment["validation_profile"])}" '
                    f'data-compatibility="{_escape(experiment["compatibility_group"])}">',
                    f'<td data-sort="{_escape(experiment["experiment_id"])}">'
                    f"<code>{_escape(experiment['experiment_id'])}</code></td>",
                    f'<td data-sort="{_escape(strategy)}"><strong>{_escape(strategy)}</strong>'
                    f"<br><code>{_escape(parameters)}</code></td>",
                    f'<td data-sort="{_escape(experiment["hypothesis_id"])}">'
                    f"{_escape(experiment['hypothesis_id'])}</td>",
                    f'<td data-sort="{_escape(campaign_names)}">{_escape(campaign_names)}</td>',
                    f'<td data-sort="{_escape(experiment["dataset_id"])}">'
                    f"{_escape(experiment['dataset_id'])}</td>",
                    f'<td data-sort="{_escape(experiment["latest_status"])}">'
                    f'<span class="badge">{_escape(experiment["latest_status"])}</span>'
                    f"{('<br><small>' + _escape(error) + '</small>') if error else ''}</td>",
                    f'<td data-sort="{_sort_value(experiment["attempts"])}">'
                    f"{_escape(experiment['attempts'])}</td>",
                    f'<td data-sort="{_sort_value(experiment["runtime_seconds"])}">'
                    f"{_format_number(experiment['runtime_seconds'], digits=3)}</td>",
                    f'<td data-sort="{_sort_value(experiment["total_return"])}">'
                    f"{_format_percent(experiment['total_return'])}</td>",
                    f'<td data-sort="{_sort_value(experiment["sharpe"])}">'
                    f"{_format_number(experiment['sharpe'])}</td>",
                    f'<td data-sort="{_sort_value(experiment["max_drawdown"])}">'
                    f"{_format_percent(experiment['max_drawdown'])}</td>",
                    f'<td data-sort="{_sort_value(experiment["turnover"])}">'
                    f"{_format_number(experiment['turnover'])}</td>",
                    f'<td data-sort="{_sort_value(experiment["worst_fold_return"])}">'
                    f"{_format_percent(experiment['worst_fold_return'])}</td>",
                    f'<td data-sort="{_sort_value(experiment["cost_1_5x_return"])}">'
                    f"{_format_percent(experiment['cost_1_5x_return'])}</td>",
                    f'<td data-sort="{_sort_value(experiment["cost_2_0x_return"])}">'
                    f"{_format_percent(experiment['cost_2_0x_return'])}</td>",
                    f'<td data-sort="{_sort_value(experiment["signal_delay_1_bar_return"])}">'
                    f"{_format_percent(experiment['signal_delay_1_bar_return'])}</td>",
                    f'<td data-sort="{_sort_value(experiment["calmar"])}">'
                    f"{_format_number(experiment['calmar'])}</td>",
                    f'<td data-sort="{_sort_value(experiment["dsr"])}">'
                    f"{_format_number(experiment['dsr'])}</td>",
                    f'<td data-sort="{_sort_value(experiment["pbo"])}">'
                    f"{_format_number(experiment['pbo'])}</td>",
                    f'<td data-sort="{_sort_value(experiment["symbol_concentration"])}">'
                    f"{_format_percent(experiment['symbol_concentration'])}</td>",
                    f'<td data-sort="{_escape(experiment["validation_profile"])}">'
                    f"{_escape(experiment['validation_profile'])}</td>",
                    f'<td data-sort="{_escape(experiment["research_stage"])}">'
                    f"{_escape(experiment['research_stage'])}</td>",
                    f'<td data-sort="{_escape(experiment["artifact_verified"])}">'
                    f"{'verified' if experiment['artifact_verified'] else 'not verified'}</td>",
                    f'<td data-sort="{_escape(experiment["promotion_gate"])}">'
                    f"{_escape(experiment['promotion_gate'])}</td>",
                    f'<td data-sort="{_escape(experiment["compatibility_group"])}"><code>'
                    f"{_escape(str(experiment['compatibility_group'])[:16])}</code></td>",
                    '<td data-sort="0">absent</td>',
                    f"<td>{links}</td>",
                    "</tr>",
                )
            )
        )
    return "".join(rows)


def _metric_card(label: str, value: str) -> str:
    return (
        '<article class="card"><span>'
        f"{_escape(label)}</span><strong>{_escape(value)}</strong></article>"
    )


def _axis_value(value: float, style: Literal["decimal", "percent"]) -> str:
    if style == "percent":
        return f"{value:.1%}"
    magnitude = abs(value)
    if magnitude >= 1_000:
        return f"{value:,.0f}"
    if magnitude >= 100:
        return f"{value:.1f}"
    if magnitude >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _tick_indices(length: int, *, maximum_ticks: int = 6) -> tuple[int, ...]:
    count = min(length, maximum_ticks)
    if count <= 1:
        return (0,)
    return tuple(dict.fromkeys(round(index * (length - 1) / (count - 1)) for index in range(count)))


def _line_svg(
    series: Mapping[str, Sequence[float]],
    *,
    title: str,
    x_labels: Sequence[str],
    x_axis_label: str,
    y_axis_label: str,
    y_style: Literal["decimal", "percent"] = "decimal",
) -> str:
    width, height = 960, 350
    left, right, top, bottom = 82, 22, 24, 68
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [float(value) for points in series.values() for value in points]
    maximum_length = max((len(points) for points in series.values()), default=0)
    if not values or maximum_length < 2 or len(x_labels) != maximum_length:
        return '<div class="empty-state">Série indisponível.</div>'
    minimum, maximum = min(values), max(values)
    if math.isclose(minimum, maximum):
        padding = max(abs(minimum) * 0.05, 0.01)
        minimum -= padding
        maximum += padding
    else:
        padding = (maximum - minimum) * 0.05
        minimum -= padding
        maximum += padding
    colors = ("#62d4c5", "#f4bd68", "#8ea7ff", "#df83f5", "#ff8f7a", "#79c68a")
    lines = []
    legends = []
    for index, (name, points) in enumerate(series.items()):
        if len(points) < 2:
            continue
        coordinates = []
        for point_index, value in enumerate(points):
            x = left + plot_width * point_index / (maximum_length - 1)
            y = top + plot_height * (maximum - float(value)) / (maximum - minimum)
            coordinates.append(f"{x:.2f},{y:.2f}")
        color = colors[index % len(colors)]
        lines.append(
            f'<polyline data-series-index="{index}" points="{" ".join(coordinates)}" '
            f'fill="none" stroke="{color}" stroke-width="2" '
            f'vector-effect="non-scaling-stroke"><title>{_escape(name)}</title></polyline>'
        )
        legends.append(
            f'<button type="button" data-series-index="{index}">'
            f'<i style="background:{color}"></i>{_escape(name)}</button>'
        )
    y_ticks = []
    for index in range(5):
        value = minimum + (maximum - minimum) * index / 4
        y = top + plot_height * (maximum - value) / (maximum - minimum)
        y_ticks.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" '
            'stroke="#2b3957" stroke-width="1"/>'
            f'<line x1="{left - 5}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" '
            'stroke="#65728c"/>'
            f'<text x="{left - 9}" y="{y + 4:.2f}" text-anchor="end" '
            f'fill="#9caac2" font-size="13">{_escape(_axis_value(value, y_style))}</text>'
        )
    x_ticks = []
    for index in _tick_indices(maximum_length):
        x = left + plot_width * index / (maximum_length - 1)
        x_ticks.append(
            f'<line x1="{x:.2f}" y1="{height - bottom}" x2="{x:.2f}" '
            f'y2="{height - bottom + 5}" stroke="#65728c"/>'
            f'<text x="{x:.2f}" y="{height - bottom + 20}" text-anchor="middle" '
            f'fill="#9caac2" font-size="13">{_escape(x_labels[index])}</text>'
        )
    return (
        f'<div class="chart"><svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_escape(title)}"><title>{_escape(title)}</title>'
        f"<desc>{_escape(title)}; valores calculados em Python, datas em UTC.</desc>"
        f"{''.join(y_ticks)}"
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" '
        f'y2="{height - bottom}" stroke="#65728c"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" '
        f'y2="{height - bottom}" stroke="#65728c"/>'
        f"{''.join(x_ticks)}"
        f'<text x="{left + plot_width / 2:.2f}" y="{height - 10}" text-anchor="middle" '
        f'fill="#edf2fb" font-size="14" font-weight="600">{_escape(x_axis_label)}</text>'
        f'<text x="18" y="{top + plot_height / 2:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 18 {top + plot_height / 2:.2f})" fill="#edf2fb" '
        f'font-size="14" font-weight="600">{_escape(y_axis_label)}</text>'
        f"{''.join(lines)}</svg>"
        f'<div class="legend">{"".join(legends)}</div></div>'
    )


def _correlation_heatmap(payload: Mapping[str, Any], *, title: str) -> str:
    labels = payload.get("labels", [])
    values = payload.get("values", [])
    observations = payload.get("observations", [])
    if not labels or not values:
        return '<div class="empty-state">Matriz indisponível.</div>'
    header = "".join(f"<th>{_escape(label)}</th>" for label in labels)
    rows = []
    for row_index, label in enumerate(labels):
        cells = []
        for column_index in range(len(labels)):
            value = values[row_index][column_index]
            count = observations[row_index][column_index] if observations else 0
            if value is None:
                cells.append('<td class="heat-na" title="not available">n/a</td>')
                continue
            intensity = min(1.0, abs(float(value)))
            hue = 164 if float(value) >= 0 else 8
            background = f"hsla({hue},65%,45%,{0.12 + intensity * 0.58:.3f})"
            cells.append(
                f'<td style="background:{background}" title="n={int(count)}">'
                f"{float(value):.3f}<small>n={int(count)}</small></td>"
            )
        rows.append(f"<tr><th>{_escape(label)}</th>{''.join(cells)}</tr>")
    return (
        f'<div class="matrix" aria-label="{_escape(title)}"><table><caption>'
        f"{_escape(title)}</caption><thead><tr><th></th>{header}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _table(
    rows: Sequence[Mapping[str, Any]],
    columns: Sequence[tuple[str, str]],
    *,
    empty: str = "Dados indisponíveis.",
) -> str:
    header = "".join(f"<th>{_escape(label)}</th>" for key, label in columns)
    if not rows:
        body = f'<tr class="empty"><td colspan="{len(columns)}">{_escape(empty)}</td></tr>'
    else:
        rendered = []
        for row in rows:
            cells = []
            for key, _ in columns:
                value = row.get(key)
                if isinstance(value, float):
                    display = f"{value:.6f}"
                elif isinstance(value, (dict, list)):
                    display = orjson.dumps(value, option=orjson.OPT_SORT_KEYS).decode()
                else:
                    display = "—" if value is None else str(value)
                cells.append(f"<td>{_escape(display)}</td>")
            rendered.append(f"<tr>{''.join(cells)}</tr>")
        body = "".join(rendered)
    return (
        '<div class="table-wrap"><table><thead><tr>'
        f"{header}</tr></thead><tbody>{body}</tbody></table></div>"
    )


def _weekday_hour_heatmap(rows: Sequence[Mapping[str, Any]]) -> str:
    values = {
        (int(row["weekday"]), int(row["hour"])): float(row["simulated_traded_weight"])
        for row in rows
    }
    maximum = max(values.values(), default=0.0)
    header = "".join(f"<th>{hour:02d}</th>" for hour in range(24))
    body = []
    for weekday, label in enumerate(("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")):
        cells = []
        for hour in range(24):
            value = values.get((weekday, hour), 0.0)
            intensity = value / maximum if maximum else 0.0
            cells.append(
                f'<td style="background:rgba(98,212,197,{0.08 + 0.72 * intensity:.3f})" '
                f'title="{label} {hour:02d}:00 UTC — {value:.6f}">{value:.3f}</td>'
            )
        body.append(f"<tr><th>{label}</th>{''.join(cells)}</tr>")
    return (
        '<div class="matrix"><table><caption>Simulated traded weight by weekday x hour UTC</caption>'
        f"<thead><tr><th>UTC</th>{header}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def _render_portfolio_panel(portfolio: Mapping[str, Any], *, selected: bool) -> str:
    hidden = "" if selected else " hidden"
    identifier = _escape(portfolio["portfolio_id"])
    if portfolio["status"] != "VALID":
        errors = "".join(f"<li>{_escape(item)}</li>" for item in portfolio["errors"])
        return (
            f'<article class="portfolio-panel" data-portfolio="{identifier}"{hidden}>'
            f"<h3>{_escape(portfolio['title'])}</h3>"
            f'<div class="alert danger"><strong>Invalid portfolio</strong><ul>{errors}</ul></div>'
            "</article>"
        )
    metrics = portfolio["metrics"]
    cards = "".join(
        (
            _metric_card("Total return", _format_percent(metrics.get("total_return"))),
            _metric_card("Annualized return", _format_percent(metrics.get("annualized_return"))),
            _metric_card("Sharpe", _format_number(metrics.get("sharpe"))),
            _metric_card("Max drawdown", _format_percent(metrics.get("max_drawdown"))),
            _metric_card("Calmar", _format_number(metrics.get("calmar"))),
            _metric_card("Turnover", _format_number(metrics.get("turnover"), digits=2)),
            _metric_card("Rebalance events", str(metrics.get("rebalance_events", "—"))),
            _metric_card("Trade legs", str(metrics.get("trade_legs", "—"))),
            _metric_card("Netting savings", _format_percent(metrics.get("netting_savings"))),
            _metric_card(
                "Effective strategies",
                _format_number(metrics.get("effective_independent_strategies"), digits=2),
            ),
        )
    )
    warnings = "".join(f"<li>{_escape(item)}</li>" for item in portfolio["warnings"])
    alignment = portfolio["alignment"]
    daily = portfolio["chart_series"]["daily"]
    daily_dates = [str(item["date"]) for item in daily]
    equity = {
        "Netted": [float(item["netted_equity"]) for item in daily],
        "Sleeve": [float(item["sleeve_equity"]) for item in daily],
    }
    for component in portfolio["components"]:
        equity[component["label"]] = [
            float(item["components"][component["label"]]) for item in daily
        ]
    drawdown = {"Drawdown": [float(item["drawdown"]) for item in daily]}
    turnover = {
        "Netted turnover": [float(item["netted_cumulative_turnover"]) for item in daily],
        "Sleeve turnover": [float(item["sleeve_cumulative_turnover"]) for item in daily],
    }
    rolling = portfolio["chart_series"]["rolling"]
    rolling_charts: list[str] = []
    for window, rows in rolling.items():
        rolling_dates = [
            datetime.fromtimestamp(int(item["decision_time_ms"]) / 1_000, tz=UTC).strftime(
                "%Y-%m-%d"
            )
            for item in rows
        ]
        rolling_charts.extend(
            (
                _line_svg(
                    {f"{window}h return": [float(item["return"]) for item in rows]},
                    title=f"Rolling {window}h compounded return — daily display, UTC",
                    x_labels=rolling_dates,
                    x_axis_label="Date (UTC)",
                    y_axis_label="Compounded return",
                    y_style="percent",
                ),
                _line_svg(
                    {
                        f"{window}h volatility": [
                            float(item["annualized_volatility"]) for item in rows
                        ]
                    },
                    title=f"Rolling {window}h annualized volatility — UTC",
                    x_labels=rolling_dates,
                    x_axis_label="Date (UTC)",
                    y_axis_label="Annualized volatility",
                    y_style="percent",
                ),
                _line_svg(
                    {f"{window}h Sharpe": [float(item["sharpe"]) for item in rows]},
                    title=f"Rolling {window}h Sharpe — UTC",
                    x_labels=rolling_dates,
                    x_axis_label="Date (UTC)",
                    y_axis_label="Sharpe ratio",
                ),
            )
        )
    correlations = portfolio["correlations"]
    position_payload = {
        "labels": portfolio["position_similarity"]["labels"],
        "values": portfolio["position_similarity"]["matrix"],
        "observations": [
            [len(daily) for _ in portfolio["position_similarity"]["labels"]]
            for _ in portfolio["position_similarity"]["labels"]
        ],
    }
    return "".join(
        (
            f'<article class="portfolio-panel" data-portfolio="{identifier}"{hidden}>',
            f"<h3>{_escape(portfolio['title'])}</h3>",
            f'<p class="lede">{_escape(portfolio["description"])}</p>',
            f'<div class="alert warning"><ul>{warnings}</ul></div>',
            '<div class="alert"><strong>Resolved analytical window</strong><br>',
            f"{_escape(_format_timestamp(alignment['start_time_ms']))} — "
            f"{_escape(_format_timestamp(alignment['end_time_ms']))}; "
            f"{_escape(alignment['periods'])} hourly periods; "
            f"policy {_escape(alignment['policy'])}; compatibility "
            f"<code>{_escape(str(alignment['compatibility_group'])[:24])}</code></div>",
            f'<section class="cards">{cards}</section>',
            "<h3>Components and provenance</h3>",
            _table(
                portfolio["components"],
                (
                    ("label", "Component"),
                    ("capital_weight", "Capital weight"),
                    ("experiment_id", "Experiment"),
                    ("run_id", "Resolved run"),
                    ("validation_profile", "Validation"),
                    ("research_stage", "Stage"),
                    ("dataset_id", "Dataset"),
                    ("code_fingerprint", "Code fingerprint"),
                    ("artifact_verified", "Artifacts verified"),
                    ("positions_available", "Positions"),
                    ("stress", "Cost/delay stresses"),
                ),
            ),
            "<h3>Source runs and verified checksums</h3>",
            _table(
                portfolio["source_runs"],
                (
                    ("experiment_id", "Experiment"),
                    ("run_id", "Run"),
                    ("result_digest", "Result digest"),
                    ("artifact_checksums", "Artifact checksums"),
                ),
            ),
            '<section class="pnl-drawdown"><h2>P&amp;L and drawdown</h2>',
            '<p class="muted">Sleeve aggregate — costs already charged inside each component; '
            "no cross-strategy order netting. Netted applies explicit costs once to the aggregate "
            "book.</p>",
            _line_svg(
                equity,
                title="Daily OOS equity in UTC",
                x_labels=daily_dates,
                x_axis_label="Date (UTC)",
                y_axis_label="Equity index",
            ),
            _line_svg(
                drawdown,
                title="Daily portfolio drawdown in UTC",
                x_labels=daily_dates,
                x_axis_label="Date (UTC)",
                y_axis_label="Drawdown",
                y_style="percent",
            ),
            "".join(rolling_charts),
            "<h3>Sleeve versus netted accounting</h3>",
            _table(
                (
                    {"view": "Sleeve", **portfolio["sleeve_metrics"]},
                    {"view": "Netted", **portfolio["netted_metrics"]},
                ),
                (
                    ("view", "View"),
                    ("total_return", "Total return"),
                    ("turnover", "Turnover"),
                    ("trading_fees", "Fees"),
                    ("spread_cost", "Spread"),
                    ("slippage_cost", "Slippage"),
                    ("average_gross_exposure", "Average gross"),
                    ("maximum_gross_exposure", "Maximum gross"),
                ),
            ),
            "<h3>P&amp;L attribution by component</h3>",
            _table(
                portfolio["component_attribution"],
                (
                    ("label", "Component"),
                    ("capital_weight", "Weight"),
                    ("price_pnl", "Price"),
                    ("funding_pnl", "Funding"),
                    ("explicit_cost", "Explicit cost"),
                    ("net_contribution", "Net contribution"),
                ),
            ),
            "</section>",
            '<section class="diversification"><h2>Diversification</h2>',
            _correlation_heatmap(
                {"labels": correlations["labels"], **correlations["daily_net"]},
                title="Daily net-return correlation",
            ),
            _correlation_heatmap(
                {"labels": correlations["labels"], **correlations["active_only"]},
                title="Active-only hourly return correlation",
            ),
            _correlation_heatmap(position_payload, title="Position cosine similarity"),
            _table(
                portfolio["position_similarity"]["pairs"],
                (
                    ("left", "Left"),
                    ("right", "Right"),
                    ("mean_cosine_similarity", "Mean cosine"),
                    ("active_overlap_ratio", "Active overlap"),
                    ("conflict_offset_ratio", "Conflict / offset"),
                    ("both_active_fraction", "Both active"),
                ),
            ),
            "</section>",
            '<section class="operations"><h2>Operations</h2>',
            '<p class="muted">Bar-based simulation: rebalance events, trade legs, and simulated '
            "traded weight are not real orders or fills.</p>",
            _line_svg(
                turnover,
                title="Cumulative simulated turnover — daily display, UTC",
                x_labels=daily_dates,
                x_axis_label="Date (UTC)",
                y_axis_label="Cumulative traded weight",
            ),
            _table(
                portfolio["trading"]["by_symbol"],
                (
                    ("key", "Symbol"),
                    ("trade_legs", "Trade legs"),
                    ("simulated_traded_weight", "Traded weight"),
                ),
            ),
            _table(
                portfolio["trading"]["by_side"],
                (
                    ("key", "Trade direction"),
                    ("trade_legs", "Trade legs"),
                    ("simulated_traded_weight", "Traded weight"),
                ),
            ),
            _weekday_hour_heatmap(portfolio["trading"]["weekday_hour_utc"]),
            _table(
                portfolio["trading"]["largest_events"],
                (
                    ("execution_time_ms", "Execution time UTC (ms)"),
                    ("fold", "Fold"),
                    ("symbol", "Symbol"),
                    ("previous_weight", "Previous"),
                    ("target_weight", "Target"),
                    ("delta_weight", "Delta"),
                    ("event_type", "Event type"),
                    ("component_contributions", "Component deltas"),
                ),
            ),
            "</section>",
            '<section class="stability-risk"><h2>Stability and risk</h2>',
            "<h3>Monthly returns</h3>",
            _table(
                portfolio["monthly_metrics"],
                (
                    ("month", "Month"),
                    ("total_return", "Return"),
                    ("sharpe", "Sharpe"),
                    ("max_drawdown", "Drawdown"),
                ),
            ),
            "<h3>Fold metrics</h3>",
            _table(
                portfolio["fold_metrics"],
                (
                    ("fold", "Fold"),
                    ("total_return", "Return"),
                    ("sharpe", "Sharpe"),
                    ("max_drawdown", "Drawdown"),
                ),
            ),
            "<h3>Regime metrics</h3>",
            _table(
                portfolio["regime_metrics"],
                (
                    ("regime", "Regime"),
                    ("total_return", "Return"),
                    ("sharpe", "Sharpe"),
                    ("max_drawdown", "Drawdown"),
                ),
            ),
            "<h3>Top drawdown episodes</h3>",
            _table(
                portfolio["drawdown_episodes"],
                (
                    ("start_time_ms", "Start UTC (ms)"),
                    ("trough_time_ms", "Trough UTC (ms)"),
                    ("recovery_time_ms", "Recovery UTC (ms)"),
                    ("depth", "Depth"),
                    ("duration_hours", "Hours"),
                    ("recovered", "Recovered"),
                ),
            ),
            "<h3>Symbol attribution and concentration</h3>",
            _table(
                portfolio["symbol_attribution"],
                (
                    ("symbol", "Symbol"),
                    ("price_pnl", "Price"),
                    ("funding_pnl", "Funding"),
                    ("net_pnl", "Sleeve net"),
                    ("turnover", "Sleeve turnover"),
                    ("note", "Netting note"),
                ),
            ),
            _table(
                (portfolio["concentration"],),
                (
                    ("capital_weight_hhi", "Capital HHI"),
                    ("maximum_component_weight", "Maximum component"),
                    ("maximum_absolute_symbol_weight", "Maximum symbol weight"),
                    ("activity_hhi_by_symbol", "Activity HHI"),
                    ("top_activity_symbol", "Top activity symbol"),
                    ("top_activity_share", "Top activity share"),
                    ("top_pnl_symbol", "Top P&L symbol"),
                    ("top_pnl_share", "Top P&L share"),
                    ("volume_participation", "Volume participation"),
                ),
            ),
            "</section></article>",
        )
    )


def _render_portfolios(portfolios: Sequence[Mapping[str, Any]]) -> str:
    if not portfolios:
        return (
            '<div class="empty-state">Nenhum arquivo de portfólio fornecido. O catálogo de '
            "campanhas e experimentos permanece disponível.</div>"
        )
    return "".join(
        _render_portfolio_panel(portfolio, selected=index == 0)
        for index, portfolio in enumerate(portfolios)
    )


_HTML_TEMPLATE = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research portfolio dashboard</title>
<style>
:root { color-scheme: dark; --bg:#090d18; --panel:#121a2a; --panel2:#172238; --line:#2b3957; --text:#edf2fb; --muted:#9caac2; --accent:#62d4c5; --warning:#f4bd68; --danger:#ff8f7a; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,sans-serif; }
main { max-width:1900px; margin:auto; padding:28px; }
h1,h2,h3 { margin:0 0 14px; } h2 { margin-top:34px; } h3 { margin-top:24px; }
.lede,.muted,small { color:var(--muted); }
.science-banner { position:sticky; top:0; z-index:5; padding:12px 16px; margin:-28px -28px 22px; background:#2b2030; border-bottom:1px solid #76506b; font-weight:700; }
.top-nav { display:flex; gap:8px; flex-wrap:wrap; margin:18px 0 24px; }
.top-nav a { text-decoration:none; padding:7px 11px; border:1px solid var(--line); border-radius:999px; background:var(--panel); }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin:22px 0; }
.card,.filters { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }
.card strong { display:block; font-size:24px; color:var(--accent); font-variant-numeric:tabular-nums; }
.filters { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-bottom:12px; }
label { color:var(--muted); } select { width:100%; margin-top:5px; padding:8px; color:var(--text); background:var(--bg); border:1px solid var(--line); }
.table-wrap { overflow:auto; border:1px solid var(--line); border-radius:10px; }
table { width:100%; border-collapse:collapse; background:var(--panel); }
th,td { padding:10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; white-space:nowrap; }
th { position:sticky; top:0; background:#1b243a; z-index:1; }
th button { border:0; color:var(--text); background:transparent; font:inherit; font-weight:700; cursor:pointer; }
td code { display:inline-block; max-width:360px; overflow:hidden; text-overflow:ellipsis; }
a { color:var(--accent); } .badge { font-size:12px; font-weight:700; }
.empty td { padding:24px; text-align:center; color:var(--muted); }
.alert,.empty-state { padding:14px 16px; border:1px solid var(--line); border-radius:10px; background:var(--panel); margin:14px 0; }
.alert.warning { border-color:#806b40; background:#292419; } .alert.danger { border-color:#8b4c4c; background:#2c1b22; }
.portfolio-picker { max-width:760px; margin:14px 0 22px; }
.portfolio-panel { border-top:1px solid var(--line); padding-top:24px; }
.chart { border:1px solid var(--line); background:var(--panel); border-radius:10px; padding:10px; margin:14px 0; }
.chart svg { display:block; width:100%; height:auto; max-height:340px; }
.legend { display:flex; flex-wrap:wrap; gap:14px; color:var(--muted); padding:6px 10px; }
.legend button { border:0; color:var(--muted); background:transparent; cursor:pointer; padding:3px; }
.legend button[aria-pressed="false"] { opacity:.4; text-decoration:line-through; }
.legend i { display:inline-block; width:14px; height:3px; margin-right:6px; vertical-align:middle; }
.matrix { overflow:auto; margin:14px 0; border:1px solid var(--line); border-radius:10px; }
.matrix caption { text-align:left; padding:12px; font-weight:700; background:var(--panel2); }
.matrix td { min-width:90px; text-align:center; font-variant-numeric:tabular-nums; }
.matrix td small { display:block; font-size:10px; } .heat-na { color:var(--muted); }
section { scroll-margin-top:70px; }
[hidden] { display:none !important; }
</style>
</head>
<body><main>
<div class="science-banner">Research only. No strategy is promoted. The evaluated final period is not an independent lockbox.</div>
<h1>Research portfolio dashboard</h1>
<p class="lede">Static, offline and deterministic evidence view. Best observed research result ≠ promoted alpha; portfolio visualization ≠ independent validation; simulated trade legs ≠ real exchange fills.</p>
<nav class="top-nav" aria-label="Dashboard sections"><a href="#overview">Overview</a><a href="#strategy-shortlist">Strategy shortlist</a><a href="#portfolio">Portfolio</a><a href="#portfolio">P&amp;L and drawdown</a><a href="#portfolio">Diversification</a><a href="#portfolio">Operations</a><a href="#portfolio">Stability and risk</a></nav>
<section class="cards">__SUMMARY_CARDS__</section>
<section id="overview"><h2>Overview</h2>
<h2>Campanhas</h2>
<div class="table-wrap"><table id="campaign-table">
<thead><tr>
<th><button data-type="text">Campanha</button></th><th><button data-type="text">Status</button></th>
<th><button data-type="text">Hipótese</button></th><th><button data-type="number">Trials</button></th>
<th><button data-type="number">Sucessos</button></th><th><button data-type="number">Falhas</button></th>
<th><button data-type="number">Criada</button></th><th><button data-type="number">Iniciada</button></th>
<th><button data-type="number">Finalizada</button></th><th><button data-type="text">Último erro</button></th><th>Links</th>
</tr></thead><tbody>__CAMPAIGN_ROWS__</tbody></table></div>
<section id="strategy-shortlist"><h2>Strategy shortlist</h2>
<div class="alert warning">Ranking exploratório. Não constitui seleção OOS independente nem promoção de alpha. Development, confirmation e full permanecem separados por badges e colunas.</div>
<section class="filters">
<label>Strategy<select id="filter-strategy"><option value="">Todas</option>__STRATEGY_OPTIONS__</select></label>
<label>Campanha<select id="filter-campaign"><option value="">Todas</option>__CAMPAIGN_OPTIONS__</select></label>
<label>Hipótese<select id="filter-hypothesis"><option value="">Todas</option>__HYPOTHESIS_OPTIONS__</select></label>
<label>Status<select id="filter-status"><option value="">Todos</option>__STATUS_OPTIONS__</select></label>
<label>Estágio<select id="filter-stage"><option value="">Todos</option>__STAGE_OPTIONS__</select></label>
<label>Validation<select id="filter-validation"><option value="">Todos</option>__VALIDATION_OPTIONS__</select></label>
<label>Compatibility group<select id="filter-compatibility"><option value="">Todos</option>__COMPATIBILITY_OPTIONS__</select></label>
</section>
<p class="muted"><span id="visible-count">0</span> experimento(s) visível(is).</p>
<div class="table-wrap"><table id="experiment-table">
<thead><tr>
<th><button data-type="text">Experimento</button></th><th><button data-type="text">Strategy / parâmetros</button></th>
<th><button data-type="text">Hipótese</button></th><th><button data-type="text">Campanha</button></th>
<th><button data-type="text">Dataset</button></th><th><button data-type="text">Último status</button></th>
<th><button data-type="number">Tentativas</button></th><th><button data-type="number">Runtime (s)</button></th>
<th><button data-type="number">Retorno</button></th><th><button data-type="number">Sharpe</button></th>
<th><button data-type="number">Drawdown</button></th><th><button data-type="number">Turnover</button></th>
<th><button data-type="number">Pior fold</button></th><th><button data-type="number">Custo 1.5x</button></th>
<th><button data-type="number">Custo 2x</button></th><th><button data-type="number">Atraso 1 barra</button></th>
<th><button data-type="number">Calmar</button></th><th><button data-type="number">DSR</button></th>
<th><button data-type="number">PBO</button></th><th><button data-type="number">Symbol concentration</button></th>
<th><button data-type="text">Validation</button></th>
<th><button data-type="text">Estágio</button></th><th><button data-type="text">Artifacts</button></th>
<th><button data-type="text">Gate</button></th><th><button data-type="text">Compatibility</button></th>
<th><button data-type="text">Lockbox</button></th><th>Links</th>
</tr></thead><tbody>__EXPERIMENT_ROWS__</tbody></table></div>
</section></section>
<section id="portfolio"><h2>Portfolio</h2>
<label class="portfolio-picker">Declared portfolio<select id="portfolio-select" aria-label="Declared portfolio">__PORTFOLIO_OPTIONS__</select></label>
__PORTFOLIO_PANELS__
</section>
</main>
<script>
(() => {
  const table = document.getElementById('experiment-table');
  const rows = [...table.querySelectorAll('tbody tr.experiment-row')];
  const filters = ['strategy','campaign','hypothesis','status','stage','validation','compatibility'];
  function applyFilters() {
    let visible = 0;
    for (const row of rows) {
      const matches = filters.every(name => {
        const selected = document.getElementById(`filter-${name}`).value;
        if (!selected) return true;
        if (name === 'campaign') return row.dataset.campaigns.split('|').includes(selected);
        return row.dataset[name] === selected;
      });
      row.hidden = !matches;
      if (matches) visible += 1;
    }
    document.getElementById('visible-count').textContent = String(visible);
  }
  for (const name of filters) document.getElementById(`filter-${name}`).addEventListener('change', applyFilters);
  const portfolioSelect = document.getElementById('portfolio-select');
  if (portfolioSelect) portfolioSelect.addEventListener('change', () => {
    for (const panel of document.querySelectorAll('.portfolio-panel')) {
      panel.hidden = panel.dataset.portfolio !== portfolioSelect.value;
    }
  });
  for (const currentTable of document.querySelectorAll('table')) {
    [...currentTable.querySelectorAll('th button')].forEach((button, index) => {
      let ascending = true;
      button.addEventListener('click', () => {
        const body = currentTable.tBodies[0];
        const currentRows = [...body.querySelectorAll('tr:not(.empty)')];
        const numeric = button.dataset.type === 'number';
        currentRows.sort((left, right) => {
          const a = left.cells[index].dataset.sort || '';
          const b = right.cells[index].dataset.sort || '';
          if (!a && b) return 1; if (a && !b) return -1;
          const order = numeric ? Number(a) - Number(b) : a.localeCompare(b, 'pt-BR');
          return ascending ? order : -order;
        });
        for (const row of currentRows) body.appendChild(row);
        ascending = !ascending;
      });
    });
  }
  for (const chart of document.querySelectorAll('.chart')) {
    for (const button of chart.querySelectorAll('.legend button')) {
      button.setAttribute('aria-pressed', 'true');
      button.addEventListener('click', () => {
        const visible = button.getAttribute('aria-pressed') === 'true';
        button.setAttribute('aria-pressed', String(!visible));
        const line = chart.querySelector(`polyline[data-series-index="${button.dataset.seriesIndex}"]`);
        if (line) line.style.display = visible ? 'none' : '';
      });
    }
  }
  applyFilters();
})();
</script>
</body></html>
"""


def render_dashboard_html(snapshot: Mapping[str, Any]) -> str:
    """Render an offline document while escaping all registry-controlled text."""

    totals = snapshot["totals"]
    cards = "".join(
        f'<article class="card"><span>{_escape(label)}</span><strong>{_escape(totals[key])}</strong></article>'
        for key, label in (
            ("hypotheses", "Hipóteses"),
            ("campaigns", "Campanhas"),
            ("experiments", "Experimentos"),
            ("successes", "Execuções com sucesso"),
            ("failures", "Execuções com falha"),
            ("portfolios", "Portfólios declarados"),
        )
    )
    campaigns = snapshot["campaigns"]
    experiments = snapshot["strategy_shortlist"]
    portfolios = snapshot.get("portfolios", [])
    portfolio_options = (
        "".join(
            f'<option value="{_escape(item["portfolio_id"])}">'
            f"{_escape(item['title'])} — {_escape(item['status'])}</option>"
            for item in portfolios
        )
        if portfolios
        else '<option value="">Nenhum portfólio declarado</option>'
    )
    html_document = _HTML_TEMPLATE
    replacements = {
        "__SUMMARY_CARDS__": cards,
        "__CAMPAIGN_ROWS__": _render_campaign_rows(campaigns),
        "__EXPERIMENT_ROWS__": _render_experiment_rows(experiments),
        "__STRATEGY_OPTIONS__": _select_options(
            [
                (
                    f"{item['strategy_id']}:{item['strategy_version']}",
                    f"{item['strategy_id']}:{item['strategy_version']}",
                )
                for item in experiments
            ]
        ),
        "__CAMPAIGN_OPTIONS__": _select_options(
            [(item["campaign_id"], item["name"]) for item in campaigns]
        ),
        "__HYPOTHESIS_OPTIONS__": _select_options(
            [(item["hypothesis_id"], item["hypothesis_id"]) for item in snapshot["hypotheses"]]
        ),
        "__STATUS_OPTIONS__": _select_options(
            [(item["latest_status"], item["latest_status"]) for item in experiments]
        ),
        "__STAGE_OPTIONS__": _select_options(
            [(item["research_stage"], item["research_stage"]) for item in experiments]
        ),
        "__VALIDATION_OPTIONS__": _select_options(
            [(item["validation_profile"], item["validation_profile"]) for item in experiments]
        ),
        "__COMPATIBILITY_OPTIONS__": _select_options(
            [
                (item["compatibility_group"], str(item["compatibility_group"])[:16])
                for item in experiments
            ]
        ),
        "__PORTFOLIO_OPTIONS__": portfolio_options,
        "__PORTFOLIO_PANELS__": _render_portfolios(portfolios),
    }
    for marker, value in replacements.items():
        html_document = html_document.replace(marker, value)
    return html_document


def _replace_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    except OSError as exc:
        if temporary.exists():
            temporary.unlink()
        raise ResearchError(f"cannot write research dashboard {path}: {exc}") from exc


def build_research_dashboard(
    *,
    store: ResearchStore,
    reports_root: Path,
    data_root: Path,
    portfolio_file: Path | None = None,
) -> DashboardBuildResult:
    """Build deterministic snapshot and offline HTML files."""

    store.initialize()
    directory = reports_root.resolve() / "research_dashboard"
    snapshot = build_dashboard_snapshot(
        store=store,
        reports_root=reports_root,
        data_root=data_root,
        portfolio_file=portfolio_file,
    )
    snapshot_path = directory / "snapshot.json"
    index_path = directory / "index.html"
    _replace_bytes(
        snapshot_path,
        orjson.dumps(snapshot, option=orjson.OPT_SORT_KEYS) + b"\n",
    )
    _replace_bytes(index_path, render_dashboard_html(snapshot).encode("utf-8"))
    return DashboardBuildResult(
        index_path=index_path,
        snapshot_path=snapshot_path,
        snapshot=snapshot,
    )


__all__ = [
    "DASHBOARD_SCHEMA_VERSION",
    "DashboardBuildResult",
    "build_dashboard_snapshot",
    "build_research_dashboard",
    "render_dashboard_html",
]
