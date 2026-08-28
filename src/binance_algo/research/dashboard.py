"""Deterministic, offline HTML dashboard for the research registry."""

# ruff: noqa: E501 -- keeping the embedded HTML/CSS/JavaScript readable is preferable here.

from __future__ import annotations

import html
import os
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import orjson

from binance_algo.common.errors import ResearchError
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

DASHBOARD_SCHEMA_VERSION = 1
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


def _relative_href(target: Path, *, dashboard_directory: Path, directory: bool = False) -> str:
    try:
        relative = os.path.relpath(target.resolve(), dashboard_directory.resolve())
    except ValueError:
        return target.resolve().as_uri()
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
        report_href, artifacts_href = _artifact_links(
            artifacts,
            data_root=data_root,
            dashboard_directory=dashboard_directory,
        )
        linked_campaigns = sorted(
            campaigns_by_experiment.get(identifier, []), key=lambda item: item.campaign_id
        )
        serialized_spec = spec.model_dump(mode="json")
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
                "signal_delay_1_bar_return": stress.get(("signal_delay_1_bar", "total_return")),
                "research_stage": _current_stage(store, identifier).value,
                "metrics_run_id": successful.run_id if successful is not None else None,
                "error_type": latest.error_type if latest is not None else None,
                "error_message": latest.error_message if latest is not None else None,
                "report_href": report_href,
                "artifacts_href": artifacts_href,
            }
        )

    return {
        "schema_version": DASHBOARD_SCHEMA_VERSION,
        "totals": totals,
        "hypotheses": [item.model_dump(mode="json") for item in hypotheses],
        "campaigns": campaign_rows,
        "experiments": experiment_rows,
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
        return '<tr class="empty"><td colspan="17">Nenhum experimento registrado.</td></tr>'
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
                    f'data-stage="{_escape(experiment["research_stage"])}">',
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
                    f'<td data-sort="{_sort_value(experiment["signal_delay_1_bar_return"])}">'
                    f"{_format_percent(experiment['signal_delay_1_bar_return'])}</td>",
                    f'<td data-sort="{_escape(experiment["research_stage"])}">'
                    f"{_escape(experiment['research_stage'])}</td>",
                    f"<td>{links}</td>",
                    "</tr>",
                )
            )
        )
    return "".join(rows)


_HTML_TEMPLATE = """<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Research dashboard</title>
<style>
:root { color-scheme: dark; --bg:#0b1020; --panel:#141b2d; --line:#2a3550; --text:#e8edf7; --muted:#9eabc2; --accent:#72d6c9; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,sans-serif; }
main { max-width:1800px; margin:auto; padding:28px; }
h1,h2 { margin:0 0 14px; } h2 { margin-top:30px; }
.lede,.muted,small { color:var(--muted); }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin:22px 0; }
.card,.filters { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; }
.card strong { display:block; font-size:26px; color:var(--accent); }
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
[hidden] { display:none !important; }
</style>
</head>
<body><main>
<h1>Research dashboard</h1>
<p class="lede">Snapshot offline do ResearchStore. Resultados negativos e falhas permanecem visíveis.</p>
<section class="cards">__SUMMARY_CARDS__</section>
<h2>Campanhas</h2>
<div class="table-wrap"><table id="campaign-table">
<thead><tr>
<th><button data-type="text">Campanha</button></th><th><button data-type="text">Status</button></th>
<th><button data-type="text">Hipótese</button></th><th><button data-type="number">Trials</button></th>
<th><button data-type="number">Sucessos</button></th><th><button data-type="number">Falhas</button></th>
<th><button data-type="number">Criada</button></th><th><button data-type="number">Iniciada</button></th>
<th><button data-type="number">Finalizada</button></th><th><button data-type="text">Último erro</button></th><th>Links</th>
</tr></thead><tbody>__CAMPAIGN_ROWS__</tbody></table></div>
<h2>Experimentos</h2>
<section class="filters">
<label>Strategy<select id="filter-strategy"><option value="">Todas</option>__STRATEGY_OPTIONS__</select></label>
<label>Campanha<select id="filter-campaign"><option value="">Todas</option>__CAMPAIGN_OPTIONS__</select></label>
<label>Hipótese<select id="filter-hypothesis"><option value="">Todas</option>__HYPOTHESIS_OPTIONS__</select></label>
<label>Status<select id="filter-status"><option value="">Todos</option>__STATUS_OPTIONS__</select></label>
<label>Estágio<select id="filter-stage"><option value="">Todos</option>__STAGE_OPTIONS__</select></label>
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
<th><button data-type="number">Atraso 1 barra</button></th><th><button data-type="text">Estágio</button></th><th>Links</th>
</tr></thead><tbody>__EXPERIMENT_ROWS__</tbody></table></div>
</main>
<script>
(() => {
  const table = document.getElementById('experiment-table');
  const rows = [...table.querySelectorAll('tbody tr.experiment-row')];
  const filters = ['strategy','campaign','hypothesis','status','stage'];
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
        )
    )
    campaigns = snapshot["campaigns"]
    experiments = snapshot["experiments"]
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
) -> DashboardBuildResult:
    """Build deterministic snapshot and offline HTML files."""

    store.initialize()
    directory = reports_root.resolve() / "research_dashboard"
    snapshot = build_dashboard_snapshot(
        store=store,
        reports_root=reports_root,
        data_root=data_root,
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
