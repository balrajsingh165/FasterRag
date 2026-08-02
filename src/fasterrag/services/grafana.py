"""Grafana auto-provisioning as code (``observability.grafana: true``).

Flipping the toggle writes provisioning manifests, starts the containers, and returns a URL.
No manual clicks, and — the rule that matters — **no application-code change at toggle time**.

Everything Grafana serves is generated from files on disk, with ``editable: false`` on
datasources and ``allowUiUpdates: false`` on dashboard providers. The UI becomes read-only
for provisioned resources on purpose: a dashboard edited in the browser is a change nobody
can review, that no repository records, and that the next provisioning run silently reverts.

**Prometheus is provisioned alongside Grafana**, because Grafana cannot scrape. Its Prometheus
datasource speaks PromQL to a Prometheus server; pointing it straight at fasterRag's
``/metrics`` exposition endpoint would produce a datasource that never returns a series.
``docs/observability.md`` §5 describes the datasource as pointing at fasterRag's metrics
endpoint, which is one hop short of workable — see TASK-0146.

Provisioning converges rather than reinstalls: manifests are rewritten, containers are
started only if absent, and a re-run changes nothing an operator has not changed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from fasterrag.config.schema import Settings
from fasterrag.observability.logging import get_logger
from fasterrag.services.provisioning import (
    ProvisionResult,
    container_state,
    docker_available,
    port_is_reachable,
    run_docker,
)

__all__ = [
    "DATASOURCE_UID",
    "DEFAULT_PROVISIONING_ROOT",
    "GRAFANA_CONTAINER",
    "GRAFANA_PORT",
    "PROMETHEUS_CONTAINER",
    "PROMETHEUS_PORT",
    "SCRAPE_INTERVAL_SECONDS",
    "UPDATE_INTERVAL_SECONDS",
    "GrafanaPlan",
    "dashboard_provider",
    "datasource_manifest",
    "prometheus_config",
    "write_manifests",
]

GRAFANA_CONTAINER: Final = "fasterrag-grafana"
PROMETHEUS_CONTAINER: Final = "fasterrag-prometheus"

GRAFANA_PORT: Final = 3001
PROMETHEUS_PORT: Final = 9090

# CRITICAL: images are pinned. A floating tag makes a provisioning run non-reproducible —
# two operators following the same instructions would get different Grafana versions, and a
# dashboard schema change would break one of them for no visible reason.
GRAFANA_IMAGE: Final = "grafana/grafana:11.6.1"
PROMETHEUS_IMAGE: Final = "prom/prometheus:v3.2.1"

DEFAULT_PROVISIONING_ROOT: Final = Path(".fasterrag") / "grafana"

# CRITICAL: the datasource uid is pinned rather than left to Grafana. Without it Grafana
# mints a random uid per installation, and every dashboard panel — which references the
# datasource by uid — resolves to nothing. The dashboards would render, empty, with no error.
DATASOURCE_UID: Final = "fasterrag-prometheus"

UPDATE_INTERVAL_SECONDS: Final = 30
SCRAPE_INTERVAL_SECONDS: Final = 15

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class GrafanaPlan:
    """Where the manifests go and what they point at."""

    root: Path
    metrics_host: str
    metrics_port: int
    grafana_port: int = GRAFANA_PORT
    prometheus_port: int = PROMETHEUS_PORT

    @property
    def provisioning(self) -> Path:
        """Return the directory mounted at Grafana's provisioning path."""
        return self.root / "provisioning"

    @property
    def dashboards(self) -> Path:
        """Return the directory holding dashboard JSON."""
        return self.root / "dashboards"

    @property
    def url(self) -> str:
        """Return the URL an operator opens."""
        return f"http://localhost:{self.grafana_port}"


def datasource_manifest() -> str:
    """Return the datasource manifest, marked non-editable.

    ``editable: false`` is what makes the repository the source of truth: a datasource
    changed in the UI would drift from the file and be reverted without explanation on the
    next provisioning run.
    """
    return "\n".join(
        [
            "apiVersion: 1",
            "",
            "datasources:",
            "  - name: fasterRag",
            f"    uid: {DATASOURCE_UID}",
            "    type: prometheus",
            "    access: proxy",
            f"    url: http://{PROMETHEUS_CONTAINER}:{PROMETHEUS_PORT}",
            "    isDefault: true",
            "    editable: false",
            "",
        ]
    )


def dashboard_provider() -> str:
    """Return the dashboard provider manifest.

    ``allowUiUpdates: false`` plus a polling interval is the combination that makes
    dashboards behave like code: edits land by changing the file, and the change applies
    without restarting Grafana.
    """
    return "\n".join(
        [
            "apiVersion: 1",
            "",
            "providers:",
            "  - name: fasterrag",
            "    orgId: 1",
            "    folder: fasterRag",
            "    type: file",
            "    disableDeletion: false",
            "    allowUiUpdates: false",
            f"    updateIntervalSeconds: {UPDATE_INTERVAL_SECONDS}",
            "    options:",
            "      path: /var/lib/grafana/dashboards",
            "      foldersFromFilesStructure: false",
            "",
        ]
    )


def prometheus_config(plan: GrafanaPlan) -> str:
    """Return the Prometheus scrape configuration for fasterRag's metrics endpoint."""
    return "\n".join(
        [
            "global:",
            f"  scrape_interval: {SCRAPE_INTERVAL_SECONDS}s",
            "",
            "scrape_configs:",
            "  - job_name: fasterrag",
            "    metrics_path: /metrics",
            "    static_configs:",
            f"      - targets: ['{plan.metrics_host}:{plan.metrics_port}']",
            "",
        ]
    )


def _panel(identifier: int, title: str, expression: str, unit: str, row: int) -> dict[str, Any]:
    """Return one dashboard panel over a PromQL expression."""
    return {
        "id": identifier,
        "title": title,
        "type": "timeseries",
        "datasource": {"type": "prometheus", "uid": DATASOURCE_UID},
        "gridPos": {"h": 8, "w": 12, "x": 0 if identifier % 2 else 12, "y": row},
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "targets": [{"expr": expression, "refId": "A"}],
    }


def dashboard() -> dict[str, Any]:
    """Return the shipped dashboard covering the six documented views.

    Every panel is built from a metric the catalogue actually declares, so a panel cannot
    render empty because it queries a series nothing emits.
    """
    return {
        "uid": "fasterrag-overview",
        "title": "fasterRag — overview",
        "tags": ["fasterrag"],
        "timezone": "browser",
        "schemaVersion": 39,
        "refresh": "30s",
        "panels": [
            _panel(
                1,
                "Query latency p50 / p95 per stage",
                "histogram_quantile(0.95, sum by (le, stage) "
                "(rate(fasterrag_stage_duration_seconds_bucket[5m])))",
                "s",
                0,
            ),
            _panel(
                2,
                "Ingestion throughput",
                "fasterrag_ingest_throughput",
                "short",
                0,
            ),
            _panel(
                3,
                "Cache hit ratio",
                "sum by (cache) (rate(fasterrag_cache_events_total{result='hit'}[5m])) / "
                "clamp_min(sum by (cache) (rate(fasterrag_cache_events_total[5m])), 1)",
                "percentunit",
                8,
            ),
            _panel(4, "Queue and DLQ depth", "fasterrag_queue_depth", "short", 8),
            _panel(5, "Circuit-breaker state", "fasterrag_circuit_state", "short", 16),
            _panel(
                6,
                "Cost per query",
                "sum(rate(fasterrag_cost_usd_total[5m])) / "
                "clamp_min(sum(rate(fasterrag_requests_total[5m])), 1)",
                "currencyUSD",
                16,
            ),
        ],
    }


def write_manifests(plan: GrafanaPlan) -> list[Path]:
    """Write every provisioning artifact, returning what was written.

    Rewritten on every run rather than written once: these files are generated from
    configuration, so an operator who edited one by hand has made a change that the next
    run must overwrite — which is exactly what provisioning-as-code means.
    """
    datasources = plan.provisioning / "datasources"
    providers = plan.provisioning / "dashboards"
    for directory in (datasources, providers, plan.dashboards):
        directory.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    targets = (
        (datasources / "fasterrag.yaml", datasource_manifest()),
        (providers / "fasterrag.yaml", dashboard_provider()),
        (plan.root / "prometheus.yml", prometheus_config(plan)),
        (plan.dashboards / "overview.json", json.dumps(dashboard(), indent=2) + "\n"),
    )
    for path, content in targets:
        path.write_text(content, encoding="utf-8")
        written.append(path)

    _logger.info(
        "wrote grafana provisioning manifests",
        extra={"root": str(plan.root), "files": [str(path) for path in written]},
    )
    return written


async def _ensure_container(name: str, arguments: list[str]) -> str:
    """Start a container if it is absent, restart it if stopped, else leave it alone."""
    state = await container_state(name)
    if state.running:
        return "already running"

    if state.exists:
        await run_docker(["start", name])
        return "restarted"

    await run_docker(arguments)
    return "created"


async def provision_grafana(settings: Settings, *, root: Path | None = None) -> ProvisionResult:
    """Provision Prometheus and Grafana from generated manifests.

    Returns:
        The result, carrying the URL an operator opens. Provisioning converges: a second run
        rewrites the manifests and leaves running containers untouched.
    """
    plan = GrafanaPlan(
        root=(root or DEFAULT_PROVISIONING_ROOT).resolve(),
        metrics_host="host.docker.internal",
        metrics_port=settings.app.port,
    )

    if not await docker_available():
        return ProvisionResult(
            tool="grafana",
            status="unavailable",
            detail="docker is not running; start it and re-run",
        )

    write_manifests(plan)

    prometheus = await _ensure_container(
        PROMETHEUS_CONTAINER,
        [
            "run",
            "-d",
            "--name",
            PROMETHEUS_CONTAINER,
            "-p",
            f"{plan.prometheus_port}:{PROMETHEUS_PORT}",
            "-v",
            f"{plan.root / 'prometheus.yml'}:/etc/prometheus/prometheus.yml:ro",
            "--add-host",
            "host.docker.internal:host-gateway",
            PROMETHEUS_IMAGE,
        ],
    )

    grafana = await _ensure_container(
        GRAFANA_CONTAINER,
        [
            "run",
            "-d",
            "--name",
            GRAFANA_CONTAINER,
            "-p",
            f"{plan.grafana_port}:3000",
            "-v",
            f"{plan.provisioning}:/etc/grafana/provisioning:ro",
            "-v",
            f"{plan.dashboards}:/var/lib/grafana/dashboards:ro",
            "-e",
            "GF_AUTH_ANONYMOUS_ENABLED=true",
            "-e",
            "GF_AUTH_ANONYMOUS_ORG_ROLE=Viewer",
            "--link",
            f"{PROMETHEUS_CONTAINER}:{PROMETHEUS_CONTAINER}",
            GRAFANA_IMAGE,
        ],
    )

    _logger.info(
        "grafana provisioning converged",
        extra={"prometheus": prometheus, "grafana": grafana, "url": plan.url},
    )
    return ProvisionResult(
        tool="grafana",
        status="running",
        url=plan.url,
        detail=f"prometheus {prometheus}, grafana {grafana}; dashboards are read-only by design",
    )


async def grafana_status(*, root: Path | None = None) -> ProvisionResult:
    """Report whether the provisioned stack is up."""
    plan = GrafanaPlan(
        root=(root or DEFAULT_PROVISIONING_ROOT).resolve(),
        metrics_host="host.docker.internal",
        metrics_port=0,
    )

    if not await docker_available():
        return ProvisionResult(tool="grafana", status="unavailable", detail="docker is not running")

    grafana = await container_state(GRAFANA_CONTAINER)
    prometheus = await container_state(PROMETHEUS_CONTAINER)
    reachable = port_is_reachable("localhost", plan.grafana_port)

    status = "running" if grafana.running and prometheus.running else "stopped"
    return ProvisionResult(
        tool="grafana",
        status=status,
        url=plan.url if reachable else None,
        detail=(
            f"grafana {'running' if grafana.running else 'stopped'}, "
            f"prometheus {'running' if prometheus.running else 'stopped'}, "
            f"port {plan.grafana_port} {'reachable' if reachable else 'not reachable'}"
        ),
    )


async def stop_grafana() -> ProvisionResult:
    """Stop the provisioned containers, leaving the manifests in place.

    Manifests survive deliberately: they are the source of truth, and deleting them on stop
    would mean a restart had to regenerate them before anything could come back.
    """
    stopped: list[str] = []
    for name in (GRAFANA_CONTAINER, PROMETHEUS_CONTAINER):
        state = await container_state(name)
        if state.running:
            await run_docker(["stop", name])
            stopped.append(name)

    return ProvisionResult(
        tool="grafana",
        status="stopped",
        detail=f"stopped {', '.join(stopped) or 'nothing'}; provisioning manifests are kept",
    )
