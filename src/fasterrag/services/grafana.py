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

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import httpx

from fasterrag.config.schema import Settings
from fasterrag.observability.logging import get_logger
from fasterrag.services.provisioning import (
    ProvisionResult,
    container_state,
    docker_available,
    port_is_reachable,
    require_provisioning_gate,
    run_docker,
)

__all__ = [
    "DASHBOARD_UID",
    "DATASOURCE_UID",
    "DEFAULT_PROVISIONING_ROOT",
    "GRAFANA_CONTAINER",
    "GRAFANA_PORT",
    "NETWORK_NAME",
    "PROMETHEUS_CONTAINER",
    "PROMETHEUS_CONTAINER_PORT",
    "PROMETHEUS_HOST_PORT",
    "SCRAPE_INTERVAL_SECONDS",
    "UPDATE_INTERVAL_SECONDS",
    "GrafanaPlan",
    "Verification",
    "dashboard_provider",
    "datasource_manifest",
    "prometheus_config",
    "verify_grafana",
    "write_manifests",
]

GRAFANA_CONTAINER: Final = "fasterrag-grafana"
PROMETHEUS_CONTAINER: Final = "fasterrag-prometheus"

GRAFANA_PORT: Final = 3001

# CRITICAL: these two are different numbers and must stay that way. Prometheus listens on
# 9090 *inside* its container and that is not configurable without overriding the image's
# command, so the datasource — which reaches it container-to-container — must use 9090. The
# host publication moves to 9099 because the Langfuse stack this repo also provisions
# publishes MinIO on 9090 (docs/observability.md §4), and two provisioners claiming one host
# port would make enabling both toggles a coin flip over which came up. Collapsing these
# into one constant publishes 9099:9099, which binds a port nothing listens on: Grafana then
# renders every panel empty with no error anywhere.
PROMETHEUS_CONTAINER_PORT: Final = 9090
PROMETHEUS_HOST_PORT: Final = 9099

# CRITICAL: a user-defined network, not ``--link``. Links are legacy, work only on the
# default bridge, and are one-directional; the datasource resolves ``fasterrag-prometheus``
# by DNS on this network instead.
NETWORK_NAME: Final = "fasterrag"

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
DASHBOARD_UID: Final = "fasterrag-overview"

UPDATE_INTERVAL_SECONDS: Final = 30
SCRAPE_INTERVAL_SECONDS: Final = 15

_VERIFY_ATTEMPTS: Final = 30
_VERIFY_INTERVAL_SECONDS: Final = 2.0
_VERIFY_TIMEOUT_SECONDS: Final = 10.0

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class GrafanaPlan:
    """Where the manifests go and what they point at."""

    root: Path
    metrics_host: str
    metrics_port: int
    grafana_port: int = GRAFANA_PORT
    prometheus_port: int = PROMETHEUS_HOST_PORT

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

    The URL is container-to-container, so it carries the *container* port. Using the host
    publication here would point Grafana at a port nothing serves inside the network.
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
            f"    url: http://{PROMETHEUS_CONTAINER}:{PROMETHEUS_CONTAINER_PORT}",
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


def _panel(
    identifier: int,
    title: str,
    expression: str,
    unit: str,
    row: int,
    description: str = "",
    legend: str = "",
) -> dict[str, Any]:
    """Return one dashboard panel over a PromQL expression.

    The description is not decoration. A panel that is legitimately empty — because the
    thing it measures has not happened yet — looks exactly like a panel that is empty
    because the datasource is broken, and the only place an operator can be told where it
    stands is on the panel itself.

    # CRITICAL: every target carries a legendFormat. Without one Grafana labels each series
    # with its whole label set, so a legend reads
    # `fasterrag_ingest_throughput{instance="host.docker.internal:8000", job="fasterrag",
    # unit="chu...` — truncated before the only label that distinguishes the series. The
    # scrape labels are identical on every series in a single-instance deployment and carry
    # no information; the one varying label is what the legend has to show.
    """
    return {
        "id": identifier,
        "title": title,
        "description": description,
        "type": "timeseries",
        "datasource": {"type": "prometheus", "uid": DATASOURCE_UID},
        "gridPos": {"h": 8, "w": 12, "x": 0 if identifier % 2 else 12, "y": row},
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"},
        },
        "targets": [{"expr": expression, "refId": "A", "legendFormat": legend or title}],
    }


def dashboard() -> dict[str, Any]:
    """Return the shipped dashboard covering the six documented views.

    Every panel is built from a metric the catalogue actually declares, so a panel cannot
    render empty because it queries a series nothing emits.
    """
    return {
        "uid": DASHBOARD_UID,
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
                "Retrieve, assemble, and generate. Empty until the first query runs.",
                "{{stage}}",
            ),
            _panel(
                2,
                "Ingestion throughput",
                "fasterrag_ingest_throughput",
                "short",
                0,
                "Published once per job, at settle. Empty until the first ingest completes.",
                "{{unit}}",
            ),
            _panel(
                3,
                "Cache hit ratio",
                "(sum by (cache) (rate(fasterrag_cache_events_total{result='hit'}[5m])) "
                "or vector(0)) / "
                "clamp_min(sum by (cache) (rate(fasterrag_cache_events_total[5m])), 1)",
                "percentunit",
                8,
                "Zero before the first cache lookup, which is a real ratio rather than a gap.",
                "{{cache}}",
            ),
            _panel(
                4,
                "Queue and DLQ depth",
                "fasterrag_queue_depth or fasterrag_dlq_depth",
                "short",
                8,
                "Queue depth is published on every enqueue and dequeue; DLQ depth once per job.",
                "{{queue}}{{collection}}",
            ),
            _panel(
                5,
                "Circuit-breaker state",
                "fasterrag_circuit_state",
                "short",
                16,
                "Empty by design for now: only the breaker's configuration exists, not the "
                "breaker (TASK-0148). This panel is not evidence of a datasource problem.",
                "{{provider}}",
            ),
            _panel(
                6,
                "Cost per query",
                "(sum(rate(fasterrag_cost_usd_total[5m])) or vector(0)) / "
                "clamp_min(sum(rate(fasterrag_requests_total[5m])), 1)",
                "currencyUSD",
                16,
                "A list-price estimate from dated published rates, not a measurement. A model "
                "with no recorded rate contributes nothing rather than a fabricated number.",
                "USD per query",
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

    # CRITICAL: the empty plugins/ and alerting/ directories are deliberate. Grafana scans
    # all four provisioning directories at boot and logs a `level=error` line for each one
    # that is absent, so a perfectly healthy install greets its first operator with two
    # errors in the log and no way to tell them apart from a real failure.
    for directory in (
        datasources,
        providers,
        plan.provisioning / "plugins",
        plan.provisioning / "alerting",
        plan.dashboards,
    ):
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


@dataclass(frozen=True, slots=True)
class Verification:
    """What provisioning could confirm about the stack it just started."""

    healthy: bool
    detail: str


async def verify_grafana(plan: GrafanaPlan) -> Verification:
    """Ask Grafana whether the dashboard loaded and the datasource answers.

    Started containers are not working containers. Provisioning that reports success because
    ``docker run`` exited zero is how a stack ends up serving six empty panels while claiming
    to be healthy, so this asks Grafana three questions it can only answer if the wiring is
    right: is it up, did the dashboard provision, and does the datasource reach Prometheus.
    """
    base = f"http://localhost:{plan.grafana_port}"

    try:
        async with httpx.AsyncClient(timeout=_VERIFY_TIMEOUT_SECONDS) as client:
            for _ in range(_VERIFY_ATTEMPTS):
                try:
                    health = await client.get(f"{base}/api/health")
                except httpx.HTTPError:
                    await asyncio.sleep(_VERIFY_INTERVAL_SECONDS)
                    continue
                if health.status_code == 200:
                    break
                await asyncio.sleep(_VERIFY_INTERVAL_SECONDS)
            else:
                return Verification(False, f"grafana did not answer on {base} — check its logs")

            found = await client.get(f"{base}/api/search", params={"type": "dash-db"})
            uids = (
                {entry.get("uid") for entry in found.json()} if found.status_code == 200 else set()
            )
            if DASHBOARD_UID not in uids:
                return Verification(
                    False,
                    f"grafana is up but the {DASHBOARD_UID} dashboard did not provision — "
                    "check that the dashboards directory is mounted",
                )

            probe = await client.get(f"{base}/api/datasources/uid/{DATASOURCE_UID}/health")
            if probe.status_code != 200:
                return Verification(
                    False,
                    "the dashboard provisioned but the datasource cannot reach prometheus — "
                    f"every panel will render empty. Grafana returned {probe.status_code}",
                )
    except httpx.HTTPError as exc:
        return Verification(False, f"could not verify grafana: {exc}")

    return Verification(
        True, "dashboard and datasource verified; dashboards are read-only by design"
    )


async def _ensure_network() -> None:
    """Create the shared network if it is absent.

    ``docker network create`` on an existing network is an error rather than a no-op, so the
    existence check happens first; the command's failure is not treated as fatal because a
    concurrent run may have won the race.
    """
    inspect = await run_docker(["network", "inspect", NETWORK_NAME])
    if inspect.ok:
        return
    await run_docker(["network", "create", NETWORK_NAME])


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

    Raises:
        ProvisioningError: If the doctor preflight fails, before anything is created.
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

    # The same gate the Qdrant provisioner uses. A port conflict reported as a fix-it string
    # before anything starts beats a container that exits seconds later for a reason only
    # `docker logs` can tell you.
    await require_provisioning_gate(settings)

    write_manifests(plan)
    await _ensure_network()

    prometheus = await _ensure_container(
        PROMETHEUS_CONTAINER,
        [
            "run",
            "-d",
            "--name",
            PROMETHEUS_CONTAINER,
            "--restart",
            "unless-stopped",
            "--network",
            NETWORK_NAME,
            "-p",
            f"{plan.prometheus_port}:{PROMETHEUS_CONTAINER_PORT}",
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
            "--restart",
            "unless-stopped",
            "--network",
            NETWORK_NAME,
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
            GRAFANA_IMAGE,
        ],
    )

    verification = await verify_grafana(plan)

    _logger.info(
        "grafana provisioning converged",
        extra={
            "prometheus": prometheus,
            "grafana": grafana,
            "url": plan.url,
            "verified": verification.healthy,
        },
    )
    return ProvisionResult(
        tool="grafana",
        status="running" if verification.healthy else "degraded",
        url=plan.url,
        detail=f"prometheus {prometheus}, grafana {grafana}; {verification.detail}",
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
