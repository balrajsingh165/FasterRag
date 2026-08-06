import json
import re
from pathlib import Path

from fasterrag.config.schema import Settings
from fasterrag.observability import metrics
from fasterrag.services.grafana import (
    DATASOURCE_UID,
    GRAFANA_PORT,
    PROMETHEUS_CONTAINER_PORT,
    PROMETHEUS_HOST_PORT,
    UPDATE_INTERVAL_SECONDS,
    GrafanaPlan,
    dashboard,
    dashboard_provider,
    datasource_manifest,
    prometheus_config,
    write_manifests,
)
from fasterrag.services.langfuse import LANGFUSE_PORTS


def plan(tmp_path: Path) -> GrafanaPlan:
    return GrafanaPlan(root=tmp_path, metrics_host="host.docker.internal", metrics_port=8000)


def test_the_datasource_is_not_editable() -> None:
    """A datasource changed in the UI drifts from the file and is silently reverted."""
    assert "editable: false" in datasource_manifest()


def test_the_datasource_uid_is_pinned() -> None:
    """Without a pinned uid Grafana mints a random one and every panel resolves to nothing."""
    assert f"uid: {DATASOURCE_UID}" in datasource_manifest()


def test_the_datasource_points_at_prometheus_not_at_the_exposition_endpoint() -> None:
    """Grafana speaks PromQL; a raw /metrics endpoint would never return a series."""
    manifest = datasource_manifest()

    assert "type: prometheus" in manifest
    assert "/metrics" not in manifest


def test_dashboards_are_not_ui_editable() -> None:
    assert "allowUiUpdates: false" in dashboard_provider()


def test_dashboards_reload_without_a_restart() -> None:
    assert f"updateIntervalSeconds: {UPDATE_INTERVAL_SECONDS}" in dashboard_provider()


def test_prometheus_scrapes_the_metrics_endpoint(tmp_path: Path) -> None:
    config = prometheus_config(plan(tmp_path))

    assert "metrics_path: /metrics" in config
    assert "host.docker.internal:8000" in config


def test_every_panel_references_the_pinned_datasource() -> None:
    for panel in dashboard()["panels"]:
        assert panel["datasource"]["uid"] == DATASOURCE_UID


def test_the_dashboard_covers_the_six_documented_views() -> None:
    titles = " ".join(panel["title"].lower() for panel in dashboard()["panels"])

    for topic in ("latency", "throughput", "cache", "queue", "circuit", "cost"):
        assert topic in titles, f"the dashboard has no {topic} panel"


def test_every_panel_queries_a_metric_the_catalogue_declares() -> None:
    """A panel over an undeclared metric renders empty forever, with no error."""
    declared = set(metrics.REGISTRY.names)

    for panel in dashboard()["panels"]:
        expression = panel["targets"][0]["expr"]
        referenced = {
            name for name in declared if name in expression or f"{name}_bucket" in expression
        }
        assert referenced, f"{panel['title']} queries no declared metric: {expression}"


# Every catalogue metric now has a write site. `fasterrag_circuit_state` was the last
# exemption and left this set when the breaker shipped (TASK-0148). Kept as an empty set
# rather than deleted: it is the seam a future declared-but-unwritten metric must be added
# to deliberately, which is what stops one from joining a growing silent exemption.
KNOWN_UNWRITTEN: set[str] = set()


def _writers() -> dict[str, bool]:
    """Return, per catalogue metric, whether any module ever writes to it.

    Declaration is not emission. A metric declared in the catalogue and written by nothing
    exports zero series, so a panel over it renders "No data" forever with no error in any
    log — which is indistinguishable from a broken datasource. This walks the source for a
    real ``.increment(``/``.set(``/``.observe(`` call site per instrument.
    """
    catalogue = Path(metrics.__file__)
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in catalogue.parent.parent.rglob("*.py")
        if path != catalogue
    )
    declarations = re.findall(
        r"^(\w+) = REGISTRY\.\w+\(\s*\n\s*\"(\w+)\"", catalogue.read_text(encoding="utf-8"), re.M
    )
    return {
        metric: re.search(rf"\b{attribute}\.(increment|set|observe)\(", source) is not None
        for attribute, metric in declarations
    }


def test_every_panel_queries_a_metric_something_actually_writes() -> None:
    written = _writers()

    for panel in dashboard()["panels"]:
        expression = panel["targets"][0]["expr"]
        referenced = [name for name in written if name in expression]
        assert referenced, f"{panel['title']} queries no catalogue metric: {expression}"
        for name in referenced:
            if name in KNOWN_UNWRITTEN:
                assert panel["description"], (
                    f"{panel['title']} is empty by design and says so nowhere; an operator "
                    "cannot tell it apart from a broken datasource"
                )
                continue
            assert written[name], f"{panel['title']} queries {name}, which nothing ever writes"


def test_no_catalogue_metric_is_declared_and_never_written() -> None:
    """A metric nobody writes is either a missing call site or a metric to delete."""
    written = _writers()
    dead = sorted(name for name, has_writer in written.items() if not has_writer)

    assert set(dead) == KNOWN_UNWRITTEN, (
        f"dead metrics changed: {dead}. Every catalogue metric needs a write site or the "
        "panel over it renders 'No data' forever"
    )


def test_every_panel_names_its_series_instead_of_dumping_the_label_set() -> None:
    """Without legendFormat a legend reads as the whole label set, truncated mid-label.

    The scrape labels — instance, job — are identical on every series in a single-instance
    deployment, so they push the one varying label off the end of the line.
    """
    for panel in dashboard()["panels"]:
        legend = panel["targets"][0].get("legendFormat", "")
        assert legend, f"{panel['title']} has no legendFormat"
        assert "instance" not in legend
        assert "job" not in legend


def test_a_multi_series_panel_legends_on_its_varying_label() -> None:
    """A fixed legend on a multi-series panel labels every line identically."""
    varying = {
        "Query latency p50 / p95 per stage": "{{stage}}",
        "Ingestion throughput": "{{unit}}",
        "Cache hit ratio": "{{cache}}",
    }

    for panel in dashboard()["panels"]:
        expected = varying.get(panel["title"])
        if expected:
            assert panel["targets"][0]["legendFormat"] == expected


def test_writing_manifests_creates_every_artifact(tmp_path: Path) -> None:
    written = write_manifests(plan(tmp_path))

    names = {path.name for path in written}
    assert names == {"fasterrag.yaml", "prometheus.yml", "overview.json"}
    assert (tmp_path / "provisioning" / "datasources" / "fasterrag.yaml").is_file()
    assert (tmp_path / "provisioning" / "dashboards" / "fasterrag.yaml").is_file()
    assert (tmp_path / "dashboards" / "overview.json").is_file()


def test_the_written_dashboard_is_valid_json(tmp_path: Path) -> None:
    write_manifests(plan(tmp_path))

    payload = json.loads((tmp_path / "dashboards" / "overview.json").read_text(encoding="utf-8"))

    assert payload["uid"] == "fasterrag-overview"
    assert payload["panels"]


def test_rewriting_manifests_is_idempotent(tmp_path: Path) -> None:
    """Provisioning converges: a second run must leave the same bytes on disk."""
    write_manifests(plan(tmp_path))
    first = (tmp_path / "prometheus.yml").read_bytes()

    write_manifests(plan(tmp_path))

    assert (tmp_path / "prometheus.yml").read_bytes() == first


def test_manifests_carry_no_secret(tmp_path: Path) -> None:
    for path in write_manifests(plan(tmp_path)):
        content = path.read_text(encoding="utf-8").lower()
        assert "api_key" not in content
        assert "password" not in content


def test_the_plan_derives_its_url_from_the_port(tmp_path: Path) -> None:
    assert plan(tmp_path).url.endswith(":3001")


def test_the_datasource_uses_the_container_port_not_the_host_publication() -> None:
    """Grafana reaches Prometheus inside the network, where it listens on 9090.

    Collapsing the two numbers into one publishes 9099:9099 and binds a port nothing serves;
    every panel then renders empty with no error in any log.
    """
    assert f":{PROMETHEUS_CONTAINER_PORT}" in datasource_manifest()
    assert f":{PROMETHEUS_HOST_PORT}" not in datasource_manifest()


def test_the_host_publication_collides_with_nothing_langfuse_claims() -> None:
    """Both toggles can be on at once, and Langfuse publishes MinIO on Prometheus's default."""
    assert PROMETHEUS_HOST_PORT not in LANGFUSE_PORTS
    assert GRAFANA_PORT not in LANGFUSE_PORTS


def test_every_provisioning_directory_grafana_scans_exists(tmp_path: Path) -> None:
    """Grafana logs a level=error line per absent directory, on an otherwise healthy boot."""
    write_manifests(plan(tmp_path))

    for name in ("datasources", "dashboards", "plugins", "alerting"):
        assert (tmp_path / "provisioning" / name).is_dir(), f"{name} is missing"


def test_the_metrics_port_follows_configuration(tmp_path: Path) -> None:
    settings = Settings.model_validate({"app": {"port": 9999}})
    configured = GrafanaPlan(
        root=tmp_path, metrics_host="host.docker.internal", metrics_port=settings.app.port
    )

    assert "host.docker.internal:9999" in prometheus_config(configured)
