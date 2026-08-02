import json
from pathlib import Path

from fasterrag.config.schema import Settings
from fasterrag.observability import metrics
from fasterrag.services.grafana import (
    DATASOURCE_UID,
    UPDATE_INTERVAL_SECONDS,
    GrafanaPlan,
    dashboard,
    dashboard_provider,
    datasource_manifest,
    prometheus_config,
    write_manifests,
)


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


def test_the_metrics_port_follows_configuration(tmp_path: Path) -> None:
    settings = Settings.model_validate({"app": {"port": 9999}})
    configured = GrafanaPlan(
        root=tmp_path, metrics_host="host.docker.internal", metrics_port=settings.app.port
    )

    assert "host.docker.internal:9999" in prometheus_config(configured)
