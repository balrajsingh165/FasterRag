from pathlib import Path

from fastapi.testclient import TestClient

from fasterrag.config.schema import Settings
from fasterrag.core.tracing import Trace
from fasterrag.observability import metrics
from fasterrag.observability.dashboard import DASHBOARD_TITLE, create_dashboard, render_page
from fasterrag.services.traces import TraceStore

WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


def store(tmp_path: Path, *traces: Trace) -> TraceStore:
    keeper = TraceStore(tmp_path)
    for trace in traces:
        keeper.store(trace)
    return keeper


def trace(trace_id: str = "a" * 32, **overrides: object) -> Trace:
    fields: dict[str, object] = {
        "trace_id": trace_id,
        "query": "what is the policy",
        "collection": "docs",
        "created_at": "2026-08-04T00:00:00+00:00",
    }
    fields.update(overrides)
    return Trace(**fields)  # type: ignore[arg-type]


def test_the_dashboard_declares_no_write_route(tmp_path: Path) -> None:
    """The hard rule from observability.md: it can never control the RAG.

    Asserted as a property of the application rather than trusted as a promise — an app
    with no write routes cannot grow one by accident.
    """
    app = create_dashboard(Settings.model_validate({}), store(tmp_path))

    for route in app.routes:
        methods: set[str] = getattr(route, "methods", set()) or set()
        assert not (methods & WRITE_METHODS), getattr(route, "path", route)


def test_the_page_renders(tmp_path: Path) -> None:
    app = create_dashboard(Settings.model_validate({}), store(tmp_path, trace()))

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert DASHBOARD_TITLE in response.text


def test_a_stored_query_is_listed(tmp_path: Path) -> None:
    app = create_dashboard(Settings.model_validate({}), store(tmp_path, trace()))

    with TestClient(app) as client:
        body = client.get("/").text

    assert "what is the policy" in body


def test_query_text_cannot_inject_markup(tmp_path: Path) -> None:
    """A trace carries user-supplied text and model output straight onto an operator's page."""
    hostile = trace(query="<script>alert('xss')</script>")
    app = create_dashboard(Settings.model_validate({}), store(tmp_path, hostile))

    with TestClient(app) as client:
        body = client.get("/").text

    assert "<script>" not in body
    assert "&lt;script&gt;" in body


def test_an_empty_deployment_still_renders(tmp_path: Path) -> None:
    """A dashboard that errors before the first query is useless exactly when it is opened."""
    app = create_dashboard(Settings.model_validate({}), store(tmp_path))

    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "no traces stored yet" in response.text


def test_the_json_view_matches_the_page(tmp_path: Path) -> None:
    app = create_dashboard(Settings.model_validate({}), store(tmp_path, trace()))

    with TestClient(app) as client:
        payload = client.get("/api/traces").json()

    assert payload["traces"][0]["query"] == "what is the policy"


def test_the_metric_names_come_from_the_registry(tmp_path: Path) -> None:
    """One source of truth: the dashboard and a scrape cannot disagree."""
    app = create_dashboard(Settings.model_validate({}), store(tmp_path))

    with TestClient(app) as client:
        names = client.get("/api/metrics").json()["metrics"]

    assert names == metrics.REGISTRY.names


def test_a_recorded_metric_reaches_the_page(tmp_path: Path) -> None:
    metrics.REQUESTS.increment(endpoint="/dash", method="GET", status="200", tenant="none")
    app = create_dashboard(Settings.model_validate({}), store(tmp_path))

    with TestClient(app) as client:
        body = client.get("/").text

    assert "/dash" in body


def test_the_tenant_is_shown(tmp_path: Path) -> None:
    app = create_dashboard(Settings.model_validate({}), store(tmp_path, trace(tenant="acme")))

    with TestClient(app) as client:
        assert "acme" in client.get("/").text


def test_rendering_needs_no_store_at_all() -> None:
    """The renderer is pure, so the page can be asserted without any I/O."""
    assert "no traces stored yet" in render_page([])


def test_the_dashboard_serves_no_schema(tmp_path: Path) -> None:
    """It is not an API surface; publishing a schema invites it to be treated as one."""
    app = create_dashboard(Settings.model_validate({}), store(tmp_path))

    with TestClient(app) as client:
        assert client.get("/openapi.json").status_code == 404


def secured(tmp_path: Path, monkeypatch, *traces: Trace, tenancy: bool = False):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("FASTERRAG_API_KEY", "dash-key:admin:acme")
    settings = Settings.model_validate({"security": {"auth": True, "multi_tenancy": tenancy}})
    return TestClient(create_dashboard(settings, store(tmp_path, *traces)))


def test_the_dashboard_is_unauthenticated_when_the_api_is(tmp_path: Path) -> None:
    """Auth off is the single-operator deployment; the dashboard must not invent a gate."""
    app = create_dashboard(Settings.model_validate({}), store(tmp_path, trace()))

    with TestClient(app) as client:
        assert client.get("/").status_code == 200


def test_the_dashboard_requires_a_key_when_the_api_does(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """It shows prompts, responses, and corpus text — at least the API's protection."""
    with secured(tmp_path, monkeypatch, trace()) as client:
        assert client.get("/").status_code == 401


def test_a_valid_key_reaches_the_dashboard(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    with secured(tmp_path, monkeypatch, trace()) as client:
        response = client.get("/", headers={"Authorization": "Bearer dash-key"})

    assert response.status_code == 200


def test_a_tenant_sees_only_its_own_traces(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """A trace carries the query text and the retrieved chunks."""
    mine = trace("a" * 32, query="my own question", tenant="acme")
    theirs = trace("b" * 32, query="a competitor secret", tenant="globex")

    with secured(tmp_path, monkeypatch, mine, theirs, tenancy=True) as client:
        body = client.get(
            "/", headers={"Authorization": "Bearer dash-key", "X-Tenant-ID": "acme"}
        ).text

    assert "my own question" in body
    assert "a competitor secret" not in body


def test_the_json_view_is_scoped_too(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Scoping the page but not its JSON twin would leave the leak one URL away."""
    mine = trace("a" * 32, query="mine", tenant="acme")
    theirs = trace("b" * 32, query="theirs", tenant="globex")

    with secured(tmp_path, monkeypatch, mine, theirs, tenancy=True) as client:
        payload = client.get(
            "/api/traces", headers={"Authorization": "Bearer dash-key", "X-Tenant-ID": "acme"}
        ).json()

    assert [item["query"] for item in payload["traces"]] == ["mine"]
