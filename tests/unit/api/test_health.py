import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fasterrag.api.health import DependencyStatus, ReadinessRegistry
from fasterrag.api.problems import PROBLEM_MEDIA_TYPE
from fasterrag.errors import ErrorCode
from tests.unit.api.conftest import StubVectorDB


async def failing_check() -> DependencyStatus:
    return DependencyStatus(name="vector_db", ready=False, detail="connection refused")


async def passing_check() -> DependencyStatus:
    return DependencyStatus(name="queue", ready=True)


def test_healthz_reports_liveness(client: TestClient) -> None:
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_healthz_ignores_failing_dependencies(app: FastAPI, client: TestClient) -> None:
    registry: ReadinessRegistry = app.state.readiness
    registry.register(failing_check)

    assert client.get("/healthz").status_code == 200


def test_readyz_reports_every_dependency(client: TestClient) -> None:
    response = client.get("/readyz")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert {"name": "config", "ready": True, "detail": "validated at startup"} in body[
        "dependencies"
    ]
    assert [entry for entry in body["dependencies"] if entry["name"] == "vector_db"]


def test_readyz_fails_when_the_vector_database_is_down(
    app: FastAPI, vector_db: StubVectorDB
) -> None:
    vector_db.healthy = False
    vector_db.detail = "connection refused"

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["code"] == ErrorCode.NOT_READY.value
    assert "vector_db" in body["detail"]
    assert {"name": "vector_db", "ready": False, "detail": "connection refused"} in body[
        "dependencies"
    ]


def test_healthz_stays_up_while_the_vector_database_is_down(
    app: FastAPI, vector_db: StubVectorDB
) -> None:
    vector_db.healthy = False

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200


def test_an_injected_adapter_is_not_closed_by_the_application(
    app: FastAPI, vector_db: StubVectorDB
) -> None:
    with TestClient(app) as client:
        client.get("/healthz")

    assert vector_db.closed is False


def test_readyz_returns_a_problem_body_when_a_dependency_is_down(
    app: FastAPI, client: TestClient
) -> None:
    registry: ReadinessRegistry = app.state.readiness
    registry.register(passing_check)
    registry.register(failing_check)

    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    body = response.json()
    assert body["code"] == ErrorCode.NOT_READY.value
    assert body["status"] == 503
    assert body["retryable"] is True
    assert "vector_db" in body["detail"]
    assert len(body["trace_id"]) == 32
    assert {"name": "vector_db", "ready": False, "detail": "connection refused"} in body[
        "dependencies"
    ]


def test_readyz_reports_all_failing_dependencies(app: FastAPI, client: TestClient) -> None:
    async def second_failure() -> DependencyStatus:
        return DependencyStatus(name="queue", ready=False)

    registry: ReadinessRegistry = app.state.readiness
    registry.register(failing_check)
    registry.register(second_failure)

    body = client.get("/readyz").json()

    assert "vector_db" in body["detail"]
    assert "queue" in body["detail"]


async def test_no_provisioning_runs_when_both_toggles_are_off() -> None:
    """The default configuration must not reach for Docker on startup."""
    from fasterrag.api.main import provision_enabled_observability
    from fasterrag.config.schema import Settings

    await provision_enabled_observability(Settings.model_validate({}))


async def test_a_failed_provision_does_not_stop_the_api_serving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refusing to answer queries because a dashboard would not start inverts the dependency."""
    from fasterrag.api import main
    from fasterrag.config.schema import Settings
    from fasterrag.errors import ProvisioningError

    async def failing(settings: object) -> None:
        raise ProvisioningError("port 3001 is taken")

    monkeypatch.setattr("fasterrag.services.grafana.provision_grafana", failing)
    settings = Settings.model_validate({"observability": {"grafana": True}})

    await main.provision_enabled_observability(settings)


ADMIN_SECRET = "readiness-admin-key-0001"
QUERY_SECRET = "readiness-query-key-0001"


def secured_app(monkeypatch: pytest.MonkeyPatch, vector_db: StubVectorDB) -> FastAPI:
    """Return an application with auth on, one admin key and one query key."""
    from fasterrag.api.main import create_app
    from fasterrag.config.schema import Settings

    monkeypatch.setenv("FASTERRAG_API_KEY", f"{ADMIN_SECRET}:admin;{QUERY_SECRET}:query")
    app = create_app(Settings.model_validate({"security": {"auth": True}}))
    app.state.vector_db = vector_db
    return app


def test_an_unauthenticated_readiness_probe_gets_the_status_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`/readyz` handed adapter detail to anyone who could reach it (TASK-0460)."""
    with TestClient(secured_app(monkeypatch, StubVectorDB())) as client:
        response = client.get("/readyz")

    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_a_failing_dependency_is_still_a_503_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The container HEALTHCHECK probes without a key; its status code must keep meaning."""
    down = StubVectorDB()
    down.healthy = False
    down.detail = "connection refused to 10.0.0.7:6333"

    with TestClient(secured_app(monkeypatch, down)) as client:
        response = client.get("/readyz")
        wrong_key = client.get("/readyz", headers={"Authorization": "Bearer not-a-key"})

    for answer in (response, wrong_key):
        assert answer.status_code == 503
        body = answer.json()
        assert body["code"] == ErrorCode.NOT_READY.value
        assert "dependencies" not in body
        assert "10.0.0.7" not in answer.text
        assert "vector_db" not in answer.text


def test_an_admin_key_sees_every_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    down = StubVectorDB()
    down.healthy = False
    down.detail = "connection refused"

    with TestClient(secured_app(monkeypatch, down)) as client:
        admin = client.get("/readyz", headers={"Authorization": f"Bearer {ADMIN_SECRET}"})
        query = client.get("/readyz", headers={"Authorization": f"Bearer {QUERY_SECRET}"})

    assert admin.status_code == 503
    assert {"name": "vector_db", "ready": False, "detail": "connection refused"} in admin.json()[
        "dependencies"
    ]
    assert query.status_code == 503
    assert "dependencies" not in query.json()
