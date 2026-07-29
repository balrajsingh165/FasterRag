from fastapi import FastAPI
from fastapi.testclient import TestClient

from fasterrag.api.health import DependencyStatus, ReadinessRegistry
from fasterrag.api.problems import PROBLEM_MEDIA_TYPE
from fasterrag.errors import ErrorCode


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
