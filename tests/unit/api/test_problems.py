import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from fasterrag.api.main import create_app
from fasterrag.api.problems import PROBLEM_MEDIA_TYPE, build_problem
from fasterrag.config.schema import Settings
from fasterrag.errors import (
    ErrorCode,
    FasterRagError,
    IngestionError,
    ProvisioningError,
    RetrievalError,
)


class Body(BaseModel):
    query: str
    top_k: int


@pytest.fixture
def failing_app() -> FastAPI:
    app = create_app(Settings())

    @app.get("/raise-typed")
    async def raise_typed() -> None:
        raise RetrievalError("both retrieval legs failed")

    @app.get("/raise-queue-full")
    async def raise_queue_full() -> None:
        raise IngestionError("chunk queue is full", code=ErrorCode.QUEUE_FULL)

    @app.get("/raise-provisioning")
    async def raise_provisioning() -> None:
        raise ProvisioningError("docker is not running", fix="Start Docker and retry.")

    @app.get("/raise-unexpected")
    async def raise_unexpected() -> None:
        raise RuntimeError("something nobody classified")

    @app.post("/validated")
    async def validated(body: Body) -> dict[str, str]:
        return {"query": body.query}

    return app


@pytest.fixture
def failing_client(failing_app: FastAPI) -> TestClient:
    return TestClient(failing_app, raise_server_exceptions=False)


def test_typed_error_becomes_a_problem_document(failing_client: TestClient) -> None:
    response = failing_client.get("/raise-typed")

    assert response.status_code == 503
    assert response.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    body = response.json()
    assert body["code"] == "RETRIEVAL_FAILED"
    assert body["type"] == "https://fasterrag.dev/problems/retrieval-failed"
    assert body["title"] == "Retrieval failed"
    assert body["status"] == 503
    assert body["detail"] == "both retrieval legs failed"
    assert body["instance"] == "/raise-typed"
    assert body["retryable"] is True
    assert len(body["trace_id"]) == 32


def test_explicit_code_drives_status_and_retryability(failing_client: TestClient) -> None:
    body = failing_client.get("/raise-queue-full").json()

    assert body["code"] == "QUEUE_FULL"
    assert body["status"] == 429
    assert body["retryable"] is True


def test_provisioning_fix_hint_reaches_the_client(failing_client: TestClient) -> None:
    body = failing_client.get("/raise-provisioning").json()

    assert body["code"] == "PROVISIONING_FAILED"
    assert "Start Docker and retry." in body["detail"]


def test_unclassified_failure_still_returns_a_problem_body(failing_client: TestClient) -> None:
    response = failing_client.get("/raise-unexpected")

    assert response.status_code == 500
    assert response.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    body = response.json()
    assert body["code"] == "INTERNAL"
    assert len(body["trace_id"]) == 32
    assert "something nobody classified" not in body["detail"]


def test_validation_failure_lists_the_offending_fields(failing_client: TestClient) -> None:
    response = failing_client.post("/validated", json={"query": "hi"})

    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "VALIDATION_FAILED"
    assert {"field": "body.top_k", "message": "Field required"} in body["errors"]


def test_unknown_path_returns_not_found_as_a_problem(client: TestClient) -> None:
    response = client.get("/v1/does-not-exist")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    assert response.json()["code"] == "NOT_FOUND"


def test_method_not_allowed_keeps_its_transport_status(client: TestClient) -> None:
    response = client.post("/healthz")

    assert response.status_code == 405
    body = response.json()
    assert body["status"] == 405
    assert body["code"] == "INTERNAL"


def test_each_request_gets_a_distinct_trace_id(client: TestClient) -> None:
    first = client.get("/v1/missing-one").json()["trace_id"]
    second = client.get("/v1/missing-two").json()["trace_id"]

    assert first != second


def test_interactive_docs_are_not_served(client: TestClient) -> None:
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 200


def test_build_problem_omits_absent_members() -> None:
    document = build_problem(FasterRagError("boom"))

    assert document.instance is None
    assert "instance" not in document.model_dump(mode="json", exclude_none=True)
