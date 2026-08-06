"""The D8 trace endpoints, including the tenant boundary they sit on.

``POST /v1/replay`` looked up its trace **unscoped** while the two ``GET`` endpoints scoped
theirs, so any authenticated tenant could replay another tenant's trace — re-executing their
question and getting a diff built from their retrieved chunks. The endpoint was at 44%
coverage with no tenancy test, which is how it went unnoticed.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings
from fasterrag.core.tracing import Span, Trace
from fasterrag.errors import ErrorCode
from fasterrag.services.traces import TraceStore
from tests.unit.api.conftest import StubVectorDB

ACME_KEY = "acme-key"
EVIL_KEY = "evil-key"
KEYS = f"{ACME_KEY}:admin:acme;{EVIL_KEY}:admin:evilcorp"
TRACE_ID = "aaaabbbbccccddddeeeeffff00001111"


def stored(root: Path, trace_id: str = TRACE_ID, tenant: str | None = "acme") -> TraceStore:
    store = TraceStore(root)
    store.store(
        Trace(
            trace_id=trace_id,
            query="the confidential question",
            tenant=tenant,
            retrieved=[{"chunk_id": "secret-1", "text": "internal only"}],
            spans=[Span("retrieval", 0.0, 1.0, {})],
            created_at="2026-08-06T10:00:00+00:00",
        )
    )
    return store


def tenanted(monkeypatch: pytest.MonkeyPatch, root: Path, **overrides: object) -> TestClient:
    monkeypatch.setenv("FASTERRAG_API_KEY", KEYS)
    payload: dict[str, object] = {
        "security": {"auth": True, "multi_tenancy": True},
        "traces": {"store": True, "replay": True},
    }
    payload.update(overrides)
    app = create_app(Settings.model_validate(payload))
    app.state.vector_db = StubVectorDB()
    app.state.traces = stored(root)
    return TestClient(app, raise_server_exceptions=False)


def headers(tenant: str, secret: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {secret}", "X-Tenant-ID": tenant}


def owner() -> dict[str, str]:
    return headers("acme", ACME_KEY)


def outsider() -> dict[str, str]:
    return headers("evilcorp", EVIL_KEY)


def test_an_owner_reads_its_own_trace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = tenanted(monkeypatch, tmp_path)

    response = client.get(f"/v1/traces/{TRACE_ID}", headers=owner())

    assert response.status_code == 200
    assert response.json()["query"] == "the confidential question"


def test_another_tenant_is_told_it_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Absent, not forbidden: 403 would confirm the id is real to someone guessing."""
    client = tenanted(monkeypatch, tmp_path)

    response = client.get(f"/v1/traces/{TRACE_ID}", headers=outsider())

    assert response.status_code == 404


def test_listing_is_scoped_to_the_caller(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = tenanted(monkeypatch, tmp_path)

    assert client.get("/v1/traces", headers=owner()).json()["traces"] == [TRACE_ID]
    assert client.get("/v1/traces", headers=outsider()).json()["traces"] == []


def test_replay_refuses_another_tenants_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The bug: replay looked its trace up unscoped while both GETs scoped theirs.

    A 404 is the refusal. Anything else means the lookup let the caller through — the run
    itself may then fail on an unreachable backend, but the disclosure has already happened.
    """
    client = tenanted(monkeypatch, tmp_path)

    response = client.post("/v1/replay", json={"trace_id": TRACE_ID}, headers=outsider())

    assert response.status_code == 404


def test_replay_does_not_echo_another_tenants_query(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Even the refusal must not repeat what it refused to hand over."""
    client = tenanted(monkeypatch, tmp_path)

    response = client.post("/v1/replay", json={"trace_id": TRACE_ID}, headers=outsider())

    assert "the confidential question" not in response.text
    assert "internal only" not in response.text


def test_replay_reaches_the_owner_s_own_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The scoping must not lock an owner out of its own trace."""
    client = tenanted(monkeypatch, tmp_path)

    response = client.post("/v1/replay", json={"trace_id": TRACE_ID}, headers=owner())

    assert response.status_code != 404


def test_replay_reports_an_unknown_trace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = tenanted(monkeypatch, tmp_path)

    response = client.post("/v1/replay", json={"trace_id": "f" * 32}, headers=owner())

    assert response.status_code == 404
    assert response.json()["code"] == ErrorCode.NOT_FOUND.value


def test_replay_disabled_is_refused_as_a_configuration_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A disabled feature answering 404 would read as a missing trace."""
    client = tenanted(monkeypatch, tmp_path, traces={"store": True, "replay": False})

    response = client.post("/v1/replay", json={"trace_id": TRACE_ID}, headers=owner())

    assert response.json()["code"] == ErrorCode.VALIDATION_FAILED.value


def test_an_unknown_trace_is_a_problem_document(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = tenanted(monkeypatch, tmp_path)

    response = client.get("/v1/traces/" + "f" * 32, headers=owner())

    assert response.status_code == 404
    assert response.json()["code"] == ErrorCode.NOT_FOUND.value


def test_the_trace_endpoints_need_a_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A trace carries the query and every retrieved chunk; it is not public."""
    client = tenanted(monkeypatch, tmp_path)

    assert client.get(f"/v1/traces/{TRACE_ID}").status_code == 401
    assert client.post("/v1/replay", json={"trace_id": TRACE_ID}).status_code == 401
