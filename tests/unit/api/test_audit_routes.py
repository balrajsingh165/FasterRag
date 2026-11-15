"""Every administrative route writes exactly one audit record, naming the key but never it.

TASK-0493: nothing recorded who dropped a collection, deleted a document, provisioned a tool,
exported or imported an archive or retried a job's dead letters. Each case drives one route
on a real application with authentication on and finds its record under ``fasterrag.audit``.
"""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from fasterrag.api import admin, collections, health, ingest, query, traces
from fasterrag.api.auth import required_scope
from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings
from fasterrag.observability.audit import AUDIT_LOGGER, KEY_ID_LENGTH
from fasterrag.services.journal import Journal
from fasterrag.services.provisioning import ProvisionResult
from tests.unit.api.conftest import StubVectorDB

SECRET = "audit-trail-secret-0123456789"
KEY_ID = hashlib.sha256(SECRET.encode()).hexdigest()[:KEY_ID_LENGTH]


class AuditedStub(StubVectorDB):
    """Holds one collection, ``kept``, and deletes documents without complaint."""

    async def drop_collection(self, name: str) -> bool:
        self.dropped.append(name)
        return name == "kept"

    async def delete(self, selector: Any) -> None:
        return None


async def _provisioned(tool: str, settings: Settings) -> ProvisionResult:
    return ProvisionResult(tool=tool, status="running")


async def _repaired(settings: Settings) -> Any:
    class Report:
        def as_dict(self) -> dict[str, Any]:
            return {"ok": True}

    return Report()


def _stub_admin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(admin, "provision_tool", _provisioned)
    monkeypatch.setattr(admin, "stop_tool", _provisioned)
    monkeypatch.setattr(admin, "apply_fixes", _repaired)


@dataclass(frozen=True)
class Case:
    """One audited request and the record it must leave."""

    id: str
    method: str
    path: str
    action: str
    target: str
    outcome: str
    code: str | None = None
    body: dict[str, Any] | None = None


CASES = [
    Case(
        "drop-unknown",
        "DELETE",
        "/v1/collections/docs?force=true",
        "collection.drop",
        "docs",
        "failed",
        "NOT_FOUND",
    ),
    Case(
        "drop-unforced",
        "DELETE",
        "/v1/collections/docs",
        "collection.drop",
        "docs",
        "failed",
        "VALIDATION_FAILED",
    ),
    Case(
        "document-delete",
        "DELETE",
        "/v1/collections/docs/documents/d_1",
        "document.delete",
        "docs/d_1",
        "succeeded",
    ),
    Case(
        "drop", "DELETE", "/v1/collections/kept?force=true", "collection.drop", "kept", "succeeded"
    ),
    Case("provision", "POST", "/v1/admin/provision/qdrant", "provision.up", "qdrant", "succeeded"),
    Case(
        "provision-down",
        "DELETE",
        "/v1/admin/provision/qdrant",
        "provision.down",
        "qdrant",
        "succeeded",
    ),
    Case("doctor-fix", "POST", "/v1/admin/doctor/fix", "doctor.fix", "deployment", "succeeded"),
    Case(
        "export",
        "POST",
        "/v1/admin/export",
        "archive.export",
        "default",
        "failed",
        body={"out": "never-written.tar"},
    ),
    Case(
        "import",
        "POST",
        "/v1/admin/import",
        "archive.import",
        "missing.tar",
        "failed",
        body={"archive": "missing.tar"},
    ),
    Case(
        "retry-dlq",
        "POST",
        "/v1/ingest/job-unknown/retry-dlq",
        "ingest.retry_dlq",
        "job-unknown",
        "failed",
        "NOT_FOUND",
    ),
]


@pytest.fixture
def secured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    monkeypatch.setenv("FASTERRAG_API_KEY", SECRET)
    monkeypatch.chdir(tmp_path)
    _stub_admin(monkeypatch)
    app = create_app(Settings.model_validate({"security": {"auth": True}}))
    app.state.vector_db = AuditedStub()
    app.state.journal = Journal(tmp_path / "journal")
    with TestClient(app) as client:
        yield client


def audit_records(caplog: pytest.LogCaptureFixture) -> list[Any]:
    return [record for record in caplog.records if record.name == AUDIT_LOGGER]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_each_administrative_route_leaves_exactly_one_audit_record(
    case: Case, secured: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """No route recorded who acted: a dropped collection had no author."""
    caplog.set_level("INFO", logger=AUDIT_LOGGER)

    response = secured.request(
        case.method, case.path, json=case.body, headers={"Authorization": f"Bearer {SECRET}"}
    )

    records = audit_records(caplog)
    assert len(records) == 1, response.text
    record = records[0]
    assert (record.action, record.target, record.outcome) == (
        case.action,
        case.target,
        case.outcome,
    )
    if case.code is not None:
        assert record.error_code == case.code
    assert record.via == "api"
    assert record.actor_key == "key-1"
    assert record.actor_key_id == KEY_ID
    assert len(record.trace_id) == 32
    assert (response.status_code < 400) == (case.outcome == "succeeded")
    for logged in caplog.records:
        assert SECRET not in logged.getMessage()
        assert SECRET not in repr(vars(logged))


def test_a_read_leaves_no_audit_record(
    secured: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger=AUDIT_LOGGER)

    secured.get("/v1/collections", headers={"Authorization": f"Bearer {SECRET}"})

    assert audit_records(caplog) == []


def test_a_refused_key_reaches_no_action_and_so_no_audit_record(
    secured: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Authentication refusals are logged by the auth middleware; nothing was attempted."""
    caplog.set_level("INFO", logger=AUDIT_LOGGER)

    secured.delete("/v1/collections/docs?force=true", headers={"Authorization": "Bearer nope"})

    assert audit_records(caplog) == []


READ_ONLY_POSTS = {
    "POST /v1/estimate": "parses sources to report a cost; writes nothing",
    "POST /v1/replay": "re-runs a stored query; writes neither cache nor trace",
}


def test_every_mutating_admin_or_collections_route_is_audited() -> None:
    """A new route under either scope must audit itself, or this lists it."""
    routers = [module.router for module in (admin, collections, ingest, query, traces, health)]
    mutating = [
        (method, route)
        for router in routers
        for route in router.routes
        if isinstance(route, APIRoute)
        for method in sorted(route.methods or set())
        if method in {"POST", "DELETE", "PUT", "PATCH"}
        and required_scope(route.path, method) in {"admin", "collections"}
    ]
    unaudited = [
        f"{method} {route.path}"
        for method, route in mutating
        if "audited(" not in inspect.getsource(route.endpoint)
        and f"{method} {route.path}" not in READ_ONLY_POSTS
    ]

    assert len(mutating) >= 7
    assert unaudited == []
