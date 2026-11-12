"""``Idempotency-Key`` header behaviour across the mutating surface (TASK-0318).

The cases that matter are the ones a naive implementation gets wrong: a replay with a
*different* body, a key reused on another endpoint, one tenant reaching another's record,
and two copies of one request arriving before either has finished. Three later cases cover
the configuration (TASK-0356) and the backend seam (TASK-0357): a shortened lease, the kill
switch writing nothing, and a store that is not a directory standing in unchanged.
"""

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fasterrag.api import admin as admin_router
from fasterrag.api import collections as collections_router
from fasterrag.api import ingest as ingest_router
from fasterrag.api.main import create_app
from fasterrag.api.problems import PROBLEM_MEDIA_TYPE
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.idempotency import (
    IN_FLIGHT_LEASE_SECONDS,
    FileIdempotencyStore,
    IdempotencyStore,
    Reservation,
    StoredResponse,
    create_idempotency_store,
    request_fingerprint,
)
from fasterrag.services.journal import JobRecord, Journal
from fasterrag.services.provisioning import ProvisionResult
from fasterrag.services.traces import TraceStore
from tests.unit.api.conftest import StubVectorDB

KEY = "11111111-2222-3333-4444-555555555555"
OTHER_KEY = "99999999-8888-7777-6666-555555555555"
SECRET = "acme-key-long-enough"
DIGEST = request_fingerprint("POST", "/v1/collections", "", b"{}")


class StubEmbedder:
    """Reports a fixed vector size without loading anything."""

    dimensions = 384

    async def embed_query(self, text: str) -> list[float]:
        return [0.0] * 8

    async def close(self) -> None:
        return None


class StubRouter:
    """Stands in for the tiered embedding router."""

    def __init__(self) -> None:
        self.default = StubEmbedder()

    async def close(self) -> None:
        return None


class RecordingIngestion:
    """Accepts and runs jobs without touching a pipeline."""

    instances: ClassVar[list["RecordingIngestion"]] = []

    def __init__(self, journal: Journal) -> None:
        self.journal = journal
        self.ran: list[str] = []
        RecordingIngestion.instances.append(self)

    async def accept(self, sources: Any, **kwargs: Any) -> JobRecord:
        return self.journal.create_job_blocking(
            kwargs.get("collection") or "default",
            [dict(source) for source in sources],
            idempotency_key=kwargs.get("idempotency_key"),
            tenant=kwargs.get("tenant"),
        )

    async def run(self, record: JobRecord, **kwargs: Any) -> JobRecord:
        self.ran.append(record.job_id)
        settled = replace(record, status="completed", counts={"total": 1, "indexed": 1})
        self.journal.save_job_blocking(settled)
        return settled

    async def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def stubbed(monkeypatch: pytest.MonkeyPatch, app: FastAPI) -> RecordingIngestion:
    RecordingIngestion.instances.clear()
    service = RecordingIngestion(app.state.journal)
    monkeypatch.setattr(
        collections_router, "create_embedding_router", lambda settings: StubRouter()
    )
    monkeypatch.setattr(ingest_router, "create_embedding_router", lambda settings: StubRouter())
    monkeypatch.setattr(
        ingest_router,
        "build_ingestion",
        lambda settings, adapter, journal, cache, **kwargs: service,
    )
    return service


def body(name: str = "handbook") -> dict[str, str]:
    return {"name": name}


def keyed(key: str = KEY) -> dict[str, str]:
    return {"Idempotency-Key": key}


def test_a_replayed_create_returns_the_original_response(client: TestClient) -> None:
    first = client.post("/v1/collections", json=body(), headers=keyed())
    second = client.post("/v1/collections", json=body(), headers=keyed())

    assert first.status_code == 201
    assert second.status_code == 201
    assert second.json() == first.json()


def test_a_replay_performs_no_second_create(client: TestClient, vector_db: StubVectorDB) -> None:
    """The point of the feature: the adapter is asked once, not twice."""
    client.post("/v1/collections", json=body(), headers=keyed())
    client.post("/v1/collections", json=body(), headers=keyed())

    assert [spec.name for spec in vector_db.created] == ["handbook"]


def test_a_replay_is_flagged_as_one(client: TestClient) -> None:
    """A client cannot tell a replay from a first call without a signal, and often must."""
    first = client.post("/v1/collections", json=body(), headers=keyed())
    second = client.post("/v1/collections", json=body(), headers=keyed())

    assert "idempotent-replayed" not in first.headers
    assert second.headers["Idempotent-Replayed"] == "true"
    assert second.headers["Idempotency-Key"] == KEY


def test_the_same_key_with_a_different_body_is_refused(client: TestClient) -> None:
    client.post("/v1/collections", json=body("handbook"), headers=keyed())

    reused = client.post("/v1/collections", json=body("payroll"), headers=keyed())

    assert reused.status_code == 422
    assert reused.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    assert reused.json()["code"] == ErrorCode.IDEMPOTENCY_KEY_REUSED.value
    assert reused.json()["retryable"] is False


def test_the_same_key_on_a_different_endpoint_is_refused(client: TestClient) -> None:
    """A key names one request, so reuse across paths is the same mistake as a new body."""
    client.post("/v1/collections", json=body(), headers=keyed())

    crossed = client.delete("/v1/collections/handbook?force=true", headers=keyed())

    assert crossed.status_code == 422
    assert crossed.json()["code"] == ErrorCode.IDEMPOTENCY_KEY_REUSED.value


def test_different_keys_do_not_collide(client: TestClient, vector_db: StubVectorDB) -> None:
    client.post("/v1/collections", json=body("a"), headers=keyed())
    client.post("/v1/collections", json=body("b"), headers=keyed(OTHER_KEY))

    assert [spec.name for spec in vector_db.created] == ["a", "b"]


def test_a_request_without_the_header_still_works(
    client: TestClient, vector_db: StubVectorDB
) -> None:
    """The header is optional: a client that never sends it is unaffected by this feature."""
    first = client.post("/v1/collections", json=body())
    second = client.post("/v1/collections", json=body())

    assert (first.status_code, second.status_code) == (201, 201)
    assert len(vector_db.created) == 2


def test_a_delete_replays_its_204(client: TestClient, vector_db: StubVectorDB) -> None:
    """A 204 carries no body, which the replay path must reproduce without inventing one."""
    client.post("/v1/collections", json=body(), headers=keyed(OTHER_KEY))

    first = client.delete("/v1/collections/handbook?force=true", headers=keyed())
    second = client.delete("/v1/collections/handbook?force=true", headers=keyed())

    assert (first.status_code, second.status_code) == (204, 204)
    assert second.content == b""
    assert vector_db.dropped == ["handbook"]


def test_a_failed_request_does_not_pin_its_key(client: TestClient) -> None:
    """A refusal is not a result: the caller must be able to change nothing and retry."""
    missing = client.delete("/v1/collections/absent?force=true", headers=keyed())
    assert missing.status_code == 404

    client.post("/v1/collections", json=body("absent"), headers=keyed(OTHER_KEY))
    retried = client.delete("/v1/collections/absent?force=true", headers=keyed())

    assert retried.status_code == 204


def test_a_read_only_post_ignores_the_header(client: TestClient) -> None:
    """``/v1/estimate`` changes nothing, so a key sent to it must not become a 4xx."""
    payload = {"sources": ["missing.txt"]}

    first = client.post("/v1/estimate", json=payload, headers=keyed())
    second = client.post("/v1/estimate", json=payload, headers=keyed())

    assert first.status_code == second.status_code
    assert "idempotent-replayed" not in second.headers


def test_an_unusable_key_is_refused_without_echoing_it(client: TestClient) -> None:
    sent = "x" * 300

    refused = client.post("/v1/collections", json=body(), headers=keyed(sent))

    assert refused.status_code == 422
    assert refused.json()["code"] == ErrorCode.VALIDATION_FAILED.value
    assert sent not in refused.text


def test_an_ingest_replay_returns_the_original_job(
    client: TestClient, stubbed: RecordingIngestion
) -> None:
    sources = {"sources": [{"type": "inline", "value": "hello"}]}

    first = client.post("/v1/ingest", json=sources, headers=keyed())
    second = client.post("/v1/ingest", json=sources, headers=keyed())

    assert first.status_code == 202
    assert second.json()["job_id"] == first.json()["job_id"]
    assert second.headers["Idempotent-Replayed"] == "true"
    assert len(stubbed.ran) == 1


def test_a_dlq_retry_is_replayable(client: TestClient) -> None:
    """The endpoint had no replay protection at all, and it starts a second ingest."""
    job_id = client.post(
        "/v1/ingest", json={"sources": [{"type": "inline", "value": "hello"}]}
    ).json()["job_id"]

    first = client.post(f"/v1/ingest/{job_id}/retry-dlq", headers=keyed())
    second = client.post(f"/v1/ingest/{job_id}/retry-dlq", headers=keyed())

    assert first.status_code == 202
    assert second.json() == first.json()
    assert second.headers["Idempotent-Replayed"] == "true"


def test_the_body_field_still_works_and_says_it_is_deprecated(client: TestClient) -> None:
    """The field is public behaviour, so it keeps working; the deprecation is announced."""
    sources = {"sources": [{"type": "inline", "value": "hello"}], "idempotency_key": "legacy-1"}

    first = client.post("/v1/ingest", json=sources)
    second = client.post("/v1/ingest", json=sources)

    assert second.json()["job_id"] == first.json()["job_id"]
    assert first.headers["Deprecation"] == "true"
    assert 'rel="deprecation"' in first.headers["Link"]


def tenanted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> TestClient:
    monkeypatch.setenv(
        "FASTERRAG_API_KEY", f"{SECRET}:admin:acme;globex-key-long-enough:admin:globex"
    )
    settings = Settings.model_validate({"security": {"auth": True, "multi_tenancy": True}})
    application = create_app(settings)
    application.state.vector_db = StubVectorDB()
    application.state.journal = Journal(tmp_path / "tenanted-journal")
    application.state.traces = TraceStore(tmp_path / "tenanted-traces")
    return TestClient(application)


def test_one_tenant_cannot_replay_anothers_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The disclosure this scoping exists to prevent: a stranger's response for your key."""
    client = tenanted(monkeypatch, tmp_path)
    acme = {"Authorization": f"Bearer {SECRET}", "X-Tenant-ID": "acme", **keyed()}
    globex = {"Authorization": "Bearer globex-key-long-enough", "X-Tenant-ID": "globex", **keyed()}

    first = client.post("/v1/collections", json=body(), headers=acme)
    second = client.post("/v1/collections", json=body(), headers=globex)

    assert (first.status_code, second.status_code) == (201, 201)
    assert "idempotent-replayed" not in second.headers


async def test_a_concurrent_duplicate_is_refused_rather_than_executed(app: FastAPI) -> None:
    """The case that usually ships broken: both copies arrive before either has finished."""
    started = asyncio.Event()
    finish = asyncio.Event()
    runs = 0

    async def slow() -> dict[str, str]:
        nonlocal runs
        runs += 1
        started.set()
        await finish.wait()
        return {"ran": "once"}

    app.add_api_route("/v1/collections/slow", slow, methods=["POST"])
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        original = asyncio.create_task(client.post("/v1/collections/slow", headers=keyed()))
        await asyncio.wait_for(started.wait(), timeout=5)
        duplicate = await client.post("/v1/collections/slow", headers=keyed())
        finish.set()
        first = await original

    assert runs == 1
    assert first.status_code == 200
    assert duplicate.status_code == 409
    assert duplicate.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    assert duplicate.json()["code"] == ErrorCode.IDEMPOTENCY_IN_PROGRESS.value
    assert duplicate.json()["retryable"] is True
    assert duplicate.headers["Retry-After"] == "1"


async def test_the_store_serialises_two_simultaneous_claims(tmp_path: Path) -> None:
    """The reservation, without HTTP: the second claim never gets a lease of its own."""
    store = FileIdempotencyStore(tmp_path)

    first = await store.begin(KEY, tenant=None, fingerprint=DIGEST)
    with pytest.raises(FasterRagError) as caught:
        await store.begin(KEY, tenant=None, fingerprint=DIGEST)

    assert isinstance(first, Reservation)
    assert caught.value.code is ErrorCode.IDEMPOTENCY_IN_PROGRESS
    assert caught.value.retryable is True


async def test_a_tenants_key_is_invisible_to_another_tenant(tmp_path: Path) -> None:
    store = FileIdempotencyStore(tmp_path)

    await store.begin(KEY, tenant="acme", fingerprint=DIGEST)
    other = await store.begin(KEY, tenant="globex", fingerprint=DIGEST)

    assert isinstance(other, Reservation)


async def test_an_abandoned_reservation_frees_its_key_on_the_lease(tmp_path: Path) -> None:
    """A worker killed mid-request must not hold a key for the full record lifetime."""
    store = FileIdempotencyStore(tmp_path)
    reservation = await store.begin(KEY, tenant=None, fingerprint=DIGEST)
    assert isinstance(reservation, Reservation)

    _age(store, reservation, IN_FLIGHT_LEASE_SECONDS + 1)

    assert isinstance(await store.begin(KEY, tenant=None, fingerprint=DIGEST), Reservation)


async def test_a_completed_record_outlives_the_in_flight_lease(tmp_path: Path) -> None:
    """A replay has to keep working long after the request that produced it finished."""
    store = FileIdempotencyStore(tmp_path)
    reservation = await store.begin(KEY, tenant=None, fingerprint=DIGEST)
    assert isinstance(reservation, Reservation)
    await store.commit(
        reservation,
        StoredResponse(status=201, media_type="application/json", body=b'{"ok":true}'),
    )

    _age(store, reservation, IN_FLIGHT_LEASE_SECONDS + 1)

    replayed = await store.begin(KEY, tenant=None, fingerprint=DIGEST)
    assert not isinstance(replayed, Reservation)
    assert replayed.body == b'{"ok":true}'


async def test_a_configured_lifetime_replaces_the_default_lease(tmp_path: Path) -> None:
    """The lease is configurable (TASK-0356), and a shorter one has to actually be shorter."""
    settings = Settings.model_validate(
        {"reliability": {"idempotency": {"in_flight_lease_seconds": 60}}}
    )
    store = create_idempotency_store(tmp_path, settings)
    reservation = await store.begin(KEY, tenant=None, fingerprint=DIGEST)
    assert isinstance(reservation, Reservation)
    assert isinstance(store, FileIdempotencyStore)

    _age(store, reservation, 61)

    assert isinstance(await store.begin(KEY, tenant=None, fingerprint=DIGEST), Reservation)


async def test_the_kill_switch_passes_a_keyed_request_straight_through(tmp_path: Path) -> None:
    """``enabled: false`` must not half-apply: no record is written and nothing is refused."""
    settings = Settings.model_validate({"reliability": {"idempotency": {"enabled": False}}})
    application = create_app(settings)
    application.state.vector_db = StubVectorDB()
    application.state.journal = Journal(tmp_path / "off-journal")

    with TestClient(application) as client:
        first = client.post("/v1/collections", json={"name": "docs"}, headers=keyed())
        second = client.post("/v1/collections", json={"name": "docs"}, headers=keyed())

    assert first.status_code == second.status_code
    assert "Idempotent-Replayed" not in second.headers
    assert not (tmp_path / "off-journal" / "idempotency").exists()


def test_the_default_keeps_the_documented_contract_true() -> None:
    """A published header contract must not be switched off by a default nobody set."""
    assert Settings().reliability.idempotency.enabled is True


def _age(store: FileIdempotencyStore, reservation: Reservation, seconds: float) -> None:
    """Rewind a record's clock, which is the only way to reach an expiry in a unit test."""
    path = store.path_for(reservation.key, reservation.tenant)
    record = store.read_record(path)
    assert record is not None
    record["created_at"] = float(record["created_at"]) - seconds
    store.write_record(path, record)


def test_an_oversized_keyed_body_is_still_a_413_problem_document(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Buffering the body moved the size check into this middleware's reach.

    A chunked upload declares no ``Content-Length``, so the ceiling is enforced as bytes
    arrive — which is now inside the buffering this feature added, outside the handler that
    used to render the refusal.
    """
    settings = Settings.model_validate({"security": {"max_request_mb": 1}})
    application = create_app(settings)
    application.state.vector_db = StubVectorDB()
    application.state.journal = Journal(tmp_path / "limit-journal")
    monkeypatch.setattr(
        collections_router, "create_embedding_router", lambda settings: StubRouter()
    )

    def chunks() -> Any:
        yield b'{"name": "'
        yield b"x" * (2 * 1024 * 1024)
        yield b'"}'

    with TestClient(application, raise_server_exceptions=False) as client:
        refused = client.post(
            "/v1/collections",
            content=chunks(),
            headers={**keyed(), "Content-Type": "application/json"},
        )

    assert refused.status_code == 413
    assert refused.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    assert refused.json()["code"] == ErrorCode.PAYLOAD_TOO_LARGE.value


def test_an_admin_route_is_covered_too(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The admin family is covered by construction; this proves one of them really is."""
    calls = 0

    async def provision(tool: str, settings: Any) -> ProvisionResult:
        nonlocal calls
        calls += 1
        return ProvisionResult(tool=tool, status="running", url="http://localhost:6333")

    monkeypatch.setattr(admin_router, "provision_tool", provision)

    first = client.post("/v1/admin/provision/qdrant", headers=keyed())
    second = client.post("/v1/admin/provision/qdrant", headers=keyed())

    assert calls == 1
    assert second.json() == first.json()
    assert second.headers["Idempotent-Replayed"] == "true"


class MemoryIdempotencyStore(IdempotencyStore):
    """A backend that is not a directory, so the seam is proven rather than asserted.

    It is deliberately the *whole* interface and nothing else: if the middleware still knew
    anything about files, paths or threads, this class could not stand in for the one that
    ships (TASK-0357).
    """

    def __init__(self) -> None:
        self.records: dict[tuple[str | None, str], dict[str, Any]] = {}

    async def begin(
        self, key: str, *, tenant: str | None, fingerprint: str
    ) -> Reservation | StoredResponse:
        existing = self.records.get((tenant, key))
        if existing is None:
            self.records[(tenant, key)] = {"fingerprint": fingerprint, "response": None}
            return Reservation(key=key, tenant=tenant, fingerprint=fingerprint, owner="memory")

        if existing["fingerprint"] != fingerprint:
            raise FasterRagError(
                "this key was already used for a different request",
                code=ErrorCode.IDEMPOTENCY_KEY_REUSED,
                retryable=False,
            )

        stored = existing["response"]
        if stored is None:
            raise FasterRagError(
                "a request with this key is still in flight",
                code=ErrorCode.IDEMPOTENCY_IN_PROGRESS,
            )
        assert isinstance(stored, StoredResponse)
        return stored

    async def commit(self, reservation: Reservation, response: StoredResponse) -> None:
        self.records[(reservation.tenant, reservation.key)]["response"] = response

    async def release(self, reservation: Reservation) -> None:
        self.records.pop((reservation.tenant, reservation.key), None)

    async def close(self) -> None:
        return None


def test_a_foreign_backend_drops_in_without_touching_the_middleware(
    app: FastAPI, tmp_path: Path
) -> None:
    """What answers TASK-0357's question: is the store substitutable, or only called one?"""
    store = MemoryIdempotencyStore()
    app.state.idempotency = store

    with TestClient(app) as client:
        first = client.post("/v1/collections", json=body(), headers=keyed())
        second = client.post("/v1/collections", json=body(), headers=keyed())
        conflicting = client.post("/v1/collections", json=body("other"), headers=keyed())

    assert second.headers["Idempotent-Replayed"] == "true"
    assert second.json() == first.json()
    assert conflicting.status_code == 422
    assert conflicting.json()["code"] == ErrorCode.IDEMPOTENCY_KEY_REUSED.value
    assert list(store.records) == [(None, KEY)]
    assert not (tmp_path / "journal" / "idempotency").exists()
