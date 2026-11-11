"""The ingest endpoints refuse a corpus the live index cannot hold (TASK-0119)."""

from typing import Any, ClassVar

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fasterrag.adapters.vectordb.base import CollectionInfo
from fasterrag.api import ingest as ingest_router
from fasterrag.api.problems import PROBLEM_MEDIA_TYPE
from fasterrag.errors import ErrorCode
from fasterrag.services.journal import JobRecord, Journal
from tests.unit.api.conftest import StubVectorDB


class RecordingIngestion:
    """Accepts jobs into the real journal without running a pipeline."""

    instances: ClassVar[list["RecordingIngestion"]] = []

    def __init__(self, journal: Journal) -> None:
        self.journal = journal
        self.accepted: list[str] = []
        RecordingIngestion.instances.append(self)

    async def accept(self, sources: Any, **kwargs: Any) -> JobRecord:
        collection = kwargs.get("collection") or "default"
        self.accepted.append(collection)
        return self.journal.create_job_blocking(
            collection,
            [{"type": "path", "value": "corpus/"}],
            idempotency_key=kwargs.get("idempotency_key"),
            tenant=kwargs.get("tenant"),
        )

    async def run(self, record: JobRecord, **kwargs: Any) -> JobRecord:
        return record

    async def close(self) -> None:
        return None


class _StubRouter:
    """Stands in for the embedding router the background task builds."""

    async def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def ingestion(monkeypatch: pytest.MonkeyPatch, app: FastAPI) -> RecordingIngestion:
    """Replace the ingestion service so acceptance is observable on its own."""
    RecordingIngestion.instances.clear()
    service = RecordingIngestion(app.state.journal)
    monkeypatch.setattr(
        ingest_router,
        "build_ingestion",
        lambda settings, adapter, journal, cache, **kwargs: service,
    )
    monkeypatch.setattr(ingest_router, "build_embedding_router", lambda settings: _StubRouter())
    return service


def dense_only(vector_db: StubVectorDB, name: str = "docs") -> None:
    """Give the backend one collection that was indexed without a keyword leg."""
    vector_db.collections = [
        CollectionInfo(name=name, vectors=10, dimensions=8, distance="cosine", sparse=False)
    ]


def hybrid(vector_db: StubVectorDB, name: str = "docs") -> None:
    """Give the backend one collection that does carry a keyword leg."""
    vector_db.collections = [
        CollectionInfo(name=name, vectors=10, dimensions=8, distance="cosine", sparse=True)
    ]


def start(client: TestClient, **body: Any) -> Any:
    """Post a one-source ingest job."""
    payload = {"sources": [{"type": "path", "value": "corpus/"}], **body}
    return client.post("/v1/ingest", json=payload)


def test_hybrid_over_a_dense_only_index_is_refused_before_the_job_exists(
    client: TestClient, vector_db: StubVectorDB, ingestion: RecordingIngestion
) -> None:
    dense_only(vector_db)

    response = start(client, collection="docs")

    assert response.status_code == 409
    assert ingestion.accepted == []


def test_the_refusal_is_a_problem_document_with_a_stable_code(
    client: TestClient, vector_db: StubVectorDB
) -> None:
    dense_only(vector_db)

    response = start(client, collection="docs")
    problem = response.json()

    assert response.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    assert problem["code"] == ErrorCode.CONFLICT.value
    assert problem["retryable"] is False
    assert problem["trace_id"]


def test_the_operator_is_told_the_real_problem_and_the_real_fix(
    client: TestClient, vector_db: StubVectorDB
) -> None:
    dense_only(vector_db)

    detail = start(client, collection="docs").json()["detail"]

    assert "retrieval.hybrid" in detail
    assert "'docs'" in detail
    assert "fasterrag index reembed docs" in detail
    assert "set retrieval.hybrid" in detail


def test_the_old_confusing_failure_no_longer_reaches_the_operator(
    client: TestClient, vector_db: StubVectorDB
) -> None:
    """Previously this request was a 202 and the job died per-document, far from its cause."""
    dense_only(vector_db)

    response = start(client, collection="docs")
    body = response.json()

    assert response.status_code != 202
    assert "job_id" not in body
    assert "carries a sparse vector" not in body["detail"]
    assert "EMBED" not in body["code"]


def test_an_index_with_a_keyword_leg_is_still_accepted(
    client: TestClient, vector_db: StubVectorDB
) -> None:
    hybrid(vector_db)

    response = start(client, collection="docs")

    assert response.status_code == 202
    assert response.json()["job_id"]


def test_a_collection_nothing_has_indexed_yet_is_still_accepted(
    client: TestClient, vector_db: StubVectorDB
) -> None:
    vector_db.collections = []

    assert start(client, collection="brand-new").status_code == 202


def test_the_preflight_follows_the_alias_to_the_collection_writes_reach(
    client: TestClient, vector_db: StubVectorDB
) -> None:
    dense_only(vector_db, name="docs-v20260101000000")
    vector_db.aliases["docs"] = "docs-v20260101000000"

    detail = start(client, collection="docs").json()["detail"]

    assert "docs-v20260101000000" in detail


def test_retrying_dead_letters_is_held_to_the_same_preflight(
    client: TestClient, app: FastAPI, vector_db: StubVectorDB
) -> None:
    job_id = start(client, collection="docs").json()["job_id"]
    journal: Journal = app.state.journal
    journal.dead_letter_blocking(
        job_id,
        document="d_1",
        source="corpus/bad.pdf",
        reason_code=ErrorCode.PARSE_FAILED,
        detail="unreadable",
        attempts=3,
    )
    dense_only(vector_db)

    response = client.post(f"/v1/ingest/{job_id}/retry-dlq")

    assert response.status_code == 409
    assert response.json()["code"] == ErrorCode.CONFLICT.value
