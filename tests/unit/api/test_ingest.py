from dataclasses import replace
from typing import Any, ClassVar

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fasterrag.api import ingest as ingest_router
from fasterrag.api.problems import PROBLEM_MEDIA_TYPE
from fasterrag.errors import ErrorCode, IngestionError
from fasterrag.services.journal import JobRecord, Journal


class RecordingIngestion:
    """Accepts and runs jobs without touching a pipeline."""

    instances: ClassVar[list["RecordingIngestion"]] = []

    def __init__(self, journal: Journal, outcome: str = "completed") -> None:
        self.journal = journal
        self.outcome = outcome
        self.ran: list[str] = []
        self.error: Exception | None = None
        RecordingIngestion.instances.append(self)

    async def accept(self, sources: Any, **kwargs: Any) -> JobRecord:
        return self.journal.create_job(
            kwargs.get("collection") or "default",
            [{"type": "path", "value": source} for source in sources],
            idempotency_key=kwargs.get("idempotency_key"),
            tenant=kwargs.get("tenant"),
        )

    async def run(self, record: JobRecord, **kwargs: Any) -> JobRecord:
        self.ran.append(record.job_id)
        if self.error is not None:
            raise self.error
        settled = replace(record, status=self.outcome, counts={"total": 1, "indexed": 1})
        self.journal.save_job(settled)
        return settled

    async def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def ingestion(monkeypatch: pytest.MonkeyPatch, app: FastAPI) -> RecordingIngestion:
    RecordingIngestion.instances.clear()
    service = RecordingIngestion(app.state.journal)
    monkeypatch.setattr(
        ingest_router,
        "build_ingestion",
        lambda settings, adapter, journal, cache, **kwargs: service,
    )
    monkeypatch.setattr(ingest_router, "build_embedding_router", lambda settings: _StubRouter())
    return service


class _StubRouter:
    async def close(self) -> None:
        return None


def start(client: TestClient, **body: Any) -> Any:
    payload = {"sources": [{"type": "path", "value": "corpus/"}], **body}
    return client.post("/v1/ingest", json=payload)


def test_ingest_is_accepted_not_awaited(client: TestClient) -> None:
    response = start(client)

    assert response.status_code == 202
    assert response.json()["job_id"].startswith("job_")


def test_the_returned_job_id_is_immediately_queryable(client: TestClient) -> None:
    job_id = start(client).json()["job_id"]

    assert client.get(f"/v1/ingest/{job_id}").status_code == 200


def test_the_job_runs_in_the_background(client: TestClient, ingestion: RecordingIngestion) -> None:
    job_id = start(client).json()["job_id"]

    assert ingestion.ran == [job_id]


def test_the_job_status_body_carries_the_documented_members(client: TestClient) -> None:
    job_id = start(client).json()["job_id"]

    body = client.get(f"/v1/ingest/{job_id}").json()

    assert set(body) == {"job_id", "status", "counts", "started_at", "finished_at"}


def test_a_completed_job_reports_its_counts(client: TestClient) -> None:
    job_id = start(client).json()["job_id"]

    body = client.get(f"/v1/ingest/{job_id}").json()

    assert body["status"] == "completed"
    assert body["counts"]["indexed"] == 1


def test_an_unknown_job_is_a_problem_document(client: TestClient) -> None:
    response = client.get("/v1/ingest/job_nonexistent")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    assert response.json()["code"] == ErrorCode.NOT_FOUND.value


def test_a_background_failure_settles_the_job_as_failed(
    client: TestClient, ingestion: RecordingIngestion
) -> None:
    ingestion.error = IngestionError("the embedding provider vanished")

    job_id = start(client).json()["job_id"]

    assert client.get(f"/v1/ingest/{job_id}").json()["status"] == "failed"


def test_a_background_failure_never_reaches_the_caller(
    client: TestClient, ingestion: RecordingIngestion
) -> None:
    ingestion.error = IngestionError("the embedding provider vanished")

    assert start(client).status_code == 202


def test_an_idempotency_key_returns_the_original_job(client: TestClient) -> None:
    first = start(client, idempotency_key="k1").json()["job_id"]
    second = start(client, idempotency_key="k1").json()["job_id"]

    assert first == second


def test_a_replayed_job_is_not_run_twice(client: TestClient, ingestion: RecordingIngestion) -> None:
    start(client, idempotency_key="k1")
    start(client, idempotency_key="k1")

    assert len(ingestion.ran) == 1


def test_at_least_one_source_is_required(client: TestClient) -> None:
    assert client.post("/v1/ingest", json={"sources": []}).status_code == 422


def test_a_missing_source_value_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/ingest", json={"sources": [{"type": "path", "value": ""}]})

    assert response.status_code == 422


def test_an_unknown_source_type_is_rejected(client: TestClient) -> None:
    response = client.post("/v1/ingest", json={"sources": [{"type": "ftp", "value": "x"}]})

    assert response.status_code == 422


def test_an_unsupported_source_type_says_which_are_supported(client: TestClient) -> None:
    response = client.post(
        "/v1/ingest", json={"sources": [{"type": "url", "value": "https://example.com/a.pdf"}]}
    )

    assert response.status_code == 422
    assert "path" in response.json()["detail"]


def test_a_misspelled_field_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/ingest",
        json={"sources": [{"type": "path", "value": "a/"}], "metadatas": {"a": "b"}},
    )

    assert response.status_code == 422


def test_documents_are_listed_for_a_job(client: TestClient, app: FastAPI) -> None:
    job_id = start(client).json()["job_id"]
    journal: Journal = app.state.journal
    journal.dead_letter(
        job_id,
        document="d_1",
        source="corpus/bad.pdf",
        reason_code=ErrorCode.PARSE_FAILED,
        detail="unreadable",
        attempts=3,
    )

    body = client.get(f"/v1/ingest/{job_id}/documents").json()

    assert body["count"] == 1
    assert body["documents"][0]["reason_code"] == ErrorCode.PARSE_FAILED.value


def test_documents_can_be_filtered_by_status(client: TestClient, app: FastAPI) -> None:
    job_id = start(client).json()["job_id"]
    journal: Journal = app.state.journal
    journal.dead_letter(
        job_id,
        document="d_1",
        source="corpus/bad.pdf",
        reason_code=ErrorCode.PARSE_FAILED,
        detail="unreadable",
        attempts=3,
    )

    matching = client.get(f"/v1/ingest/{job_id}/documents?status=dead_lettered").json()
    other = client.get(f"/v1/ingest/{job_id}/documents?status=indexed").json()

    assert matching["count"] == 1
    assert other["count"] == 0


def test_documents_of_an_unknown_job_are_404_not_an_empty_list(client: TestClient) -> None:
    response = client.get("/v1/ingest/job_nonexistent/documents")

    assert response.status_code == 404


def test_retrying_dead_letters_starts_a_new_job(client: TestClient, app: FastAPI) -> None:
    job_id = start(client).json()["job_id"]
    journal: Journal = app.state.journal
    journal.dead_letter(
        job_id,
        document="d_1",
        source="corpus/bad.pdf",
        reason_code=ErrorCode.PARSE_FAILED,
        detail="unreadable",
        attempts=3,
    )

    response = client.post(f"/v1/ingest/{job_id}/retry-dlq")
    body = response.json()

    assert response.status_code == 202
    assert body["retried"] == 1
    assert body["job_id"] != job_id
    assert body["of"] == job_id


def test_retrying_a_job_with_no_dead_letters_retries_nothing(client: TestClient) -> None:
    job_id = start(client).json()["job_id"]

    body = client.post(f"/v1/ingest/{job_id}/retry-dlq").json()

    assert body["retried"] == 0
    assert body["job_id"] == job_id


def test_retrying_an_unknown_job_is_404(client: TestClient) -> None:
    assert client.post("/v1/ingest/job_nonexistent/retry-dlq").status_code == 404


def test_the_original_job_is_not_mutated_by_a_retry(client: TestClient, app: FastAPI) -> None:
    job_id = start(client).json()["job_id"]
    journal: Journal = app.state.journal
    journal.dead_letter(
        job_id,
        document="d_1",
        source="corpus/bad.pdf",
        reason_code=ErrorCode.PARSE_FAILED,
        detail="unreadable",
        attempts=3,
    )
    before = journal.load_job(job_id).status

    client.post(f"/v1/ingest/{job_id}/retry-dlq")

    assert journal.load_job(job_id).status == before
