"""Listing the documents in a collection, on every surface (TASK-0469).

The only listing was per job (``GET /v1/ingest/{job_id}/documents``), so a caller who had
lost a job id had no way to find a document id to delete. These cases drive the real
``DocumentService`` against an in-memory backend and a real ``Journal`` on disk, then each
surface that calls it — REST, the CLI and both facades — and require all of them to return
the same page, with a deleted document gone from it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fasterrag import facade, sync
from fasterrag.adapters.vectordb.base import CollectionSpec
from fasterrag.api.main import create_app
from fasterrag.cli.commands import documents as documents_command
from fasterrag.cli.main import main
from fasterrag.cli.output import ExitCode
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.documents import MAXIMUM_PAGE_SIZE, DocumentService
from fasterrag.services.journal import Journal
from tests.unit.api.test_query_tenancy import MemoryRouter
from tests.unit.services.test_documents import DIMENSIONS, MemoryAdapter, RecordingCache, chunk

HELD = {"d_3": "hash-3", "d_1": "hash-1", "d_2": "hash-2"}


@pytest.fixture
def backend() -> MemoryAdapter:
    """Return a backend holding three documents in ``docs`` and an empty ``other``."""
    store = MemoryAdapter(Settings())
    for name in ("docs", "other"):
        store.specs.append(CollectionSpec(name=name, dimensions=DIMENSIONS))
    store.points = [chunk(document, 0) for document in HELD]
    return store


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    """Return a journal that remembers the three documents, d_2 deleted and re-ingested."""
    opened = Journal(tmp_path / "journal")
    for document, digest in HELD.items():
        opened.remember_content_blocking("docs", digest, document)
    return opened


@pytest.fixture
def service(backend: MemoryAdapter, journal: Journal) -> DocumentService:
    """Return the service over the backend and the journal."""
    return DocumentService(backend, journal, cache=RecordingCache())


async def test_every_document_is_listed_in_id_order(service: DocumentService) -> None:
    """There was no way to list a collection's documents at all."""
    listing = await service.list_documents("docs")

    assert [entry.document_id for entry in listing.documents] == ["d_1", "d_2", "d_3"]
    assert [entry.content_hash for entry in listing.documents] == ["hash-1", "hash-2", "hash-3"]
    assert listing.total == 3
    assert listing.next_offset is None


async def test_pages_cover_the_collection_without_overlap(service: DocumentService) -> None:
    first = await service.list_documents("docs", limit=2)
    second = await service.list_documents("docs", offset=first.next_offset or 0, limit=2)

    assert [entry.document_id for entry in first.documents] == ["d_1", "d_2"]
    assert first.next_offset == 2
    assert [entry.document_id for entry in second.documents] == ["d_3"]
    assert second.next_offset is None
    assert second.total == 3


async def test_a_deleted_document_is_not_listed(service: DocumentService) -> None:
    """The tombstone a delete writes must hide the document, or the listing lies."""
    await service.delete("docs", "d_2")

    listing = await service.list_documents("docs")

    assert [entry.document_id for entry in listing.documents] == ["d_1", "d_3"]
    assert listing.total == 2


async def test_a_reingested_document_is_listed_with_its_new_hash(
    service: DocumentService, journal: Journal
) -> None:
    await service.delete("docs", "d_2")
    journal.remember_content_blocking("docs", "hash-2b", "d_2")

    listing = await service.list_documents("docs")

    assert listing.documents[1].as_dict() == {"document_id": "d_2", "content_hash": "hash-2b"}


async def test_an_empty_collection_lists_nothing(service: DocumentService) -> None:
    listing = await service.list_documents("other")

    assert listing.documents == []
    assert listing.total == 0
    assert listing.next_offset is None


async def test_a_missing_collection_is_not_found(service: DocumentService) -> None:
    """A journal outlives a dropped collection; listing its stale record would mislead."""
    with pytest.raises(FasterRagError) as caught:
        await service.list_documents("absent")

    assert caught.value.code is ErrorCode.NOT_FOUND
    assert "'absent'" in caught.value.detail


async def test_an_alias_is_listed_as_the_collection(
    service: DocumentService, backend: MemoryAdapter, journal: Journal
) -> None:
    await backend.set_alias("served", "docs")
    journal.remember_content_blocking("served", "hash-9", "d_9")

    listing = await service.list_documents("served")

    assert [entry.document_id for entry in listing.documents] == ["d_9"]


@pytest.mark.parametrize(("offset", "limit"), [(-1, 10), (0, 0), (0, MAXIMUM_PAGE_SIZE + 1)])
async def test_an_impossible_page_is_refused(
    service: DocumentService, offset: int, limit: int
) -> None:
    with pytest.raises(FasterRagError) as caught:
        await service.list_documents("docs", offset=offset, limit=limit)

    assert caught.value.code is ErrorCode.VALIDATION_FAILED


@pytest.fixture
def app(
    backend: MemoryAdapter, journal: Journal, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> FastAPI:
    """Return the application over the backend and the journal."""
    monkeypatch.chdir(tmp_path)
    application = create_app(Settings())
    application.state.vector_db = backend
    application.state.journal = journal
    application.state.cache = RecordingCache()
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    """Return a client that runs the application's lifespan."""
    with TestClient(app) as test_client:
        yield test_client


def test_rest_lists_a_page_of_documents(client: TestClient) -> None:
    response = client.get("/v1/collections/docs/documents", params={"limit": 2})

    assert response.status_code == 200
    assert response.json() == {
        "collection": "docs",
        "documents": [
            {"document_id": "d_1", "content_hash": "hash-1"},
            {"document_id": "d_2", "content_hash": "hash-2"},
        ],
        "total": 3,
        "offset": 0,
        "limit": 2,
        "next_offset": 2,
    }


def test_rest_stops_listing_a_document_once_it_is_deleted(client: TestClient) -> None:
    client.delete("/v1/collections/docs/documents/d_1")

    response = client.get("/v1/collections/docs/documents")

    assert [entry["document_id"] for entry in response.json()["documents"]] == ["d_2", "d_3"]


def test_rest_answers_a_missing_collection_with_404(client: TestClient) -> None:
    response = client.get("/v1/collections/absent/documents")

    assert response.status_code == 404
    assert response.json()["code"] == "NOT_FOUND"


@pytest.mark.parametrize("query", [{"limit": 0}, {"limit": MAXIMUM_PAGE_SIZE + 1}, {"offset": -1}])
def test_rest_refuses_an_impossible_page(client: TestClient, query: dict[str, int]) -> None:
    response = client.get("/v1/collections/docs/documents", params=query)

    assert response.status_code == 422
    assert response.json()["code"] == "VALIDATION_FAILED"


KEYS = "acme-secret:collections:acme;reader-secret:query:acme"


@pytest.fixture
def tenanted(
    backend: MemoryAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """Return a client with auth and multi-tenancy on; acme's ``docs`` is ``acme__docs``."""
    backend.specs.append(CollectionSpec(name="acme__docs", dimensions=DIMENSIONS))
    journal = Journal(tmp_path / "tenanted-journal")
    journal.remember_content_blocking("acme__docs", "hash-a", "d_acme")
    journal.remember_content_blocking("docs", "hash-1", "d_1")
    monkeypatch.setenv("FASTERRAG_API_KEY", KEYS)
    monkeypatch.chdir(tmp_path)
    application = create_app(
        Settings.model_validate({"security": {"auth": True, "multi_tenancy": True}})
    )
    application.state.vector_db = backend
    application.state.journal = journal
    application.state.cache = RecordingCache()
    with TestClient(application) as test_client:
        yield test_client


def test_a_tenant_lists_its_own_collection_under_the_name_it_chose(
    tenanted: TestClient,
) -> None:
    """Resolved like every sibling /v1/collections route: the prefix is applied, never shown."""
    response = tenanted.get(
        "/v1/collections/docs/documents",
        headers={"Authorization": "Bearer acme-secret", "X-Tenant-ID": "acme"},
    )

    assert response.status_code == 200
    assert response.json()["collection"] == "docs"
    assert [entry["document_id"] for entry in response.json()["documents"]] == ["d_acme"]
    assert "acme__" not in response.text


def test_a_key_without_the_collections_scope_cannot_list(tenanted: TestClient) -> None:
    response = tenanted.get(
        "/v1/collections/docs/documents",
        headers={"Authorization": "Bearer reader-secret", "X-Tenant-ID": "acme"},
    )

    assert response.status_code == 403
    assert response.json()["code"] == ErrorCode.AUTH_SCOPE.value


@pytest.fixture
def cli_wired(backend: MemoryAdapter, journal: Journal, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the documents command at the backend and the journal."""
    monkeypatch.setattr(documents_command, "create_vector_db_adapter", lambda settings: backend)
    monkeypatch.setattr(documents_command, "create_journal", lambda settings: journal)
    monkeypatch.setattr(documents_command, "create_lock_store", lambda settings: None)
    monkeypatch.setattr(
        documents_command, "SemanticCache", lambda settings, store: RecordingCache()
    )


def test_the_cli_prints_one_line_per_document(
    config: str, cli_wired: None, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--config", config, "--collection", "docs", "documents", "list"])

    printed = capsys.readouterr().out
    assert code == ExitCode.SUCCESS
    assert [line.split()[0] for line in printed.splitlines() if line.startswith("d_")] == [
        "d_1",
        "d_2",
        "d_3",
    ]
    assert "3 of 3" in printed


def test_the_cli_says_how_to_reach_the_next_page(
    config: str, cli_wired: None, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--config", config, "--collection", "docs", "documents", "list", "--limit", "2"])

    assert code == ExitCode.SUCCESS
    assert "--offset 2" in capsys.readouterr().out


def test_the_cli_reports_a_missing_collection(
    config: str, cli_wired: None, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--config", config, "--collection", "absent", "documents", "list"])

    assert code == ExitCode.FAILURE
    assert "[NOT_FOUND]" in capsys.readouterr().err


@pytest.fixture
def facade_wired(backend: MemoryAdapter, journal: Journal, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point both facades' factories at the backend and the journal."""
    monkeypatch.setattr(facade, "create_vector_db_adapter", lambda settings: backend)
    monkeypatch.setattr(facade, "create_embedding_router", lambda settings: MemoryRouter())
    monkeypatch.setattr(facade, "create_journal", lambda settings: journal)


def configured(collection: str) -> Settings:
    """Return settings whose default collection is ``collection``."""
    return Settings.model_validate({"vector_db": {"collection": {"default_name": collection}}})


async def test_the_async_facade_lists_the_default_collection(facade_wired: None) -> None:
    async with facade.FasterRag.from_settings(configured("docs")) as rag:
        listing = await rag.documents(limit=2)
        other = await rag.documents("other")

    assert [entry.document_id for entry in listing.documents] == ["d_1", "d_2"]
    assert listing.next_offset == 2
    assert other.total == 0


def test_the_blocking_facade_lists_the_same_page(facade_wired: None) -> None:
    with sync.FasterRag.from_settings(configured("docs")) as rag:
        listing = rag.documents(offset=1)

    assert [entry.document_id for entry in listing.documents] == ["d_2", "d_3"]


async def test_the_facade_checks_it_was_started() -> None:
    with pytest.raises(FasterRagError) as caught:
        await facade.FasterRag.from_settings(Settings()).documents()

    assert caught.value.code is ErrorCode.USAGE_ERROR


def test_rest_cli_and_both_facades_return_the_same_listing(
    client: TestClient,
    config: str,
    cli_wired: None,
    facade_wired: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The acceptance case: one service, so one answer, whichever surface asks."""
    client.delete("/v1/collections/docs/documents/d_3")

    rest = client.get("/v1/collections/docs/documents", params={"offset": 1, "limit": 5}).json()
    capsys.readouterr()
    main(
        [
            "--config",
            config,
            "--json",
            "--collection",
            "docs",
            "documents",
            "list",
            "--offset",
            "1",
            "--limit",
            "5",
        ]
    )
    cli: dict[str, Any] = json.loads(capsys.readouterr().out)
    with sync.FasterRag.from_settings(configured("docs")) as rag:
        blocking = rag.documents(offset=1, limit=5).as_dict()

    assert rest == cli == blocking
    assert rest["documents"] == [{"document_id": "d_2", "content_hash": "hash-2"}]
    assert rest["total"] == 2
