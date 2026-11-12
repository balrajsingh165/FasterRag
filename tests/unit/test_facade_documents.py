"""``FasterRag.delete_document`` on both facades, through the real ``DocumentService``.

The facade is a composition layer, so what matters is that it composes the *same* service
over *its own* adapter, journal and cache, and that the blocking facade reaches the same
method. Only the backend is faked; the journal is a real one on disk (TASK-0443).
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from fasterrag import facade, sync
from fasterrag.adapters.vectordb.base import CollectionSpec
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.documents import DocumentDeletion
from fasterrag.services.journal import Journal
from tests.unit.services.test_documents import DIMENSIONS, MemoryAdapter, RecordingCache, chunk


@pytest.fixture
def backend() -> MemoryAdapter:
    """Return a backend holding one document in the default collection and one elsewhere."""
    store = MemoryAdapter(Settings())
    for name in ("default", "contracts"):
        store.specs.append(CollectionSpec(name=name, dimensions=DIMENSIONS))
    store.points = [
        chunk("d_1", 0, collection="default", tenant="acme"),
        chunk("d_1", 1, collection="default", tenant="globex"),
        chunk("d_2", 0, collection="default"),
        chunk("d_1", 0, collection="contracts"),
    ]
    return store


@pytest.fixture
def started(
    backend: MemoryAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> facade.FasterRag:
    """Return an async facade that is started over the fake backend and a real journal."""
    monkeypatch.chdir(tmp_path)
    journal = Journal(tmp_path / "journal")
    journal.remember_content_blocking("default", "hash-1", "d_1")

    rag: Any = facade.FasterRag.from_settings(Settings())
    rag._started = True
    rag._vector_db = backend
    rag._journal = journal
    rag._cache = RecordingCache()
    rag._embeddings = object()
    built: facade.FasterRag = rag
    return built


async def test_a_document_is_deleted_from_the_default_collection(
    started: facade.FasterRag, backend: MemoryAdapter
) -> None:
    """The library could ingest a document and had no way to remove one."""
    deletion = await started.delete_document("d_1")

    assert isinstance(deletion, DocumentDeletion)
    assert deletion.collection == "default"
    assert deletion.known is True
    assert backend.chunks_of("default", "d_1") == []
    assert backend.chunks_of("default", "d_2") == [0]
    assert backend.chunks_of("contracts", "d_1") == [0]


async def test_a_named_collection_is_the_one_deleted_from(
    started: facade.FasterRag, backend: MemoryAdapter
) -> None:
    deletion = await started.delete_document("d_1", collection="contracts")

    assert deletion.known is False
    assert backend.chunks_of("contracts", "d_1") == []
    assert backend.chunks_of("default", "d_1") == [0, 1]


async def test_a_tenant_limits_the_delete_to_its_own_chunks(
    started: facade.FasterRag, backend: MemoryAdapter
) -> None:
    await started.delete_document("d_1", tenant="acme")

    assert [point.payload.get("tenant") for point in backend.held("default")] == ["globex", None]


async def test_the_facades_own_journal_and_cache_are_the_ones_updated(
    started: facade.FasterRag,
) -> None:
    """A service built over fresh ones would clear nothing the next query or ingest reads."""
    await started.delete_document("d_1")

    cache: Any = started.cache
    assert await started.journal.is_duplicate("default", "hash-1") is False
    assert len(cache.reasons) == 1


async def test_deleting_a_document_that_does_not_exist_is_not_an_error(
    started: facade.FasterRag, backend: MemoryAdapter
) -> None:
    deletion = await started.delete_document("d_never_ingested")

    assert deletion.known is False
    assert len(backend.points) == 4


async def test_a_missing_collection_raises_not_found(started: facade.FasterRag) -> None:
    with pytest.raises(FasterRagError) as caught:
        await started.delete_document("d_1", collection="absent")

    assert caught.value.code is ErrorCode.NOT_FOUND


async def test_it_checks_the_facade_was_started() -> None:
    rag = facade.FasterRag.from_settings(Settings())

    with pytest.raises(FasterRagError, match="not been started"):
        await rag.delete_document("d_1")


def test_both_facades_take_the_same_arguments() -> None:
    """A parameter on one and not the other is a feature only half the callers can reach."""
    asynchronous = inspect.signature(facade.FasterRag.delete_document)
    blocking = inspect.signature(sync.FasterRag.delete_document)

    assert list(blocking.parameters) == list(asynchronous.parameters)
    assert not inspect.iscoroutinefunction(sync.FasterRag.delete_document)


@pytest.fixture
def blocking(started: facade.FasterRag) -> Iterator[sync.FasterRag]:
    """Return a blocking facade driving the started async one on its own loop."""
    rag = sync.FasterRag(started)
    rag._loop = asyncio.new_event_loop()
    try:
        yield rag
    finally:
        rag._loop.close()
        rag._loop = None


def test_the_blocking_facade_deletes_through_the_same_service(
    blocking: sync.FasterRag, backend: MemoryAdapter
) -> None:
    deletion = blocking.delete_document("d_1", tenant="acme")

    assert deletion.known is True
    assert [point.payload.get("tenant") for point in backend.held("default")] == ["globex", None]


def test_the_blocking_facade_unstarted_names_the_context_manager() -> None:
    rag = sync.FasterRag.from_settings(Settings())

    with pytest.raises(FasterRagError, match="not been started"):
        rag.delete_document("d_1")
