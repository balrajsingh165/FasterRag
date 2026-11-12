"""Deleting one document, against a backend that actually deletes.

Until TASK-0443 nothing outside the adapters called ``VectorDBAdapter.delete``, so a
document could be added to a collection and never taken out of it. These cases drive
``DocumentService`` against an in-memory adapter that honours the documented filter grammar
and a real ``Journal`` on disk, so what is left in the index and what the dedup record still
claims are both observed rather than asserted about a stub.

The contract suite (``tests/contract/vectordb.py``) proves the same selector against the
real backends; what is proven here is the orchestration around it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fasterrag.adapters.embeddings.tiering import matches
from fasterrag.adapters.vectordb.base import (
    CollectionInfo,
    CollectionSpec,
    Point,
    PointSelector,
    UpsertResult,
)
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError, RetrievalError
from fasterrag.services.documents import DocumentDeletion, DocumentService, document_filter
from fasterrag.services.journal import Journal
from fasterrag.services.lockfile import LockStore, build_lock
from tests.unit.workers.test_indexer import RecordingAdapter

DIMENSIONS = 4


class MemoryAdapter(RecordingAdapter):
    """Holds points by id and honours ``delete``, so a test can see what is left."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.selectors: list[PointSelector] = []
        self.failures: list[FasterRagError] = []

    async def list_collections(self) -> list[CollectionInfo]:
        return [
            CollectionInfo(
                name=spec.name,
                vectors=len(self.held(spec.name)),
                sparse=spec.sparse,
            )
            for spec in self.specs
        ]

    async def upsert(self, points: list[Point]) -> UpsertResult:
        merged = {(point.collection, point.point_id): point for point in self.points}
        merged.update({(point.collection, point.point_id): point for point in points})
        self.points = list(merged.values())
        return UpsertResult(upserted=len(points))

    async def delete(self, selector: PointSelector) -> None:
        self.selectors.append(selector)
        if self.failures:
            raise self.failures.pop(0)
        if selector.collection not in {spec.name for spec in self.specs}:
            raise FasterRagError(
                f"collection {selector.collection!r} does not exist",
                code=ErrorCode.NOT_FOUND,
                retryable=False,
            )
        self.points = [point for point in self.points if not _selected(selector, point)]

    def held(self, collection: str) -> list[Point]:
        """Return the points a collection holds, in chunk order."""
        return sorted(
            (point for point in self.points if point.collection == collection),
            key=lambda point: (
                str(point.payload.get("document_id")),
                int(point.payload.get("chunk_index", 0)),
            ),
        )

    def chunks_of(self, collection: str, document: str) -> list[int]:
        """Return the chunk indexes a document still has in a collection."""
        return [
            int(point.payload["chunk_index"])
            for point in self.held(collection)
            if point.payload.get("document_id") == document
        ]


def _selected(selector: PointSelector, point: Point) -> bool:
    """Return whether a selector names a point, by id or by pushed-down filter."""
    if point.collection != selector.collection:
        return False
    if selector.point_ids is not None:
        return point.point_id in selector.point_ids
    return matches(selector.filters or {}, point.payload)


class RecordingCache:
    """Records invalidation, which is what a corpus change is supposed to trigger."""

    def __init__(self) -> None:
        self.reasons: list[str] = []

    async def invalidate(self, reason: str) -> None:
        self.reasons.append(reason)

    async def close(self) -> None:
        return None


def chunk(document: str, index: int, *, collection: str = "docs", **payload: Any) -> Point:
    """Build one indexed chunk the way ``workers.indexer`` writes it."""
    return Point(
        point_id=f"{document}-{index}",
        collection=collection,
        vector=[1.0, 0.0, 0.0, 0.0],
        payload={"document_id": document, "chunk_index": index, **payload},
    )


@pytest.fixture
def adapter() -> MemoryAdapter:
    """Return a backend holding two documents in one collection."""
    backend = MemoryAdapter(Settings())
    backend.specs.append(CollectionSpec(name="docs", dimensions=DIMENSIONS))
    backend.points = [chunk("d_1", index) for index in range(3)] + [
        chunk("d_2", index) for index in range(2)
    ]
    return backend


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    """Return a journal that remembers both documents as indexed."""
    journal = Journal(tmp_path / "journal")
    journal.remember_content_blocking("docs", "hash-1", "d_1")
    journal.remember_content_blocking("docs", "hash-2", "d_2")
    return journal


async def test_deleting_a_document_removes_every_chunk_of_it_and_no_other(
    adapter: MemoryAdapter, journal: Journal
) -> None:
    """Nothing outside the adapters called ``delete``, so a document could never be removed."""
    service = DocumentService(adapter, journal)

    await service.delete("docs", "d_1")

    assert adapter.chunks_of("docs", "d_1") == []
    assert adapter.chunks_of("docs", "d_2") == [0, 1]


async def test_the_delete_is_one_payload_filter_on_the_document_id(
    adapter: MemoryAdapter, journal: Journal
) -> None:
    """One pushed-down filter, not a scan for ids followed by a delete of each."""
    service = DocumentService(adapter, journal)

    await service.delete("docs", "d_1")

    assert adapter.selectors == [PointSelector(collection="docs", filters={"document_id": "d_1"})]


async def test_a_deleted_document_stops_being_a_duplicate(
    adapter: MemoryAdapter, journal: Journal
) -> None:
    """The dedup record outlived the chunks, so deleted content could never be re-added."""
    service = DocumentService(adapter, journal)

    await service.delete("docs", "d_1")

    assert await journal.is_duplicate("docs", "hash-1") is False
    assert await journal.is_duplicate("docs", "hash-2") is True


async def test_the_result_says_the_journal_knew_the_document(
    adapter: MemoryAdapter, journal: Journal
) -> None:
    service = DocumentService(adapter, journal)

    deletion = await service.delete("docs", "d_1")

    assert isinstance(deletion, DocumentDeletion)
    assert deletion.known is True
    assert deletion.as_dict() == {
        "collection": "docs",
        "document_id": "d_1",
        "known": True,
        "deleted_at": deletion.deleted_at,
    }


async def test_deleting_invalidates_the_semantic_cache(
    adapter: MemoryAdapter, journal: Journal
) -> None:
    """Only ingest completion invalidated, so a cached answer could cite a deleted chunk."""
    cache = RecordingCache()
    service = DocumentService(adapter, journal, cache=cache)

    await service.delete("docs", "d_1")

    assert len(cache.reasons) == 1
    assert "d_1" in cache.reasons[0]


async def test_deleting_drops_the_document_from_the_lockfile(
    adapter: MemoryAdapter, journal: Journal, tmp_path: Path
) -> None:
    """A lockfile still listing the document would report the delete as corpus drift."""
    locks = LockStore(tmp_path / "locks")
    locks.write(
        build_lock(
            "docs",
            Settings(),
            embedding_model="m",
            embedding_model_version="1",
            dimensions=DIMENSIONS,
            document_hashes={"d_1": "hash-1", "d_2": "hash-2"},
        )
    )
    service = DocumentService(adapter, journal, locks=locks)

    await service.delete("docs", "d_1")

    lock = locks.read("docs")
    assert lock is not None
    assert lock.document_hashes == {"d_2": "hash-2"}
    assert lock.document_hashes == await journal.document_hashes("docs")


async def test_a_collection_without_a_lockfile_is_not_given_one(
    adapter: MemoryAdapter, journal: Journal, tmp_path: Path
) -> None:
    locks = LockStore(tmp_path / "locks")
    service = DocumentService(adapter, journal, locks=locks)

    await service.delete("docs", "d_1")

    assert locks.read("docs") is None


async def test_deleting_an_unknown_document_succeeds_and_says_so(
    adapter: MemoryAdapter, journal: Journal
) -> None:
    """A delete is idempotent: the second one finds nothing and is not an error."""
    service = DocumentService(adapter, journal)
    await service.delete("docs", "d_1")

    again = await service.delete("docs", "d_1")
    never = await service.delete("docs", "d_never_ingested")

    assert again.known is False
    assert never.known is False
    assert adapter.chunks_of("docs", "d_2") == [0, 1]
    assert await journal.document_hashes("docs") == {"d_2": "hash-2"}


async def test_a_document_the_journal_never_saw_is_still_removed_from_the_index(
    adapter: MemoryAdapter, tmp_path: Path
) -> None:
    """With dedup off or an imported archive the journal holds no record of what is indexed."""
    service = DocumentService(adapter, Journal(tmp_path / "empty-journal"))

    deletion = await service.delete("docs", "d_1")

    assert deletion.known is False
    assert adapter.chunks_of("docs", "d_1") == []


async def test_a_tenant_deletes_only_chunks_carrying_its_own_tag(
    adapter: MemoryAdapter, journal: Journal
) -> None:
    """A document id is derivable from a source path, so the id alone is not ownership."""
    adapter.points = [
        chunk("d_shared", 0, tenant="acme"),
        chunk("d_shared", 1, tenant="globex"),
        chunk("d_untagged", 0),
    ]
    service = DocumentService(adapter, journal)

    await service.delete("docs", "d_shared", tenant="acme")
    await service.delete("docs", "d_untagged", tenant="acme")

    assert [point.payload.get("tenant") for point in adapter.held("docs")] == ["globex", None]
    assert adapter.selectors[0].filters == {"document_id": "d_shared", "tenant": "acme"}


def test_the_filter_carries_no_tenant_when_there_is_none() -> None:
    assert document_filter("d_1") == {"document_id": "d_1"}
    assert document_filter("d_1", "acme") == {"document_id": "d_1", "tenant": "acme"}


async def test_a_missing_collection_is_not_found_under_the_name_the_caller_used(
    adapter: MemoryAdapter, journal: Journal
) -> None:
    """The backend's name carries the tenant prefix, which must never reach a response."""
    service = DocumentService(adapter, journal)

    with pytest.raises(FasterRagError) as caught:
        await service.delete("acme__absent", "d_1", display="absent")

    assert caught.value.code is ErrorCode.NOT_FOUND
    assert "'absent'" in caught.value.detail
    assert "acme__" not in caught.value.detail


@pytest.mark.parametrize("blank", ["", "   "])
async def test_a_blank_document_id_is_refused_before_anything_is_touched(
    adapter: MemoryAdapter, journal: Journal, blank: str
) -> None:
    service = DocumentService(adapter, journal)

    with pytest.raises(FasterRagError) as caught:
        await service.delete("docs", blank)

    assert caught.value.code is ErrorCode.VALIDATION_FAILED
    assert adapter.selectors == []


async def test_a_delete_the_backend_refused_can_simply_be_repeated(
    adapter: MemoryAdapter, journal: Journal
) -> None:
    """A failure halfway must not strand chunks behind a journal that already forgot them."""
    cache = RecordingCache()
    adapter.failures.append(RetrievalError("backend unreachable", retryable=True))
    service = DocumentService(adapter, journal, cache=cache)

    with pytest.raises(RetrievalError):
        await service.delete("docs", "d_1")
    assert adapter.chunks_of("docs", "d_1") == [0, 1, 2]

    retried = await service.delete("docs", "d_1")

    assert adapter.chunks_of("docs", "d_1") == []
    assert retried.known is False
    assert await journal.is_duplicate("docs", "hash-1") is False
    assert len(cache.reasons) == 1
