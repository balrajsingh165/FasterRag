"""Updating and re-adding a document through the whole ingestion pipeline (TASK-0443).

Chunk ids are deterministic in the document id and the chunk index, so re-ingesting a
changed document overwrites its chunks in place. That is only an update when the new version
is at least as long as the old one: a document that shrank from five chunks to two overwrote
two and left three behind, still retrievable, still citing text the source no longer holds.

These cases run ``IngestionService`` with both worker pools against an in-memory backend that
really overwrites and really deletes, so what is asserted is what a query could retrieve
afterwards — not that some method was called.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.vectordb.base import Point, PointSelector, UpsertResult
from fasterrag.config.schema import Settings
from fasterrag.core.identity import document_id
from fasterrag.errors import FasterRagError, RetrievalError
from fasterrag.services.documents import DocumentService
from fasterrag.services.ingestion import IngestionService
from fasterrag.services.journal import Journal
from fasterrag.services.lockfile import LockStore
from tests.unit.services.test_documents import MemoryAdapter
from tests.unit.services.test_ingestion import StubEmbedder, settings, threads

COLLECTION = "default"

LONG = "\n\n".join(
    f"## Section {number}\n\n"
    + " ".join(f"Clause {number}.{line} sets out obligation {line} in full." for line in range(40))
    for number in range(6)
)
SHORT = "## Section 0\n\nThe agreement was replaced by a single sentence.\n"


class OrderedAdapter(MemoryAdapter):
    """Also records the order of writes and deletes, which is what "after" is about."""

    def __init__(self, configured: Settings) -> None:
        super().__init__(configured)
        self.operations: list[tuple[str, str]] = []

    async def upsert(self, points: list[Point]) -> UpsertResult:
        for written in points:
            self.operations.append(("upsert", str(written.payload["document_id"])))
        return await super().upsert(points)

    async def delete(self, selector: PointSelector) -> None:
        self.operations.append(("delete", str((selector.filters or {}).get("document_id"))))
        await super().delete(selector)


def build(
    tmp_path: Path, configured: Settings | None = None, **kwargs: Any
) -> tuple[IngestionService, OrderedAdapter]:
    """Return an ingestion service over a backend that overwrites and deletes for real."""
    resolved = configured or settings()
    backend = OrderedAdapter(resolved)
    service = IngestionService(
        resolved,
        journal=Journal(tmp_path / "journal", checkpoint_every=1),
        adapter=backend,
        router=TieringRouter(StubEmbedder(resolved)),
        executor_factory=threads,
        **kwargs,
    )
    return service, backend


@pytest.fixture
def contract(tmp_path: Path) -> Path:
    """Return a document long enough to be several chunks."""
    path = tmp_path / "corpus" / "contract.md"
    path.parent.mkdir()
    path.write_text(LONG, encoding="utf-8")
    return path


@pytest.fixture
def handbook(tmp_path: Path, contract: Path) -> Path:
    """Return a second document, which no change to the first may touch."""
    path = contract.parent / "handbook.md"
    path.write_text("# Leave\n\nParental leave is 26 weeks at full pay.\n", encoding="utf-8")
    return path


async def ingest(service: IngestionService, *paths: Path) -> dict[str, int]:
    """Ingest files and return the settled job's counts."""
    record = await service.run(await service.accept([str(path) for path in paths]))
    assert record.status == "completed"
    return record.counts


async def test_a_document_that_shrank_leaves_no_stale_chunk(tmp_path: Path, contract: Path) -> None:
    """The tail of the longer version stayed in the index, retrievable and citable."""
    service, adapter = build(tmp_path)
    document = document_id(str(contract))
    await ingest(service, contract)
    before = adapter.chunks_of(COLLECTION, document)
    assert len(before) >= 3

    contract.write_text(SHORT, encoding="utf-8")
    counts = await ingest(service, contract)

    after = adapter.chunks_of(COLLECTION, document)
    assert after == list(range(counts["chunked"]))
    assert len(after) < len(before)
    assert [index for index in after if index >= counts["chunked"]] == []


async def test_only_the_new_version_of_a_shrunken_document_remains(
    tmp_path: Path, contract: Path
) -> None:
    """Every surviving chunk must belong to the version the source now holds."""
    service, adapter = build(tmp_path)
    await ingest(service, contract)

    contract.write_text(SHORT, encoding="utf-8")
    await ingest(service, contract)

    hashes = {point.payload["content_hash"] for point in adapter.held(COLLECTION)}
    texts = " ".join(str(point.payload["text"]) for point in adapter.held(COLLECTION))
    assert len(hashes) == 1
    assert "Clause" not in texts


async def test_the_tail_is_trimmed_only_after_the_new_chunks_are_written(
    tmp_path: Path, contract: Path
) -> None:
    """Trimming first would open a window in which the document has no chunks at all."""
    service, adapter = build(tmp_path)
    document = document_id(str(contract))
    await ingest(service, contract)
    adapter.operations.clear()

    contract.write_text(SHORT, encoding="utf-8")
    counts = await ingest(service, contract)

    assert adapter.operations == [("upsert", document)] * counts["chunked"] + [("delete", document)]
    assert adapter.selectors[-1] == PointSelector(
        collection=COLLECTION,
        filters={"document_id": document, "chunk_index": {"$gte": counts["chunked"]}},
    )


async def test_a_first_ingest_deletes_nothing(
    tmp_path: Path, contract: Path, handbook: Path
) -> None:
    """A bulk load must not pay a filtered delete per document for tails that cannot exist."""
    service, adapter = build(tmp_path)

    await ingest(service, contract, handbook)

    assert adapter.selectors == []


async def test_an_unchanged_document_is_neither_rewritten_nor_trimmed(
    tmp_path: Path, contract: Path
) -> None:
    service, adapter = build(tmp_path)
    await ingest(service, contract)
    adapter.operations.clear()

    counts = await ingest(service, contract)

    assert counts["deduplicated"] == 1
    assert adapter.operations == []


async def test_a_document_that_grew_keeps_every_chunk(tmp_path: Path, contract: Path) -> None:
    service, adapter = build(tmp_path)
    document = document_id(str(contract))
    contract.write_text(SHORT, encoding="utf-8")
    await ingest(service, contract)

    contract.write_text(LONG, encoding="utf-8")
    counts = await ingest(service, contract)

    assert adapter.chunks_of(COLLECTION, document) == list(range(counts["chunked"]))
    assert counts["chunked"] >= 3


async def test_trimming_one_document_leaves_another_alone(
    tmp_path: Path, contract: Path, handbook: Path
) -> None:
    service, adapter = build(tmp_path)
    other = document_id(str(handbook))
    await ingest(service, contract, handbook)
    untouched = adapter.chunks_of(COLLECTION, other)

    contract.write_text(SHORT, encoding="utf-8")
    await ingest(service, contract, handbook)

    assert adapter.chunks_of(COLLECTION, other) == untouched
    assert untouched


async def test_a_shrunken_document_invalidates_the_cache_and_updates_the_lockfile(
    tmp_path: Path, contract: Path
) -> None:
    """An update is a corpus change like any other, whatever it wrote."""
    from tests.unit.services.test_documents import RecordingCache

    cache = RecordingCache()
    locks = LockStore(tmp_path / "locks")
    service, _ = build(tmp_path, cache=cache, locks=locks)
    document = document_id(str(contract))
    await ingest(service, contract)
    first = locks.read(COLLECTION)
    cache.reasons.clear()

    contract.write_text(SHORT, encoding="utf-8")
    await ingest(service, contract)

    second = locks.read(COLLECTION)
    assert first is not None
    assert second is not None
    assert second.document_hashes[document] != first.document_hashes[document]
    assert cache.reasons


async def test_a_trim_the_backend_refuses_fails_the_job_rather_than_keeping_the_tail(
    tmp_path: Path, contract: Path
) -> None:
    """Recording the update as indexed with its tail still present would make it permanent."""
    service, adapter = build(tmp_path)
    document = document_id(str(contract))
    await ingest(service, contract)

    contract.write_text(SHORT, encoding="utf-8")
    adapter.failures.append(RetrievalError("backend unreachable", retryable=False))
    with pytest.raises(FasterRagError):
        await service.run(await service.accept([str(contract)]))

    counts = await ingest(service, contract)

    assert counts["deduplicated"] == 0
    assert adapter.chunks_of(COLLECTION, document) == list(range(counts["chunked"]))


async def test_a_retryable_trim_failure_is_retried(tmp_path: Path, contract: Path) -> None:
    configured = settings(
        reliability={"retries": {"max_attempts": 2, "backoff_base_ms": 1, "jitter": False}}
    )
    service, adapter = build(tmp_path, configured)
    document = document_id(str(contract))
    await ingest(service, contract)

    contract.write_text(SHORT, encoding="utf-8")
    adapter.failures.append(RetrievalError("backend busy", retryable=True))
    counts = await ingest(service, contract)

    assert adapter.chunks_of(COLLECTION, document) == list(range(counts["chunked"]))


async def test_a_deleted_document_is_indexed_again_by_the_next_ingest(
    tmp_path: Path, contract: Path, handbook: Path
) -> None:
    """The dedup record outlived the chunks, so a deleted document came back 'deduplicated'."""
    service, adapter = build(tmp_path)
    document = document_id(str(contract))
    await ingest(service, contract, handbook)
    written = adapter.chunks_of(COLLECTION, document)

    deletion = await DocumentService(adapter, service.journal).delete(COLLECTION, document)
    assert adapter.chunks_of(COLLECTION, document) == []
    counts = await ingest(service, contract, handbook)

    assert deletion.known is True
    assert counts["deduplicated"] == 1
    assert counts["parsed"] == 1
    assert adapter.chunks_of(COLLECTION, document) == written


async def test_the_lockfile_written_after_a_delete_does_not_list_the_document(
    tmp_path: Path, contract: Path, handbook: Path
) -> None:
    """``_write_lockfile`` reads ``document_hashes``, which never forgot anything."""
    locks = LockStore(tmp_path / "locks")
    service, adapter = build(tmp_path, locks=locks)
    await ingest(service, contract, handbook)
    await DocumentService(adapter, service.journal, locks=locks).delete(
        COLLECTION, document_id(str(contract))
    )

    handbook.write_text("# Leave\n\nParental leave is now 30 weeks.\n", encoding="utf-8")
    await ingest(service, handbook)

    lock = locks.read(COLLECTION)
    assert lock is not None
    assert set(lock.document_hashes) == {document_id(str(handbook))}


async def test_a_document_restored_to_an_earlier_version_is_indexed_again(
    tmp_path: Path, contract: Path
) -> None:
    """The superseded hash stayed remembered, so reverting an edit was 'deduplicated' away."""
    service, adapter = build(tmp_path)
    document = document_id(str(contract))
    await ingest(service, contract)
    original = adapter.chunks_of(COLLECTION, document)
    contract.write_text(SHORT, encoding="utf-8")
    await ingest(service, contract)

    contract.write_text(LONG, encoding="utf-8")
    counts = await ingest(service, contract)

    assert counts["deduplicated"] == 0
    assert adapter.chunks_of(COLLECTION, document) == original


async def test_a_known_document_that_became_empty_loses_every_chunk(
    tmp_path: Path, contract: Path
) -> None:
    """An emptied source has a new length of zero, so its whole old version is the tail."""
    service, adapter = build(tmp_path)
    document = document_id(str(contract))
    await ingest(service, contract)
    assert adapter.chunks_of(COLLECTION, document)

    contract.write_text("", encoding="utf-8")
    await service.run(await service.accept([str(contract)]))

    assert adapter.chunks_of(COLLECTION, document) == []
