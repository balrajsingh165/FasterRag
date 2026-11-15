"""Deduplication is kept per vector database, not per collection name (TASK-0550).

The dedup journal remembers which content a collection already holds. It used to be keyed
by the collection name alone, so after ``vector_db.provider`` changed, a re-ingest found
every file "already indexed" and skipped it, and the new backend never received anything.
These cases run the whole ingestion service against two in-memory backends.
"""

from __future__ import annotations

from pathlib import Path

from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.config.schema import Settings
from fasterrag.core.identity import document_id
from fasterrag.services.ingestion import IngestionService
from fasterrag.services.journal import Journal, backend_identity, create_journal
from tests.unit.services.test_documents import MemoryAdapter
from tests.unit.services.test_ingestion import StubEmbedder, settings, threads

COLLECTION = "default"


def on(provider: str, **vector_db: object) -> Settings:
    """Return test settings that name ``provider`` as the vector database."""
    return settings(vector_db={"provider": provider, **vector_db})


def service(configured: Settings, journal: Journal) -> tuple[IngestionService, MemoryAdapter]:
    """Return an ingestion service writing to a fresh in-memory backend."""
    backend = MemoryAdapter(configured)
    built = IngestionService(
        configured,
        journal=journal,
        adapter=backend,
        router=TieringRouter(StubEmbedder(configured)),
        executor_factory=threads,
    )
    return built, backend


def corpus(tmp_path: Path) -> list[str]:
    """Write two small documents and return their paths."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "leave.md").write_text("# Leave\n\nParental leave is 26 weeks.\n", encoding="utf-8")
    (root / "meals.md").write_text("# Meals\n\nThe allowance is 41 pounds.\n", encoding="utf-8")
    return [str(root / "leave.md"), str(root / "meals.md")]


async def test_a_new_vector_database_receives_every_document_the_old_one_held(
    tmp_path: Path,
) -> None:
    """After switching provider the dedup journal skipped every file; the new store stayed empty."""
    sources = corpus(tmp_path)
    root = tmp_path / "journal"
    first, old = service(on("qdrant"), create_journal(on("qdrant"), root))
    await first.run(await first.accept(sources))

    second, new = service(on("chroma"), create_journal(on("chroma"), root))
    settled = await second.run(await second.accept(sources))

    assert settled.counts["deduplicated"] == 0
    assert settled.counts["parsed"] == 2
    assert {str(point.payload["document_id"]) for point in new.held(COLLECTION)} == {
        document_id(source) for source in sources
    }
    assert old.held(COLLECTION)


async def test_returning_to_the_first_vector_database_still_deduplicates(tmp_path: Path) -> None:
    sources = corpus(tmp_path)
    root = tmp_path / "journal"
    first, _ = service(on("qdrant"), create_journal(on("qdrant"), root))
    await first.run(await first.accept(sources))
    second, _ = service(on("chroma"), create_journal(on("chroma"), root))
    await second.run(await second.accept(sources))

    again, _ = service(on("qdrant"), create_journal(on("qdrant"), root))
    settled = await again.run(await again.accept(sources))

    assert settled.counts["deduplicated"] == 2


def test_the_same_provider_at_another_address_is_another_backend() -> None:
    assert backend_identity(on("qdrant")) == backend_identity(on("qdrant"))
    assert backend_identity(on("qdrant")) != backend_identity(on("qdrant", host="qdrant.internal"))
    assert backend_identity(on("chroma")) != backend_identity(
        on("chroma", chroma={"path": "elsewhere"})
    )
    assert backend_identity(on("qdrant")) != backend_identity(on("chroma"))


def test_a_journal_written_before_the_change_belongs_to_the_backend_configured_now(
    tmp_path: Path,
) -> None:
    """A dedup file from an earlier release is read as the current backend's, and only its."""
    root = tmp_path / "journal"
    Journal(root).remember_content_blocking(COLLECTION, "hash-a", "doc-a")

    current = create_journal(on("qdrant"), root)
    assert current.known_content_blocking(COLLECTION) == {"hash-a": "doc-a"}
    assert not (root / "collections" / COLLECTION / "content-hashes.jsonl").exists()

    other = create_journal(on("chroma"), root)
    assert other.known_content_blocking(COLLECTION) == {}
    assert create_journal(on("qdrant"), root).known_content_blocking(COLLECTION) == {
        "hash-a": "doc-a"
    }


def test_the_retention_pass_compacts_a_backend_scoped_dedup_file(tmp_path: Path) -> None:
    backend = backend_identity(on("qdrant"))
    journal = Journal(tmp_path / "journal", retention_days=1, backend=backend)
    journal.remember_content_blocking(COLLECTION, "hash-a", "doc-a")
    journal.remember_content_blocking(COLLECTION, "hash-b", "doc-a")

    report = journal.prune_blocking()

    assert report.dedup_lines == 1
    assert journal.known_content_blocking(COLLECTION) == {"hash-b": "doc-a"}
