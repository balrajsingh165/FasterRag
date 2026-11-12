"""An ingest through a provider that refuses large batches (TASK-0478).

Every batch the provider refused for its size used to be dead-lettered whole, so a provider
whose limit sat below ``embeddings.batch_size`` indexed nothing at all.
"""

from collections.abc import Sequence
from pathlib import Path

from fasterrag.adapters.embeddings.base import EmbeddingResult, EmbedTooLargeError
from fasterrag.config.schema import Settings
from tests.unit.services.test_ingestion import StubEmbedder, build, settings

DOCUMENTS = 6


class ScriptedLimitEmbedder(StubEmbedder):
    """Refuses a batch of more than ``limit`` texts, or any text containing ``poison``."""

    def __init__(self, configured: Settings, *, limit: int, poison: str | None = None) -> None:
        super().__init__(configured)
        self.limit = limit
        self.poison = poison
        self.refusals = 0

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        poisoned = self.poison is not None and any(self.poison in text for text in texts)
        if len(texts) > self.limit or poisoned:
            self.refusals += 1
            raise EmbedTooLargeError("scripted provider refused the request as too large")
        return await super().embed_documents(texts)


def corpus_of(root: Path) -> list[str]:
    root.mkdir()
    paths = []
    for index in range(DOCUMENTS):
        path = root / f"policy_{index}.md"
        path.write_text(f"# Policy {index}\n\nRule number {index} applies.\n", encoding="utf-8")
        paths.append(str(path))
    return paths


async def test_a_provider_that_refuses_large_batches_is_ingested_through(tmp_path: Path) -> None:
    configured = settings()
    embedder = ScriptedLimitEmbedder(configured, limit=2)
    service, adapter = build(tmp_path, configured, embedder=embedder)

    settled = await service.run(await service.accept(corpus_of(tmp_path / "corpus")))

    assert embedder.refusals > 0
    assert settled.status == "completed"
    assert settled.counts["dead_lettered"] == 0
    assert settled.counts["indexed"] == DOCUMENTS
    assert len(adapter.points) == settled.counts["chunked"]


async def test_only_the_document_that_cannot_fit_is_dead_lettered(tmp_path: Path) -> None:
    configured = settings()
    embedder = ScriptedLimitEmbedder(configured, limit=8, poison="Rule number 3")
    service, _ = build(tmp_path, configured, embedder=embedder)

    record = await service.accept(corpus_of(tmp_path / "corpus"))
    settled = await service.run(record)
    dead = await service.journal.dead_lettered(record.job_id)

    assert settled.counts["dead_lettered"] == 1
    assert settled.counts["indexed"] == DOCUMENTS - 1
    assert [Path(entry.source).name for entry in dead] == ["policy_3.md"]
