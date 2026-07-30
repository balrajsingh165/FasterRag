"""End-to-end ingestion against a real Qdrant.

Proves the chain the architecture describes actually runs: parse, chunk, embed, index, and
then retrieve what was written — over both the dense and the keyword leg. Every component
has unit tests; this is the test that they compose.

The embedder is a deterministic stand-in rather than a real model. The point here is the
pipeline, not the quality of anyone's vectors, and a fake keeps the test fast and free of a
multi-gigabyte download.
"""

from collections.abc import AsyncIterator, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path

import pytest

from fasterrag.adapters.embeddings.base import EmbeddingAdapter, EmbeddingResult
from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.vectordb.base import HealthStatus, SearchQuery
from fasterrag.adapters.vectordb.qdrant import QdrantAdapter
from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.bm25 import encode_query
from fasterrag.services.ingestion import IngestionService
from fasterrag.services.journal import Journal
from tests.integration.conftest import TEST_VOLUME

pytestmark = pytest.mark.integration

DIMENSIONS = 8

TERMINATION = """\
# Vendor Agreement

## 3. Termination

Either party may terminate this agreement with thirty days written notice.
"""

PAYMENT = """\
# Vendor Agreement

## 5. Payment

Invoices are payable within forty five days of receipt.
"""


class DeterministicEmbedder(EmbeddingAdapter):
    """Embeds by hashing terms into a fixed-width vector, so results are reproducible."""

    provider = "deterministic"

    @property
    def model(self) -> str:
        return "deterministic-test"

    @property
    def model_version(self) -> str:
        return "deterministic-test-v1"

    @property
    def dimensions(self) -> int:
        return DIMENSIONS

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * DIMENSIONS
        for token in text.lower().split():
            vector[hash(token) % DIMENSIONS] += 1.0
        total = sum(value * value for value in vector) ** 0.5
        return [value / total for value in vector] if total else [1.0] + [0.0] * (DIMENSIONS - 1)

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[self._vector(text) for text in texts],
            model=self.model,
            model_version=self.model_version,
        )

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True)

    async def close(self) -> None:
        return None


def threads(workers: int) -> Executor:
    return ThreadPoolExecutor(max_workers=workers)


def pipeline_settings(collection: str) -> Settings:
    return Settings.model_validate(
        {
            "vector_db": {
                "mode": "docker",
                "docker": {"volume": TEST_VOLUME},
                "collection": {"default_name": collection},
            },
            "chunking": {"chunk_size": 64, "overlap": 8},
            "embeddings": {"batch_size": 4},
            "workers": {"embedding_pool_size": 2, "queue_depth": 32},
        }
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "termination.md").write_text(TERMINATION, encoding="utf-8")
    (tmp_path / "payment.md").write_text(PAYMENT, encoding="utf-8")
    return tmp_path


@pytest.fixture
async def service(
    qdrant: Settings, tmp_path: Path, collection_name: str
) -> AsyncIterator[IngestionService]:
    settings = pipeline_settings(collection_name)
    adapter = QdrantAdapter(settings)
    built = IngestionService(
        settings,
        journal=Journal(tmp_path / "journal", checkpoint_every=1),
        adapter=adapter,
        router=TieringRouter(DeterministicEmbedder(settings)),
        executor_factory=threads,
    )
    yield built

    await adapter.client.delete_collection(collection_name)
    await built.close()


def sources(corpus: Path) -> list[str]:
    return [str(corpus / "termination.md"), str(corpus / "payment.md")]


async def test_a_corpus_is_parsed_chunked_embedded_and_indexed(
    service: IngestionService, corpus: Path
) -> None:
    record = await service.ingest(sources(corpus))

    assert record.status == "completed"
    assert record.counts["total"] == 2
    assert record.counts["parsed"] == 2
    assert record.counts["indexed"] == record.counts["chunked"]
    assert record.counts["indexed"] > 0
    assert record.finished_at is not None


async def test_indexed_chunks_are_retrievable_over_the_dense_leg(
    service: IngestionService, corpus: Path, collection_name: str
) -> None:
    await service.ingest(sources(corpus))

    vector = await service.router.default.embed_query("terminate the agreement")
    hits = await service.adapter.search(
        SearchQuery(collection=collection_name, vector=vector, limit=5)
    )

    assert hits
    assert all(hit.point_id.startswith("c_") for hit in hits)
    assert all("document_id" in hit.payload for hit in hits)


async def test_the_keyword_leg_finds_the_right_document(
    service: IngestionService, corpus: Path, collection_name: str
) -> None:
    await service.ingest(sources(corpus))

    hits = await service.adapter.search(
        SearchQuery(collection=collection_name, sparse=encode_query("invoices payable"), limit=5)
    )

    assert hits
    assert "payment.md" in hits[0].payload["source_uri"]


async def test_a_citation_can_be_resolved_from_what_was_indexed(
    service: IngestionService, corpus: Path, collection_name: str
) -> None:
    await service.ingest(sources(corpus))

    hits = await service.adapter.search(
        SearchQuery(collection=collection_name, sparse=encode_query("termination"), limit=1)
    )
    payload = hits[0].payload

    assert payload["span"]["end"] > payload["span"]["start"]
    assert payload["source_uri"].endswith("termination.md")
    assert payload["embedding_model"] == "deterministic-test"
    assert payload["section"]


async def test_re_ingesting_the_same_corpus_writes_nothing_new(
    service: IngestionService, corpus: Path
) -> None:
    first = await service.ingest(sources(corpus))
    second = await service.ingest(sources(corpus))

    assert second.counts["deduplicated"] == 2
    assert second.counts["indexed"] == 0
    assert second.counts["chunked"] == 0
    assert first.counts["indexed"] > 0


async def test_a_broken_document_is_dead_lettered_and_the_rest_still_index(
    service: IngestionService, corpus: Path
) -> None:
    (corpus / "broken.zip").write_bytes(b"PK\x03\x04 not a document")
    record = await service.ingest([*sources(corpus), str(corpus / "broken.zip")])

    assert record.status == "partial"
    assert record.counts["dead_lettered"] == 1
    assert record.counts["indexed"] > 0

    entries = service.journal.dead_lettered(record.job_id)
    assert [entry.reason_code for entry in entries] == ["PARSE_FAILED"]


async def test_a_replayed_idempotency_key_does_not_start_a_second_job(
    service: IngestionService, corpus: Path
) -> None:
    first = await service.accept(sources(corpus), idempotency_key="key-1")
    second = await service.accept(sources(corpus), idempotency_key="key-1")

    assert first.job_id == second.job_id


async def test_progress_is_checkpointed_as_documents_complete(
    service: IngestionService, corpus: Path
) -> None:
    record = await service.ingest(sources(corpus))
    reloaded = service.journal.load_job(record.job_id)

    assert reloaded.checkpoint is not None
    assert reloaded.checkpoint.last_document_index == 1
    assert reloaded.status == "completed"
