from collections.abc import Sequence

import pytest

from fasterrag.adapters.embeddings.base import EmbeddingAdapter, EmbeddingResult
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.core.cache.embedding import (
    CachingEmbeddingAdapter,
    decode_vector,
    embedding_key,
    encode_vector,
)
from fasterrag.core.cache.store import CacheStore, MemoryStore
from fasterrag.errors import CacheError, EmbedError


class CountingEmbedder(EmbeddingAdapter):
    """Returns a deterministic vector per text and counts what it was asked to embed."""

    provider = "counting"

    def __init__(self, settings: Settings, version: str = "1.0") -> None:
        super().__init__(settings)
        self.embedded: list[str] = []
        self.calls = 0
        self._version = version

    @property
    def model(self) -> str:
        return "counting-model"

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def dimensions(self) -> int | None:
        return 3

    def _vector(self, text: str) -> list[float]:
        return [float(len(text)), float(sum(map(ord, text)) % 97), 1.0]

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        self.calls += 1
        self.embedded.extend(texts)
        return EmbeddingResult(
            vectors=[self._vector(text) for text in texts],
            model=self.model,
            model_version=self.model_version,
            total_tokens=len(texts),
        )

    async def embed_query(self, text: str) -> list[float]:
        self.calls += 1
        self.embedded.append(text)
        return self._vector(text)

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True)

    async def close(self) -> None:
        return None


class BrokenStore(MemoryStore):
    """Fails every operation, as a downed Redis would."""

    async def get(self, key: str) -> bytes | None:
        raise CacheError("backend is unreachable")

    async def set(self, key: str, value: bytes, *, ttl: int | None = None) -> None:
        raise CacheError("backend is unreachable")


def build(store: CacheStore | None = None) -> tuple[CachingEmbeddingAdapter, CountingEmbedder]:
    settings = Settings()
    inner = CountingEmbedder(settings)
    return CachingEmbeddingAdapter(inner, store or MemoryStore()), inner


def test_a_vector_round_trips_through_its_packing() -> None:
    vector = [0.5, -1.25, 3.0]

    assert decode_vector(encode_vector(vector)) == pytest.approx(vector)


def test_a_truncated_vector_decodes_to_nothing() -> None:
    assert decode_vector(b"abc") is None
    assert decode_vector(b"") is None


def test_the_key_separates_models() -> None:
    assert embedding_key("text", "model-a", "1.0") != embedding_key("text", "model-b", "1.0")


def test_the_key_separates_model_versions() -> None:
    assert embedding_key("text", "model", "1.0") != embedding_key("text", "model", "2.0")


def test_the_key_separates_texts() -> None:
    assert embedding_key("a", "model", "1.0") != embedding_key("b", "model", "1.0")


def test_the_key_never_contains_the_text_itself() -> None:
    assert "secret passage" not in embedding_key("secret passage", "model", "1.0")


async def test_a_repeated_text_is_embedded_once() -> None:
    adapter, inner = build()

    await adapter.embed_documents(["hello"])
    await adapter.embed_documents(["hello"])

    assert inner.embedded == ["hello"]


async def test_only_the_uncached_part_of_a_batch_reaches_the_provider() -> None:
    adapter, inner = build()

    await adapter.embed_documents(["a", "b"])
    await adapter.embed_documents(["a", "b", "c"])

    assert inner.embedded == ["a", "b", "c"]


async def test_a_partially_cached_batch_returns_vectors_in_the_given_order() -> None:
    adapter, inner = build()

    first = await adapter.embed_documents(["alpha", "beta", "gamma"])
    second = await adapter.embed_documents(["gamma", "alpha", "delta"])

    assert second.vectors[0] == pytest.approx(first.vectors[2])
    assert second.vectors[1] == pytest.approx(first.vectors[0])
    assert second.vectors[2] == pytest.approx(inner._vector("delta"))


async def test_a_fully_cached_batch_never_calls_the_provider() -> None:
    adapter, inner = build()
    await adapter.embed_documents(["a", "b"])
    calls = inner.calls

    await adapter.embed_documents(["a", "b"])

    assert inner.calls == calls


async def test_a_fully_cached_batch_reports_no_tokens_billed() -> None:
    adapter, _ = build()
    await adapter.embed_documents(["a"])

    assert (await adapter.embed_documents(["a"])).total_tokens == 0


async def test_a_repeated_query_is_embedded_once() -> None:
    adapter, inner = build()

    first = await adapter.embed_query("what is the notice period?")
    second = await adapter.embed_query("what is the notice period?")

    assert first == pytest.approx(second)
    assert inner.embedded == ["what is the notice period?"]


async def test_a_query_and_a_document_share_one_cached_vector() -> None:
    adapter, inner = build()

    await adapter.embed_documents(["shared text"])
    await adapter.embed_query("shared text")

    assert inner.embedded == ["shared text"]


async def test_hits_and_misses_are_counted() -> None:
    adapter, _ = build()

    await adapter.embed_documents(["a", "b"])
    await adapter.embed_documents(["a", "c"])

    assert adapter.stats.misses == 3
    assert adapter.stats.hits == 1
    assert adapter.stats.hit_rate == pytest.approx(0.25)


async def test_a_different_model_version_never_serves_the_old_vector() -> None:
    settings = Settings()
    store = MemoryStore()
    old = CachingEmbeddingAdapter(CountingEmbedder(settings, version="1.0"), store)
    await old.embed_documents(["text"])

    new_inner = CountingEmbedder(settings, version="2.0")
    await CachingEmbeddingAdapter(new_inner, store).embed_documents(["text"])

    assert new_inner.embedded == ["text"]


async def test_a_downed_cache_still_embeds() -> None:
    adapter, inner = build(BrokenStore())

    result = await adapter.embed_documents(["a"])

    assert result.vectors[0] == pytest.approx(inner._vector("a"))
    assert adapter.stats.errors > 0


async def test_a_downed_cache_still_answers_queries() -> None:
    adapter, _ = build(BrokenStore())

    assert len(await adapter.embed_query("q")) == 3


async def test_a_provider_failure_is_not_swallowed_by_the_cache() -> None:
    class FailingEmbedder(CountingEmbedder):
        async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
            raise EmbedError("provider is down")

    adapter = CachingEmbeddingAdapter(FailingEmbedder(Settings()), MemoryStore())

    with pytest.raises(EmbedError):
        await adapter.embed_documents(["a"])


async def test_the_model_identity_passes_through() -> None:
    adapter, inner = build()

    assert adapter.model == inner.model
    assert adapter.model_version == inner.model_version
    assert adapter.dimensions == inner.dimensions


async def test_health_reports_the_provider_not_the_cache() -> None:
    adapter, _ = build(BrokenStore())

    assert (await adapter.health()).healthy is True


async def test_an_empty_batch_never_calls_the_provider() -> None:
    adapter, inner = build()

    result = await adapter.embed_documents([])

    assert result.vectors == []
    assert inner.calls == 0
