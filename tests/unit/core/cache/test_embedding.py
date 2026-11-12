import hashlib
from collections.abc import Sequence

import pytest

from fasterrag.adapters.embeddings.base import EmbeddingAdapter, EmbeddingResult
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.core.cache.embedding import (
    CachingEmbeddingAdapter,
    InputType,
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


class OtherProviderEmbedder(CountingEmbedder):
    """The same model name served by a different provider, with different vectors."""

    provider = "other"

    def _vector(self, text: str) -> list[float]:
        return [-1.0, -1.0, -1.0]


def build(store: CacheStore | None = None) -> tuple[CachingEmbeddingAdapter, CountingEmbedder]:
    settings = Settings()
    inner = CountingEmbedder(settings)
    return CachingEmbeddingAdapter(inner, store or MemoryStore()), inner


def key(
    text: str = "text",
    *,
    provider: str = "provider",
    model: str = "model",
    model_version: str = "1.0",
    dimensions: int | None = None,
    input_type: InputType = "document",
) -> str:
    return embedding_key(
        text,
        provider=provider,
        model=model,
        model_version=model_version,
        dimensions=dimensions,
        input_type=input_type,
    )


def test_a_vector_round_trips_through_its_packing() -> None:
    vector = [0.5, -1.25, 3.0]

    assert decode_vector(encode_vector(vector)) == pytest.approx(vector)


def test_a_truncated_vector_decodes_to_nothing() -> None:
    assert decode_vector(b"abc") is None
    assert decode_vector(b"") is None


def test_the_key_separates_models() -> None:
    assert key(model="model-a") != key(model="model-b")


def test_the_key_separates_model_versions() -> None:
    assert key(model_version="1.0") != key(model_version="2.0")


def test_the_key_separates_texts() -> None:
    assert key("a") != key("b")


def test_the_key_separates_input_types() -> None:
    """The key ignored the input type, so a query was served a document's vector."""
    assert key(input_type="query") != key(input_type="document")


def test_the_key_separates_providers() -> None:
    """The key ignored the provider, so two providers' same-named models shared vectors."""
    assert key(provider="openai") != key(provider="ollama")


def test_the_key_separates_dimension_settings() -> None:
    """The key ignored embeddings.dimensions, so a shortened vector outlived the setting."""
    assert key(dimensions=None) != key(dimensions=256)
    assert key(dimensions=256) != key(dimensions=512)


def test_the_key_cannot_be_forged_by_moving_a_separator_between_fields() -> None:
    """Model ids contain colons, so joining fields with one is not enough to tell them apart."""
    assert key(model="a:b", model_version="c") != key(model="a", model_version="b:c")
    assert key(provider="a:b", model="c") != key(provider="a", model="b:c")


def test_the_key_never_contains_the_text_itself() -> None:
    assert "secret passage" not in key("secret passage")


def test_the_key_is_stable_across_calls() -> None:
    assert key("text", dimensions=256, input_type="query") == key(
        "text", dimensions=256, input_type="query"
    )


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


async def test_a_query_and_a_document_of_the_same_text_are_separate_entries() -> None:
    """One entry served both, though a provider may embed a query and a passage differently."""
    adapter, inner = build()

    await adapter.embed_documents(["shared text"])
    await adapter.embed_query("shared text")
    await adapter.embed_documents(["shared text"])
    await adapter.embed_query("shared text")

    assert inner.embedded == ["shared text", "shared text"]
    assert inner.calls == 2


async def test_two_providers_sharing_a_store_never_share_a_vector() -> None:
    """A second provider serving a model of the same name was handed the first one's vector."""
    settings = Settings()
    store = MemoryStore()
    await CachingEmbeddingAdapter(CountingEmbedder(settings), store).embed_documents(["text"])

    other = OtherProviderEmbedder(settings)
    result = await CachingEmbeddingAdapter(other, store).embed_documents(["text"])

    assert other.embedded == ["text"]
    assert result.vectors == [[-1.0, -1.0, -1.0]]


async def test_a_changed_dimension_setting_never_serves_the_old_vector() -> None:
    """A vector cached at one embeddings.dimensions was served after the setting changed."""
    store = MemoryStore()
    native = CountingEmbedder(Settings())
    await CachingEmbeddingAdapter(native, store).embed_documents(["text"])

    shortened = CountingEmbedder(Settings.model_validate({"embeddings": {"dimensions": 256}}))
    await CachingEmbeddingAdapter(shortened, store).embed_documents(["text"])
    await CachingEmbeddingAdapter(shortened, store).embed_query("text")

    assert shortened.embedded == ["text", "text"]


async def test_an_entry_written_under_the_old_key_format_is_never_served() -> None:
    """Entries keyed without provider, dimensions and input type must not survive the fix."""
    adapter, inner = build()
    digest = hashlib.sha256(b"text").hexdigest()
    await adapter.store.set(
        f"emb:{inner.model}:{inner.model_version}:{digest}", encode_vector([9.0, 9.0, 9.0])
    )

    result = await adapter.embed_documents(["text"])
    vector = await adapter.embed_query("text")

    assert inner.embedded == ["text", "text"]
    assert result.vectors[0] == pytest.approx(inner._vector("text"))
    assert vector == pytest.approx(inner._vector("text"))


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
