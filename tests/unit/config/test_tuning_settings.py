"""The settings that only matter if something downstream actually reads them.

Each test here pins a knob to a value and asserts the behaviour changes. A setting that
validates but is never consumed is worse than no setting: it reads as a supported control
and silently does nothing.
"""

from typing import Any

from fasterrag.config.schema import Settings
from fasterrag.core.cache import create_embedding_store, create_semantic_store
from fasterrag.core.chunking import create_chunker
from fasterrag.core.retrieval.bm25 import encode_document


def settings(**sections: Any) -> Settings:
    return Settings.model_validate(sections)


def test_bm25_saturation_changes_the_encoded_weights() -> None:
    text = "policy policy policy travel allowance"

    flat = encode_document(text, k1=0.1)
    steep = encode_document(text, k1=2.5)

    assert flat.values != steep.values


def test_bm25_length_normalisation_changes_the_encoded_weights() -> None:
    text = "the travel allowance policy covers meals and lodging for contractors"

    off = encode_document(text, b=0.0)
    full = encode_document(text, b=1.0)

    assert off.values != full.values


def test_bm25_defaults_match_the_schema_defaults() -> None:
    """A default that drifts between the schema and the encoder is a silent reindex."""
    configured = settings()
    text = "policy policy travel"

    assert encode_document(text) == encode_document(
        text, k1=configured.retrieval.bm25_k1, b=configured.retrieval.bm25_b
    )


def test_the_semantic_percentile_reaches_the_chunker() -> None:
    class Embedder:
        def encode(self, texts: list[str]) -> list[list[float]]:
            return [[float(index), 1.0] for index, _ in enumerate(texts)]

    chunker = create_chunker(
        settings(chunking={"strategy": "semantic", "semantic_percentile": 0.6}),
        embedder=Embedder(),  # type: ignore[arg-type]
    )

    assert chunker._percentile == 0.6  # type: ignore[attr-defined]


def test_the_cache_ceiling_reaches_the_semantic_store() -> None:
    store = create_semantic_store(settings(cache={"backend": "memory", "max_entries": 7}))

    assert store._maximum == 7  # type: ignore[attr-defined]


def test_the_cache_ceiling_reaches_the_embedding_store() -> None:
    store = create_embedding_store(
        settings(embeddings={"cache": {"backend": "memory", "max_entries": 3}})
    )

    assert store._maximum == 3  # type: ignore[attr-defined]


async def test_a_bounded_store_actually_evicts() -> None:
    """A ceiling nothing enforces is a memory leak with a number next to it."""
    store = create_semantic_store(settings(cache={"backend": "memory", "max_entries": 2}))

    for index in range(5):
        await store.set(f"key-{index}", b"value", ttl=3600)

    assert await store.get("key-0") is None
    assert await store.get("key-4") is not None
