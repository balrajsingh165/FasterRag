from typing import Any

import pytest

from fasterrag.config.schema import Settings
from fasterrag.core.cache.semantic import SemanticCache, cosine_similarity
from fasterrag.core.cache.store import CacheStore, MemoryStore
from fasterrag.errors import CacheError


class BrokenStore(MemoryStore):
    """Fails every operation, as a downed Redis would."""

    async def set(self, key: str, value: bytes, *, ttl: int | None = None) -> None:
        raise CacheError("backend is unreachable")

    async def items(self) -> list[tuple[str, bytes]]:
        raise CacheError("backend is unreachable")


def build(
    *,
    threshold: float = 0.95,
    ttl: int = 3600,
    store: CacheStore | None = None,
    enabled: bool = True,
) -> SemanticCache:
    settings = Settings.model_validate(
        {"cache": {"semantic": enabled, "similarity_threshold": threshold, "ttl": ttl}}
    )
    return SemanticCache(settings, store or MemoryStore())


def response(answer: str = "thirty days") -> dict[str, Any]:
    return {
        "answer": answer,
        "citations": [{"chunk_id": "c_a", "source": "s3://a.pdf", "span": {"start": 0, "end": 4}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        "mode": "full",
        "faithfulness": 0.9,
    }


def test_identical_vectors_are_perfectly_similar() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_a_scaled_vector_points_the_same_way() -> None:
    assert cosine_similarity([1.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0)


def test_orthogonal_vectors_are_unrelated() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_a_zero_vector_matches_nothing() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == pytest.approx(0.0)


def test_vectors_of_different_lengths_do_not_match() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(0.0)


async def test_an_empty_cache_is_a_miss() -> None:
    cache = build()

    assert await cache.lookup([1.0, 0.0, 0.0]) is None
    assert cache.stats.misses == 1


async def test_the_same_question_hits() -> None:
    cache = build()
    await cache.store_response("q", [1.0, 0.0, 0.0], response())

    hit = await cache.lookup([1.0, 0.0, 0.0])

    assert hit is not None
    assert hit.response["answer"] == "thirty days"
    assert hit.similarity == pytest.approx(1.0)
    assert cache.stats.hits == 1


async def test_a_close_paraphrase_hits() -> None:
    cache = build(threshold=0.95)
    await cache.store_response("q", [1.0, 0.0, 0.0], response())

    assert await cache.lookup([0.99, 0.1, 0.0]) is not None


async def test_a_different_question_misses() -> None:
    cache = build(threshold=0.95)
    await cache.store_response("q", [1.0, 0.0, 0.0], response())

    assert await cache.lookup([0.0, 1.0, 0.0]) is None


async def test_a_tighter_threshold_rejects_what_a_looser_one_accepts() -> None:
    stored, query = [1.0, 0.0, 0.0], [0.97, 0.24, 0.0]

    loose = build(threshold=0.95)
    await loose.store_response("q", stored, response())
    tight = build(threshold=0.99)
    await tight.store_response("q", stored, response())

    assert await loose.lookup(query) is not None
    assert await tight.lookup(query) is None


async def test_the_closest_entry_wins_when_several_clear_the_threshold() -> None:
    cache = build(threshold=0.9)
    await cache.store_response("far", [0.95, 0.3, 0.0], response("far answer"))
    await cache.store_response("near", [1.0, 0.0, 0.0], response("near answer"))

    hit = await cache.lookup([1.0, 0.0, 0.0])

    assert hit is not None
    assert hit.response["answer"] == "near answer"


async def test_a_hit_reports_its_similarity_to_the_caller() -> None:
    cache = build()
    await cache.store_response("q", [1.0, 0.0, 0.0], response())

    hit = await cache.lookup([1.0, 0.0, 0.0])

    assert hit is not None
    assert hit.as_dict() == {"semantic_hit": True, "similarity": 1.0}


async def test_the_stored_body_survives_the_round_trip() -> None:
    cache = build()
    await cache.store_response("q", [1.0, 0.0, 0.0], response())

    hit = await cache.lookup([1.0, 0.0, 0.0])

    assert hit is not None
    assert hit.response == response()


async def test_the_question_is_recorded_with_the_entry() -> None:
    cache = build()
    await cache.store_response("what is the notice period?", [1.0, 0.0, 0.0], response())

    hit = await cache.lookup([1.0, 0.0, 0.0])

    assert hit is not None
    assert hit.question == "what is the notice period?"


async def test_a_disabled_cache_never_stores_or_hits() -> None:
    cache = build(enabled=False)
    await cache.store_response("q", [1.0, 0.0, 0.0], response())

    assert await cache.lookup([1.0, 0.0, 0.0]) is None
    assert cache.stats.lookups == 0


async def test_an_expired_entry_misses() -> None:
    import time

    cache = build(ttl=1)
    await cache.store_response("q", [1.0, 0.0, 0.0], response())
    time.sleep(1.05)

    assert await cache.lookup([1.0, 0.0, 0.0]) is None


async def test_a_corpus_change_drops_every_entry() -> None:
    cache = build()
    await cache.store_response("a", [1.0, 0.0, 0.0], response())
    await cache.store_response("b", [0.0, 1.0, 0.0], response())

    await cache.invalidate("ingest job j_1")

    assert await cache.lookup([1.0, 0.0, 0.0]) is None
    assert cache.stats.invalidations == 2


async def test_invalidating_an_empty_cache_is_harmless() -> None:
    cache = build()

    await cache.invalidate("ingest job j_1")

    assert cache.stats.invalidations == 0


async def test_a_downed_backend_reads_as_a_miss_rather_than_an_error() -> None:
    cache = build(store=BrokenStore())

    assert await cache.lookup([1.0, 0.0, 0.0]) is None
    assert cache.stats.errors == 1


async def test_a_downed_backend_never_fails_a_write() -> None:
    cache = build(store=BrokenStore())

    await cache.store_response("q", [1.0, 0.0, 0.0], response())

    assert cache.stats.errors == 1


async def test_a_downed_backend_never_fails_an_invalidation() -> None:
    cache = build(store=BrokenStore())

    await cache.invalidate("ingest job j_1")

    assert cache.stats.errors == 1


async def test_a_corrupt_entry_is_skipped_rather_than_served() -> None:
    store = MemoryStore()
    cache = build(store=store)
    await cache.store_response("good", [1.0, 0.0, 0.0], response("real answer"))
    await store.set("sem:corrupt", b"not an entry at all")

    hit = await cache.lookup([1.0, 0.0, 0.0])

    assert hit is not None
    assert hit.response["answer"] == "real answer"


async def test_a_vector_from_another_model_never_matches() -> None:
    cache = build()
    await cache.store_response("q", [1.0, 0.0, 0.0], response())

    assert await cache.lookup([1.0, 0.0, 0.0, 0.0]) is None


async def test_rewriting_a_question_replaces_its_entry() -> None:
    cache = build()
    await cache.store_response("q", [1.0, 0.0, 0.0], response("old"))
    await cache.store_response("q", [1.0, 0.0, 0.0], response("new"))

    hit = await cache.lookup([1.0, 0.0, 0.0])

    assert hit is not None
    assert hit.response["answer"] == "new"
