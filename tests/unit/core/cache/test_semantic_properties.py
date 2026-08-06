"""Semantic cache isolation, generated rather than reasoned about.

Lookup compares the query vector against *every* stored entry, so the tenant check cannot
live on the key — the scan never reads keys. It lives on the entry, and this suite asserts
the consequence over generated vectors: no similarity, however high, returns another
tenant's answer.

That failure mode is the quiet kind. It arrives with a cache hit's latency, a plausible
answer, and nothing in the logs.
"""

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from fasterrag.config.schema import Settings
from fasterrag.core.cache import MemoryStore
from fasterrag.core.cache.semantic import SemanticCache

DIMENSIONS = 8

VECTOR = st.lists(
    st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    min_size=DIMENSIONS,
    max_size=DIMENSIONS,
).filter(lambda values: any(abs(value) > 1e-6 for value in values))

TENANT = st.one_of(st.none(), st.sampled_from(["acme", "globex", "initech"]))


def cache() -> SemanticCache:
    settings = Settings.model_validate(
        {"cache": {"semantic": True, "similarity_threshold": 0.9, "backend": "memory"}}
    )
    return SemanticCache(settings, MemoryStore())


def answer(marker: str) -> dict[str, Any]:
    return {"answer": f"answer for {marker}", "citations": [], "usage": {}}


@given(vector=VECTOR, owner=TENANT, other=TENANT)
async def test_an_identical_vector_never_crosses_tenants(
    vector: list[float], owner: str | None, other: str | None
) -> None:
    """The strongest form: similarity 1.0 must still not cross the boundary."""
    if owner == other:
        return

    subject = cache()
    await subject.store_response("the question", vector, answer(str(owner)), tenant=owner)

    assert await subject.lookup(vector, tenant=other) is None


@given(vector=VECTOR, owner=TENANT)
async def test_a_tenant_reaches_its_own_entry(vector: list[float], owner: str | None) -> None:
    """Isolation that also blocks the owner is an outage, not isolation."""
    subject = cache()
    await subject.store_response("the question", vector, answer("mine"), tenant=owner)

    hit = await subject.lookup(vector, tenant=owner)

    assert hit is not None
    assert hit.response["answer"] == "answer for mine"


@given(vector=VECTOR)
async def test_untenanted_and_tenanted_entries_are_distinct_owners(
    vector: list[float],
) -> None:
    """`None` is an owner, not a wildcard; a shared deployment would leak into every tenant."""
    subject = cache()
    await subject.store_response("q", vector, answer("shared"), tenant=None)

    assert await subject.lookup(vector, tenant="acme") is None
    assert await subject.lookup(vector, tenant=None) is not None


@given(vector=VECTOR)
async def test_two_tenants_asking_the_same_question_keep_both_answers(
    vector: list[float],
) -> None:
    """The key carries the tenant too, or one tenant's entry evicts the other's."""
    subject = cache()
    await subject.store_response("same question", vector, answer("acme"), tenant="acme")
    await subject.store_response("same question", vector, answer("globex"), tenant="globex")

    first = await subject.lookup(vector, tenant="acme")
    second = await subject.lookup(vector, tenant="globex")

    assert first is not None
    assert second is not None
    assert first.response["answer"] == "answer for acme"
    assert second.response["answer"] == "answer for globex"


@given(vector=VECTOR, owner=TENANT)
async def test_an_orthogonal_vector_is_a_miss_for_its_own_tenant(
    vector: list[float], owner: str | None
) -> None:
    """A cache that hits on anything is worse than none; the threshold has to bind."""
    subject = cache()
    await subject.store_response("q", vector, answer("mine"), tenant=owner)

    opposite = [-value for value in vector]

    assert await subject.lookup(opposite, tenant=owner) is None


@given(
    vector=VECTOR,
    owner=st.sampled_from(["acme", "globex"]),
    others=st.lists(st.sampled_from(["initech", "umbrella"]), min_size=1, max_size=4),
)
async def test_a_crowded_cache_still_isolates(
    vector: list[float], owner: str, others: list[str]
) -> None:
    """The scan walks every entry, so isolation must hold at the end of a long list."""
    subject = cache()
    for index, tenant in enumerate(others):
        await subject.store_response(f"q{index}", vector, answer(tenant), tenant=tenant)
    await subject.store_response("mine", vector, answer(owner), tenant=owner)

    hit = await subject.lookup(vector, tenant=owner)

    assert hit is not None
    assert hit.response["answer"] == f"answer for {owner}"


@pytest.mark.parametrize("tenant", ["acme", None])
async def test_a_malformed_entry_is_skipped_rather_than_served(tenant: str | None) -> None:
    """A corrupt entry must not become a hit for whoever asks next."""
    subject = cache()
    await subject.store.set("sem:acme:junk", b"not an entry", ttl=60)

    assert await subject.lookup([1.0] + [0.0] * (DIMENSIONS - 1), tenant=tenant) is None
