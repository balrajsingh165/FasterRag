"""Identity invariants, generated rather than hand-picked.

Chunk and document ids drive deduplication (D3) and lockfile drift detection (D1). A
collision there is silent data loss — two distinct documents become one, and the second is
dropped as a duplicate — so the injectivity these ids depend on is asserted over generated
inputs instead of the one hand-picked pair the example suite carries.
"""

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from fasterrag.core.identity import (
    chunk_id,
    content_hash,
    document_id,
    normalise_source,
    text_hash,
)
from fasterrag.errors import IngestionError

# Deliberately unrestricted: a source URI is whatever a caller passed to `ingest`, and the
# separator these ids are built on is a control character, so a strategy that generated only
# printable text could never reach the collision it is meant to rule out.
SEPARATOR = chr(0)

# Unrestricted apart from the one byte a source may not contain — which is itself an
# invariant, tested separately below. Restricting further would put the collision this
# suite exists to rule out beyond reach of every generator.
SOURCE = st.text(min_size=1, max_size=60).filter(lambda value: SEPARATOR not in value)
TENANT = st.one_of(st.none(), st.text(min_size=1, max_size=20))
FORGED = st.text(min_size=1, max_size=40).map(lambda value: f"{value}{SEPARATOR}{value}")


@given(source=SOURCE, tenant=TENANT)
def test_a_document_id_is_deterministic(source: str, tenant: str | None) -> None:
    assert document_id(source, tenant) == document_id(source, tenant)


@given(source=SOURCE, tenant=TENANT)
def test_a_document_id_has_the_documented_shape(source: str, tenant: str | None) -> None:
    """The prefix is what makes an id self-describing in a log line or a payload."""
    identifier = document_id(source, tenant)

    assert identifier.startswith("d_")
    assert len(identifier) == len("d_") + 16


@given(first=SOURCE, second=SOURCE, tenant=TENANT)
def test_distinct_sources_get_distinct_documents(first: str, second: str, tenant: str) -> None:
    """Two documents sharing an id means the second is dropped as a duplicate."""
    assume(normalise_source(first) != normalise_source(second))

    assert document_id(first, tenant) != document_id(second, tenant)


@given(
    source=SOURCE, first=st.text(min_size=1, max_size=20), second=st.text(min_size=1, max_size=20)
)
def test_one_source_under_two_tenants_stays_two_documents(
    source: str, first: str, second: str
) -> None:
    """A cross-tenant id collision leaks one tenant's document into another's corpus."""
    assume(first != second)

    assert document_id(source, first) != document_id(source, second)


@given(source=FORGED, tenant=st.text(min_size=1, max_size=20))
def test_a_source_carrying_the_separator_is_refused(source: str, tenant: str) -> None:
    """The classic ambiguity: joining fields must not let one field impersonate two.

    The digest separates its parts with a NUL, which separates only while no part contains
    one. `document_id(a + NUL + b, c)` reached the same digest as
    `document_id(a, b + NUL + c)` — a real collision, found by generating unrestricted text
    rather than the printable range. Tenant validation made it unreachable in practice, but
    an invariant enforced only by a validator two modules away is one refactor from being
    false, so the source is refused at the canonicalisation point instead (TASK-0209).
    """
    with pytest.raises(IngestionError):
        document_id(source, tenant)


@given(source=SOURCE)
def test_normalising_is_idempotent(source: str) -> None:
    """A second pass must not move the answer, or the id depends on how often it was called."""
    once = normalise_source(source)

    assert normalise_source(once) == once


@given(
    document=st.text(min_size=1, max_size=30),
    index=st.integers(0, 10_000),
    chunker=st.text(min_size=1, max_size=30),
)
def test_a_chunk_id_is_a_pure_function(document: str, index: int, chunker: str) -> None:
    assert chunk_id(document, index, chunker) == chunk_id(document, index, chunker)


@given(
    document=st.text(min_size=1, max_size=30),
    first=st.integers(0, 5_000),
    second=st.integers(0, 5_000),
    chunker=st.text(min_size=1, max_size=30),
)
def test_two_positions_in_one_document_are_two_chunks(
    document: str, first: int, second: int, chunker: str
) -> None:
    """Colliding positions would drop a chunk on upsert and leave a gap nothing reports."""
    assume(first != second)

    assert chunk_id(document, first, chunker) != chunk_id(document, second, chunker)


@given(
    document=st.text(min_size=1, max_size=30),
    index=st.integers(0, 5_000),
    first=st.text(min_size=1, max_size=30),
    second=st.text(min_size=1, max_size=30),
)
def test_rechunking_differently_produces_different_ids(
    document: str, index: int, first: str, second: str
) -> None:
    """The chunker hash is what makes a re-chunk a new build rather than an overwrite."""
    assume(first != second)

    assert chunk_id(document, index, first) != chunk_id(document, index, second)


@given(data=st.binary(max_size=200))
def test_a_content_hash_is_deterministic(data: bytes) -> None:
    assert content_hash(data) == content_hash(data)


@given(first=st.binary(max_size=200), second=st.binary(max_size=200))
def test_distinct_bytes_get_distinct_hashes(first: bytes, second: bytes) -> None:
    """Dedup compares these; a collision silently discards the second document."""
    assume(first != second)

    assert content_hash(first) != content_hash(second)


@given(text=st.text(max_size=200))
def test_a_text_hash_is_deterministic(text: str) -> None:
    assert text_hash(text) == text_hash(text)
