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
    IDENTITY_VERSION,
    _digest,
    chunk_id,
    content_hash,
    document_id,
    normalise_source,
    text_hash,
)
from fasterrag.errors import IngestionError

# Deliberately unrestricted: a source URI is whatever a caller passed to `ingest`. The byte
# that once served as the digest's separator is a control character, so a strategy that
# generated only printable text could never have reached the collision version 2 had.
NUL = chr(0)

# Unrestricted apart from the one byte a source may not contain — which is itself an
# invariant, tested separately below. Restricting further would put the collision this
# suite exists to rule out beyond reach of every generator.
SOURCE = st.text(min_size=1, max_size=60).filter(lambda value: NUL not in value)
TENANT = st.one_of(st.none(), st.text(min_size=1, max_size=20))
FORGED = st.text(min_size=1, max_size=40).map(lambda value: f"{value}{NUL}{value}")


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
def test_a_source_carrying_a_nul_is_refused_as_invalid_input(source: str, tenant: str) -> None:
    """A NUL in a source URI is refused because no filesystem or URL permits one.

    Until identity version 3 this refusal was also the only thing standing between a NUL and
    a forged document id — the digest NUL-joined its fields, so `document_id(a + NUL + b, c)`
    reached the same digest as `document_id(a, b + NUL + c)`, a real collision found by
    generating unrestricted text (TASK-0209). The digest is now length-prefixed and cannot be
    forged by any byte (TASK-0210, the properties below), so this is plain input validation
    and would be correct even if the identity guarantee did not exist.
    """
    with pytest.raises(IngestionError):
        document_id(source, tenant)


# Truly unrestricted, NUL included: the point of the properties below is that the digest no
# longer needs its inputs to avoid any byte at all.
FIELD = st.text(min_size=0, max_size=40)
FIELDS = st.lists(FIELD, min_size=1, max_size=4)


@given(fields=st.lists(FIELD, min_size=2, max_size=4))
def test_the_version_two_forgery_is_impossible_by_construction(fields: list[str]) -> None:
    """Folding a NUL into a field to swallow its neighbour must change the digest.

    The exact shape that collided under version 2: every tuple whose NUL-join equals the
    original's reached the original's digest, so ``(x + NUL + y, z)`` hashed like
    ``(x, y, z)`` and so did the one-field ``(x + NUL + y + NUL + z,)``. Each such forgery
    is built here and must differ. This property was run against the version-2 digest before
    landing and failed on its first example, which is what makes it evidence.
    """
    original = _digest(*fields)
    for boundary in range(1, len(fields)):
        merged = [*fields[: boundary - 1], NUL.join(fields[boundary - 1 : boundary + 1])]
        forged = [*merged, *fields[boundary + 1 :]]
        assert _digest(*forged) != original
    assert _digest(NUL.join(fields)) != original


@given(left=FIELDS, right=FIELDS)
def test_distinct_tuples_never_share_a_digest(left: list[str], right: list[str]) -> None:
    """Different tuples give different digests, whatever bytes the fields contain.

    This is the guarantee the length prefix buys: a colon, a digit, or a NUL inside a field
    cannot masquerade as a boundary, so ``("3:abc",)`` and ``("3", "abc")`` differ as surely
    as ``("a",)`` and ``("b",)`` do.
    """
    assume(left != right)

    assert _digest(*left) != _digest(*right)


@given(fields=FIELDS)
def test_the_digest_is_deterministic(fields: list[str]) -> None:
    assert _digest(*fields) == _digest(*fields)


def test_the_identity_version_is_three() -> None:
    """Pinned because the lockfile records it.

    An index built under version 2 must report drift rather than silently mix two id
    schemes in one collection, so a change to the encoding that forgets the bump is caught
    here before it is caught by an adopter.
    """
    assert IDENTITY_VERSION == 3


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
