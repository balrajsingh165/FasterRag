"""The five chunker invariants, asserted for every strategy.

Property-based, per ``docs/testing-strategy.md`` §1.2. A chunker that violates any of
these silently ruins retrieval quality, and the damage only shows up much later as poor
answers, so the invariants are checked against generated documents rather than against
hand-picked examples.
"""

from collections.abc import Sequence
from itertools import pairwise

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fasterrag.core.chunking import (
    Chunker,
    FixedChunker,
    LateChunker,
    LayoutChunker,
    RecursiveChunker,
    SemanticChunker,
)
from fasterrag.core.chunking.models import EstimatingTokenCounter, TextChunk
from fasterrag.core.parsing import parse_markdown, parse_plaintext

CHUNK_SIZE = 24
OVERLAP = 4
COUNTER = EstimatingTokenCounter()
CHAR_LIMIT = CHUNK_SIZE * COUNTER.chars_per_token
OVERLAP_CHARS = OVERLAP * COUNTER.chars_per_token


class RotatingEmbedder:
    """Returns vectors that shift every few sentences, forcing semantic boundaries."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [
            [1.0, 0.0, 0.0] if index % 3 == 0 else [0.0, 1.0, 0.0] for index in range(len(texts))
        ]


def chunkers() -> list[Chunker]:
    return [
        FixedChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP, counter=COUNTER),
        RecursiveChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP, counter=COUNTER),
        LayoutChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP, counter=COUNTER),
        LateChunker(chunk_size=CHUNK_SIZE, overlap=OVERLAP, counter=COUNTER),
        SemanticChunker(
            RotatingEmbedder(), chunk_size=CHUNK_SIZE, overlap=OVERLAP, counter=COUNTER
        ),
    ]


def ids() -> list[str]:
    return [chunker.strategy for chunker in chunkers()]


ALL_CHUNKERS = pytest.mark.parametrize("chunker", chunkers(), ids=ids())

TEXT = st.text(
    alphabet=st.characters(min_codepoint=32, max_codepoint=126),
    min_size=1,
    max_size=600,
)


def reconstruct(chunks: list[TextChunk]) -> str:
    """Rebuild the source text from chunks by dropping each chunk's overlap."""
    pieces: list[str] = []
    for index, chunk in enumerate(chunks):
        repeated = 0 if index == 0 else chunks[index - 1].end - chunk.start
        pieces.append(chunk.text[repeated:])
    return "".join(pieces)


@ALL_CHUNKERS
@given(body=TEXT)
@settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_invariant_1_chunks_reconstruct_the_source(chunker: Chunker, body: str) -> None:
    document = parse_plaintext(body.encode())
    chunks = chunker.split(document)

    if not chunks:
        assert not document.text.strip()
        return

    assert reconstruct(chunks) == document.text


@ALL_CHUNKERS
@given(body=TEXT)
@settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_invariant_2_no_chunk_is_empty(chunker: Chunker, body: str) -> None:
    for chunk in chunker.split(parse_plaintext(body.encode())):
        assert chunk.text.strip()


@ALL_CHUNKERS
@given(body=TEXT)
@settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_invariant_3_offsets_are_monotonic_and_in_bounds(chunker: Chunker, body: str) -> None:
    document = parse_plaintext(body.encode())
    chunks = chunker.split(document)

    for chunk in chunks:
        assert 0 <= chunk.start < chunk.end <= len(document.text)
        assert document.text[chunk.start : chunk.end] == chunk.text

    for earlier, later in pairwise(chunks):
        assert earlier.start < later.start
        assert earlier.end <= later.end


@ALL_CHUNKERS
@given(body=TEXT)
@settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_invariant_4_overlap_is_respected(chunker: Chunker, body: str) -> None:
    document = parse_plaintext(body.encode())

    for earlier, later in pairwise(chunker.split(document)):
        repeated = earlier.end - later.start
        assert 0 <= repeated <= OVERLAP_CHARS


@ALL_CHUNKERS
@given(body=TEXT)
@settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_invariant_5_chunks_stay_within_the_configured_size(chunker: Chunker, body: str) -> None:
    for chunk in chunker.split(parse_plaintext(body.encode())):
        assert chunk.token_count <= CHUNK_SIZE + OVERLAP


@ALL_CHUNKERS
def test_chunk_indexes_are_sequential_from_zero(chunker: Chunker) -> None:
    document = parse_plaintext((("sentence one. " * 40).strip() + ".").encode())
    chunks = chunker.split(document)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.strategy == chunker.strategy for chunk in chunks)


@ALL_CHUNKERS
def test_an_empty_document_yields_no_chunks(chunker: Chunker) -> None:
    assert chunker.split(parse_plaintext(b"   \n\n  ")) == []


@ALL_CHUNKERS
def test_structured_documents_keep_their_section_and_page(chunker: Chunker) -> None:
    source = "# Title\n\n## Section A\n\n" + ("body sentence. " * 30)
    chunks = chunker.split(parse_markdown(source.encode()))

    assert chunks
    assert any(chunk.section for chunk in chunks)


DENSE_BUDGET = 40
DENSE_OVERLAP = 6


# The invariants above generate ASCII only. The token-budget pass added under TASK-0114
# exists precisely for text where characters and tokens diverge — CJK, accented scripts,
# emoji — so it had never been property-tested against the inputs it was written for.
# CRITICAL: no whitespace in the alphabet, and a minimum length above the character limit
# these chunkers derive (40 tokens x 4 chars). A strategy that emitted spaces would let the
# separator splitting break every run into small pieces, so the budget pass would never
# trigger and every assertion below would hold whether or not it existed. The first version
# of this strategy did exactly that: disabling the budget pass failed none of these tests.
DENSE_TEXT = st.text(
    alphabet=st.characters(
        min_codepoint=0x4E00,
        max_codepoint=0x9FFF,
        blacklist_categories=("Cs", "Cc", "Zs"),
    ),
    min_size=DENSE_BUDGET * 4 + 1,
    max_size=400,
)


class DenseCounter:
    """Every non-space character is a token, as CJK very nearly is.

    The estimating counter can never trigger the budget pass, because the character limit
    is derived from the same ratio it counts with. This one forces it on every document.
    """

    def count(self, text: str) -> int:
        return len(text.strip())

    @property
    def chars_per_token(self) -> int:
        return 4


def dense_chunkers() -> list[Chunker]:
    counter = DenseCounter()
    return [
        FixedChunker(chunk_size=DENSE_BUDGET, overlap=DENSE_OVERLAP, counter=counter),
        RecursiveChunker(chunk_size=DENSE_BUDGET, overlap=DENSE_OVERLAP, counter=counter),
        LayoutChunker(chunk_size=DENSE_BUDGET, overlap=DENSE_OVERLAP, counter=counter),
        LateChunker(chunk_size=DENSE_BUDGET, overlap=DENSE_OVERLAP, counter=counter),
    ]


DENSE_CHUNKERS = pytest.mark.parametrize(
    "chunker", dense_chunkers(), ids=[c.strategy for c in dense_chunkers()]
)


@DENSE_CHUNKERS
@given(body=DENSE_TEXT)
@settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_dense_text_still_reconstructs_the_source(chunker: Chunker, body: str) -> None:
    """The budget pass re-splits segments; a tiling that stops being gapless loses text."""
    document = parse_plaintext(body.encode())
    chunks = chunker.split(document)

    if not chunks:
        assert not document.text.strip()
        return

    assert reconstruct(chunks) == document.text


@DENSE_CHUNKERS
@given(body=DENSE_TEXT)
@settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_dense_text_keeps_offsets_exact(chunker: Chunker, body: str) -> None:
    """Re-splitting must not move an offset off the text it names."""
    document = parse_plaintext(body.encode())

    for chunk in chunker.split(document):
        assert 0 <= chunk.start < chunk.end <= len(document.text)
        assert document.text[chunk.start : chunk.end] == chunk.text


@DENSE_CHUNKERS
@given(body=DENSE_TEXT)
@settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_dense_text_respects_the_token_budget(chunker: Chunker, body: str) -> None:
    """The whole point of TASK-0114: a "40-token" chunk must not really be 200."""
    counter = DenseCounter()

    for chunk in chunker.split(parse_plaintext(body.encode())):
        assert counter.count(chunk.text) <= DENSE_BUDGET + DENSE_OVERLAP


@DENSE_CHUNKERS
@given(body=DENSE_TEXT)
@settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_dense_text_bounds_the_overlap_in_tokens(chunker: Chunker, body: str) -> None:
    """A character reach-back runs several times the configured tokens on dense text."""
    document = parse_plaintext(body.encode())
    counter = DenseCounter()

    for earlier, later in pairwise(chunker.split(document)):
        repeated = document.text[later.start : earlier.end]
        assert counter.count(repeated) <= DENSE_OVERLAP


@DENSE_CHUNKERS
@given(body=DENSE_TEXT)
@settings(max_examples=60, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_dense_text_terminates_without_degenerate_chunks(chunker: Chunker, body: str) -> None:
    """The budget pass halves toward a floor; without one it would split to one character."""
    chunks = chunker.split(parse_plaintext(body.encode()))

    assert all(chunk.text.strip() for chunk in chunks)
    assert len(chunks) <= max(len(body), 1)
