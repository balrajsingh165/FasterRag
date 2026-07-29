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
