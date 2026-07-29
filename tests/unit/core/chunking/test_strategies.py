from collections.abc import Sequence

import pytest

from fasterrag.chunking import RecursiveChunker as PublicRecursiveChunker
from fasterrag.config.schema import Settings
from fasterrag.core.chunking import (
    FixedChunker,
    LateChunker,
    LayoutChunker,
    RecursiveChunker,
    SemanticChunker,
    create_chunker,
)
from fasterrag.core.chunking.late import LATE_POOLING_KEY
from fasterrag.core.chunking.models import EstimatingTokenCounter
from fasterrag.core.parsing import parse_markdown, parse_plaintext
from fasterrag.errors import ConfigError

COUNTER = EstimatingTokenCounter()


class ShiftingEmbedder:
    """Reports a topic change exactly once, at the third sentence."""

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[1.0, 0.0] if index < 2 else [0.0, 1.0] for index in range(len(texts))]


def paragraphs(count: int, words: int = 12) -> bytes:
    body = "\n\n".join(" ".join(f"word{index}" for index in range(words)) for _ in range(count))
    return body.encode()


def test_fixed_windows_are_close_to_uniform() -> None:
    document = parse_plaintext(paragraphs(20))
    chunks = FixedChunker(chunk_size=20, overlap=0, counter=COUNTER).split(document)

    assert len(chunks) > 2
    lengths = [len(chunk.text) for chunk in chunks[:-1]]
    assert max(lengths) - min(lengths) <= COUNTER.chars_per_token * 20


def test_fixed_windows_do_not_split_mid_word() -> None:
    document = parse_plaintext(paragraphs(20))
    chunks = FixedChunker(chunk_size=20, overlap=0, counter=COUNTER).split(document)

    for chunk in chunks[:-1]:
        assert chunk.text.endswith((" ", "\n"))


def test_recursive_prefers_paragraph_boundaries() -> None:
    document = parse_plaintext(paragraphs(6, words=8))
    chunks = RecursiveChunker(chunk_size=30, overlap=0, counter=COUNTER).split(document)

    assert len(chunks) > 1
    for chunk in chunks[:-1]:
        assert chunk.text.rstrip(" ").endswith("\n") or chunk.text.endswith(" ")


def test_layout_starts_a_chunk_at_each_heading() -> None:
    source = (
        "# Title\n\nintro paragraph here\n\n## Section A\n\n"
        + ("alpha " * 20)
        + "\n\n## Section B\n\n"
        + ("beta " * 20)
    )
    chunks = LayoutChunker(chunk_size=40, overlap=0, counter=COUNTER).split(
        parse_markdown(source.encode())
    )

    starts = [chunk.text.strip().splitlines()[0] for chunk in chunks]
    assert any(line.startswith("Section A") for line in starts)
    assert any(line.startswith("Section B") for line in starts)


def test_layout_keeps_a_small_table_whole() -> None:
    source = "## Data\n\n| Term | Days |\n| --- | --- |\n| Notice | 30 |\n| Cure | 15 |\n"
    chunks = LayoutChunker(chunk_size=200, overlap=0, counter=COUNTER).split(
        parse_markdown(source.encode())
    )

    table_chunks = [chunk for chunk in chunks if "Notice | 30" in chunk.text]
    assert len(table_chunks) == 1
    assert "Cure | 15" in table_chunks[0].text


def test_layout_falls_back_when_a_document_has_no_blocks() -> None:
    document = parse_plaintext(paragraphs(10))
    chunker = LayoutChunker(chunk_size=20, overlap=0, counter=COUNTER)

    assert chunker.split(document)


def test_semantic_breaks_where_the_topic_shifts() -> None:
    document = parse_plaintext(b"First one. Second one. Third one. Fourth one.")
    chunker = SemanticChunker(ShiftingEmbedder(), chunk_size=200, overlap=0, counter=COUNTER)

    chunks = chunker.split(document)

    assert len(chunks) == 2
    assert "Third one." in chunks[1].text


def test_semantic_falls_back_for_very_short_documents() -> None:
    chunker = SemanticChunker(ShiftingEmbedder(), chunk_size=200, overlap=0, counter=COUNTER)

    chunks = chunker.split(parse_plaintext(b"Only one sentence here."))

    assert len(chunks) == 1


def test_late_marks_chunks_for_pooled_embedding() -> None:
    chunks = LateChunker(chunk_size=20, overlap=0, counter=COUNTER).split(
        parse_plaintext(paragraphs(6))
    )

    assert chunks
    assert all(chunk.metadata[LATE_POOLING_KEY] is True for chunk in chunks)
    assert all(chunk.strategy == "late" for chunk in chunks)


def test_late_boundaries_match_the_recursive_baseline() -> None:
    document = parse_plaintext(paragraphs(6))
    late = LateChunker(chunk_size=20, overlap=4, counter=COUNTER).split(document)
    recursive = RecursiveChunker(chunk_size=20, overlap=4, counter=COUNTER).split(document)

    assert [(chunk.start, chunk.end) for chunk in late] == [
        (chunk.start, chunk.end) for chunk in recursive
    ]


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("fixed", FixedChunker),
        ("recursive", RecursiveChunker),
        ("layout", LayoutChunker),
        ("late", LateChunker),
    ],
)
def test_the_factory_builds_each_strategy(strategy: str, expected: type) -> None:
    settings = Settings.model_validate({"chunking": {"strategy": strategy}})

    assert isinstance(create_chunker(settings), expected)


def test_the_factory_passes_the_configured_size_through() -> None:
    settings = Settings.model_validate({"chunking": {"chunk_size": 128, "overlap": 8}})
    chunker = create_chunker(settings)

    chunks = chunker.split(parse_plaintext(paragraphs(60)))
    assert all(chunk.token_count <= 128 + 8 for chunk in chunks)


def test_semantic_requires_an_embedder() -> None:
    settings = Settings.model_validate({"chunking": {"strategy": "semantic"}})

    with pytest.raises(ConfigError, match="needs an embedding model"):
        create_chunker(settings)


def test_semantic_is_built_when_an_embedder_is_supplied() -> None:
    settings = Settings.model_validate({"chunking": {"strategy": "semantic"}})

    chunker = create_chunker(settings, embedder=ShiftingEmbedder())

    assert isinstance(chunker, SemanticChunker)


def test_the_public_surface_exports_the_documented_names() -> None:
    assert PublicRecursiveChunker is RecursiveChunker
