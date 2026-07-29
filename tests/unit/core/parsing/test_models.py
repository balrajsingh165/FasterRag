from itertools import pairwise

from hypothesis import given
from hypothesis import strategies as st

from fasterrag.core.parsing.models import DocumentBuilder, ParseFlag


def build(*blocks: tuple[str, str]) -> DocumentBuilder:
    builder = DocumentBuilder(mime_type="text/plain", parser="test")
    for kind, text in blocks:
        if kind == "heading":
            builder.heading(text, 1)
        else:
            builder.add("paragraph", text)
    return builder


def test_offsets_index_into_the_document_text() -> None:
    document = build(("paragraph", "first"), ("paragraph", "second")).build()

    for block in document.blocks:
        assert document.text[block.start : block.end] == block.text


def test_blocks_are_separated_in_the_joined_text() -> None:
    document = build(("paragraph", "first"), ("paragraph", "second")).build()

    assert document.text == "first\n\nsecond"


def test_empty_blocks_are_dropped() -> None:
    builder = DocumentBuilder(mime_type="text/plain", parser="test")
    builder.add("paragraph", "   ")
    builder.add("paragraph", "\n\n")
    builder.add("paragraph", "kept")

    document = builder.build()
    assert len(document.blocks) == 1
    assert document.text == "kept"


def test_heading_path_nests_and_truncates() -> None:
    builder = DocumentBuilder(mime_type="text/markdown", parser="test")
    builder.heading("3. Termination", 2)
    builder.heading("3.2 Notice", 3)
    builder.add("paragraph", "body under 3.2")
    builder.heading("4. Payment", 2)
    builder.add("paragraph", "body under 4")

    sections = [block.section for block in builder.build().blocks]
    assert sections[2] == "3. Termination > 3.2 Notice"
    assert sections[4] == "4. Payment"


def test_section_and_page_lookups_track_position() -> None:
    builder = DocumentBuilder(mime_type="application/pdf", parser="test")
    builder.heading("Chapter 1", 1, page=1)
    builder.add("paragraph", "first page body", page=1)
    builder.heading("Chapter 2", 1, page=2)
    builder.add("paragraph", "second page body", page=2)
    document = builder.build()

    first = document.text.index("first page body")
    second = document.text.index("second page body")

    assert document.section_at(first) == "Chapter 1"
    assert document.section_at(second) == "Chapter 2"
    assert document.page_at(first) == 1
    assert document.page_at(second) == 2


def test_block_at_finds_the_containing_block() -> None:
    document = build(("paragraph", "alpha"), ("paragraph", "beta")).build()

    first = document.block_at(0)
    second = document.block_at(document.text.index("beta"))

    assert first is not None
    assert second is not None
    assert first.text == "alpha"
    assert second.text == "beta"


def test_block_at_returns_nothing_inside_a_separator() -> None:
    document = build(("paragraph", "alpha"), ("paragraph", "beta")).build()

    assert document.block_at(len("alpha")) is None


def test_flags_are_recorded_once() -> None:
    builder = DocumentBuilder(mime_type="text/plain", parser="test")
    builder.flag(ParseFlag.TABLES_DETECTED)
    builder.flag(ParseFlag.TABLES_DETECTED)

    assert builder.build().flags == ("tables_detected",)


def test_empty_metadata_values_are_ignored() -> None:
    builder = DocumentBuilder(mime_type="text/plain", parser="test")
    builder.meta(title="Kept", author=None, subject="", tags=[])

    assert builder.build().metadata == {"title": "Kept"}


@given(st.lists(st.text(min_size=1, max_size=40), min_size=1, max_size=15))
def test_offset_invariant_holds_for_arbitrary_blocks(texts: list[str]) -> None:
    builder = DocumentBuilder(mime_type="text/plain", parser="test")
    for text in texts:
        builder.add("paragraph", text)
    document = builder.build()

    for block in document.blocks:
        assert document.text[block.start : block.end] == block.text
        assert block.start < block.end
        assert block.end <= len(document.text)


@given(st.lists(st.text(min_size=1, max_size=40), min_size=1, max_size=15))
def test_offsets_are_monotonic(texts: list[str]) -> None:
    builder = DocumentBuilder(mime_type="text/plain", parser="test")
    for text in texts:
        builder.add("paragraph", text)

    for earlier, later in pairwise(builder.build().blocks):
        assert earlier.end <= later.start
