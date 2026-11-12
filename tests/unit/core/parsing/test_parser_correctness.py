"""One regression test per parser defect named in TASK-0459.

Each test was written before its fix and failed on ``main`` at ``b97ce52`` for the reason
its docstring gives.
"""

from fasterrag.core.parsing import parse_html
from fasterrag.core.parsing.models import ParsedDocument


def blocks(document: ParsedDocument) -> list[tuple[str, str]]:
    return [(block.kind, block.text) for block in document.blocks]


def test_a_paragraph_inside_a_list_item_is_emitted_once() -> None:
    """``find_all`` returned the ``li`` and the ``p`` inside it, so the text was indexed twice."""
    document = parse_html(b"<ul><li><p>only once</p></li></ul>")

    assert blocks(document) == [("list_item", "only once")]


def test_text_directly_inside_a_div_or_span_is_kept() -> None:
    """Only listed content tags were read, so text a page puts straight in a ``div`` vanished."""
    document = parse_html(
        b"<body><div>hello <b>world</b>!</div><span>alone</span><p>para</p></body>"
    )

    assert blocks(document) == [
        ("paragraph", "hello world!"),
        ("paragraph", "alone"),
        ("paragraph", "para"),
    ]


def test_loose_text_in_sibling_containers_stays_in_separate_blocks() -> None:
    """Two adjacent ``div``s are two blocks of a page, not one run-on paragraph."""
    document = parse_html(b"<div>first</div><div>second<br>line</div>")

    assert blocks(document) == [("paragraph", "first"), ("paragraph", "second line")]


def test_comments_and_the_head_are_not_indexed_as_loose_text() -> None:
    """Reading loose text must not start reading what a browser never shows."""
    document = parse_html(
        b"<html><head><title>T</title></head><body><!-- hidden -->shown</body></html>"
    )

    assert blocks(document) == [("paragraph", "shown")]
    assert document.metadata["title"] == "T"


def test_ten_thousand_nested_elements_parse_without_recursion() -> None:
    """A recursive walk over the tree would raise ``RecursionError`` long before this depth."""
    depth = 10_000
    data = b"<div>" * depth + b"deep text" + b"</div>" * depth

    assert blocks(parse_html(data)) == [("paragraph", "deep text")]
