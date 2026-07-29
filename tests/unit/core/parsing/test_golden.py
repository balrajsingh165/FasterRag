"""Golden-file parser tests.

The committed snapshots pin reading order, table structure, heading paths, and metadata.
A parser change that alters any of them shows up as a reviewed diff rather than as a
silent retrieval regression (``docs/testing-strategy.md`` §1.3).
"""

from fasterrag.core.parsing import (
    parse_csv,
    parse_docx,
    parse_html,
    parse_json,
    parse_markdown,
    parse_pdf,
)
from tests.unit.core.parsing.conftest import (
    CSV,
    HTML,
    JSON_SOURCE,
    MARKDOWN,
    assert_matches_golden,
    docx_bytes,
    pdf_bytes,
)


def test_markdown_golden() -> None:
    assert_matches_golden("markdown", parse_markdown(MARKDOWN.encode()))


def test_html_golden() -> None:
    assert_matches_golden("html", parse_html(HTML.encode()))


def test_csv_golden() -> None:
    assert_matches_golden("csv", parse_csv(CSV.encode()))


def test_json_golden() -> None:
    assert_matches_golden("json", parse_json(JSON_SOURCE.encode()))


def test_docx_golden() -> None:
    assert_matches_golden("docx", parse_docx(docx_bytes()))


def test_pdf_golden() -> None:
    assert_matches_golden("pdf", parse_pdf(pdf_bytes(pages=2)))
