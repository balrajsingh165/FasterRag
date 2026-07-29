import pytest

from fasterrag.core.parsing import (
    parse_csv,
    parse_docx,
    parse_html,
    parse_json,
    parse_markdown,
    parse_pdf,
    parse_plaintext,
)
from fasterrag.core.parsing.models import ParsedDocument, ParseFlag
from fasterrag.errors import ParseError
from tests.unit.core.parsing.conftest import (
    CSV,
    HTML,
    JSON_SOURCE,
    MARKDOWN,
    blank_pdf_bytes,
    docx_bytes,
    pdf_bytes,
)


def kinds(document: ParsedDocument) -> list[str]:
    return [block.kind for block in document.blocks]


def texts(document: ParsedDocument) -> list[str]:
    return [block.text for block in document.blocks]


def test_plaintext_splits_on_blank_lines() -> None:
    document = parse_plaintext(b"First paragraph.\n\nSecond paragraph.\n")

    assert texts(document) == ["First paragraph.", "Second paragraph."]


def test_plaintext_flags_an_encoding_fallback() -> None:
    document = parse_plaintext(b"caf\xe9 latte")

    assert ParseFlag.ENCODING_FALLBACK.value in document.flags
    assert document.text


def test_markdown_classifies_every_structure() -> None:
    document = parse_markdown(MARKDOWN.encode())

    assert kinds(document).count("heading") == 3
    assert "list_item" in kinds(document)
    assert "code" in kinds(document)
    assert "table" in kinds(document)
    assert ParseFlag.TABLES_DETECTED.value in document.flags


def test_markdown_builds_a_nested_heading_path() -> None:
    document = parse_markdown(MARKDOWN.encode())
    body = next(
        block for block in document.blocks if block.text.startswith("Either party may terminate")
    )

    assert body.section == "Vendor Agreement > 3. Termination > 3.2 Notice"


def test_markdown_table_keeps_rows_separate() -> None:
    document = parse_markdown(MARKDOWN.encode())
    table = next(block for block in document.blocks if block.kind == "table")

    assert table.text.splitlines() == ["Term | Days", "Notice | 30", "Cure | 15"]


def test_markdown_records_the_title() -> None:
    assert parse_markdown(MARKDOWN.encode()).metadata["title"] == "Vendor Agreement"


def test_html_drops_non_content_elements() -> None:
    document = parse_html(HTML.encode())

    assert "skip this navigation" not in document.text
    assert "skip this footer" not in document.text
    assert "console.log" not in document.text


def test_html_preserves_reading_order_and_structure() -> None:
    document = parse_html(HTML.encode())

    assert kinds(document)[:4] == ["heading", "paragraph", "heading", "paragraph"]
    assert "table" in kinds(document)
    assert kinds(document).count("list_item") == 2


def test_html_reads_document_metadata() -> None:
    document = parse_html(HTML.encode())

    assert document.metadata["title"] == "Vendor Agreement"
    assert document.metadata["author"] == "Legal Team"


def test_csv_rows_describe_their_columns() -> None:
    document = parse_csv(CSV.encode())

    assert "name: Alice; department: legal; year: 2024" in document.text
    assert document.metadata["columns"] == ["name", "department", "year"]
    assert document.metadata["row_count"] == 2


def test_csv_detects_tab_separated_files() -> None:
    document = parse_csv(b"name\tyear\nAlice\t2024\n")

    assert "name: Alice; year: 2024" in document.text


def test_json_keeps_the_key_path_with_each_value() -> None:
    document = parse_json(JSON_SOURCE.encode())

    assert "notice_days: 30" in document.text
    assert "agreement.parties[0]: Acme" in document.text


def test_json_uses_keys_as_the_heading_path() -> None:
    document = parse_json(JSON_SOURCE.encode())
    notice = next(block for block in document.blocks if "notice_days" in block.text)

    assert notice.section is not None
    assert "termination" in notice.section


def test_invalid_json_is_a_parse_error() -> None:
    with pytest.raises(ParseError, match="invalid JSON"):
        parse_json(b"{not json")


def test_docx_interleaves_tables_with_paragraphs_in_reading_order() -> None:
    document = parse_docx(docx_bytes())
    order = kinds(document)

    assert order.index("table") < order.index("paragraph", order.index("table"))
    assert texts(document)[-1] == "Text that follows the table."


def test_docx_reads_outline_levels_and_metadata() -> None:
    document = parse_docx(docx_bytes())
    body = next(
        block for block in document.blocks if block.text.startswith("Either party may terminate")
    )

    assert body.section == "Vendor Agreement > 3. Termination"
    assert document.metadata["title"] == "Vendor Agreement"
    assert document.metadata["author"] == "Legal Team"


def test_docx_serializes_table_rows() -> None:
    document = parse_docx(docx_bytes())
    table = next(block for block in document.blocks if block.kind == "table")

    assert table.text.splitlines() == ["Term | Days", "Notice | 30"]
    assert ParseFlag.TABLES_DETECTED.value in document.flags


def test_pdf_infers_headings_from_type_size() -> None:
    document = parse_pdf(pdf_bytes())
    headings = [block.text for block in document.blocks if block.kind == "heading"]

    assert "Vendor Agreement" in headings
    assert "3. Termination" in headings


def test_pdf_records_pages_and_metadata() -> None:
    document = parse_pdf(pdf_bytes(pages=2))

    assert document.metadata["page_count"] == 2
    assert document.metadata["title"] == "Vendor Agreement"
    assert {block.page for block in document.blocks} == {1, 2}


def test_pdf_body_text_carries_the_heading_path() -> None:
    document = parse_pdf(pdf_bytes())
    body = next(
        block for block in document.blocks if block.text.startswith("Either party may terminate")
    )

    assert body.section == "Vendor Agreement > 3. Termination"


def test_a_pdf_with_no_extractable_text_fails_so_it_can_be_dead_lettered() -> None:
    with pytest.raises(ParseError, match="no extractable text"):
        parse_pdf(blank_pdf_bytes())


def test_a_corrupt_pdf_becomes_a_typed_parse_error() -> None:
    with pytest.raises(ParseError, match="could not be read"):
        parse_pdf(b"%PDF-1.7 this is not really a pdf")
