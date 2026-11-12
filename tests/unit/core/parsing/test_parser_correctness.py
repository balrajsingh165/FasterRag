"""One regression test per parser defect named in TASK-0459.

Each test was written before its fix and failed on ``main`` at ``b97ce52`` for the reason
its docstring gives.
"""

import io
import struct
import zipfile

import pytest
from docx import Document
from docx.document import Document as DocxDocument

from fasterrag.core.parsing import parse_docx, parse_html, parse_markdown, parse_plaintext
from fasterrag.core.parsing.docx import MAX_UNZIPPED_BYTES
from fasterrag.core.parsing.models import ParsedDocument
from fasterrag.errors import ErrorCode, IngestionError, ParseError
from tests.unit.core.parsing.conftest import docx_bytes


def blocks(document: ParsedDocument) -> list[tuple[str, str]]:
    return [(block.kind, block.text) for block in document.blocks]


def texts(document: ParsedDocument) -> list[str]:
    return [block.text for block in document.blocks]


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


_FOOTNOTES_XML = (
    '<w:footnotes xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
    '<w:footnote w:type="separator" w:id="-1"><w:p><w:r><w:separator/></w:r></w:p></w:footnote>'
    '<w:footnote w:type="continuationSeparator" w:id="0"><w:p><w:r>'
    "<w:continuationSeparator/></w:r></w:p></w:footnote>"
    '<w:footnote w:id="1"><w:p><w:r><w:t>See the 2024 audit, </w:t></w:r>'
    "<w:r><w:t>page 12.</w:t></w:r></w:p></w:footnote>"
    "</w:footnotes>"
)
_FOOTNOTES_RELATIONSHIP = (
    '<Relationship Id="rIdNotes" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
    'relationships/footnotes" Target="footnotes.xml"/></Relationships>'
)
_FOOTNOTES_CONTENT_TYPE = (
    '<Override PartName="/word/footnotes.xml" ContentType="application/vnd.openxmlformats-'
    'officedocument.wordprocessingml.footnotes+xml"/></Types>'
)


def _saved(document: DocxDocument) -> bytes:
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def _with_footnotes(data: bytes) -> bytes:
    """Add a footnotes part, which python-docx can read around but cannot write."""
    source = zipfile.ZipFile(io.BytesIO(data))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as target:
        for member in source.infolist():
            content = source.read(member.filename)
            if member.filename == "word/_rels/document.xml.rels":
                content = content.replace(b"</Relationships>", _FOOTNOTES_RELATIONSHIP.encode())
            elif member.filename == "[Content_Types].xml":
                content = content.replace(b"</Types>", _FOOTNOTES_CONTENT_TYPE.encode())
            target.writestr(member, content)
        target.writestr("word/footnotes.xml", _FOOTNOTES_XML)
    return buffer.getvalue()


def test_docx_headers_and_footers_are_read_once() -> None:
    """Only the body was walked, so a header naming the document's owner was never indexed."""
    document = Document()
    document.sections[0].header.paragraphs[0].text = "ACME CONFIDENTIAL"
    document.sections[0].footer.paragraphs[0].text = "Contract 7781 footer"
    document.add_paragraph("Body paragraph.")
    document.add_section()
    document.add_paragraph("Second section body.")

    parsed = parse_docx(_saved(document))

    assert texts(parsed) == [
        "ACME CONFIDENTIAL",
        "Contract 7781 footer",
        "Body paragraph.",
        "Second section body.",
    ]
    assert parsed.blocks[0].section is None


def test_docx_footnotes_are_read_after_the_body() -> None:
    """Footnotes live in their own part; a citation's source often sits only there."""
    document = Document()
    document.add_paragraph("A claim that needs a source.")

    parsed = parse_docx(_with_footnotes(_saved(document)))

    assert texts(parsed) == ["A claim that needs a source.", "See the 2024 audit, page 12."]


def test_a_table_nested_in_a_cell_keeps_its_text() -> None:
    """``cell.text`` reads a cell's paragraphs only, so a nested table vanished entirely."""
    document = Document()
    outer = document.add_table(rows=1, cols=2)
    outer.cell(0, 0).text = "Fees"
    inner = outer.cell(0, 1).add_table(rows=2, cols=2)
    for row, (term, amount) in enumerate([("Setup", "100"), ("Monthly", "20")]):
        inner.cell(row, 0).text = term
        inner.cell(row, 1).text = amount

    parsed = parse_docx(_saved(document))

    assert texts(parsed) == ["Fees | Setup / 100; Monthly / 20"]


def test_a_docx_zip_bomb_is_refused_before_it_is_inflated() -> None:
    """``Document()`` inflated every member it was handed; 300 KB here unpacks to 300 MiB."""
    data = io.BytesIO(docx_bytes())
    with (
        zipfile.ZipFile(data, "a", zipfile.ZIP_DEFLATED) as archive,
        archive.open("word/media/padding.bin", "w", force_zip64=True) as member,
    ):
        for _ in range(300):
            member.write(bytes(1 << 20))

    with pytest.raises(IngestionError) as caught:
        parse_docx(data.getvalue())

    assert caught.value.code is ErrorCode.PAYLOAD_TOO_LARGE
    assert caught.value.retryable is False
    assert len(data.getvalue()) < MAX_UNZIPPED_BYTES // 100


def test_a_docx_member_that_lies_about_its_size_is_a_parse_error() -> None:
    """The declared size is what the cap trusts, so a member larger than declared must fail."""
    archive = bytearray(docx_bytes())
    header = archive.find(b"PK\x03\x04")
    central = archive.find(b"PK\x01\x02")
    struct.pack_into("<I", archive, header + 22, 1)
    struct.pack_into("<I", archive, central + 24, 1)

    with pytest.raises(ParseError):
        parse_docx(bytes(archive))


def test_composed_and_decomposed_accents_index_as_the_same_text() -> None:
    """Only ``.strip()`` was applied, so a decomposed accent never matched a composed one."""
    decomposed = parse_plaintext("cafe\u0301 au lait".encode())
    composed = parse_plaintext("caf\u00e9 au lait".encode())

    assert decomposed.text == composed.text == "caf\u00e9 au lait"


def test_nul_and_other_control_codes_are_removed_from_block_text() -> None:
    """A NUL reached the index, where PostgreSQL refuses it in a text column outright."""
    document = parse_plaintext(b"before\x00after\x07 and\x1b[0m done")

    assert document.text == "beforeafter and[0m done"


def test_carriage_returns_inside_a_block_become_line_feeds() -> None:
    """A CRLF file and its LF twin indexed different text for the same paragraph."""
    crlf = parse_markdown(b"line one\r\nline two\r\n")
    lf = parse_markdown(b"line one\nline two\n")

    assert crlf.text == lf.text == "line one\nline two"
