"""Inputs the pipeline had never been shown (TASK-0472) and hostile ones (TASK-0489).

Each test was written before its fix and failed on ``main`` at ``b97ce52`` for the reason
its docstring gives.
"""

import codecs
import importlib.util
import io
import shutil
import sys
from types import ModuleType
from typing import cast

import pytest
from PIL import Image
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas

from fasterrag.core.parsing import (
    DEFAULT_PARSING_OPTIONS,
    parse_csv,
    parse_markdown,
    parse_pdf,
    parse_plaintext,
)
from fasterrag.core.parsing.models import ParseFlag
from fasterrag.errors import ErrorCode, ParseError
from tests.unit.core.parsing.conftest import encrypted_pdf_bytes, scanned_pdf_bytes

LATIN_1_TEXT = "Caf\xe9 cr\xe8me br\xfbl\xe9e, \xa3 12 \xb7 na\xefve fa\xe7ade"


def test_latin_1_text_decodes_to_its_characters() -> None:
    """Latin-1 bytes became U+FFFD and the replacement characters were indexed."""
    document = parse_plaintext(LATIN_1_TEXT.encode("latin-1"))

    assert document.text == LATIN_1_TEXT
    assert ParseFlag.ENCODING_FALLBACK.value in document.flags


def test_windows_1252_punctuation_decodes_to_its_characters() -> None:
    """Curly quotes and the euro sign are the bytes a Windows editor writes most often."""
    text = "\u201cquoted\u201d \u2014 \u20ac5 \u2019tis"

    assert parse_plaintext(text.encode("cp1252")).text == text


@pytest.mark.parametrize("codec", ["utf-16", "utf-16-le", "utf-16-be", "utf-32"])
def test_text_with_a_utf_16_or_utf_32_byte_order_mark_decodes(codec: str) -> None:
    """A UTF-16 file (what PowerShell's ``>`` writes) decoded as UTF-8 was mostly U+FFFD."""
    text = "First paragraph, caf\xe9.\n\nSecond paragraph \u65e5\u672c."
    bom = {
        "utf-16-le": codecs.BOM_UTF16_LE,
        "utf-16-be": codecs.BOM_UTF16_BE,
    }.get(codec, b"")

    document = parse_plaintext(bom + text.encode(codec))

    assert [block.text for block in document.blocks] == [
        "First paragraph, caf\xe9.",
        "Second paragraph \u65e5\u672c.",
    ]
    assert ParseFlag.ENCODING_FALLBACK.value not in document.flags


def test_a_utf_16_csv_keeps_its_header() -> None:
    """The column names are the first thing a wrong decode destroys."""
    document = parse_csv("name,year\nAlice,2024\n".encode("utf-16"))

    assert document.metadata["columns"] == ["name", "year"]


@pytest.mark.parametrize(
    "data",
    [
        bytes(range(0x80, 0x100)) * 4,
        b"\x81\x8d\x8f\x90\x9d" * 40 + b"tail",
        b"\x00\x01\x02\x03\x04\x05\x06\x07\xe9" * 30,
        codecs.BOM_UTF8 + b"\xff\xfe\xfd" * 50,
    ],
    ids=["high-bytes", "undefined-in-cp1252", "control-bytes", "bom-then-garbage"],
)
def test_a_document_that_is_mostly_unreadable_is_refused(data: bytes) -> None:
    """A binary file read as text became a block of replacement characters and was indexed."""
    for parse in (parse_plaintext, parse_markdown, parse_csv):
        with pytest.raises(ParseError) as caught:
            parse(data)

        assert caught.value.code is ErrorCode.ENCODING_UNREADABLE
        assert caught.value.retryable is False


def test_valid_utf_8_is_never_refused_however_odd_it_looks() -> None:
    """The refusal is a verdict on a failed decode, not on content a file really holds."""
    data = ("\ufffd" * 20 + "\x00\x01" * 10).encode("utf-8")

    assert parse_plaintext(data).flags == ()


def test_a_password_protected_pdf_is_refused_with_its_own_reason_code() -> None:
    """It fell into the corrupt-file catch: ``PARSE_FAILED``, as if the file were damaged."""
    with pytest.raises(ParseError) as caught:
        parse_pdf(encrypted_pdf_bytes(user_password="secret"))

    assert caught.value.code is ErrorCode.PDF_ENCRYPTED
    assert caught.value.retryable is False
    assert "password" in caught.value.detail


def test_a_pdf_encrypted_only_against_editing_still_parses() -> None:
    """An empty user password is how most "protected" PDFs ship, and anyone may read them."""
    document = parse_pdf(encrypted_pdf_bytes(user_password=""))

    assert "Secret body text" in document.text


class FakeTesseract:
    """Stands in for ``pytesseract``, recording the page images it is handed."""

    class TesseractNotFoundError(Exception):
        """Mirrors the real module's error, which the parser names in an ``except``."""

    def __init__(self) -> None:
        self.sizes: list[tuple[int, int]] = []

    def image_to_string(self, image: Image.Image) -> str:
        self.sizes.append(image.size)
        return "Recognised invoice text"


def test_a_scanned_page_without_ocr_is_refused_rather_than_indexed_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fixture has no text layer, so without OCR the document must reach the DLQ."""
    monkeypatch.setitem(sys.modules, "pytesseract", None)

    with pytest.raises(ParseError, match="no extractable text"):
        parse_pdf(scanned_pdf_bytes())


def test_a_scanned_page_is_rasterised_and_handed_to_ocr(monkeypatch: pytest.MonkeyPatch) -> None:
    """The OCR path had no scanned fixture at all (failure-modes.md row 2)."""
    tesseract = FakeTesseract()
    monkeypatch.setitem(sys.modules, "pytesseract", cast(ModuleType, tesseract))

    document = parse_pdf(scanned_pdf_bytes())

    assert ParseFlag.OCR_APPLIED.value in document.flags
    assert document.text == "Recognised invoice text"
    assert len(tesseract.sizes) == 1
    width, _ = tesseract.sizes[0]
    assert width == round(LETTER[0] / 72 * DEFAULT_PARSING_OPTIONS.ocr_resolution)


def test_real_ocr_reads_the_scanned_fixture() -> None:
    """End to end through tesseract. Runs only where the ``ocr`` extra and binary exist.

    No CI lane installs either (the unit lanes install ``.[dev]``), so in CI this skips;
    the rasterisation half of the path is proven above with a stand-in recogniser.
    """
    if importlib.util.find_spec("pytesseract") is None or shutil.which("tesseract") is None:
        pytest.skip(
            "OCR needs the 'ocr' extra and the tesseract binary; neither is installed in "
            "any CI lane, so this case runs only on a machine that has both"
        )

    document = parse_pdf(scanned_pdf_bytes())

    assert ParseFlag.OCR_APPLIED.value in document.flags
    assert "4471" in document.text


def test_a_page_with_no_text_beside_one_with_text_is_flagged_low_text_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing asserted the flag: a half-scanned document must say so, not index silently."""
    monkeypatch.setitem(sys.modules, "pytesseract", None)

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    pdf.drawString(72, 720, "The first page has a text layer and enough words to count.")
    pdf.showPage()
    pdf.rect(72, 72, 300, 300, fill=1)
    pdf.save()

    document = parse_pdf(buffer.getvalue())

    assert document.metadata["page_count"] == 2
    assert ParseFlag.LOW_TEXT_YIELD.value in document.flags
    assert {block.page for block in document.blocks} == {1}
