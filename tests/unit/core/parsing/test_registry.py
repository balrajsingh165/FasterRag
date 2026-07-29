from pathlib import Path

import pytest

from fasterrag.core.parsing import SUPPORTED_SUFFIXES, parse_bytes, parse_document
from fasterrag.errors import ErrorCode, IngestionError, ParseError


@pytest.mark.parametrize(
    ("filename", "parser"),
    [
        ("notes.txt", "plaintext"),
        ("agreement.md", "markdown"),
        ("agreement.html", "html"),
        ("people.csv", "csv"),
        ("agreement.json", "json"),
        ("agreement.docx", "docx"),
        ("agreement.pdf", "pdf-layout"),
    ],
)
def test_every_documented_format_dispatches_to_its_parser(
    sources: Path, filename: str, parser: str
) -> None:
    document = parse_document(sources / filename)

    assert document.parser == parser
    assert document.text.strip()


def test_dispatch_is_case_insensitive(sources: Path) -> None:
    upper = sources / "AGREEMENT.MD"
    upper.write_bytes((sources / "agreement.md").read_bytes())

    assert parse_document(upper).parser == "markdown"


def test_all_documented_formats_are_registered() -> None:
    assert {".pdf", ".html", ".md", ".docx", ".txt", ".csv", ".json"} <= SUPPORTED_SUFFIXES


def test_an_unsupported_type_names_what_is_supported(tmp_path: Path) -> None:
    unsupported = tmp_path / "archive.zip"
    unsupported.write_bytes(b"PK\x03\x04")

    with pytest.raises(ParseError, match="no parser is registered") as caught:
        parse_document(unsupported)
    assert ".pdf" in caught.value.detail


def test_a_missing_file_is_a_parse_error(tmp_path: Path) -> None:
    with pytest.raises(ParseError, match="does not exist"):
        parse_document(tmp_path / "absent.md")


def test_an_oversized_document_is_rejected_before_parsing() -> None:
    with pytest.raises(IngestionError, match="above the configured limit") as caught:
        parse_bytes(b"x" * 2048, filename="big.txt", max_bytes=1024)

    assert caught.value.code is ErrorCode.PAYLOAD_TOO_LARGE
    assert caught.value.status == 413


def test_a_document_at_the_limit_is_accepted() -> None:
    document = parse_bytes(b"hello there", filename="ok.txt", max_bytes=11)

    assert document.text == "hello there"


def test_inline_content_parses_without_touching_the_disk() -> None:
    document = parse_bytes(b"# Title\n\nBody text.", filename="inline.md")

    assert document.parser == "markdown"
    assert document.metadata["title"] == "Title"
