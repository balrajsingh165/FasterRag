"""The settings that only matter if something downstream actually reads them.

Each test here pins a knob to a value and asserts the behaviour changes. A setting that
validates but is never consumed is worse than no setting: it reads as a supported control
and silently does nothing.
"""

import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pdfplumber
import pytest

from fasterrag.config.schema import Settings
from fasterrag.core.cache import create_embedding_store, create_semantic_store
from fasterrag.core.chunking import create_chunker
from fasterrag.core.parsing import ParseFlag, ParsingOptions, create_parsing_options
from fasterrag.core.parsing.pdf import parse_pdf
from fasterrag.core.parsing.tabular import parse_csv
from fasterrag.core.retrieval.bm25 import encode_document
from fasterrag.workers.cpu_pool import CpuWorkerPool, parse_and_chunk
from tests.unit.core.parsing.conftest import pdf_bytes

ROWS = "name,year\n" + "".join(f"Person{index},20{index:02d}\n" for index in range(6))


def settings(**sections: Any) -> Settings:
    return Settings.model_validate(sections)


class FakeTesseract:
    """Stands in for the optional `pytesseract` module, which need not be installed."""

    TesseractNotFoundError = RuntimeError

    @staticmethod
    def image_to_string(_image: object) -> str:
        return "recognized text"


def install_fake_ocr(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Make the OCR path runnable in-process, recording the DPI of every page render."""
    rendered: list[int] = []

    def to_image(page: object, *, resolution: int) -> object:
        rendered.append(resolution)
        return SimpleNamespace(original=object())

    monkeypatch.setitem(sys.modules, "pytesseract", cast("ModuleType", FakeTesseract))
    monkeypatch.setattr(pdfplumber.page.Page, "to_image", to_image)
    return rendered


def headings(options: ParsingOptions) -> list[str]:
    document = parse_pdf(pdf_bytes(), options=options)
    return [block.text for block in document.blocks if block.kind == "heading"]


def test_bm25_saturation_changes_the_encoded_weights() -> None:
    text = "policy policy policy travel allowance"

    flat = encode_document(text, k1=0.1)
    steep = encode_document(text, k1=2.5)

    assert flat.values != steep.values


def test_bm25_length_normalisation_changes_the_encoded_weights() -> None:
    text = "the travel allowance policy covers meals and lodging for contractors"

    off = encode_document(text, b=0.0)
    full = encode_document(text, b=1.0)

    assert off.values != full.values


def test_bm25_defaults_match_the_schema_defaults() -> None:
    """A default that drifts between the schema and the encoder is a silent reindex."""
    configured = settings()
    text = "policy policy travel"

    assert encode_document(text) == encode_document(
        text, k1=configured.retrieval.bm25_k1, b=configured.retrieval.bm25_b
    )


def test_the_semantic_percentile_reaches_the_chunker() -> None:
    class Embedder:
        def encode(self, texts: list[str]) -> list[list[float]]:
            return [[float(index), 1.0] for index, _ in enumerate(texts)]

    chunker = create_chunker(
        settings(chunking={"strategy": "semantic", "semantic_percentile": 0.6}),
        embedder=Embedder(),  # type: ignore[arg-type]
    )

    assert chunker._percentile == 0.6  # type: ignore[attr-defined]


def test_the_cache_ceiling_reaches_the_semantic_store() -> None:
    store = create_semantic_store(settings(cache={"backend": "memory", "max_entries": 7}))

    assert store._maximum == 7  # type: ignore[attr-defined]


def test_the_cache_ceiling_reaches_the_embedding_store() -> None:
    store = create_embedding_store(
        settings(embeddings={"cache": {"backend": "memory", "max_entries": 3}})
    )

    assert store._maximum == 3  # type: ignore[attr-defined]


async def test_a_bounded_store_actually_evicts() -> None:
    """A ceiling nothing enforces is a memory leak with a number next to it."""
    store = create_semantic_store(settings(cache={"backend": "memory", "max_entries": 2}))

    for index in range(5):
        await store.set(f"key-{index}", b"value", ttl=3600)

    assert await store.get("key-0") is None
    assert await store.get("key-4") is not None


def test_the_parsing_defaults_are_the_schema_defaults() -> None:
    """`ParsingOptions` restates the shipped defaults; drift would change behaviour."""
    assert create_parsing_options(settings()) == ParsingOptions()


def test_every_parsing_setting_is_carried_into_the_options() -> None:
    """A field added to the section and forgotten in the factory is a dead setting."""
    configured = settings(
        parsing={
            "minimum_chars_per_page": 7,
            "ocr_resolution": 321,
            "heading_size_ratio": 2.5,
            "max_heading_chars": 33,
            "rows_per_block": 3,
        }
    )

    assert asdict(create_parsing_options(configured)) == configured.parsing.model_dump()


def test_the_ocr_trigger_reaches_the_pdf_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    """The knob that matters most: which pages are treated as scans."""
    rendered = install_fake_ocr(monkeypatch)

    born_digital = parse_pdf(pdf_bytes())

    assert rendered == []
    assert ParseFlag.OCR_APPLIED.value not in born_digital.flags

    treated_as_a_scan = parse_pdf(
        pdf_bytes(), options=ParsingOptions(minimum_chars_per_page=10_000)
    )

    assert rendered == [200]
    assert ParseFlag.OCR_APPLIED.value in treated_as_a_scan.flags
    assert "recognized text" in treated_as_a_scan.text


def test_the_ocr_resolution_reaches_the_render(monkeypatch: pytest.MonkeyPatch) -> None:
    rendered = install_fake_ocr(monkeypatch)

    parse_pdf(
        pdf_bytes(),
        options=ParsingOptions(minimum_chars_per_page=10_000, ocr_resolution=350),
    )

    assert rendered == [350]


def test_the_heading_ratio_reaches_the_pdf_parser() -> None:
    assert headings(ParsingOptions()) == ["Vendor Agreement", "3. Termination"]
    assert headings(ParsingOptions(heading_size_ratio=4.0)) == []


def test_the_heading_length_ceiling_reaches_the_pdf_parser() -> None:
    assert headings(ParsingOptions(max_heading_chars=5)) == []


def test_rows_per_block_reaches_the_csv_parser() -> None:
    grouped = parse_csv(ROWS.encode())
    per_row = parse_csv(ROWS.encode(), options=ParsingOptions(rows_per_block=1))

    assert len(grouped.blocks) == 1
    assert len(per_row.blocks) == 6


def test_the_parsing_options_reach_the_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pipeline path, not just the parser: settings must survive to the worker."""
    rendered = install_fake_ocr(monkeypatch)
    source = tmp_path / "scan.pdf"
    source.write_bytes(pdf_bytes())
    task = CpuWorkerPool.tasks_for([str(source)])[0]

    assert ParseFlag.OCR_APPLIED.value not in parse_and_chunk(task, settings()).parse_flags
    assert rendered == []

    outcome = parse_and_chunk(
        task, settings(parsing={"minimum_chars_per_page": 10_000, "ocr_resolution": 300})
    )

    assert rendered == [300]
    assert ParseFlag.OCR_APPLIED.value in outcome.parse_flags
