"""How a fetched or inline source chooses its parser (TASK-0459).

Each test was written before its fix and failed on ``main`` at ``b97ce52`` for the reason
its docstring gives. The parse runs through ``parse_and_chunk`` — the worker's own entry
point — because the defect was in which name the worker chose a parser from, not only in
what was staged.
"""

import base64
from collections.abc import Callable
from ipaddress import IPv4Address, IPv6Address, ip_address
from pathlib import Path

import httpx
import pytest

from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, IngestionError
from fasterrag.services.sources import ResolvedSources, resolve_sources
from fasterrag.workers.cpu_pool import CpuWorkerPool, parse_and_chunk
from fasterrag.workers.queues import ParseOutcome
from tests.unit.core.parsing.conftest import docx_bytes, pdf_bytes

PAGE = b"<html><body><h1>About us</h1><p>We make widgets.</p></body></html>"


def settings() -> Settings:
    return Settings.model_validate({})


def serve(
    handler: Callable[[httpx.Request], httpx.Response], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route every AsyncClient the module builds through an in-memory transport."""
    real_client = httpx.AsyncClient

    def client(
        *, timeout: float, follow_redirects: bool, limits: httpx.Limits
    ) -> httpx.AsyncClient:
        return real_client(
            transport=httpx.MockTransport(handler),
            timeout=timeout,
            follow_redirects=follow_redirects,
            limits=limits,
        )

    async def addresses(host: str) -> list[IPv4Address | IPv6Address]:
        return [ip_address("93.184.216.34")]

    monkeypatch.setattr("fasterrag.services.sources.httpx.AsyncClient", client)
    monkeypatch.setattr("fasterrag.services.sources._addresses", addresses)


def parsed(resolved: ResolvedSources) -> ParseOutcome:
    """Parse the one resolved source the way a worker would."""
    tasks = CpuWorkerPool.tasks_for(resolved.sources, locations=resolved.locations)
    try:
        return parse_and_chunk(tasks[0], settings())
    finally:
        resolved.cleanup()


def texts(outcome: ParseOutcome) -> str:
    return " ".join(payload.chunk.text for payload in outcome.chunks)


@pytest.mark.parametrize(
    "url", ["https://example.com/about", "https://example.com/index.php", "https://example.com/"]
)
async def test_a_page_served_as_html_parses_as_html_whatever_its_url(
    url: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The parser came from the URL's name: ``/about`` and ``.php`` had no parser at all."""
    serve(
        lambda _: httpx.Response(
            200, content=PAGE, headers={"content-type": "text/html; charset=utf-8"}
        ),
        monkeypatch,
    )

    outcome = parsed(await resolve_sources([{"type": "url", "value": url}], settings()))

    assert outcome.parser == "html"
    assert "<h1>" not in texts(outcome)
    assert "We make widgets." in texts(outcome)


async def test_a_url_with_a_query_string_still_parses_by_its_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``doc.pdf?download=1`` was looked up as the suffix ``.pdf?download=1``."""
    serve(
        lambda _: httpx.Response(
            200, content=pdf_bytes(), headers={"content-type": "application/pdf"}
        ),
        monkeypatch,
    )

    outcome = parsed(
        await resolve_sources(
            [{"type": "url", "value": "https://example.com/doc.pdf?download=1"}], settings()
        )
    )

    assert outcome.parser == "pdf-layout"


async def test_a_generic_content_type_defers_to_a_known_url_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw file hosts serve everything as ``text/plain``; a ``.md`` URL is still Markdown."""
    serve(
        lambda _: httpx.Response(
            200, content=b"# Title\n\nBody.", headers={"content-type": "text/plain"}
        ),
        monkeypatch,
    )

    outcome = parsed(
        await resolve_sources(
            [{"type": "url", "value": "https://example.com/notes.md"}], settings()
        )
    )

    assert outcome.parser == "markdown"


async def test_inline_text_that_looks_like_a_filename_is_still_text() -> None:
    """The suffix came from the payload: ``Release 1.2`` was a file of type ``.2``."""
    resolved = await resolve_sources(
        [{"type": "inline", "value": "Shipped in Release 1.2"}], settings()
    )

    outcome = parsed(resolved)

    assert outcome.parser == "plaintext"
    assert texts(outcome) == "Shipped in Release 1.2"


async def test_a_word_that_happens_to_be_base64_is_kept_as_text() -> None:
    """``Test`` is valid base64 and decoded to three bytes of noise."""
    resolved = await resolve_sources([{"type": "inline", "value": "Test"}], settings())

    assert Path(resolved.locations[resolved.sources[0]]).read_bytes() == b"Test"
    resolved.cleanup()


async def test_an_inline_media_type_chooses_the_parser() -> None:
    """Without a field to say so, inline Markdown could only ever be parsed as plain text."""
    resolved = await resolve_sources(
        [{"type": "inline", "value": "# Title\n\nBody.", "media_type": "text/markdown"}],
        settings(),
    )

    outcome = parsed(resolved)

    assert outcome.parser == "markdown"


async def test_an_inline_filename_chooses_the_parser_and_requires_base64_for_binaries() -> None:
    """A base64 DOCX staged as ``.txt`` was parsed as plain text and indexed as noise."""
    payload = base64.b64encode(docx_bytes()).decode("ascii")
    resolved = await resolve_sources(
        [{"type": "inline", "value": payload, "filename": "agreement.docx"}], settings()
    )

    outcome = parsed(resolved)

    assert outcome.parser == "docx"
    assert "Vendor Agreement" in texts(outcome)


async def test_a_base64_pdf_without_a_declared_type_is_recognised_by_its_signature() -> None:
    """Base64 binaries were promised to work undeclared; a PDF was staged and parsed as text."""
    payload = base64.b64encode(pdf_bytes()).decode("ascii")

    outcome = parsed(await resolve_sources([{"type": "inline", "value": payload}], settings()))

    assert outcome.parser == "pdf-layout"


@pytest.mark.parametrize(
    "extra",
    [
        {"media_type": "application/x-unknown"},
        {"filename": "notes.rtf"},
        {"filename": "scan.pdf"},
    ],
    ids=["unknown-media-type", "unsupported-filename", "pdf-that-is-not-base64"],
)
async def test_an_inline_declaration_that_cannot_be_honoured_is_refused(
    extra: dict[str, str],
) -> None:
    """An explicit declaration is a promise; quietly parsing as text would break it."""
    with pytest.raises(IngestionError) as caught:
        await resolve_sources(
            [{"type": "inline", "value": "not base64 at all!", **extra}], settings()
        )

    assert caught.value.code is ErrorCode.VALIDATION_FAILED
    assert caught.value.retryable is False
