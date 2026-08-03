import base64
from collections.abc import AsyncIterator, Callable
from pathlib import Path

import httpx
import pytest

from fasterrag.config.schema import Settings
from fasterrag.core.identity import document_id
from fasterrag.errors import IngestionError, ParseError
from fasterrag.services.sources import INLINE_SCHEME, resolve_sources, typed_source
from fasterrag.workers.cpu_pool import CpuWorkerPool


def settings() -> Settings:
    return Settings.model_validate({})


def serve(
    handler: Callable[[httpx.Request], httpx.Response], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route every AsyncClient the module builds through an in-memory transport.

    The fake accepts exactly the keyword arguments production passes, by name, so a new
    argument in ``_fetch`` fails here loudly instead of silently diverging from the suite.
    The real class is captured before patching — ``sources.httpx`` is the global module, so
    the patched name is also the one this helper would otherwise resolve.
    """
    real_client = httpx.AsyncClient

    def client(*, timeout: float, follow_redirects: bool) -> httpx.AsyncClient:
        return real_client(
            transport=httpx.MockTransport(handler),
            timeout=timeout,
            follow_redirects=follow_redirects,
        )

    monkeypatch.setattr("fasterrag.services.sources.httpx.AsyncClient", client)


async def test_a_path_source_is_passed_through_untouched(tmp_path: Path) -> None:
    """Nothing is staged for a file that can already be read where it stands."""
    source = {"type": "path", "value": str(tmp_path / "a.md")}
    resolved = await resolve_sources([source], settings())

    assert resolved.sources == [str(tmp_path / "a.md")]
    assert resolved.locations == {}
    assert resolved.staging is None


async def test_plain_strings_are_treated_as_paths() -> None:
    """The CLI passes bare paths; both shapes must resolve identically."""
    resolved = await resolve_sources(["a.md", "b.md"], settings())

    assert resolved.sources == ["a.md", "b.md"]


async def test_an_inline_source_is_staged_and_readable() -> None:
    resolved = await resolve_sources(
        [{"type": "inline", "value": "# Title\n\nBody text."}], settings()
    )

    staged = Path(resolved.locations[resolved.sources[0]])
    assert staged.read_text(encoding="utf-8") == "# Title\n\nBody text."
    resolved.cleanup()


async def test_base64_inline_content_is_decoded() -> None:
    payload = base64.b64encode(b"# Encoded\n").decode("ascii")

    resolved = await resolve_sources([{"type": "inline", "value": payload}], settings())

    assert Path(resolved.locations[resolved.sources[0]]).read_bytes() == b"# Encoded\n"
    resolved.cleanup()


async def test_the_same_inline_payload_is_the_same_document_twice() -> None:
    """The whole reason this was a blocker: identity must survive staging.

    A canonical URI keyed on the temp path — or on the position in the request — would mint
    a new document id on every ingest, so deduplication would never fire and the corpus
    would grow a fresh copy each run with nothing reporting a problem.
    """
    first = await resolve_sources([{"type": "inline", "value": "same text"}], settings())
    second = await resolve_sources([{"type": "inline", "value": "same text"}], settings())

    assert first.sources == second.sources
    assert document_id(first.sources[0]) == document_id(second.sources[0])
    assert first.locations != second.locations

    first.cleanup()
    second.cleanup()


async def test_different_inline_payloads_are_different_documents() -> None:
    resolved = await resolve_sources(
        [{"type": "inline", "value": "one"}, {"type": "inline", "value": "two"}], settings()
    )

    assert len(set(resolved.sources)) == 2
    resolved.cleanup()


async def test_an_inline_source_carries_a_recognisable_scheme() -> None:
    resolved = await resolve_sources([{"type": "inline", "value": "text"}], settings())

    assert resolved.sources[0].startswith(INLINE_SCHEME)
    resolved.cleanup()


async def test_an_unknown_source_type_is_rejected() -> None:
    with pytest.raises(IngestionError, match="path, url, inline"):
        await resolve_sources([{"type": "ftp", "value": "ftp://x/a"}], settings())


async def test_cleanup_removes_the_staging_directory() -> None:
    resolved = await resolve_sources([{"type": "inline", "value": "text"}], settings())
    staging = resolved.staging
    assert staging is not None
    assert staging.is_dir()

    resolved.cleanup()

    assert not staging.exists()


async def test_cleanup_is_safe_to_call_twice() -> None:
    """A job that fails after cleanup must not fail again during teardown."""
    resolved = await resolve_sources([{"type": "inline", "value": "text"}], settings())

    resolved.cleanup()
    resolved.cleanup()


async def test_the_task_reads_from_the_location_but_is_identified_by_the_source() -> None:
    """The two fields must not be conflated — that conflation is the defect this prevents."""
    resolved = await resolve_sources([{"type": "inline", "value": "text"}], settings())

    task = CpuWorkerPool.tasks_for(resolved.sources, locations=resolved.locations)[0]

    assert task.source == resolved.sources[0]
    assert task.readable == resolved.locations[resolved.sources[0]]
    assert task.document_id == document_id(resolved.sources[0])
    resolved.cleanup()


def test_a_plain_path_task_reads_from_its_source() -> None:
    """No staging, so location is absent and the two collapse back into one."""
    task = CpuWorkerPool.tasks_for(["docs/a.md"])[0]

    assert task.location is None
    assert task.readable == "docs/a.md"


async def test_a_url_source_is_fetched_and_staged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The canonical URI stays the URL; only the bytes move to a local file."""
    serve(lambda _: httpx.Response(200, content=b"# Fetched\n"), monkeypatch)

    resolved = await resolve_sources(
        [{"type": "url", "value": "https://example.com/notes.md"}], settings()
    )

    assert resolved.sources == ["https://example.com/notes.md"]
    staged = Path(resolved.locations["https://example.com/notes.md"])
    assert staged.read_bytes() == b"# Fetched\n"
    assert staged.suffix == ".md"
    resolved.cleanup()


async def test_a_url_declaring_an_oversized_body_is_refused_before_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oversized = str(2 * 1024 * 1024)
    serve(
        lambda _: httpx.Response(200, headers={"content-length": oversized}, content=b""),
        monkeypatch,
    )
    limited = Settings.model_validate({"ingestion": {"max_document_mb": 1}})

    with pytest.raises(IngestionError, match="declares"):
        await resolve_sources([{"type": "url", "value": "https://example.com/big.pdf"}], limited)


async def test_a_url_lying_about_its_size_is_cut_off_while_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A limit enforced only on a header the peer controls is not a limit.

    The body arrives chunked with no ``content-length`` at all — the case the declared-size
    check cannot see — so only the count taken while streaming can stop it.
    """

    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(32):
            yield b"x" * 65536

    serve(lambda _: httpx.Response(200, content=chunks()), monkeypatch)
    limited = Settings.model_validate({"ingestion": {"max_document_mb": 1}})

    with pytest.raises(IngestionError, match="while downloading"):
        await resolve_sources([{"type": "url", "value": "https://example.com/big.bin"}], limited)


async def test_an_http_error_is_a_parse_error_naming_the_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serve(lambda _: httpx.Response(404), monkeypatch)

    with pytest.raises(ParseError, match="404"):
        await resolve_sources([{"type": "url", "value": "https://example.com/gone.md"}], settings())


def test_a_bare_https_string_is_classified_as_a_url() -> None:
    """The CLI documents ``ingest <path|url>``; a pasted URL must not be read as a path."""
    assert typed_source("https://example.com/a.pdf") == {
        "type": "url",
        "value": "https://example.com/a.pdf",
    }
    assert typed_source("http://example.com/a.pdf")["type"] == "url"


def test_a_windows_drive_spelling_stays_a_path() -> None:
    r"""``C:\docs`` parses as a one-letter URL scheme; it must classify as a path anyway."""
    assert typed_source("C:\\docs\\a.md")["type"] == "path"
    assert typed_source("docs/a.md")["type"] == "path"
