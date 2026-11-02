import base64
from collections.abc import AsyncIterator, Callable
from ipaddress import IPv4Address, IPv6Address, ip_address
from pathlib import Path

import httpx
import pytest

from fasterrag.config.schema import Settings
from fasterrag.core.identity import document_id
from fasterrag.errors import IngestionError, ParseError
from fasterrag.services.sources import (
    INLINE_SCHEME,
    _addresses,
    resolve_sources,
    typed_source,
)
from fasterrag.workers.cpu_pool import CpuWorkerPool


def settings() -> Settings:
    return Settings.model_validate({})


def serve(
    handler: Callable[[httpx.Request], httpx.Response], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route every AsyncClient the module builds through an in-memory transport."""
    real_client = httpx.AsyncClient

    def client(*, timeout: float, follow_redirects: bool) -> httpx.AsyncClient:
        return real_client(
            transport=httpx.MockTransport(handler),
            timeout=timeout,
            follow_redirects=follow_redirects,
        )

    monkeypatch.setattr("fasterrag.services.sources.httpx.AsyncClient", client)

    # CRITICAL: the SSRF guard resolves every host for real, so without this a unit
    # test naming a domain would do a live DNS lookup — slow, offline-hostile, and
    # flaky when resolution is rate-limited. Tests that exercise the guard itself use
    # literal addresses or set their own table, so stubbing here weakens nothing.
    async def addresses(host: str) -> list[IPv4Address | IPv6Address]:
        return [ip_address("93.184.216.34")]

    monkeypatch.setattr("fasterrag.services.sources._addresses", addresses)


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
    """The whole reason this was a blocker: identity must survive staging."""
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


def resolve_as(monkeypatch: pytest.MonkeyPatch, mapping: dict[str, str]) -> None:
    """Resolve hosts from a table, so the guard can be exercised without a network."""

    async def addresses(host: str) -> list[IPv4Address | IPv6Address]:
        return [ip_address(mapping.get(host, host))]

    monkeypatch.setattr("fasterrag.services.sources._addresses", addresses)


async def test_a_literal_address_resolves_without_a_network() -> None:
    """The guard's resolver must handle a bare IP, or every literal check is untested."""
    assert await _addresses("127.0.0.1") == [ip_address("127.0.0.1")]


@pytest.mark.parametrize(
    "uri",
    [
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        "http://127.0.0.1:6333/collections",
        "http://10.0.0.5/internal",
        "http://192.168.1.1/admin",
        "http://[::1]:6379/",
    ],
)
async def test_a_non_public_address_is_refused(uri: str) -> None:
    """The SSRF boundary: cloud metadata and anything inside the perimeter are unreachable."""
    with pytest.raises(IngestionError, match="non-public address"):
        await resolve_sources([{"type": "url", "value": uri}], settings())


@pytest.mark.parametrize("uri", ["file:///etc/passwd", "gopher://x/", "ftp://host/f"])
async def test_a_non_http_scheme_is_refused(uri: str) -> None:
    """Only http(s) is fetched; a file:// source would read the server's disk."""
    with pytest.raises(IngestionError, match="is not supported"):
        await resolve_sources([{"type": "url", "value": uri}], settings())


async def test_a_redirect_into_the_perimeter_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hop is guarded, not just the first URL — otherwise the guard is decoration."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "example.com":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/"})
        return httpx.Response(200, content=b"metadata")

    serve(handler, monkeypatch)
    resolve_as(monkeypatch, {"example.com": "93.184.216.34"})

    with pytest.raises(IngestionError, match="non-public address"):
        await resolve_sources([{"type": "url", "value": "http://example.com/doc.md"}], settings())


async def test_a_public_url_is_still_fetched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Control case: the guard must not break the feature it protects."""
    serve(lambda request: httpx.Response(200, content=b"# hello"), monkeypatch)
    resolve_as(monkeypatch, {"example.com": "93.184.216.34"})

    resolved = await resolve_sources(
        [{"type": "url", "value": "http://example.com/doc.md"}], settings()
    )

    try:
        location = resolved.locations["http://example.com/doc.md"]
        assert Path(location).read_bytes() == b"# hello"
    finally:
        resolved.cleanup()


async def test_a_redirect_chain_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """An endless redirect must end as a parse error, not as a hang."""
    serve(
        lambda request: httpx.Response(302, headers={"location": "http://example.com/next"}),
        monkeypatch,
    )
    resolve_as(monkeypatch, {"example.com": "93.184.216.34"})

    with pytest.raises(ParseError, match="redirects"):
        await resolve_sources([{"type": "url", "value": "http://example.com/a"}], settings())


def served() -> Settings:
    """A deployment that authenticates callers, i.e. one serving parties who are not you."""
    return Settings.model_validate({"security": {"auth": True}})


async def test_an_unconfined_path_source_is_refused_on_a_served_deployment() -> None:
    """Arbitrary server-side file read: the caller names the path, the query reads it back."""
    with pytest.raises(IngestionError, match="allowed_source_roots is empty"):
        await resolve_sources([{"type": "path", "value": "/etc/passwd"}], served())


async def test_a_path_outside_every_configured_root_is_refused(tmp_path: Path) -> None:
    """A configured root is a boundary, not a hint."""
    allowed = tmp_path / "corpus"
    allowed.mkdir()
    settings = Settings.model_validate(
        {"security": {"auth": True}, "ingestion": {"allowed_source_roots": [str(allowed)]}}
    )

    with pytest.raises(IngestionError, match="outside every directory"):
        await resolve_sources([{"type": "path", "value": "/etc/passwd"}], settings)


async def test_traversal_out_of_an_allowed_root_is_refused(tmp_path: Path) -> None:
    """`..` must be collapsed before the comparison, or the root is decorative."""
    allowed = tmp_path / "corpus"
    allowed.mkdir()
    (tmp_path / "secret.env").write_text("TOKEN=x", encoding="utf-8")
    settings = Settings.model_validate(
        {"security": {"auth": True}, "ingestion": {"allowed_source_roots": [str(allowed)]}}
    )

    escape = str(allowed / ".." / "secret.env")
    with pytest.raises(IngestionError, match="outside every directory"):
        await resolve_sources([{"type": "path", "value": escape}], settings)


async def test_a_path_inside_an_allowed_root_is_accepted(tmp_path: Path) -> None:
    """Control case: confinement must not break the feature it protects."""
    allowed = tmp_path / "corpus"
    allowed.mkdir()
    document = allowed / "a.md"
    document.write_text("hello", encoding="utf-8")
    settings = Settings.model_validate(
        {"security": {"auth": True}, "ingestion": {"allowed_source_roots": [str(allowed)]}}
    )

    resolved = await resolve_sources([{"type": "path", "value": str(document)}], settings)

    assert resolved.sources == [str(document)]


async def test_an_unauthenticated_deployment_still_ingests_local_paths(tmp_path: Path) -> None:
    """The CLI case. Auth off means nobody but the operator can reach it."""
    document = tmp_path / "a.md"
    document.write_text("hello", encoding="utf-8")

    resolved = await resolve_sources([{"type": "path", "value": str(document)}], settings())

    assert resolved.sources == [str(document)]
