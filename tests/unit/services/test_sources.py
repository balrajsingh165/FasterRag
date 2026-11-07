import base64
import hashlib
import sys
from collections.abc import AsyncIterator, Callable
from ipaddress import IPv4Address, IPv6Address, ip_address
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from fasterrag.config.schema import Settings
from fasterrag.core.identity import document_id
from fasterrag.errors import ErrorCode, FasterRagError, IngestionError, ParseError
from fasterrag.services.sources import (
    _MAXIMUM_REDIRECTS,
    INLINE_SCHEME,
    ResolvedSources,
    _addresses,
    guard_archive_path,
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

    def client(
        *, timeout: float, follow_redirects: bool, limits: httpx.Limits
    ) -> httpx.AsyncClient:
        return real_client(
            transport=httpx.MockTransport(handler),
            timeout=timeout,
            follow_redirects=follow_redirects,
            limits=limits,
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
        if request.headers["host"] == "example.com":
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


_REFUSAL_IS_REPORTABLE: Final = sys.platform == "win32"
"""Whether a refused symlink is a privilege the account lacks rather than a defect."""


def link(target: Path, name: Path) -> bool:
    """Create a symlink; refusal is reportable on Windows and nowhere else.

    A local Windows account without ``SeCreateSymbolicLinkPrivilege`` refuses and the
    callers skip. Catching the same error on POSIX would let the confinement guard's
    symlink leg vanish into a silent skip on the only runners that prove it (TASK-0328).
    """
    try:
        name.symlink_to(target)
    except (NotImplementedError, OSError):
        if _REFUSAL_IS_REPORTABLE:
            return False
        raise
    return True


async def test_a_symlink_out_of_an_allowed_root_is_refused(tmp_path: Path) -> None:
    """The other half of resolve-before-compare, which `..` alone does not prove.

    A comparison made on the path as written accepts a link that sits inside the root,
    and the read then follows it to whatever it points at.
    """
    allowed = tmp_path / "corpus"
    allowed.mkdir()
    (tmp_path / "secret.env").write_text("TOKEN=x", encoding="utf-8")
    if not link(tmp_path / "secret.env", allowed / "innocent.md"):
        pytest.skip(
            "this Windows account lacks SeCreateSymbolicLinkPrivilege; both CI runners "
            "create the link and run this case"
        )
    configured = Settings.model_validate(
        {"security": {"auth": True}, "ingestion": {"allowed_source_roots": [str(allowed)]}}
    )

    with pytest.raises(IngestionError, match="outside every directory"):
        await resolve_sources([{"type": "path", "value": str(allowed / "innocent.md")}], configured)


def test_a_symlink_out_of_an_allowed_archive_root_is_refused(tmp_path: Path) -> None:
    """The same escape against the verb that WRITES the file the caller names."""
    allowed = tmp_path / "archives"
    allowed.mkdir()
    (tmp_path / "config.yaml").write_text("app: {}", encoding="utf-8")
    if not link(tmp_path / "config.yaml", allowed / "backup.fragx"):
        pytest.skip(
            "this Windows account lacks SeCreateSymbolicLinkPrivilege; both CI runners "
            "create the link and run this case"
        )
    configured = Settings.model_validate(
        {"security": {"auth": True, "allowed_archive_roots": [str(allowed)]}}
    )

    with pytest.raises(FasterRagError, match="outside every directory"):
        guard_archive_path(str(allowed / "backup.fragx"), configured)


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


def rebinding(monkeypatch: pytest.MonkeyPatch, first: str, later: str) -> list[str]:
    """Answer ``first`` once and ``later`` after that, the shape of a TTL-0 rebind."""
    asked: list[str] = []

    async def addresses(host: str) -> list[IPv4Address | IPv6Address]:
        asked.append(host)
        return [ip_address(first if len(asked) == 1 else later)]

    monkeypatch.setattr("fasterrag.services.sources._addresses", addresses)
    return asked


def recorder() -> tuple[list[httpx.Request], Callable[[httpx.Request], httpx.Response]]:
    """Return a request log and a handler that fills it."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=b"# body")

    return seen, handler


async def test_the_address_that_was_checked_is_the_address_dialled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DNS rebinding: a second answer must not be able to redirect the connection."""
    seen, handler = recorder()
    serve(handler, monkeypatch)
    asked = rebinding(monkeypatch, "93.184.216.34", "169.254.169.254")

    resolved = await resolve_sources(
        [{"type": "url", "value": "https://example.com/doc.md"}], settings()
    )
    resolved.cleanup()

    assert asked == ["example.com"]
    assert seen[0].url.host == "93.184.216.34"


async def test_pinning_keeps_the_host_header_and_the_certificate_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinning must not cost virtual hosting, nor verify the certificate against an IP."""
    seen, handler = recorder()
    serve(handler, monkeypatch)
    resolve_as(monkeypatch, {"example.com": "93.184.216.34"})

    resolved = await resolve_sources(
        [{"type": "url", "value": "https://example.com:8443/doc.md"}], settings()
    )
    resolved.cleanup()

    assert seen[0].headers["host"] == "example.com:8443"
    assert seen[0].extensions["sni_hostname"] == "example.com"


async def test_an_ipv6_answer_is_dialled_as_a_bracketed_literal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An IPv6 literal written into a URL unbracketed is not a URL."""
    seen, handler = recorder()
    serve(handler, monkeypatch)
    resolve_as(monkeypatch, {"example.com": "2606:4700:4700::1111"})

    resolved = await resolve_sources(
        [{"type": "url", "value": "http://example.com/doc.md"}], settings()
    )
    resolved.cleanup()

    assert str(seen[0].url) == "http://[2606:4700:4700::1111]/doc.md"


async def test_a_relative_redirect_is_joined_against_the_name_not_the_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Joining against the pinned URL would lose the name the next hop must be checked by."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(302, headers={"location": "/next"})
        return httpx.Response(200, content=b"# body")

    serve(handler, monkeypatch)
    resolve_as(monkeypatch, {"example.com": "93.184.216.34"})

    resolved = await resolve_sources(
        [{"type": "url", "value": "https://example.com/doc.md"}], settings()
    )
    resolved.cleanup()

    assert [request.url.path for request in seen] == ["/doc.md", "/next"]
    assert seen[1].headers["host"] == "example.com"
    assert seen[1].extensions["sni_hostname"] == "example.com"


async def test_an_unconfined_archive_path_is_refused_on_a_served_deployment() -> None:
    """Export writes where the caller says, so unconfined is a write primitive."""
    with pytest.raises(FasterRagError, match="allowed_archive_roots is empty"):
        guard_archive_path("/etc/fasterrag/config.yaml", served())


async def test_an_archive_path_outside_every_configured_root_is_refused(tmp_path: Path) -> None:
    """A configured root is a boundary, not a hint."""
    allowed = tmp_path / "archives"
    allowed.mkdir()
    configured = Settings.model_validate(
        {"security": {"auth": True, "allowed_archive_roots": [str(allowed)]}}
    )

    with pytest.raises(FasterRagError, match="outside every directory"):
        guard_archive_path(str(tmp_path / "elsewhere.fragx"), configured)


async def test_traversal_out_of_an_allowed_archive_root_is_refused(tmp_path: Path) -> None:
    """`..` must be collapsed before the comparison, or the root is decorative."""
    allowed = tmp_path / "archives"
    allowed.mkdir()
    configured = Settings.model_validate(
        {"security": {"auth": True, "allowed_archive_roots": [str(allowed)]}}
    )

    escape = str(allowed / ".." / "config.yaml")
    with pytest.raises(FasterRagError, match="outside every directory"):
        guard_archive_path(escape, configured)


async def test_an_archive_path_inside_an_allowed_root_is_returned_resolved(
    tmp_path: Path,
) -> None:
    """Control case: confinement must not break the feature it protects."""
    allowed = tmp_path / "archives"
    allowed.mkdir()
    configured = Settings.model_validate(
        {"security": {"auth": True, "allowed_archive_roots": [str(allowed)]}}
    )

    assert (
        guard_archive_path(str(allowed / "backup.fragx"), configured)
        == (allowed / "backup.fragx").resolve()
    )


async def test_an_unauthenticated_deployment_still_writes_archives_anywhere(
    tmp_path: Path,
) -> None:
    """The CLI case. Auth off means nobody but the operator can reach it."""
    assert (
        guard_archive_path(str(tmp_path / "backup.fragx"), settings())
        == (tmp_path / "backup.fragx").resolve()
    )


def serve_recording(
    handler: Callable[[httpx.Request], httpx.Response], monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Serve like :func:`serve`, and hand back the arguments the client was built with."""
    built: dict[str, Any] = {}
    real_client = httpx.AsyncClient

    def client(
        *, timeout: float, follow_redirects: bool, limits: httpx.Limits
    ) -> httpx.AsyncClient:
        built.update(timeout=timeout, follow_redirects=follow_redirects, limits=limits)
        return real_client(
            transport=httpx.MockTransport(handler),
            timeout=timeout,
            follow_redirects=follow_redirects,
            limits=limits,
        )

    monkeypatch.setattr("fasterrag.services.sources.httpx.AsyncClient", client)
    resolve_as(monkeypatch, {"example.com": "93.184.216.34"})
    return built


async def test_the_fetch_client_pools_nothing_and_follows_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinning only holds per connection.

    httpcore pools by origin and a pinned origin is `(scheme, IP, port)`, so a kept-alive
    connection would carry a second host's request over the first host's verified session;
    and a client that followed redirects itself would route hops past the guard entirely.
    """
    built = serve_recording(lambda _: httpx.Response(200, content=b"# body"), monkeypatch)

    resolved = await resolve_sources(
        [{"type": "url", "value": "http://example.com/a.md"}], settings()
    )
    resolved.cleanup()

    assert built["follow_redirects"] is False
    assert built["limits"].max_keepalive_connections == 0
    assert built["limits"].max_connections == 10
    assert built["timeout"] == 60.0


@pytest.mark.parametrize(
    "uri", ["http://224.0.0.1/admin", "http://239.1.2.3/admin", "http://[ff02::1]/"]
)
async def test_a_multicast_address_is_refused(uri: str) -> None:
    """`ip_address(...).is_global` is True for multicast, so only the multicast term refuses it."""
    with pytest.raises(IngestionError, match="non-public address"):
        await resolve_sources([{"type": "url", "value": uri}], settings())


async def test_one_private_answer_among_public_ones_refuses_the_whole_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every record is checked; a public answer must not launder a private sibling."""
    seen, handler = recorder()
    serve(handler, monkeypatch)

    async def addresses(host: str) -> list[IPv4Address | IPv6Address]:
        return [ip_address("93.184.216.34"), ip_address("127.0.0.1")]

    monkeypatch.setattr("fasterrag.services.sources._addresses", addresses)

    with pytest.raises(IngestionError, match="non-public address"):
        await resolve_sources([{"type": "url", "value": "http://example.com/a"}], settings())

    assert seen == []


async def test_the_first_checked_answer_is_the_one_dialled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pin has to name an address the guard actually examined and approved."""
    seen, handler = recorder()
    serve(handler, monkeypatch)

    async def addresses(host: str) -> list[IPv4Address | IPv6Address]:
        return [ip_address("93.184.216.34"), ip_address("1.1.1.1")]

    monkeypatch.setattr("fasterrag.services.sources._addresses", addresses)

    resolved = await resolve_sources([{"type": "url", "value": "http://example.com/a"}], settings())
    resolved.cleanup()

    assert seen[0].url.host == "93.184.216.34"


async def test_the_scheme_refusal_names_what_arrived() -> None:
    """An operator reading the refusal needs the scheme that was sent, or that none was."""
    with pytest.raises(IngestionError, match="scheme 'ftp' is not supported"):
        await resolve_sources([{"type": "url", "value": "ftp://host/f"}], settings())

    with pytest.raises(IngestionError, match=r"scheme '\(none\)' is not supported"):
        await resolve_sources([{"type": "url", "value": "example.com/f"}], settings())


async def test_five_redirects_are_followed_and_the_sixth_hop_is_the_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bound is a number, not merely "finite": moving it moves how far a caller is led."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if len(seen) <= _MAXIMUM_REDIRECTS:
            return httpx.Response(302, headers={"location": f"/hop{len(seen)}"})
        return httpx.Response(200, content=b"# body")

    serve(handler, monkeypatch)

    resolved = await resolve_sources([{"type": "url", "value": "http://example.com/a"}], settings())
    resolved.cleanup()

    assert _MAXIMUM_REDIRECTS == 5
    assert len(seen) == 6


async def test_a_sixth_redirect_ends_the_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    """One request per allowed hop plus the first, then a refusal naming the bound."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(302, headers={"location": f"/hop{len(seen)}"})

    serve(handler, monkeypatch)

    with pytest.raises(ParseError, match="exceeded 5 redirects"):
        await resolve_sources([{"type": "url", "value": "http://example.com/a"}], settings())

    assert len(seen) == 6


async def test_a_body_of_exactly_the_limit_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The documented maximum is inclusive, and a megabyte is 1024 * 1024 bytes."""
    serve(lambda _: httpx.Response(200, content=b"x" * (1024 * 1024)), monkeypatch)
    limited = Settings.model_validate({"ingestion": {"max_document_mb": 1}})

    resolved = await resolve_sources(
        [{"type": "url", "value": "https://example.com/exact.bin"}], limited
    )

    staged = Path(resolved.locations["https://example.com/exact.bin"])
    assert staged.stat().st_size == 1024 * 1024
    resolved.cleanup()


async def test_one_byte_over_the_limit_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other side of the same boundary, which is what makes the first test a boundary."""
    serve(lambda _: httpx.Response(200, content=b"x" * (1024 * 1024 + 1)), monkeypatch)
    limited = Settings.model_validate({"ingestion": {"max_document_mb": 1}})

    with pytest.raises(IngestionError, match="above the configured limit"):
        await resolve_sources([{"type": "url", "value": "https://example.com/over.bin"}], limited)


async def test_a_ten_character_extension_survives_and_an_eleventh_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parsing dispatches on the staged filename, so the suffix window is a real boundary."""
    serve(lambda _: httpx.Response(200, content=b"body"), monkeypatch)

    kept = await resolve_sources(
        [{"type": "url", "value": "https://example.com/a.abcdefghi"}], settings()
    )
    dropped = await resolve_sources(
        [{"type": "url", "value": "https://example.com/a.abcdefghij"}], settings()
    )

    assert Path(kept.locations["https://example.com/a.abcdefghi"]).suffix == ".abcdefghi"
    assert Path(dropped.locations["https://example.com/a.abcdefghij"]).suffix == ".txt"
    kept.cleanup()
    dropped.cleanup()


async def test_a_uri_ending_in_a_bare_dot_stages_as_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`Path("report.").suffix` is "." — one character, and not an extension any parser knows."""
    serve(lambda _: httpx.Response(200, content=b"body"), monkeypatch)

    resolved = await resolve_sources(
        [{"type": "url", "value": "https://example.com/report."}], settings()
    )

    assert Path(resolved.locations["https://example.com/report."]).suffix == ".txt"
    resolved.cleanup()


async def test_a_staged_file_is_named_by_a_truncated_digest_of_its_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two sources staged for one job must not collide, and the filename is that promise.

    The width is part of it: a collision silently serves one document's bytes under the
    other's id, which nothing downstream can detect.
    """
    serve(lambda _: httpx.Response(200, content=b"body"), monkeypatch)
    uri = "https://example.com/a.md"

    resolved = await resolve_sources([{"type": "url", "value": uri}], settings())

    assert Path(resolved.locations[uri]).stem == hashlib.sha256(uri.encode()).hexdigest()[:16]
    resolved.cleanup()


def test_cleanup_removes_what_it_can_rather_than_stopping_at_the_first_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One undeletable entry must not abandon the rest of the staged copy on disk.

    `suppress(OSError)` hides the difference from every caller, so the flag that decides
    whether removal continues is pinned here instead.
    """
    captured: dict[str, Any] = {}

    def rmtree(path: Path, ignore_errors: bool = False) -> None:
        captured["ignore_errors"] = ignore_errors

    monkeypatch.setattr("fasterrag.services.sources.shutil.rmtree", rmtree)
    resolved = ResolvedSources(staging=tmp_path)

    resolved.cleanup()

    assert captured["ignore_errors"] is True
    assert resolved.staging is None


async def test_plain_text_is_not_mistaken_for_base64() -> None:
    """Unvalidated, `b64decode` discards the space and decodes the rest into unrelated bytes."""
    resolved = await resolve_sources([{"type": "inline", "value": "same text"}], settings())

    staged = Path(resolved.locations[resolved.sources[0]])
    assert staged.read_text(encoding="utf-8") == "same text"
    resolved.cleanup()


async def test_the_inline_uri_carries_a_fixed_length_digest() -> None:
    """The canonical URI is what a document id derives from, so its shape is a contract."""
    resolved = await resolve_sources([{"type": "inline", "value": "text"}], settings())

    digest = resolved.sources[0].removeprefix(INLINE_SCHEME).removesuffix(".txt")
    assert len(digest) == 32
    assert all(character in "0123456789abcdef" for character in digest)
    resolved.cleanup()


def test_a_resolved_source_set_refuses_an_unknown_field() -> None:
    """Three fields that must stay in step; a typo must fail rather than create a fourth."""
    resolved = ResolvedSources()
    typo = "stagin"

    with pytest.raises(AttributeError):
        setattr(resolved, typo, Path("."))


@pytest.mark.parametrize(
    "source",
    [
        {"type": "url", "value": "ftp://host/f"},
        {"type": "url", "value": "http:///nohost"},
        {"type": "url", "value": "http://169.254.169.254/"},
        {"type": "ftp", "value": "ftp://host/f"},
    ],
)
async def test_a_refused_source_is_not_advertised_as_retryable(source: dict[str, str]) -> None:
    """`retryable` reaches the problem document and the CLI exit code.

    A guard refusal that says "retry" turns a single refused request into a loop against
    the guard, and tells an operator the deployment is flaky rather than the input wrong.
    """
    with pytest.raises(IngestionError) as caught:
        await resolve_sources([source], settings())

    assert caught.value.code is ErrorCode.VALIDATION_FAILED
    assert caught.value.retryable is False


async def test_a_refused_path_is_not_advertised_as_retryable(tmp_path: Path) -> None:
    """Both path refusals: unconfined on a served deployment, and outside every root."""
    allowed = tmp_path / "corpus"
    allowed.mkdir()
    configured = Settings.model_validate(
        {"security": {"auth": True}, "ingestion": {"allowed_source_roots": [str(allowed)]}}
    )

    with pytest.raises(IngestionError) as unconfined:
        await resolve_sources([{"type": "path", "value": "/etc/passwd"}], served())
    with pytest.raises(IngestionError) as outside:
        await resolve_sources([{"type": "path", "value": "/etc/passwd"}], configured)

    assert unconfined.value.retryable is False
    assert outside.value.retryable is False


def test_a_refused_archive_path_is_not_advertised_as_retryable(tmp_path: Path) -> None:
    """Both archive refusals, which gate a caller-named write as well as a read."""
    allowed = tmp_path / "archives"
    allowed.mkdir()
    configured = Settings.model_validate(
        {"security": {"auth": True, "allowed_archive_roots": [str(allowed)]}}
    )

    with pytest.raises(FasterRagError) as unconfined:
        guard_archive_path("/etc/fasterrag/config.yaml", served())
    with pytest.raises(FasterRagError) as outside:
        guard_archive_path(str(tmp_path / "elsewhere.fragx"), configured)

    assert unconfined.value.retryable is False
    assert outside.value.retryable is False


async def test_an_oversized_source_is_not_advertised_as_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All three size refusals: declared, counted while streaming, and measured inline."""
    limited = Settings.model_validate({"ingestion": {"max_document_mb": 1}})

    async def chunks() -> AsyncIterator[bytes]:
        for _ in range(32):
            yield b"x" * 65536

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/declared.pdf":
            return httpx.Response(
                200, headers={"content-length": str(4 * 1024 * 1024)}, content=b""
            )
        return httpx.Response(200, content=chunks())

    serve(handler, monkeypatch)

    with pytest.raises(IngestionError) as declared:
        await resolve_sources(
            [{"type": "url", "value": "https://example.com/declared.pdf"}], limited
        )
    with pytest.raises(IngestionError) as streamed:
        await resolve_sources(
            [{"type": "url", "value": "https://example.com/streamed.bin"}], limited
        )
    with pytest.raises(IngestionError) as inline:
        await resolve_sources([{"type": "inline", "value": "y" * (2 * 1024 * 1024)}], limited)

    assert declared.value.code is ErrorCode.PAYLOAD_TOO_LARGE
    assert declared.value.retryable is False
    assert streamed.value.retryable is False
    assert inline.value.retryable is False
