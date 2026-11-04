"""Resolving the three documented source types into readable bytes.

``docs/api-reference.md`` specifies ``path``, ``url``, and ``inline``. Only ``path`` can be
read where it stands; the other two must be staged to disk first, because parsing happens in
a worker *process* and a URL or a base64 blob cannot be handed across that boundary as a file.

**Staging must never change a document's identity.** A document id derives from its canonical
URI, so if a staged file's temp path became the source, re-ingesting the same URL would mint a
new id on every run and deduplication would never fire — the corpus would grow a fresh copy
each time and nothing would report a problem. The canonical URI is therefore preserved as the
``source`` and the temp path travels separately as the ``location``
(:class:`~fasterrag.workers.queues.DocumentTask`).

Staged files live under one directory per job and are removed when the job settles. They are
a transport detail, not a cache: keeping them would mean a second copy of the corpus on disk
that nothing tracks and nothing prunes.

This module also owns the confinement applied to every *caller-supplied server path*, which
is why :func:`guard_archive_path` lives here beside the ingestion guard rather than next to
the archive service: both answer the same question with the same rule, and a second shape of
the same check is a second place for it to be wrong.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import shutil
import socket
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from ipaddress import IPv4Address, IPv6Address, ip_address
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlparse

import httpx

from fasterrag.config.schema import Settings
from fasterrag.core.identity import text_hash
from fasterrag.errors import ErrorCode, FasterRagError, IngestionError, ParseError
from fasterrag.observability.logging import get_logger

__all__ = [
    "INLINE_SCHEME",
    "ResolvedSources",
    "guard_archive_path",
    "resolve_sources",
    "typed_source",
]

INLINE_SCHEME: Final = "inline:"

_MEGABYTE: Final = 1024 * 1024
_FETCH_TIMEOUT_SECONDS: Final = 60.0
_ALLOWED_SCHEMES: Final = frozenset({"http", "https"})
_MAXIMUM_REDIRECTS: Final = 5
_NO_REUSE: Final = httpx.Limits(max_connections=10, max_keepalive_connections=0)

_logger = get_logger(__name__)


@dataclass(slots=True)
class ResolvedSources:
    """Canonical URIs plus where each one's bytes can be read from."""

    sources: list[str] = field(default_factory=list)
    locations: dict[str, str] = field(default_factory=dict)
    staging: Path | None = None

    def cleanup(self) -> None:
        """Remove the staging directory, if one was created.

        Failures are suppressed: a leftover temp directory is untidy, but raising here would
        turn a completed ingest into a failed one over housekeeping.
        """
        if self.staging is None:
            return
        with suppress(OSError):
            shutil.rmtree(self.staging, ignore_errors=True)
        self.staging = None


def _extension_for(uri: str) -> str:
    """Return a filename suffix so the parser can recognise the format.

    Parsing dispatches on the filename, so a staged file with no extension would be parsed
    as plain text no matter what it actually is.
    """
    suffix = Path(urlparse(uri).path).suffix
    return suffix if 1 < len(suffix) <= 10 else ".txt"


def typed_source(value: str) -> dict[str, str]:
    """Classify a bare string the way the CLI and library accept them."""
    kind = "url" if value.startswith(("http://", "https://")) else "path"
    return {"type": kind, "value": value}


def _decode_inline(value: str) -> bytes:
    """Return the bytes an inline source carries."""
    with suppress(binascii.Error, ValueError):
        return base64.b64decode(value, validate=True)

    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ParseError(f"an inline source is neither base64 nor valid UTF-8: {exc}") from exc


def _within(value: str, roots: Sequence[str]) -> Path | None:
    """Return the resolved path when it sits inside a root, else ``None``.

    ``resolve()`` is what makes the comparison sound: it collapses ``..`` and follows
    symlinks, so neither traversal nor a link planted inside a root can escape it.
    """
    candidate = Path(value).resolve()
    if any(candidate.is_relative_to(Path(root).resolve()) for root in roots):
        return candidate
    return None


def _guard_path(value: str, settings: Settings) -> None:
    """Refuse a path source that falls outside ``ingestion.allowed_source_roots``.

    Raises:
        IngestionError: If path sources are unconfined on a served deployment, or the
            path resolves outside every configured root.
    """
    # CRITICAL: `value` names a file on the SERVER and arrives from whoever called
    # `POST /v1/ingest`. Unconfined, it reads `.env`, `config.yaml` or an SSH key into a
    # collection the same caller can query back. The resolve-before-compare lives in
    # `_within`. When no roots are configured the deployment is either single-operator
    # (auth off — the CLI case, left alone) or served (auth on — refused here), because a
    # control this important must not be satisfied by forgetting it.
    roots = settings.ingestion.allowed_source_roots
    if not roots:
        if settings.security.auth:
            raise IngestionError(
                "path sources are refused while security.auth is on and "
                "ingestion.allowed_source_roots is empty; list the directories this "
                "deployment may ingest from, or send the document as an inline source",
                code=ErrorCode.VALIDATION_FAILED,
                retryable=False,
            )
        return

    if _within(value, roots) is not None:
        return

    raise IngestionError(
        f"source path {value!r} is outside every directory in ingestion.allowed_source_roots",
        code=ErrorCode.VALIDATION_FAILED,
        retryable=False,
    )


def guard_archive_path(value: str, settings: Settings) -> Path:
    """Return a caller-supplied archive path confined to ``security.allowed_archive_roots``.

    Raises:
        FasterRagError: If archive paths are unconfined on a served deployment, or the
            path resolves outside every configured root.
    """
    # CRITICAL: `value` names a file on the SERVER and arrives in the body of
    # `POST /v1/admin/export` (which WRITES it) or `POST /v1/admin/import` (which reads it).
    # Unconfined, export overwrites anything the process can write — `config.yaml`, a web
    # root, an authorized_keys file — and import reads any archive on the host. The `admin`
    # scope is trusted to drop collections and drive provisioning; in a tenant-operator
    # deployment that is not the same as filesystem access. One list covers both verbs
    # because they name the same artifact in the same directory — the round trip is the
    # normal case — so two lists would be set identically and could only ever diverge by
    # mistake. Empty-means-refused-once-auth-is-on mirrors `_guard_path`, for its reasons.
    roots = settings.security.allowed_archive_roots
    if not roots:
        if settings.security.auth:
            raise FasterRagError(
                "archive paths are refused while security.auth is on and "
                "security.allowed_archive_roots is empty; list the directories this "
                "deployment may read and write archives in",
                code=ErrorCode.VALIDATION_FAILED,
                retryable=False,
            )
        return Path(value).resolve()

    confined = _within(value, roots)
    if confined is not None:
        return confined

    raise FasterRagError(
        f"archive path {value!r} is outside every directory in security.allowed_archive_roots",
        code=ErrorCode.VALIDATION_FAILED,
        retryable=False,
    )


async def _addresses(host: str) -> list[IPv4Address | IPv6Address]:
    """Resolve a hostname to every address it currently answers with."""
    loop = asyncio.get_running_loop()
    try:
        records = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ParseError(f"{host} could not be resolved: {type(exc).__name__}") from exc
    return [ip_address(str(record[4][0])) for record in records]


async def _guard_url(uri: str) -> IPv4Address | IPv6Address:
    """Refuse a URL that does not name a public http(s) endpoint, returning its address.

    Raises:
        IngestionError: If the scheme, host, or resolved address is not permitted.
    """
    # CRITICAL: this is the SSRF boundary and it must stay in front of every request,
    # including each redirect hop. `value` arrives from an API caller who is authenticated
    # but is NOT trusted with the host: without this check `POST /v1/ingest` will fetch
    # http://169.254.169.254/ (cloud instance metadata, i.e. IAM credentials), loopback
    # services the operator believes are unreachable, and anything else inside the network
    # perimeter — and then hand the body back through an ordinary query. Scheme-checking
    # alone is not enough, because a name resolving to a private address defeats it, and
    # checking only the first URL is not enough, because a redirect moves the target.
    parsed = urlparse(uri)
    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise IngestionError(
            f"source URL scheme {parsed.scheme or '(none)'!r} is not supported; "
            f"use {' or '.join(sorted(_ALLOWED_SCHEMES))}",
            code=ErrorCode.VALIDATION_FAILED,
            retryable=False,
        )

    host = parsed.hostname
    if not host:
        raise IngestionError(
            f"source URL {uri!r} names no host",
            code=ErrorCode.VALIDATION_FAILED,
            retryable=False,
        )

    checked = await _addresses(host)
    if not checked:
        raise ParseError(f"{host} resolved to no addresses")

    for address in checked:
        if not address.is_global or address.is_multicast:
            raise IngestionError(
                f"source URL {uri!r} resolves to the non-public address {address}; "
                "ingestion fetches public endpoints only",
                code=ErrorCode.VALIDATION_FAILED,
                retryable=False,
            )
    return checked[0]


def _pinned(
    uri: str, address: IPv4Address | IPv6Address
) -> tuple[httpx.URL, dict[str, str], dict[str, Any]]:
    """Dial the address that was checked, preserving ``Host`` and the TLS name."""
    # CRITICAL: this is what closes DNS rebinding, and the two extras are not optional.
    # Checking a name and then handing the name back to the connector resolves it a second
    # time, so a record with a near-zero TTL can answer public to `_guard_url` and private
    # to the socket — the guard above would be decoration. The connection is therefore
    # addressed to the literal that was checked. `Host` keeps virtual hosting working and
    # `sni_hostname` keeps TLS correct: httpcore passes it as `server_hostname`, which is
    # both the SNI sent and the name the certificate is verified against, so pinning the
    # address does NOT weaken verification to the IP. Changing either back reintroduces
    # the rebinding window or silently breaks certificate checking. Both read the IDNA
    # form (`netloc`, `raw_host`) rather than `url.host`, which hands back decoded unicode
    # for an internationalised name — exactly the two values httpx and httpcore would have
    # derived themselves, so pinning changes the destination and nothing else.
    url = httpx.URL(uri)
    headers = {"Host": url.netloc.decode("ascii")}
    extensions: dict[str, Any] = {"sni_hostname": url.raw_host.decode("ascii")}
    return url.copy_with(host=str(address)), headers, extensions


async def _read_body(response: httpx.Response, uri: str, limit: int) -> bytes:
    """Stream a response body, refusing one larger than the configured limit."""
    declared = response.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > limit:
        raise IngestionError(
            f"{uri} declares {declared} bytes, above the configured limit of {limit}",
            code=ErrorCode.PAYLOAD_TOO_LARGE,
            retryable=False,
        )

    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        total += len(chunk)
        if total > limit:
            raise IngestionError(
                f"{uri} exceeded the configured limit of {limit} bytes while downloading",
                code=ErrorCode.PAYLOAD_TOO_LARGE,
                retryable=False,
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def _fetch(uri: str, limit: int) -> bytes:
    """Download a URL, guarding every hop and refusing an over-large body."""
    try:
        # CRITICAL: redirects are followed by hand, not by httpx. `follow_redirects=True`
        # would send the later hops without passing them through `_guard_url`, so a public
        # URL could redirect to the metadata endpoint and the guard above would be decoration.
        # CRITICAL: no connection is kept alive between hops. httpcore pools connections by
        # origin, and after pinning an origin is `(scheme, IP, port)` — so two hosts sharing
        # an address would share a connection, sending the second host's request over a TLS
        # session established and verified for the first. Zero keepalive makes every hop its
        # own connection, which is what keeps `_pinned`'s SNI guarantee true per request.
        client = httpx.AsyncClient(
            timeout=_FETCH_TIMEOUT_SECONDS,
            follow_redirects=False,
            limits=_NO_REUSE,
        )
        async with client:
            current = uri
            for _ in range(_MAXIMUM_REDIRECTS + 1):
                address = await _guard_url(current)
                url, headers, extensions = _pinned(current, address)
                async with client.stream(
                    "GET", url, headers=headers, extensions=extensions
                ) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise ParseError(f"{current} redirected without a Location header")
                        # CRITICAL: a relative `Location` is joined against the URL the
                        # caller named, never against `response.url` — that one carries the
                        # pinned literal, so joining there would turn the next hop into an
                        # address and lose the name the guard and the certificate check need.
                        current = str(httpx.URL(current).join(location))
                        continue
                    response.raise_for_status()
                    return await _read_body(response, uri, limit)
            raise ParseError(f"{uri} exceeded {_MAXIMUM_REDIRECTS} redirects")
    except httpx.HTTPStatusError as exc:
        raise ParseError(f"{uri} returned HTTP {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise ParseError(f"{uri} could not be fetched: {type(exc).__name__}") from exc


async def resolve_sources(
    sources: Sequence[Mapping[str, Any]] | Sequence[str], settings: Settings
) -> ResolvedSources:
    """Resolve typed sources into canonical URIs and readable locations."""
    limit = settings.ingestion.max_document_mb * _MEGABYTE
    resolved = ResolvedSources()

    for entry in sources:
        if isinstance(entry, str):
            resolved.sources.append(entry)
            continue

        kind = str(entry.get("type", "path"))
        value = str(entry.get("value", ""))

        if kind == "path":
            _guard_path(value, settings)
            resolved.sources.append(value)
            continue
        if kind not in {"url", "inline"}:
            raise IngestionError(
                f"source type {kind!r} is not one of path, url, inline",
                code=ErrorCode.VALIDATION_FAILED,
                retryable=False,
            )

        if resolved.staging is None:
            resolved.staging = Path(tempfile.mkdtemp(prefix="fasterrag-sources-"))

        if kind == "url":
            uri = value
            data = await _fetch(value, limit)
        else:
            data = _decode_inline(value)
            # CRITICAL: the canonical URI hashes the *content*, not the position in the
            # request. Two identical inline payloads must be one document — keying on an
            # index would make re-sending the same snippet a new document every time.
            uri = f"{INLINE_SCHEME}{text_hash(value)[:32]}{_extension_for(value)}"

        if len(data) > limit:
            raise IngestionError(
                f"{uri} is {len(data)} bytes, above the configured limit of {limit}",
                code=ErrorCode.PAYLOAD_TOO_LARGE,
                retryable=False,
            )

        staged = resolved.staging / f"{hashlib.sha256(uri.encode()).hexdigest()[:16]}"
        staged = staged.with_suffix(_extension_for(uri))
        staged.write_bytes(data)

        resolved.sources.append(uri)
        resolved.locations[uri] = str(staged)

    if resolved.locations:
        _logger.info(
            "staged non-path sources",
            extra={"staged": len(resolved.locations), "total": len(resolved.sources)},
        )
    return resolved
