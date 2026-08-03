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
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from urllib.parse import urlparse

import httpx

from fasterrag.config.schema import Settings
from fasterrag.core.identity import text_hash
from fasterrag.errors import ErrorCode, IngestionError, ParseError
from fasterrag.observability.logging import get_logger

__all__ = ["INLINE_SCHEME", "ResolvedSources", "resolve_sources", "typed_source"]

INLINE_SCHEME: Final = "inline:"

_MEGABYTE: Final = 1024 * 1024
_FETCH_TIMEOUT_SECONDS: Final = 60.0

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
    r"""Classify a bare string the way the CLI and library accept them.

    ``fasterrag ingest`` documents ``<path|url>`` and a facade caller passes plain strings;
    both arrive as untyped text. Only an explicit http(s) scheme reads as a URL — a Windows
    drive spelling like ``C:\docs`` parses as a one-letter scheme and must stay a path.
    """
    kind = "url" if value.startswith(("http://", "https://")) else "path"
    return {"type": kind, "value": value}


def _decode_inline(value: str) -> bytes:
    """Return the bytes an inline source carries.

    The value is the document content itself — the ``type`` field already said it is inline,
    so nothing is stripped from it. Base64 is tried first, falling back to UTF-8: callers
    send both — a base64 blob for binary documents, and raw text for the common case of
    pasting a snippet — and guessing wrong in either direction produces a document full of
    mojibake rather than an error.

    Raises:
        ParseError: If the payload is neither decodable form.
    """
    with suppress(binascii.Error, ValueError):
        return base64.b64decode(value, validate=True)

    try:
        return value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ParseError(f"an inline source is neither base64 nor valid UTF-8: {exc}") from exc


async def _fetch(uri: str, limit: int) -> bytes:
    """Download a URL, refusing a body larger than the configured document limit.

    The size is checked against the declared length *before* reading, and again while
    streaming, because a server may under-report or omit ``Content-Length`` entirely — and a
    limit enforced only on a header a peer controls is not a limit.

    Raises:
        IngestionError: If the body exceeds ``ingestion.max_document_mb``.
        ParseError: If the URL cannot be fetched.
    """
    try:
        client = httpx.AsyncClient(timeout=_FETCH_TIMEOUT_SECONDS, follow_redirects=True)
        async with client, client.stream("GET", uri) as response:
            response.raise_for_status()

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
    except httpx.HTTPStatusError as exc:
        raise ParseError(f"{uri} returned HTTP {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise ParseError(f"{uri} could not be fetched: {type(exc).__name__}") from exc


async def resolve_sources(
    sources: Sequence[Mapping[str, Any]] | Sequence[str], settings: Settings
) -> ResolvedSources:
    """Resolve typed sources into canonical URIs and readable locations.

    Args:
        sources: Either the API's ``{"type", "value"}`` mappings, or plain strings, which are
            treated as paths — the CLI's shape.
        settings: Validated configuration; ``ingestion.max_document_mb`` bounds every fetch.

    Returns:
        The resolution. Call :meth:`ResolvedSources.cleanup` when the job settles.

    Raises:
        IngestionError: If a source type is unrecognised, or a fetched body is too large.
        ParseError: If a URL cannot be fetched or an inline payload cannot be decoded.
    """
    limit = settings.ingestion.max_document_mb * _MEGABYTE
    resolved = ResolvedSources()

    for entry in sources:
        if isinstance(entry, str):
            resolved.sources.append(entry)
            continue

        kind = str(entry.get("type", "path"))
        value = str(entry.get("value", ""))

        if kind == "path":
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
