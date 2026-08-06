"""Request body size limiting (``security.max_request_mb``).

An unbounded request body is a memory exhaustion vector: the server reads whatever it is
sent before any handler decides it was unreasonable. ``docs/security.md`` §5 documents this
limit as enforced on every endpoint, returning ``413 PAYLOAD_TOO_LARGE``.

Two checks, because either alone can be bypassed:

* **``Content-Length``**, refused before the body is read at all. This is the case worth
  optimising — the server never allocates anything.
* **Bytes as they arrive**, because ``Content-Length`` is optional. A chunked upload omits
  it entirely, so a header-only check is a limit a client opts into.

The limit is deliberately separate from ``ingestion.max_document_mb``. That one bounds a
*document* a job was told to read, which can legitimately be far larger than any request;
this one bounds what a caller may send in one HTTP body.
"""

from __future__ import annotations

from typing import Final

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from fasterrag.api.problems import problem_response
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError

__all__ = ["BodyLimitMiddleware"]

_MEGABYTE: Final = 1024 * 1024

# Methods that carry no body worth bounding. A GET with a body is legal and meaningless, and
# refusing one on size would be a surprising failure on a request nothing reads.
_UNBOUNDED_METHODS: Final[frozenset[str]] = frozenset({"GET", "HEAD", "OPTIONS", "DELETE"})


def _too_large(limit_bytes: int) -> FasterRagError:
    """Return the refusal, naming the limit rather than only the fact of it.

    The size that was sent is deliberately not echoed: it is attacker-controlled and would
    end up in logs and problem bodies verbatim.
    """
    return FasterRagError(
        f"the request body exceeds the configured limit of "
        f"{limit_bytes // _MEGABYTE} MB (security.max_request_mb)",
        code=ErrorCode.PAYLOAD_TOO_LARGE,
        retryable=False,
    )


class BodyLimitMiddleware:
    """Refuses a request body larger than ``security.max_request_mb``."""

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        """Wrap the downstream application with the configured ceiling."""
        self.app = app
        self.limit_bytes = settings.security.max_request_mb * _MEGABYTE

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Enforce the ceiling on anything that carries a body."""
        if scope["type"] != "http" or scope.get("method", "") in _UNBOUNDED_METHODS:
            await self.app(scope, receive, send)
            return

        declared = _content_length(scope)
        if declared is not None and declared > self.limit_bytes:
            await self._refuse(scope, send)
            return

        await self.app(scope, _counted(receive, self.limit_bytes), send)

    async def _refuse(self, scope: Scope, send: Send) -> None:
        """Answer with the problem document, without reading the body."""
        response = problem_response(_too_large(self.limit_bytes), instance=scope.get("path", ""))
        await response(scope, _empty_receive, send)


def _content_length(scope: Scope) -> int | None:
    """Return the declared body size, or ``None`` when absent or unparseable."""
    for name, value in scope.get("headers", []):
        if name == b"content-length":
            try:
                return int(value)
            except ValueError:
                return None
    return None


async def _empty_receive() -> Message:
    """Return a body-less receive, for a response sent before the body was read."""
    return {"type": "http.request", "body": b"", "more_body": False}


def _counted(receive: Receive, limit_bytes: int) -> Receive:
    """Wrap ``receive`` so the body is refused the moment it passes the ceiling.

    # CRITICAL: raising here rather than returning a truncated body. A handler handed a
    # short read would parse whatever arrived and answer as though the request were
    # complete, which turns a rejected upload into a silently partial one.
    """
    seen = 0

    async def counted() -> Message:
        nonlocal seen
        message = await receive()
        if message["type"] == "http.request":
            seen += len(message.get("body", b""))
            if seen > limit_bytes:
                raise _too_large(limit_bytes)
        return message

    return counted
