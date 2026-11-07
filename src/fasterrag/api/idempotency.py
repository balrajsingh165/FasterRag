"""``Idempotency-Key`` handling for every mutating endpoint (TASK-0318).

Implemented as ASGI middleware for the reason :mod:`fasterrag.api.auth` gives for doing the
same with authentication: a dependency has to be attached to each route, so the failure mode
of forgetting one is an endpoint that is silently *not* idempotent. Middleware sees every
request, so a mutating router added later is covered the moment it is mounted and a gap has
to be introduced deliberately — which :data:`READ_ONLY_PATHS` is, and names.

The middleware holds no idempotency logic of its own. It buffers the body, asks
:class:`~fasterrag.services.idempotency.IdempotencyStore` what this key means, and either
replays what it is handed or runs the request and hands back the response. Every decision —
scope, lifetime, conflict, the concurrent-duplicate race — belongs to the service, which is
also what the CLI and library would call (``docs/structure.md``).

**It runs innermost**, nearest the router: a replay still passes through metrics, so a
duplicate that cost nothing is still visible as a request, and both auth and the correlation
id are already bound, so a record is written against the tenant the key belongs to and a
refusal carries the same trace id as everything else.

**The header is optional and inert.** A request without it is passed straight through and
touches no store, so this changes nothing for any existing client; only a request that opts
in pays a buffered body and two small file writes.
"""

from __future__ import annotations

import asyncio
from typing import Final

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from fasterrag.api.problems import problem_response
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.idempotency import (
    IDEMPOTENCY_HEADER,
    MAXIMUM_KEY_LENGTH,
    MAXIMUM_STORED_BODY_BYTES,
    REPLAYED_HEADER,
    IdempotencyStore,
    Reservation,
    StoredResponse,
    create_idempotency_store,
    request_fingerprint,
)
from fasterrag.services.journal import DEFAULT_JOURNAL_ROOT

__all__ = ["MUTATING_METHODS", "READ_ONLY_PATHS", "IdempotencyMiddleware"]

MUTATING_METHODS: Final[frozenset[str]] = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# CRITICAL: these four are POSTs because they carry a body, not because they change
# anything. `/v1/query` and `/v1/retrieve` read the index, `/v1/estimate` parses a corpus
# without embedding it, and `/v1/replay` deliberately writes neither the cache nor the trace
# store. Replaying one is free, and `/v1/query` can be a `text/event-stream` that cannot be
# captured and handed back at all. A key sent to one of them is ignored rather than refused,
# so a client that sets the header on every request keeps working.
READ_ONLY_PATHS: Final[frozenset[str]] = frozenset(
    {"/v1/query", "/v1/retrieve", "/v1/estimate", "/v1/replay"}
)

_HEADER_NAME: Final = IDEMPOTENCY_HEADER.lower().encode("ascii")
_SUCCESS_FLOOR: Final = 200
_SUCCESS_CEILING: Final = 300
_BODYLESS_STATUSES: Final[frozenset[int]] = frozenset({204, 304})


class IdempotencyMiddleware:
    """Replays a keyed mutating request instead of performing it twice."""

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the downstream application."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Serve a keyed mutating request from its stored response, or record a new one."""
        if scope["type"] != "http" or scope.get("method", "") not in MUTATING_METHODS:
            await self.app(scope, receive, send)
            return

        path = str(scope.get("path", ""))
        key = _sent_key(scope)
        if key is None or path in READ_ONLY_PATHS:
            await self.app(scope, receive, send)
            return

        store = _store_for(scope)
        if not store.enabled:
            await self.app(scope, receive, send)
            return

        try:
            _validate(key)
            body = await _buffer(receive)
            outcome = await asyncio.to_thread(
                store.begin,
                key,
                tenant=_tenant(scope),
                fingerprint=request_fingerprint(
                    str(scope.get("method", "")),
                    path,
                    bytes(scope.get("query_string", b"")).decode("latin-1"),
                    body,
                ),
            )
        except FasterRagError as exc:
            await _refuse(scope, send, exc, path)
            return

        if isinstance(outcome, StoredResponse):
            await _replay(send, outcome, key)
            return

        await self._record(scope, send, store=store, reservation=outcome, body=body, key=key)

    async def _record(
        self,
        scope: Scope,
        send: Send,
        *,
        store: IdempotencyStore,
        reservation: Reservation,
        body: bytes,
        key: str,
    ) -> None:
        """Run the request once and keep its response, releasing the key if it failed."""
        recorder = _Recorder(send, key)
        committed = False
        try:
            await self.app(scope, _rewound(body), recorder)
            if recorder.storable:
                await asyncio.to_thread(store.commit, reservation, recorder.captured())
                committed = True
        finally:
            if not committed:
                await asyncio.to_thread(store.release, reservation)


class _Recorder:
    """Passes a response on while keeping a copy small enough to replay."""

    def __init__(self, send: Send, key: str) -> None:
        """Wrap ``send`` for exactly one response."""
        self._send = send
        self._key = key
        self.status = 0
        self.media_type = ""
        self._chunks: list[bytes] = []
        self._size = 0
        self._overflowed = False

    @property
    def storable(self) -> bool:
        """Return whether this response may be replayed later."""
        return not self._overflowed and _SUCCESS_FLOOR <= self.status < _SUCCESS_CEILING

    def captured(self) -> StoredResponse:
        """Return the response as it went out."""
        return StoredResponse(
            status=self.status, media_type=self.media_type, body=b"".join(self._chunks)
        )

    async def __call__(self, message: Message) -> None:
        """Record one ASGI message and send it on, echoing the key back to the client."""
        if message["type"] == "http.response.start":
            headers = [(name, value) for name, value in message.get("headers", [])]
            self.status = int(message["status"])
            self.media_type = next(
                (value.decode("latin-1") for name, value in headers if name == b"content-type"),
                "",
            )
            message = {**message, "headers": [*headers, (_HEADER_NAME, self._key.encode("ascii"))]}
        elif message["type"] == "http.response.body":
            self._keep(bytes(message.get("body", b"")))

        await self._send(message)

    def _keep(self, chunk: bytes) -> None:
        """Keep a body chunk unless the response has outgrown the replay limit."""
        if self._overflowed:
            return

        self._size += len(chunk)
        if self._size > MAXIMUM_STORED_BODY_BYTES:
            self._overflowed = True
            self._chunks.clear()
            return

        self._chunks.append(chunk)


def _sent_key(scope: Scope) -> str | None:
    """Return the ``Idempotency-Key`` the caller sent, if any."""
    for name, value in scope.get("headers", []):
        if name == _HEADER_NAME:
            return bytes(value).decode("latin-1")
    return None


def _validate(key: str) -> None:
    """Refuse a key that is empty, oversized, or not printable ASCII.

    The offending value is never echoed back, by the rule ``api/problems.py`` records: a
    credential sent in the wrong header must not be reflected into a problem body or a log.
    """
    if not key or len(key) > MAXIMUM_KEY_LENGTH or not (key.isascii() and key.isprintable()):
        raise FasterRagError(
            f"the {IDEMPOTENCY_HEADER} header must be 1 to {MAXIMUM_KEY_LENGTH} printable "
            "ASCII characters",
            code=ErrorCode.VALIDATION_FAILED,
            retryable=False,
        )


def _tenant(scope: Scope) -> str | None:
    """Return the tenant the auth middleware resolved, which scopes every record."""
    tenant = scope.get("state", {}).get("tenant")
    return str(tenant) if tenant else None


def _store_for(scope: Scope) -> IdempotencyStore:
    """Return the application's store, built once and rooted beside its journal."""
    state = scope["app"].state
    store: IdempotencyStore | None = getattr(state, "idempotency", None)
    if store is None:
        journal = getattr(state, "journal", None)
        root = DEFAULT_JOURNAL_ROOT if journal is None else journal.root
        store = create_idempotency_store(root)
        state.idempotency = store
    return store


async def _buffer(receive: Receive) -> bytes:
    """Read the whole request body, so it can be fingerprinted and then replayed."""
    chunks: list[bytes] = []
    more_body = True
    while more_body:
        message = await receive()
        if message["type"] == "http.disconnect":
            raise FasterRagError(
                "the client disconnected before the request body arrived",
                code=ErrorCode.VALIDATION_FAILED,
                retryable=False,
            )
        chunks.append(bytes(message.get("body", b"")))
        more_body = bool(message.get("more_body", False))

    return b"".join(chunks)


def _rewound(body: bytes) -> Receive:
    """Return a receive channel that hands the already-read body to the route once."""
    delivered = False

    async def receive() -> Message:
        nonlocal delivered
        if delivered:
            return {"type": "http.disconnect"}
        delivered = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


async def _empty_receive() -> Message:
    """Return a body-less receive, for a response sent before the body was read."""
    return {"type": "http.request", "body": b"", "more_body": False}


async def _refuse(scope: Scope, send: Send, error: FasterRagError, path: str) -> None:
    """Answer with the problem document for a key the store would not accept."""
    headers = {"Retry-After": "1"} if error.code is ErrorCode.IDEMPOTENCY_IN_PROGRESS else None
    response = problem_response(error, instance=path, headers=headers)
    await response(scope, _empty_receive, send)


async def _replay(send: Send, stored: StoredResponse, key: str) -> None:
    """Return the original response, flagged so the client knows nothing ran again."""
    headers: list[tuple[bytes, bytes]] = [
        (REPLAYED_HEADER.lower().encode("ascii"), b"true"),
        (_HEADER_NAME, key.encode("ascii")),
    ]
    if stored.media_type:
        headers.append((b"content-type", stored.media_type.encode("latin-1")))
    if stored.status not in _BODYLESS_STATUSES:
        headers.append((b"content-length", str(len(stored.body)).encode("ascii")))

    await send({"type": "http.response.start", "status": stored.status, "headers": headers})
    await send({"type": "http.response.body", "body": stored.body, "more_body": False})
