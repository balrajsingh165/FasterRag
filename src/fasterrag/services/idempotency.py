"""Durable ``Idempotency-Key`` records for the mutating REST surface (TASK-0318).

A retried ``POST`` is indistinguishable from a second one. The only thing that tells them
apart is a key the client chooses, so this store remembers what each key did and replays it
rather than doing the work again.

**Scope is ``(tenant, key)``**, not ``(tenant, endpoint, key)``. A key is therefore the name
of *one request*, and reusing it for a different method, path, query string, or body is a
mistake rather than a second namespace — which is why the request fingerprint is stored
beside the response and a mismatch is refused with ``IDEMPOTENCY_KEY_REUSED`` instead of
silently returning the wrong endpoint's answer. The tenant is part of the path, so one
tenant's key can never reach another's record even if both clients pick ``"retry-1"``.

**Only a 2xx is remembered.** A refusal is not a result worth pinning to a key: a caller that
hits a 503 must be able to retry the same key and get through, and a caller that hits a 422
will fix the body and resend it — which the fingerprint would then refuse if the failure had
been stored. Errors therefore release the key.

**Concurrency is the case this exists for.** Two copies of one request arriving at once is
the normal shape of a client retry, and a store that checks-then-writes returns two 2xx
responses for them. The reservation is taken with ``O_CREAT | O_EXCL``, which is atomic on
every supported platform, so exactly one caller creates the record and the loser is refused
``IDEMPOTENCY_IN_PROGRESS`` (409, retryable) rather than executing. The loser is told to
retry because the winner's response will be replayable by then.

**Records live on disk, under the ingestion journal's root**, for the reason the journal
itself is on disk: production runs Gunicorn with several Uvicorn workers, and a per-process
dictionary would make every key hold only for the worker that happened to serve the first
request — replay protection that silently does nothing N-1 times out of N. A shared
filesystem covers one host; a deployment spread over several hosts needs a shared volume, and
that is a documented limit rather than a solved problem (``docs/api-reference.md``).

Expiry is lazy: a record older than its lifetime is taken over by the next request that
reaches it, so nothing has to sweep the directory. An in-flight reservation uses a much
shorter lease than a completed record, so a worker that died mid-request frees its key in
minutes rather than holding it for the full day.
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from secrets import token_hex
from typing import Any, Final

from fasterrag.core.identity import text_hash
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger
from fasterrag.services.journal import DEFAULT_JOURNAL_ROOT, envelope, open_envelope

__all__ = [
    "DEFAULT_TTL_SECONDS",
    "IDEMPOTENCY_HEADER",
    "IN_FLIGHT_LEASE_SECONDS",
    "MAXIMUM_KEY_LENGTH",
    "MAXIMUM_STORED_BODY_BYTES",
    "REPLAYED_HEADER",
    "IdempotencyStore",
    "Reservation",
    "StoredResponse",
    "create_idempotency_store",
    "request_fingerprint",
]

IDEMPOTENCY_HEADER: Final = "Idempotency-Key"
REPLAYED_HEADER: Final = "Idempotent-Replayed"

MAXIMUM_KEY_LENGTH: Final = 255
MAXIMUM_STORED_BODY_BYTES: Final = 256 * 1024
DEFAULT_TTL_SECONDS: Final = 24 * 60 * 60

# CRITICAL: an in-flight reservation is held under this lease, not under the record
# lifetime. A worker killed between taking the key and writing its response would otherwise
# make that key answer 409 for a full day, and the client's only remedy would be to invent a
# new key — which is exactly the duplicate request the key existed to prevent.
IN_FLIGHT_LEASE_SECONDS: Final = 15 * 60

_RECORDS_DIR: Final = "idempotency"
_UNTENANTED: Final = "_shared"
_IN_PROGRESS: Final = "in_progress"
_COMPLETED: Final = "completed"

_logger = get_logger(__name__)


def _now() -> float:
    """Return the current UTC time as a POSIX timestamp."""
    return datetime.now(tz=UTC).timestamp()


def request_fingerprint(method: str, path: str, query: str, body: bytes) -> str:
    """Return the digest identifying one request, for replay comparison.

    Fields are length-prefixed before hashing for the reason ``core.identity`` records: no
    field can then forge a boundary with a separator the next one happens to contain.
    """
    encoded = base64.b64encode(body).decode("ascii")
    parts = (method, path, query, encoded)
    return text_hash("|".join(f"{len(part)}:{part}" for part in parts))


@dataclass(frozen=True, slots=True)
class StoredResponse:
    """The response a replay returns verbatim."""

    status: int
    media_type: str
    body: bytes


@dataclass(frozen=True, slots=True)
class Reservation:
    """A claim on one key, held while the original request runs."""

    key: str
    tenant: str | None
    fingerprint: str
    owner: str
    path: Path


class IdempotencyStore:
    """Remembers what each ``Idempotency-Key`` did, so a replay repeats nothing."""

    def __init__(
        self,
        root: str | Path = DEFAULT_JOURNAL_ROOT,
        *,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        enabled: bool = True,
    ) -> None:
        """Open a store rooted beside ``root``."""
        self.root = Path(root) / _RECORDS_DIR
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled

    def begin(
        self, key: str, *, tenant: str | None, fingerprint: str
    ) -> Reservation | StoredResponse:
        """Claim ``key`` for this request, or return what it returned the first time.

        Raises:
            FasterRagError: ``IDEMPOTENCY_KEY_REUSED`` for a different request under the
                same key, ``IDEMPOTENCY_IN_PROGRESS`` while the original is still running.
        """
        path = self._path(key, tenant)
        reservation = Reservation(
            key=key, tenant=tenant, fingerprint=fingerprint, owner=token_hex(8), path=path
        )
        record = {
            "tenant": tenant,
            "fingerprint": fingerprint,
            "owner": reservation.owner,
            "state": _IN_PROGRESS,
            "created_at": _now(),
            "response": None,
        }

        try:
            self._create_exclusively(path, record)
        except FileExistsError:
            existing = self.read_record(path)
            if existing is not None and not self._expired(existing):
                return self._resolve(existing, fingerprint=fingerprint)
            _logger.info("taking over an expired idempotency record", extra={"tenant": tenant})
            self.write_record(path, record)

        return reservation

    def commit(self, reservation: Reservation, response: StoredResponse) -> None:
        """Store the response ``reservation``'s request produced, for later replays."""
        if not self._owned(reservation):
            return

        self.write_record(
            reservation.path,
            {
                "tenant": reservation.tenant,
                "fingerprint": reservation.fingerprint,
                "owner": reservation.owner,
                "state": _COMPLETED,
                "created_at": _now(),
                "response": {
                    "status": response.status,
                    "media_type": response.media_type,
                    "body": base64.b64encode(response.body).decode("ascii"),
                },
            },
        )

    def release(self, reservation: Reservation) -> None:
        """Free ``reservation``'s key, so the client may retry it immediately."""
        if not self._owned(reservation):
            return

        try:
            reservation.path.unlink(missing_ok=True)
        except OSError as exc:
            _logger.warning(
                "an idempotency reservation could not be released; it expires on its lease",
                extra={"detail": str(exc)},
            )

    def _path(self, key: str, tenant: str | None) -> Path:
        """Return the record path, hashed so a client-chosen key cannot escape the root."""
        bucket = text_hash(tenant) if tenant else _UNTENANTED
        return self.root / bucket / f"{text_hash(key)}.json"

    def _resolve(
        self, existing: dict[str, Any], *, fingerprint: str
    ) -> Reservation | StoredResponse:
        """Turn a live record into a replay, or into the refusal it deserves."""
        if existing.get("fingerprint") != fingerprint:
            raise FasterRagError(
                f"this {IDEMPOTENCY_HEADER} was already used for a different request; "
                "a key names one request, so resend with a new key",
                code=ErrorCode.IDEMPOTENCY_KEY_REUSED,
                retryable=False,
            )

        response = existing.get("response")
        if existing.get("state") != _COMPLETED or not isinstance(response, dict):
            raise FasterRagError(
                f"a request with this {IDEMPOTENCY_HEADER} is still in flight; "
                "retry shortly and the original response will be replayed",
                code=ErrorCode.IDEMPOTENCY_IN_PROGRESS,
            )

        return StoredResponse(
            status=int(response["status"]),
            media_type=str(response["media_type"]),
            body=base64.b64decode(str(response["body"])),
        )

    def _expired(self, existing: dict[str, Any]) -> bool:
        """Return whether a record may be taken over by a new request."""
        created = existing.get("created_at")
        if not isinstance(created, (int, float)):
            return True

        completed = existing.get("state") == _COMPLETED
        lease = self.ttl_seconds if completed else IN_FLIGHT_LEASE_SECONDS
        return _now() - float(created) > lease

    def _owned(self, reservation: Reservation) -> bool:
        """Return whether this reservation still holds its key."""
        existing = self.read_record(reservation.path)
        return existing is not None and existing.get("owner") == reservation.owner

    def read_record(self, path: Path) -> dict[str, Any] | None:
        """Return the record at ``path``, or ``None`` when it is gone or failed its checksum."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            _logger.warning(
                "an idempotency record could not be read and is treated as absent",
                extra={"detail": str(exc)},
            )
            return None

        return open_envelope(text)

    def _create_exclusively(self, path: Path, record: dict[str, Any]) -> None:
        """Create the record only if nobody else has, which is what serialises replays.

        Raises:
            FileExistsError: When another request already holds this key.
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise
        except OSError as exc:
            raise _unwritable(path, exc) from exc

        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(envelope(record))

    def write_record(self, path: Path, record: dict[str, Any]) -> None:
        """Replace a record atomically, so a reader never sees a half-written one."""
        temporary = path.with_name(f"{path.name}.{token_hex(4)}.tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(envelope(record), encoding="utf-8")
            os.replace(temporary, path)
        except OSError as exc:
            _discard(temporary)
            raise _unwritable(path, exc) from exc


def _discard(temporary: Path) -> None:
    """Remove a half-written record without replacing the error being reported."""
    try:
        temporary.unlink(missing_ok=True)
    except OSError as exc:
        _logger.warning(
            "a temporary idempotency record could not be removed",
            extra={"detail": str(exc)},
        )


def _unwritable(path: Path, exc: OSError) -> FasterRagError:
    """Return the refusal for a store that cannot be written."""
    return FasterRagError(
        f"the idempotency record store at {path.parent} could not be written: {exc}",
        code=ErrorCode.INTERNAL,
        retryable=True,
    )


def create_idempotency_store(root: str | Path = DEFAULT_JOURNAL_ROOT) -> IdempotencyStore:
    """Build the store that serves one application, rooted beside its journal."""
    return IdempotencyStore(root)
