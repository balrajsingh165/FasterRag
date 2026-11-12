"""Control-plane authentication, scopes, and per-key rate limiting.

``security.auth: true`` requires ``Authorization: Bearer <key>`` on every control-plane
endpoint (``docs/security.md`` §2). Keys carry scopes — ``ingest``, ``query``,
``collections``, ``admin`` — and an endpoint refuses a key that lacks its scope.

Implemented as ASGI middleware rather than a FastAPI dependency, deliberately. A dependency
has to be attached to each route, so the failure mode of *forgetting one* is an endpoint
that is silently public. Middleware sees every request, so a new router is protected the
moment it is mounted and a gap has to be introduced on purpose.

**An entry is ``<secret>[:<scopes>[:<tenant>]]`` and an omitted scope field is not an empty
one.** A bare secret is the single-operator case and still carries all four scopes; a field
that is written and left blank (``k:``, ``k::acme``, ``k: , ``) is refused at startup,
because defaulting it granted ``admin`` to the one spelling that could name a tenant. Ask
for every scope in writing with ``*`` — ``k:*:acme`` (TASK-0327, ``docs/security.md`` §2).

**A key is held as its SHA-256 digest only.** An entry may give the digest itself
(``sha256:<hex>``) so the key never sits in the environment, may end in ``expires=<ISO 8601>``,
and may live in ``security.api_key_file``, which is re-read when it changes and on ``SIGHUP``
where the platform has one, so a key rotates without a restart (TASK-0492).

**Failures say nothing about whether a key exists.** ``AUTH_MISSING`` means no credential
arrived, ``AUTH_INVALID`` means one did and was refused — an unknown key and a revoked key
are indistinguishable, so the endpoint cannot be used to enumerate valid keys.

Rate limiting is per key and turns on with auth, because a limit keyed on something the
caller chooses is not a limit. **Where the window lives is a deployment choice**, made by
``security.rate_limit_backend`` and implemented in :mod:`fasterrag.core.rate_limit`: the
default ``memory`` window is per process, which is correct for one replica and grants
``limit`` x N across N of them, and ``redis`` counts every replica into one shared window.
This module holds no limiting logic of its own; it owns the limiter's lifetime and asks it.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import secrets
import signal
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from starlette.types import ASGIApp, Receive, Scope, Send

from fasterrag.api.problems import problem_response
from fasterrag.config.schema import Settings
from fasterrag.core.rate_limit import RateLimiter, create_rate_limiter
from fasterrag.errors import ConfigError, ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger

__all__ = [
    "ALL_SCOPES",
    "ALL_SCOPES_TOKEN",
    "CROSS_TENANT_SCOPE",
    "HASHED_PREFIX",
    "PUBLIC_PATHS",
    "ApiKey",
    "AuthMiddleware",
    "KeyRegistry",
    "load_keys",
    "required_scope",
]

ALL_SCOPES: Final[frozenset[str]] = frozenset({"ingest", "query", "collections", "admin"})

# CRITICAL: the only way to ask for every scope in writing. A bare secret still means all
# four, but once an entry names a tenant it must also name its scopes, and this token is
# what it writes to keep them all. Without it, failing an empty scope field closed would
# leave `secret::tenant` with no replacement spelling at all.
ALL_SCOPES_TOKEN: Final[str] = "*"

# CRITICAL: under `security.multi_tenancy`, the only thing that lets a key with no tenant act
# for one. It is not in ALL_SCOPES and `*` does not grant it, and `admin` does not imply it:
# acting for every tenant has to be written into the key entry by name (TASK-0460).
CROSS_TENANT_SCOPE: Final[str] = "cross-tenant"

# CRITICAL: liveness and readiness are unauthenticated on purpose. A load balancer or
# orchestrator probes them without credentials, and a health check that needs a key reports
# the service as down whenever the key is wrong — turning a credential mistake into an
# outage. Neither endpoint reveals corpus content. `/metrics` is NOT here: it exposes
# per-endpoint volumes and costs, so it is protected like any other surface.
PUBLIC_PATHS: Final[frozenset[str]] = frozenset({"/healthz", "/readyz", "/openapi.json"})

READINESS_PATH: Final = "/readyz"

# CRITICAL: the request-state key `/readyz` reads to decide whether to show adapter detail.
# It is set by this middleware alone and only while auth is on; with auth off every caller
# may already read everything the deployment holds, so the detail is not withheld.
READINESS_DETAIL: Final = "readiness_detail"

_SCOPE_BY_PREFIX: Final[tuple[tuple[str, str], ...]] = (
    ("/v1/ingest", "ingest"),
    ("/v1/collections", "collections"),
    ("/v1/admin", "admin"),
    ("/v1/query", "query"),
    ("/v1/retrieve", "query"),
    ("/v1/traces", "admin"),
    ("/metrics", "admin"),
    ("/api/traces", "admin"),
    ("/api/metrics", "admin"),
)

MINIMUM_KEY_LENGTH: Final = 16

# CRITICAL: `.env.example` ships `FASTERRAG_API_KEY=change-me`. A copied template that was
# never edited would otherwise turn `security.auth: true` into a key every reader of this
# repository holds, so the placeholder is refused by name even though the length rule would
# refuse it too: the message has to say which mistake was made.
PLACEHOLDER_KEYS: Final[frozenset[str]] = frozenset({"change-me", "changeme", "change_me"})

_COLLECTION_ITSELF: Final = re.compile(r"^/v1/collections/[^/]+/?$")

# CRITICAL: an entry whose secret field starts with this is a SHA-256 digest of the key, not
# the key. It is the only form that keeps a credential out of the environment and the key
# file; the plaintext form stays valid so no existing key file breaks (TASK-0492).
HASHED_PREFIX: Final = "sha256:"

KEY_FILE_CHECK_SECONDS: float = 1.0

_EXPIRES: Final = re.compile(r"^(?P<head>.*?)\s+expires=(?P<when>\S+)$")
_HEX_DIGEST: Final = re.compile(r"^[0-9a-fA-F]{64}$")

_BEARER: Final = "bearer"

_logger = get_logger(__name__)


def _digest(secret: str) -> bytes:
    """Return the SHA-256 digest a key is stored and compared as."""
    return hashlib.sha256(secret.encode("utf-8", "surrogatepass")).digest()


@dataclass(frozen=True, slots=True)
class ApiKey:
    """One credential and what it may do; only the key's digest is held, never the key."""

    digest: bytes
    scopes: frozenset[str]
    name: str = "default"
    tenant: str | None = None
    expires: datetime | None = None

    @classmethod
    def from_secret(
        cls,
        secret: str,
        scopes: frozenset[str],
        *,
        name: str = "default",
        tenant: str | None = None,
        expires: datetime | None = None,
    ) -> ApiKey:
        """Build a key from its plaintext, keeping only the digest."""
        return cls(_digest(secret), scopes, name=name, tenant=tenant, expires=expires)

    @property
    def identity(self) -> str:
        """Return what the rate limiter counts this key as: its digest, never the key."""
        return self.digest.hex()

    def expired(self, now: datetime) -> bool:
        """Return whether the key's ``expires=`` moment has passed."""
        return self.expires is not None and now >= self.expires

    def permits(self, scope: str) -> bool:
        """Return whether this key carries ``scope``.

        ``admin`` implies every other scope: a key trusted to delete collections and drive
        provisioning is already trusted to query.
        """
        return "admin" in self.scopes or scope in self.scopes


def required_scope(path: str, method: str = "GET") -> str | None:
    """Return the scope an endpoint needs, or ``None`` when it is public.

    Dropping a whole collection is destructive, so ``DELETE /v1/collections/{name}`` needs
    ``admin`` although its prefix grants ``collections`` (TASK-0520).
    """
    if path in PUBLIC_PATHS:
        return None
    if method.upper() == "DELETE" and _COLLECTION_ITSELF.match(path):
        return "admin"
    for prefix, scope in _SCOPE_BY_PREFIX:
        if path == prefix or path.startswith(f"{prefix}/"):
            return scope
    # CRITICAL: an unmapped path requires `admin` rather than defaulting to open. A new
    # router that nobody added to the table above is then refused rather than exposed, so
    # the failure of omission is a 403 instead of an unauthenticated endpoint.
    return "admin"


def _parse_scopes(raw: str, where: str = "this key entry") -> frozenset[str]:
    """Parse a present scope field, rejecting an undefined or an empty one.

    An empty field used to default to every scope, so ``k::acme`` — the only way the old
    grammar could name a tenant — was a silent `admin` key. Say ``*`` to mean all.
    """
    scopes = {part.strip() for part in raw.split(",") if part.strip()}
    if not scopes:
        raise ConfigError(
            f"{where} has an empty scope field; it used to grant every scope including "
            f"'admin', so it is now refused. Write '{ALL_SCOPES_TOKEN}' for all scopes "
            f"(as in 'secret:{ALL_SCOPES_TOKEN}:tenant'), or name the scopes this key "
            f"needs ({', '.join(sorted(ALL_SCOPES))}), or drop the colon entirely for a "
            f"bare operator key"
        )
    unknown = scopes - ALL_SCOPES - {ALL_SCOPES_TOKEN, CROSS_TENANT_SCOPE}
    if unknown:
        raise ConfigError(
            f"unknown scope(s) {', '.join(sorted(unknown))}; "
            f"valid scopes are {', '.join(sorted(ALL_SCOPES | {CROSS_TENANT_SCOPE}))}"
        )
    if ALL_SCOPES_TOKEN in scopes:
        return ALL_SCOPES | (scopes & {CROSS_TENANT_SCOPE})
    return frozenset(scopes)


def _refuse_weak_secret(secret: str, where: str) -> None:
    """Refuse the shipped placeholder and any key too short to resist guessing.

    The message names the entry and the rule, never the key (TASK-0460).
    """
    if secret.strip().lower() in PLACEHOLDER_KEYS:
        raise ConfigError(
            f"{where} is the placeholder shipped in .env.example, which anyone who has read "
            f"that file can present; generate a real key, for example with "
            f'`python -c "import secrets; print(secrets.token_urlsafe(32))"`'
        )
    if len(secret) < MINIMUM_KEY_LENGTH:
        raise ConfigError(
            f"{where} is shorter than {MINIMUM_KEY_LENGTH} characters, which is short enough "
            f"to guess; generate a longer key, for example with "
            f'`python -c "import secrets; print(secrets.token_urlsafe(32))"`'
        )


def _parse_expiry(raw: str, where: str) -> datetime:
    """Parse an ``expires=`` value; a value with no zone is read as UTC."""
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ConfigError(
            f"{where} has an expires= value that is not an ISO 8601 date or date-time"
        ) from exc
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _parse_digest(raw: str, where: str) -> bytes:
    """Parse the hex digest of a ``sha256:`` entry, refusing the placeholder's digest."""
    if not _HEX_DIGEST.match(raw):
        raise ConfigError(f"{where} has a {HASHED_PREFIX} key that is not 64 hexadecimal digits")
    digest = bytes.fromhex(raw)
    if digest in {_digest(placeholder) for placeholder in PLACEHOLDER_KEYS}:
        raise ConfigError(
            f"{where} is the digest of the placeholder shipped in .env.example; hash a real key"
        )
    return digest


def _parse_entry(entry: str, where: str, name: str) -> ApiKey:
    """Parse one ``<secret>[:<scopes>[:<tenant>]][ expires=<when>]`` entry."""
    expires: datetime | None = None
    trailer = _EXPIRES.match(entry)
    if trailer is not None:
        entry = trailer.group("head").strip()
        expires = _parse_expiry(trailer.group("when"), where)
    hashed = entry[: len(HASHED_PREFIX)].lower() == HASHED_PREFIX
    body = entry[len(HASHED_PREFIX) :] if hashed else entry
    credential, separator, remainder = body.partition(":")
    scopes_raw, _, tenant = remainder.partition(":")
    if not credential:
        raise ConfigError(f"{where} has an empty key")
    # CRITICAL: a bare secret has no scope field at all and still means all four, the
    # single-operator case. A field that is present and empty is a different thing and
    # is refused — the two were indistinguishable here until TASK-0327.
    scopes = _parse_scopes(scopes_raw, where) if separator else ALL_SCOPES
    if hashed:
        digest = _parse_digest(credential, where)
    else:
        _refuse_weak_secret(credential, where)
        digest = _digest(credential)
    return ApiKey(digest, scopes, name=name, tenant=tenant.strip() or None, expires=expires)


def _read_key_file(path: Path) -> list[tuple[int, str]]:
    """Return ``(line number, entry)`` for every entry in the key file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"security.api_key_file {path} could not be read ({type(exc).__name__}); "
            f"create it or unset security.api_key_file"
        ) from exc
    entries: list[tuple[int, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            entries.extend((number, part.strip()) for part in stripped.split(";") if part.strip())
    return entries


def load_keys(settings: Settings) -> list[ApiKey]:
    """Load API keys from the environment variable and the key file configuration names."""
    env_name = settings.security.api_key_env
    key_file = settings.security.api_key_file
    raw = os.environ.get(env_name, "").strip() if env_name is not None else ""

    keys: list[ApiKey] = []
    for index, entry in enumerate(part.strip() for part in raw.split(";")):
        if entry:
            keys.append(_parse_entry(entry, f"{env_name} entry {index + 1}", f"key-{index + 1}"))
    if key_file is not None:
        path = Path(key_file)
        for number, entry in _read_key_file(path):
            keys.append(_parse_entry(entry, f"{path.name} line {number}", f"file-key-{number}"))

    if keys:
        return keys
    if key_file is None and env_name is not None and not raw:
        raise ConfigError(
            f"security.auth is true but {env_name} is unset, so no request could ever be "
            f"authenticated; set {env_name} in .env or set security.auth to false"
        )
    sources = " or ".join(
        source for source in (env_name, key_file and f"security.api_key_file {key_file}") if source
    )
    raise ConfigError(f"security.auth is true but {sources} contains no usable key")


class KeyRegistry:
    """Resolves a presented secret to the key that owns it."""

    def __init__(self, keys: list[ApiKey]) -> None:
        """Index keys for lookup."""
        self._keys = keys

    def resolve(self, presented: str) -> ApiKey | None:
        """Return the matching key, or ``None``."""
        # CRITICAL: compared as digests, which are bytes. `secrets.compare_digest` raises
        # `TypeError` on a `str` holding a non-ASCII character, and the Authorization header
        # is latin-1 decoded, so one such byte in a bearer token turned a 401 into an
        # unhandled 500 for an unauthenticated caller. Hashing first also makes a plaintext
        # entry and a `sha256:` entry the same comparison (TASK-0492).
        candidate = _digest(presented)
        found: ApiKey | None = None
        for key in self._keys:
            if secrets.compare_digest(key.digest, candidate):
                found = key
        return found


class AuthMiddleware:
    """Authenticates every request, checks its scope, and applies the rate limit."""

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        """Wrap the application, loading keys once at startup."""
        self.app = app
        self.enabled = settings.security.auth
        self.multi_tenancy = settings.security.multi_tenancy
        self.tenant_header = settings.security.tenant_header.lower().encode("latin-1")
        self._settings = settings
        self._registry: KeyRegistry | None = None
        self._limiter: RateLimiter | None = None
        key_file = settings.security.api_key_file
        self._key_file = Path(key_file) if key_file is not None and self.enabled else None
        self._key_stamp: tuple[int, int] | None = None
        self._next_key_check = 0.0
        self._reload_requested = False

        if self.enabled:
            keys = load_keys(settings)
            if self._key_file is not None:
                self._key_stamp = _stamp(self._key_file)
            self._registry = KeyRegistry(keys)
            _announce_expired_keys(keys)
            self._limiter = create_rate_limiter(settings)
            _logger.info(
                "control-plane authentication enabled",
                extra={
                    "keys": len(keys),
                    "rate_limit_per_minute": settings.security.rate_limit_per_minute,
                    "rate_limit_backend": settings.security.rate_limit_backend,
                },
            )
            if self.multi_tenancy:
                _announce_unbound_keys(keys)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Authenticate an HTTP request, or pass it through when auth is off."""
        if scope["type"] == "lifespan":
            # CRITICAL: the limiter may hold a connection pool, and this middleware is the
            # only thing that owns one. The lifespan scope traverses the whole middleware
            # stack and this call returns on shutdown, so it is the one place a limiter
            # built in `__init__` can be released without the application factory knowing
            # that a limiter has resources at all.
            hangup = self._install_reload_signal()
            try:
                await self.app(scope, receive, send)
            finally:
                if hangup is not None:
                    asyncio.get_running_loop().remove_signal_handler(hangup)
                if self._limiter is not None:
                    await self._limiter.close()
            return

        if not self.enabled or scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        await self._refresh_keys()
        path = scope.get("path", "")
        needed = required_scope(path, scope.get("method", "GET"))
        if needed is None:
            if path == READINESS_PATH:
                scope["state"] = {
                    **scope.get("state", {}),
                    READINESS_DETAIL: self._may_read_readiness_detail(scope),
                }
            await self.app(scope, receive, send)
            return

        presented = _bearer_token(scope)
        if presented is None:
            await self._refuse(
                scope,
                receive,
                send,
                FasterRagError(
                    "this endpoint requires an Authorization: Bearer <key> header",
                    code=ErrorCode.AUTH_MISSING,
                ),
                path,
            )
            return

        key = self._authenticate(presented)
        if key is None:
            await self._refuse(
                scope,
                receive,
                send,
                FasterRagError("the presented key was refused", code=ErrorCode.AUTH_INVALID),
                path,
            )
            return

        if not key.permits(needed):
            await self._refuse(
                scope,
                receive,
                send,
                FasterRagError(
                    f"this endpoint requires the {needed!r} scope",
                    code=ErrorCode.AUTH_SCOPE,
                ),
                path,
            )
            return

        assert self._limiter is not None
        decision = await self._limiter.allow(key.identity)
        if not decision.allowed:
            await self._refuse(
                scope,
                receive,
                send,
                FasterRagError(
                    "the per-key rate limit was exceeded",
                    code=ErrorCode.RATE_LIMITED,
                ),
                path,
                headers={"Retry-After": str(decision.retry_after)},
            )
            return

        tenant = key.tenant
        if self.multi_tenancy:
            declared = _header(scope, self.tenant_header)
            refusal = _tenant_refusal(key, declared)
            if refusal is not None:
                await self._refuse(scope, receive, send, refusal, path)
                return
            tenant = declared or key.tenant

        scope["state"] = {**scope.get("state", {}), "api_key": key, "tenant": tenant}
        await self.app(scope, receive, send)

    def _may_read_readiness_detail(self, scope: Scope) -> bool:
        """Return whether ``/readyz`` may show per-dependency detail to this caller.

        The probe stays public; only an ``admin`` key sees adapter detail (TASK-0460).
        A wrong key is not refused here, so a probe's status code never depends on it.
        """
        presented = _bearer_token(scope)
        if presented is None:
            return False
        key = self._authenticate(presented)
        return key is not None and key.permits("admin")

    def _authenticate(self, presented: str) -> ApiKey | None:
        """Return the key a presented secret resolves to, or ``None`` if unknown or expired.

        An expired key is refused exactly as an unknown one is; only the log names it.
        """
        if self._registry is None:
            return None
        key = self._registry.resolve(presented)
        if key is not None and key.expired(datetime.now(UTC)):
            _logger.warning("an expired api key was presented", extra={"key": key.name})
            return None
        return key

    def request_reload(self) -> None:
        """Re-read the key file on the next request, changed or not: the SIGHUP handler."""
        self._reload_requested = True
        _logger.info("api key reload requested")

    def _install_reload_signal(self) -> signal.Signals | None:
        """Route ``SIGHUP`` to :meth:`request_reload` where the platform and thread allow it.

        Windows has no ``SIGHUP``; there the key file's change is the only trigger.
        """
        hangup: signal.Signals | None = getattr(signal, "SIGHUP", None)
        if self._key_file is None or hangup is None:
            return None
        if signal.getsignal(hangup) is not signal.SIG_DFL:
            _logger.info("SIGHUP already has a handler; api keys reload on key-file change")
            return None
        try:
            asyncio.get_running_loop().add_signal_handler(hangup, self.request_reload)
        except (NotImplementedError, RuntimeError, ValueError) as exc:
            _logger.info(
                "SIGHUP cannot be handled here; api keys reload on key-file change",
                extra={"reason": type(exc).__name__},
            )
            return None
        return hangup

    async def _refresh_keys(self) -> None:
        """Reload the key file when it changed or a reload was requested.

        A file that fails to load leaves the keys already in force in place (TASK-0492).
        """
        if self._key_file is None:
            return
        now = time.monotonic()
        if not self._reload_requested and now < self._next_key_check:
            return
        self._next_key_check = now + KEY_FILE_CHECK_SECONDS
        requested, self._reload_requested = self._reload_requested, False
        try:
            stamp = await asyncio.to_thread(_stamp, self._key_file)
        except OSError as exc:
            _logger.warning(
                "the api key file could not be checked; the keys already loaded stay in force",
                extra={"path": str(self._key_file), "error": type(exc).__name__},
            )
            return
        if stamp == self._key_stamp and not requested:
            return
        try:
            keys = await asyncio.to_thread(load_keys, self._settings)
        except ConfigError as exc:
            self._key_stamp = stamp
            _logger.error(
                "the api key file changed but could not be loaded; the keys already loaded "
                "stay in force",
                extra={"path": str(self._key_file), "detail": exc.detail},
            )
            return
        self._registry = KeyRegistry(keys)
        self._key_stamp = stamp
        _logger.info(
            "api keys reloaded",
            extra={"keys": len(keys), "trigger": "signal" if requested else "file change"},
        )
        _announce_expired_keys(keys)
        if self.multi_tenancy:
            _announce_unbound_keys(keys)

    async def _refuse(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        error: FasterRagError,
        path: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Send an RFC 9457 problem document and log the refusal without the credential."""
        _logger.warning(
            "request refused at the control-plane boundary",
            extra={"path": path, "code": error.code.value},
        )
        response = problem_response(error, instance=path, headers=headers)
        await response(scope, receive, send)


def _announce_unbound_keys(keys: Sequence[ApiKey]) -> None:
    """Warn about every key with no tenant, naming the fix and never the secret.

    Without ``cross-tenant`` such a key is refused on every request; with it, it may act for
    any tenant, which is the grant worth seeing at startup (TASK-0460).
    """
    refused = [key.name for key in keys if key.tenant is None and not _crosses(key)]
    crossing = [key.name for key in keys if key.tenant is None and _crosses(key)]
    if refused:
        _logger.warning(
            "keys with no tenant are refused on every request while security.multi_tenancy "
            "is on; bind each to a tenant (secret:scopes:tenant) or, for an operator key "
            "that may act for any tenant, add the 'cross-tenant' scope",
            extra={"keys": ", ".join(refused), "count": len(refused)},
        )
    if crossing:
        _logger.warning(
            "keys holding the 'cross-tenant' scope may act for any tenant",
            extra={"keys": ", ".join(crossing), "count": len(crossing)},
        )


def _announce_expired_keys(keys: Sequence[ApiKey]) -> None:
    """Warn about every key whose expiry has already passed, by name only."""
    now = datetime.now(UTC)
    expired = [key.name for key in keys if key.expired(now)]
    if expired:
        _logger.warning(
            "keys past their expires= moment are loaded but refused on every request",
            extra={"keys": ", ".join(expired), "count": len(expired)},
        )


def _stamp(path: Path) -> tuple[int, int]:
    """Return what decides whether the key file changed: its mtime and size."""
    status = path.stat()
    return status.st_mtime_ns, status.st_size


def _crosses(key: ApiKey) -> bool:
    """Return whether ``key`` was granted the explicit cross-tenant scope."""
    return CROSS_TENANT_SCOPE in key.scopes


def _tenant_refusal(key: ApiKey, declared: str | None) -> FasterRagError | None:
    """Return the error for a tenant mismatch, or ``None`` when the request is in bounds."""
    # CRITICAL: before TASK-0460 an omitted tenant field was an operator credential for every
    # tenant. It now needs the scope by name, so leaving a field off narrows, never widens.
    if key.tenant is None and not _crosses(key):
        return FasterRagError(
            f"this key is bound to no tenant and does not hold the {CROSS_TENANT_SCOPE!r} "
            f"scope, so it may not act for a tenant",
            code=ErrorCode.TENANT_FORBIDDEN,
        )
    if declared is None:
        return FasterRagError(
            "security.multi_tenancy is on, so every request must carry a tenant header",
            code=ErrorCode.TENANT_FORBIDDEN,
        )
    if key.tenant is not None and declared != key.tenant:
        return FasterRagError(
            "this key may not act for the requested tenant",
            code=ErrorCode.TENANT_FORBIDDEN,
        )
    return None


def _header(scope: Scope, name: bytes) -> str | None:
    """Return a request header's value, or ``None`` when absent or blank."""
    headers: Sequence[tuple[bytes, bytes]] = scope.get("headers", ())
    for key, value in headers:
        if key.lower() == name:
            decoded = value.decode("latin-1").strip()
            return decoded or None
    return None


def _bearer_token(scope: Scope) -> str | None:
    """Return the presented bearer token, or ``None`` when absent or malformed."""
    headers: Sequence[tuple[bytes, bytes]] = scope.get("headers", ())
    for name, value in headers:
        if name.lower() != b"authorization":
            continue
        raw = value.decode("latin-1").strip()
        kind, _, token = raw.partition(" ")
        if kind.lower() == _BEARER and token.strip():
            return token.strip()
    return None
