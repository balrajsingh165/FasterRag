"""Control-plane authentication, scopes, and per-key rate limiting.

``security.auth: true`` requires ``Authorization: Bearer <key>`` on every control-plane
endpoint (``docs/security.md`` §2). Keys carry scopes — ``ingest``, ``query``,
``collections``, ``admin`` — and an endpoint refuses a key that lacks its scope.

Implemented as ASGI middleware rather than a FastAPI dependency, deliberately. A dependency
has to be attached to each route, so the failure mode of *forgetting one* is an endpoint
that is silently public. Middleware sees every request, so a new router is protected the
moment it is mounted and a gap has to be introduced on purpose.

**Failures say nothing about whether a key exists.** ``AUTH_MISSING`` means no credential
arrived, ``AUTH_INVALID`` means one did and was refused — an unknown key and a revoked key
are indistinguishable, so the endpoint cannot be used to enumerate valid keys.

Rate limiting is per key and turns on with auth, because a limit keyed on something the
caller chooses is not a limit. It is an in-process fixed window: correct for a single
server, and deliberately not claimed to work across replicas — a distributed limiter needs
shared state this deployment does not have (TASK-0124).
"""

from __future__ import annotations

import os
import secrets
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

from starlette.types import ASGIApp, Receive, Scope, Send

from fasterrag.api.problems import problem_response
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError, ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger

__all__ = [
    "ALL_SCOPES",
    "PUBLIC_PATHS",
    "ApiKey",
    "AuthMiddleware",
    "KeyRegistry",
    "RateLimiter",
    "load_keys",
    "required_scope",
]

ALL_SCOPES: Final[frozenset[str]] = frozenset({"ingest", "query", "collections", "admin"})

# CRITICAL: liveness and readiness are unauthenticated on purpose. A load balancer or
# orchestrator probes them without credentials, and a health check that needs a key reports
# the service as down whenever the key is wrong — turning a credential mistake into an
# outage. Neither endpoint reveals corpus content. `/metrics` is NOT here: it exposes
# per-endpoint volumes and costs, so it is protected like any other surface.
PUBLIC_PATHS: Final[frozenset[str]] = frozenset({"/healthz", "/readyz", "/openapi.json"})

_SCOPE_BY_PREFIX: Final[tuple[tuple[str, str], ...]] = (
    ("/v1/ingest", "ingest"),
    ("/v1/collections", "collections"),
    ("/v1/admin", "admin"),
    ("/v1/query", "query"),
    ("/v1/retrieve", "query"),
    ("/v1/traces", "admin"),
    ("/metrics", "admin"),
)

_BEARER: Final = "bearer"

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ApiKey:
    """One credential and what it may do."""

    secret: str
    scopes: frozenset[str]
    name: str = "default"
    tenant: str | None = None

    def permits(self, scope: str) -> bool:
        """Return whether this key carries ``scope``.

        ``admin`` implies every other scope: a key trusted to delete collections and drive
        provisioning is already trusted to query.
        """
        return "admin" in self.scopes or scope in self.scopes


def required_scope(path: str) -> str | None:
    """Return the scope an endpoint needs, or ``None`` when it is public."""
    if path in PUBLIC_PATHS:
        return None
    for prefix, scope in _SCOPE_BY_PREFIX:
        if path == prefix or path.startswith(f"{prefix}/"):
            return scope
    # CRITICAL: an unmapped path requires `admin` rather than defaulting to open. A new
    # router that nobody added to the table above is then refused rather than exposed, so
    # the failure of omission is a 403 instead of an unauthenticated endpoint.
    return "admin"


def _parse_scopes(raw: str) -> frozenset[str]:
    """Parse a comma-separated scope list, rejecting anything undefined.

    Raises:
        ConfigError: If a scope is not one of the four documented ones. A typo like
            ``collection`` would otherwise produce a key that silently permits nothing.
    """
    scopes = {part.strip() for part in raw.split(",") if part.strip()}
    unknown = scopes - ALL_SCOPES
    if unknown:
        raise ConfigError(
            f"unknown scope(s) {', '.join(sorted(unknown))}; "
            f"valid scopes are {', '.join(sorted(ALL_SCOPES))}"
        )
    return frozenset(scopes or ALL_SCOPES)


def load_keys(settings: Settings) -> list[ApiKey]:
    """Load API keys from the environment variable configuration names.

    Keys are separated by ``;`` and scopes within a key by ``,``, each entry optionally
    carrying scopes and then a tenant after a colon::

        FASTERRAG_API_KEY=secret-one:query,ingest:acme; secret-two:admin

    # CRITICAL: the two separators must differ. Using a comma for both makes
    # `a:query,ingest` ambiguous — it reads equally as one key with two scopes or as two
    # keys, and the parser would silently pick one. A bare secret with no scopes gets all
    # four, which is the single-operator case.

    Raises:
        ConfigError: If the variable is unset or empty while auth is on. Starting with
            authentication enabled and no keys would refuse every request, which reads as a
            broken deployment rather than a configuration mistake.
    """
    name = settings.security.api_key_env
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise ConfigError(
            f"security.auth is true but {name} is unset, so no request could ever be "
            f"authenticated; set {name} in .env or set security.auth to false"
        )

    keys: list[ApiKey] = []
    for index, entry in enumerate(part.strip() for part in raw.split(";")):
        if not entry:
            continue
        secret, _, remainder = entry.partition(":")
        scopes_raw, _, tenant = remainder.partition(":")
        if not secret:
            raise ConfigError(f"{name} entry {index + 1} has an empty key")
        keys.append(
            ApiKey(
                secret=secret,
                scopes=_parse_scopes(scopes_raw),
                name=f"key-{index + 1}",
                tenant=tenant.strip() or None,
            )
        )

    if not keys:
        raise ConfigError(f"{name} is set but contains no usable key")
    return keys


class KeyRegistry:
    """Resolves a presented secret to the key that owns it."""

    def __init__(self, keys: list[ApiKey]) -> None:
        """Index keys for lookup."""
        self._keys = keys

    def resolve(self, presented: str) -> ApiKey | None:
        """Return the matching key, or ``None``.

        # CRITICAL: every key is compared with `secrets.compare_digest`, and the loop does
        # not break early. A dict lookup or an early return leaks how much of a guess was
        # correct through response timing, which is exactly what a constant-time compare
        # exists to prevent.
        """
        found: ApiKey | None = None
        for key in self._keys:
            if secrets.compare_digest(key.secret, presented):
                found = key
        return found


@dataclass
class RateLimiter:
    """A fixed-window, per-key request limiter."""

    limit: int
    window_seconds: float = 60.0
    _seen: dict[str, deque[float]] = field(default_factory=dict)

    def allow(self, identity: str, now: float | None = None) -> tuple[bool, int]:
        """Record a request and report whether it is permitted.

        Returns:
            Whether to allow it, and the seconds a refused caller should wait — the value
            that becomes ``Retry-After``. Telling a client to back off without saying how
            long invites an immediate retry, which is the behaviour the limit exists to stop.
        """
        moment = time.monotonic() if now is None else now
        window = self._seen.setdefault(identity, deque())

        while window and moment - window[0] >= self.window_seconds:
            window.popleft()

        if len(window) >= self.limit:
            retry_after = self.window_seconds - (moment - window[0])
            return False, max(1, int(retry_after) + 1)

        window.append(moment)
        return True, 0


class AuthMiddleware:
    """Authenticates every request, checks its scope, and applies the rate limit."""

    def __init__(self, app: ASGIApp, settings: Settings) -> None:
        """Wrap the application, loading keys once at startup.

        Raises:
            ConfigError: If auth is enabled with no usable key. Failing at construction
                means a misconfigured deployment refuses to start rather than refusing
                every request afterwards.
        """
        self.app = app
        self.enabled = settings.security.auth
        self.multi_tenancy = settings.security.multi_tenancy
        self.tenant_header = settings.security.tenant_header.lower().encode("latin-1")
        self._registry: KeyRegistry | None = None
        self._limiter: RateLimiter | None = None

        if self.enabled:
            keys = load_keys(settings)
            self._registry = KeyRegistry(keys)
            self._limiter = RateLimiter(settings.security.rate_limit_per_minute)
            _logger.info(
                "control-plane authentication enabled",
                extra={
                    "keys": len(keys),
                    "rate_limit_per_minute": settings.security.rate_limit_per_minute,
                },
            )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Authenticate an HTTP request, or pass it through when auth is off."""
        if not self.enabled or scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        needed = required_scope(path)
        if needed is None:
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

        assert self._registry is not None
        key = self._registry.resolve(presented)
        if key is None:
            # Deliberately identical in wording and timing to a revoked key: an attacker must
            # not be able to tell a wrong key from an unknown one.
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
        allowed, retry_after = self._limiter.allow(key.secret)
        if not allowed:
            await self._refuse(
                scope,
                receive,
                send,
                FasterRagError(
                    "the per-key rate limit was exceeded",
                    code=ErrorCode.RATE_LIMITED,
                ),
                path,
                headers={"Retry-After": str(retry_after)},
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

        # Downstream reads the authenticated identity from here rather than re-parsing the
        # header, so exactly one place decides who the caller is and which tenant they are.
        scope["state"] = {**scope.get("state", {}), "api_key": key, "tenant": tenant}
        await self.app(scope, receive, send)

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


def _tenant_refusal(key: ApiKey, declared: str | None) -> FasterRagError | None:
    """Return the error for a tenant mismatch, or ``None`` when the request is in bounds.

    Three rules, and the third is the one that matters: a key **bound to a tenant** may only
    act as that tenant. A key with no tenant is an operator credential and may act as any,
    but must still say which — an unstated tenant would otherwise silently read and write
    the untenanted namespace while multi-tenancy is supposedly on.
    """
    if declared is None:
        return FasterRagError(
            "security.multi_tenancy is on, so every request must carry a tenant header",
            code=ErrorCode.TENANT_FORBIDDEN,
        )
    if key.tenant is not None and declared != key.tenant:
        # Deliberately does not echo the key's tenant: that would let a caller enumerate
        # which tenants exist by trying values and reading the error.
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
