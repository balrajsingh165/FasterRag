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

import os
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
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
    ("/api/traces", "admin"),
    ("/api/metrics", "admin"),
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
    unknown = scopes - ALL_SCOPES - {ALL_SCOPES_TOKEN}
    if unknown:
        raise ConfigError(
            f"unknown scope(s) {', '.join(sorted(unknown))}; "
            f"valid scopes are {', '.join(sorted(ALL_SCOPES))}"
        )
    if ALL_SCOPES_TOKEN in scopes:
        return ALL_SCOPES
    return frozenset(scopes)


def load_keys(settings: Settings) -> list[ApiKey]:
    """Load API keys from the environment variable configuration names."""
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
        secret, separator, remainder = entry.partition(":")
        scopes_raw, _, tenant = remainder.partition(":")
        if not secret:
            raise ConfigError(f"{name} entry {index + 1} has an empty key")
        # CRITICAL: a bare secret has no scope field at all and still means all four, the
        # single-operator case. A field that is present and empty is a different thing and
        # is refused — the two were indistinguishable here until TASK-0327.
        scopes = _parse_scopes(scopes_raw, f"{name} entry {index + 1}") if separator else ALL_SCOPES
        keys.append(
            ApiKey(
                secret=secret,
                scopes=scopes,
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
        """Return the matching key, or ``None``."""
        # CRITICAL: compared as bytes. `secrets.compare_digest` raises `TypeError` on a
        # `str` holding a non-ASCII character, and the Authorization header is latin-1
        # decoded, so one such byte in a bearer token turned a 401 into an unhandled 500
        # for an unauthenticated caller.
        candidate = presented.encode("utf-8", "surrogatepass")
        found: ApiKey | None = None
        for key in self._keys:
            if secrets.compare_digest(key.secret.encode("utf-8", "surrogatepass"), candidate):
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
        self._registry: KeyRegistry | None = None
        self._limiter: RateLimiter | None = None

        if self.enabled:
            keys = load_keys(settings)
            self._registry = KeyRegistry(keys)
            self._limiter = create_rate_limiter(settings)
            _logger.info(
                "control-plane authentication enabled",
                extra={
                    "keys": len(keys),
                    "rate_limit_per_minute": settings.security.rate_limit_per_minute,
                    "rate_limit_backend": settings.security.rate_limit_backend,
                },
            )
            # CRITICAL: a key whose entry omits the third field parses as `tenant=None`,
            # which `_tenant_refusal` treats as an operator credential free to act for ANY
            # tenant. That is deliberate (docs/security.md), but the dangerous state is the
            # one you get by leaving a field off, so it is announced rather than inferred.
            # Never log the secret — only the positional name the parser assigned.
            unbound = [key.name for key in keys if key.tenant is None]
            if self.multi_tenancy and unbound:
                _logger.warning(
                    "keys with no tenant may act for any tenant; bind them or drop the "
                    "operator capability",
                    extra={"keys": ", ".join(unbound), "count": len(unbound)},
                )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Authenticate an HTTP request, or pass it through when auth is off."""
        if scope["type"] == "lifespan":
            # CRITICAL: the limiter may hold a connection pool, and this middleware is the
            # only thing that owns one. The lifespan scope traverses the whole middleware
            # stack and this call returns on shutdown, so it is the one place a limiter
            # built in `__init__` can be released without the application factory knowing
            # that a limiter has resources at all.
            try:
                await self.app(scope, receive, send)
            finally:
                if self._limiter is not None:
                    await self._limiter.close()
            return

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
        decision = await self._limiter.allow(key.secret)
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
    """Return the error for a tenant mismatch, or ``None`` when the request is in bounds."""
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
