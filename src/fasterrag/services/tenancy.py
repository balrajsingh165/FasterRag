"""Tenant scoping for collection names.

Collections are the one piece of tenant state that lives in the *backend* rather than in a
payload we control, so isolation cannot be a filter on a field — a vector database has no
notion of our tenants. The scope therefore lives in the name: a tenant's collections are
prefixed, and a tenant may only address names carrying its own prefix.

That makes isolation a property of the identifier rather than of a check somebody has to
remember to write. A missing filter fails open; a name that does not resolve fails closed.

Deliberately *not* a hash. An operator reading `docs` in the Qdrant dashboard needs to see
which tenant owns it, and an opaque prefix would make every support conversation start with
a lookup table.
"""

from __future__ import annotations

import re
from typing import Final

from fasterrag.errors import ErrorCode, FasterRagError

__all__ = ["SEPARATOR", "scoped_name", "unscoped_name", "visible_to"]

SEPARATOR: Final = "__"

# CRITICAL: the separator must not be legal inside a tenant id, or `a__b` and `a` owning a
# collection called `b` become the same string and one tenant can address the other's data.
_TENANT_PATTERN: Final = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")


def _validated(tenant: str) -> str:
    """Return the tenant id, refusing anything that could forge a prefix.

    Raises:
        FasterRagError: With ``TENANT_FORBIDDEN`` if the id is empty or contains the
            separator. A tenant named ``acme__x`` could otherwise address ``acme``'s
            collection ``x``.
    """
    if not _TENANT_PATTERN.match(tenant) or SEPARATOR in tenant:
        raise FasterRagError(
            "a tenant id must be alphanumeric with dots, dashes, or underscores, and may "
            f"not contain {SEPARATOR!r}",
            code=ErrorCode.TENANT_FORBIDDEN,
            retryable=False,
        )
    return tenant


def scoped_name(name: str, tenant: str | None) -> str:
    """Return the backend collection name a tenant's request addresses.

    Untenanted deployments pass ``None`` and get the name unchanged, so enabling
    multi-tenancy is the only thing that ever moves a collection.
    """
    if tenant is None:
        return name
    return f"{_validated(tenant)}{SEPARATOR}{name}"


def unscoped_name(name: str, tenant: str | None) -> str:
    """Return the name a tenant sees, with its own prefix removed.

    A tenant should never learn that prefixing exists: it named the collection ``docs`` and
    every response must call it ``docs``.
    """
    if tenant is None:
        return name
    prefix = f"{_validated(tenant)}{SEPARATOR}"
    return name.removeprefix(prefix)


def visible_to(name: str, tenant: str | None) -> bool:
    """Return whether a backend collection belongs to ``tenant``.

    An untenanted caller sees everything, which is the single-operator deployment. A tenant
    sees only its own prefix — including *not* seeing untenanted collections, because those
    predate tenancy and belong to no one who is asking.
    """
    if tenant is None:
        return True
    return name.startswith(f"{_validated(tenant)}{SEPARATOR}")
