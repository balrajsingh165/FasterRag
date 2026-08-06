"""Every registered route's required scope, pinned.

``required_scope`` is tested in isolation elsewhere. What is asserted here is the mapping it
produces for the routes the application *actually registers* — the thing a reviewer needs to
see change. A new router inherits ``admin`` by omission, which is the safe direction, but a
route landing under an existing prefix silently inherits that prefix's scope, and a route
added to ``PUBLIC_PATHS`` becomes unauthenticated with no other signal.

Updating this table is the point. It should be a deliberate line in a diff, not a discovery.
"""

from typing import Any

from fastapi import FastAPI

from fasterrag.api.auth import PUBLIC_PATHS, required_scope
from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings

# path -> (scope or None for public, sorted methods)
EXPECTED: dict[str, tuple[str | None, list[str]]] = {
    "/healthz": (None, ["GET"]),
    "/readyz": (None, ["GET"]),
    "/openapi.json": (None, ["GET"]),
    "/metrics": ("admin", ["GET"]),
    "/v1/admin/doctor": ("admin", ["GET"]),
    "/v1/admin/export": ("admin", ["POST"]),
    "/v1/admin/import": ("admin", ["POST"]),
    "/v1/admin/provision/{tool}": ("admin", ["POST"]),
    "/v1/admin/provision/{tool}/status": ("admin", ["GET"]),
    "/v1/collections": ("collections", ["POST"]),
    "/v1/collections/{name}": ("collections", ["DELETE"]),
    "/v1/estimate": ("admin", ["POST"]),
    "/v1/ingest": ("ingest", ["POST"]),
    "/v1/ingest/{job_id}": ("ingest", ["GET"]),
    "/v1/ingest/{job_id}/documents": ("ingest", ["GET"]),
    "/v1/ingest/{job_id}/retry-dlq": ("ingest", ["POST"]),
    "/v1/query": ("query", ["POST"]),
    "/v1/replay": ("admin", ["POST"]),
    "/v1/retrieve": ("query", ["POST"]),
    "/v1/traces": ("admin", ["GET"]),
    "/v1/traces/{trace_id}": ("admin", ["GET"]),
}

_MUTATING = {"POST", "PUT", "PATCH", "DELETE"}


def registered(app: FastAPI) -> dict[str, list[str]]:
    """Return every routable path and its methods.

    Walks nested routers rather than reading ``app.routes``: this FastAPI version wraps
    included routers, so the flat list holds only what was registered on the application
    itself — which would make this whole suite pass by inspecting almost nothing.
    """
    found: dict[str, list[str]] = {}

    def walk(router: Any, depth: int = 0) -> None:
        for route in getattr(router, "routes", []):
            path = getattr(route, "path", None)
            methods = getattr(route, "methods", None)
            if path and methods:
                found[path] = sorted(method for method in methods if method != "HEAD")
            if depth >= 4:
                continue
            for attribute in ("original_router", "router", "app"):
                child = getattr(route, attribute, None)
                if child is not None and hasattr(child, "routes"):
                    walk(child, depth + 1)

    walk(app.router)
    return found


def app() -> FastAPI:
    return create_app(Settings())


def test_the_route_inventory_is_what_is_expected() -> None:
    """A route appearing or disappearing here should be a reviewed change."""
    assert set(registered(app())) == set(EXPECTED)


def test_every_route_requires_the_expected_scope() -> None:
    for path, (scope, _) in EXPECTED.items():
        assert required_scope(path) == scope, f"{path} resolves to {required_scope(path)!r}"


def test_the_methods_are_what_is_expected() -> None:
    """A GET endpoint quietly gaining a POST is a new write surface under an old scope."""
    found = registered(app())
    for path, (_, methods) in EXPECTED.items():
        assert found[path] == methods, f"{path} serves {found[path]}"


def test_only_the_three_probes_are_public() -> None:
    """Everything else exposes corpus content, costs, or control; none of it is public."""
    public = {path for path in registered(app()) if required_scope(path) is None}

    assert public == {"/healthz", "/readyz", "/openapi.json"}
    assert public == set(PUBLIC_PATHS)


def test_no_mutating_route_is_public() -> None:
    """A public write endpoint is a takeover, not a disclosure."""
    for path, methods in registered(app()).items():
        if set(methods) & _MUTATING:
            assert required_scope(path) is not None, f"{path} mutates and needs no key"


def test_every_route_resolves_to_a_known_scope() -> None:
    """A typo in the prefix table would resolve a route to a scope no key can hold."""
    from fasterrag.api.auth import ALL_SCOPES

    for path in registered(app()):
        scope = required_scope(path)
        assert scope is None or scope in ALL_SCOPES, f"{path} wants unknown scope {scope!r}"


def test_reading_a_trace_needs_admin_not_query() -> None:
    """A trace holds the query text and every retrieved chunk of whoever ran it."""
    assert required_scope("/v1/traces") == "admin"
    assert required_scope("/v1/traces/abc") == "admin"
    assert required_scope("/v1/replay") == "admin"


def test_estimating_needs_admin_because_it_reads_server_paths() -> None:
    """`/v1/estimate` parses whatever paths it is handed on the server's filesystem."""
    assert required_scope("/v1/estimate") == "admin"
