"""Fail when the served OpenAPI schema and ``docs/api-reference.md`` disagree.

Endpoint drift is silent in both directions. A route added without a doc entry ships an
undocumented surface that clients discover by accident; a doc entry with no route promises
something that returns 404. Neither shows up in a test suite, because both halves are
individually correct.

This compares the two sets and reports each difference with the direction it points, so the
fix is obvious from the failure. It checks *existence* only — method and path — deliberately.
Comparing parameters or response bodies would mean parsing prose, and a gate that guesses at
prose produces failures nobody trusts.

Exit codes: ``0`` when the two agree, ``1`` on any drift.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
API_REFERENCE: Final = REPOSITORY_ROOT / "docs" / "api-reference.md"

METHODS: Final = ("GET", "POST", "PUT", "PATCH", "DELETE")

# Routes that exist for machines rather than for API consumers. Documenting the schema
# endpoint inside the document it describes is circular, and the exposition endpoint is
# specified in observability.md instead.
UNDOCUMENTED_BY_DESIGN: Final[frozenset[str]] = frozenset(
    {
        "GET /openapi.json",
        "GET /metrics",
    }
)


def _normalise(path: str) -> str:
    """Return a path with parameter names erased, so ``{job_id}`` and ``{id}`` compare equal.

    The two documents name parameters independently and both spellings are correct; a gate
    that failed on the difference would be reporting a synonym as a defect.
    """
    return re.sub(r"\{[^}]*\}", "{}", path.rstrip("/") or "/")


def served_routes() -> set[str]:
    """Return every ``METHOD /path`` the application actually serves."""
    from fasterrag.api.main import create_app
    from fasterrag.config.schema import Settings

    schema: dict[str, Any] = create_app(Settings.model_validate({})).openapi()

    return {
        f"{method.upper()} {_normalise(path)}"
        for path, operations in schema.get("paths", {}).items()
        for method in operations
        if method.upper() in METHODS
    }


def documented_routes(reference: Path = API_REFERENCE) -> tuple[set[str], set[str]]:
    """Return the routes the reference presents as shipped, and those marked unbuilt.

    A specification legitimately describes endpoints ahead of their implementation — that is
    what makes it a specification. The two are distinguished by an explicit **Not yet
    implemented** marker on the line, so an unbuilt endpoint is visible to a reader rather
    than merely tolerated by this script.

    Returns:
        The shipped set and the planned set, both as ``METHOD /path``.
    """
    pattern = rf"\b({'|'.join(METHODS)})\s+(/[A-Za-z0-9_\-/{{}}.]*)"
    shipped: set[str] = set()
    planned: set[str] = set()

    for line in reference.read_text(encoding="utf-8").splitlines():
        target = planned if "not yet implemented" in line.lower() else shipped
        for method, path in re.findall(pattern, line):
            target.add(f"{method} {_normalise(path)}")

    return shipped, planned


def check(reference: Path = API_REFERENCE) -> list[str]:
    """Return one message per drifting route, empty when the two agree."""
    served = served_routes() - UNDOCUMENTED_BY_DESIGN
    shipped, planned = documented_routes(reference)
    shipped -= UNDOCUMENTED_BY_DESIGN

    return [
        *(f"served but undocumented: {route}" for route in sorted(served - shipped - planned)),
        *(f"documented as shipped but not served: {route}" for route in sorted(shipped - served)),
        # The reverse of the usual drift: something got built and the "not yet implemented"
        # note was left behind, so the reference now understates the system.
        *(
            f"marked 'not yet implemented' but served: {route}"
            for route in sorted(planned & served)
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the check, reporting every drifting route at once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    drift = check()
    if not drift:
        return 0

    sys.stderr.write("the API and docs/api-reference.md disagree about the endpoint list:\n")
    for entry in drift:
        sys.stderr.write(f"  {entry}\n")
    sys.stderr.write(
        "\nadd the route to the reference, remove it from the code, or add it to "
        "UNDOCUMENTED_BY_DESIGN with a reason.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
