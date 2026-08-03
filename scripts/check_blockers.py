"""Fail when ``docs/blockers.md`` disagrees with ``docs/todo.md``.

``blockers.md`` is a view, not a second task file, and a view that drifts is worse than no
view: it either hides a blocker that is still real or presents a resolved one as outstanding,
and a maintainer reading it has no way to tell which.

Two invariants, both cheap and both silent if unchecked:

* every id cited there exists in ``todo.md`` — a typo'd id is a blocker pointing at nothing;
* every id cited there is still **open** — a ticked task listed as blocking is stale.

Deliberately one-directional. It does not require every blocked task to appear in the view,
because "blocked" is a judgement the ledger does not encode in a machine-readable way, and a
gate that guessed at it would fire on entries a human deliberately left out.

Exit codes: ``0`` when the view is faithful, ``1`` on any drift.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
TODO: Final = REPOSITORY_ROOT / "docs" / "todo.md"
BLOCKERS: Final = REPOSITORY_ROOT / "docs" / "blockers.md"

_ID = re.compile(r"\b(TASK-\d{4}|AUDIT-\d{4})\b")


def _ledger_state(todo: Path) -> tuple[set[str], set[str]]:
    """Return the open and completed ids the ledger declares."""
    open_ids: set[str] = set()
    done_ids: set[str] = set()

    for line in todo.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith(("- [ ]", "- [x]")):
            continue
        # The id being *defined* is the first on the line; later ones are cross-references.
        found = _ID.search(stripped)
        if found is None:
            continue
        (open_ids if stripped.startswith("- [ ]") else done_ids).add(found.group(0))

    return open_ids, done_ids


def check(todo: Path = TODO, blockers: Path = BLOCKERS) -> list[str]:
    """Return one message per drifting citation, empty when the view is faithful."""
    if not blockers.is_file():
        return []

    open_ids, done_ids = _ledger_state(todo)

    # A completed task is legitimate *context* — "TASK-0174 made the wait visible; this
    # decides whether it should exist" is exactly the history a reader needs. It is marked
    # with the same ✅ the ledger uses, so the claim is visible to a human rather than a
    # silent exemption in this script.
    cited: set[str] = set()
    for line in blockers.read_text(encoding="utf-8").splitlines():
        if "✅" in line:
            continue
        cited.update(_ID.findall(line))

    return [
        *(
            f"{identifier} is cited in blockers.md but appears in no todo.md entry"
            for identifier in sorted(cited - open_ids - done_ids)
        ),
        *(
            f"{identifier} is listed as blocking but is already ticked in todo.md"
            for identifier in sorted(cited & done_ids - open_ids)
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the check, reporting every drifting citation at once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    drift = check()
    if not drift:
        return 0

    sys.stderr.write("docs/blockers.md has drifted from docs/todo.md:\n")
    for entry in drift:
        sys.stderr.write(f"  {entry}\n")
    sys.stderr.write(
        "\nblockers.md is a view over todo.md. Fix the id, or delete the entry now that "
        "the blocker has cleared.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
