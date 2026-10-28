"""internals.md's spec→module map must name modules and tests that exist.

internals.md is the first document someone rebuilding the system reads: its §2 table maps
each documented behavior to the module implementing it and the test proving it. Nothing
checked those two columns, so a renamed or deleted module left the map quietly lying — and
a map that lies is worse than no map, because it is the one artifact a rebuilder trusts.

The scope asserted here is deliberately narrower than `test_reference_parity.py`'s, and the
difference is honest rather than lazy. internals.md nowhere claims to name every module, so
a tree→doc check at module granularity would invent a promise the document does not make;
`services/benchmark.py` is a real module the map omits on purpose. What §1 *does* promise is
the layer set, so the only
exhaustiveness checked is that every top-level name under `src/fasterrag/` — package *and*
module, the two directions TASK-0255 asks for — is reached by some row, with the standalone
re-export shims exempted by name below.

Only §2 is parsed, and every assertion here is made against the parsed rows rather than the
raw document, so that deleting the table turns this file red instead of leaving a mention in
prose to satisfy it. §2's conventions are stated in the document (implementation paths
relative to `src/fasterrag/`, test paths relative to `tests/`); the prose sections mix
repo-root paths and globs with those, and a parser guessing between the two would produce
false reds, which is how a gate gets disabled.

What a resolved token does *not* prove: a bare filename is matched against every directory
the same cell named earlier, so the assertion is "this cell's tokens all name real files
somewhere the cell points at", not "each file sits under the directory its author meant".
The document gives no way to recover that intent, and a stricter rule would invent one.
"""

import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[3]
INTERNALS = ROOT / "docs" / "internals.md"
SRC = ROOT / "src" / "fasterrag"
TESTS = ROOT / "tests"

_SECTION = re.compile(r"^## 2\..*?(?=^## )", re.M | re.S)
_TOKEN = re.compile(r"`([^`]+)`")
_HEADER = "Contract (doc)"

EXEMPT_TOP_LEVEL = frozenset({"chunking.py", "evals.py", "parsing.py", "rerank.py", "retrieval.py"})
"""Root re-export shims documented in python-api.md, not in internals §2.

Each is a `from fasterrag.core... import ...` surface with no behavior of its own, so the
§2 row for the subsystem already points at the code a rebuilder needs. They are listed
here rather than given filler rows, for the same reason `services/benchmark.py` is not
forced into the map: a row that exists to satisfy a test teaches a reader nothing. A
top-level name absent from both the map and this set is a genuine gap.
"""


def _map_rows() -> list[tuple[str, str]]:
    """Return the (implementation, proof) cells of every data row in the §2 table."""
    section = _SECTION.search(INTERNALS.read_text(encoding="utf-8"))
    if section is None:
        raise AssertionError("internals.md has no '## 2.' spec→module section")
    rows: list[tuple[str, str]] = []
    for line in section.group(0).splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3 or cells[0] == _HEADER or set(cells[0]) <= {"-"}:
            continue
        rows.append((cells[1], cells[2]))
    return rows


def _is_path_token(token: str) -> bool:
    """Backticks also wrap symbols and config keys; only path shapes are checked."""
    return token.endswith((".py", "/", "/*"))


def _sibling_group(token: str) -> list[str] | None:
    """Expand the `fixed/recursive/.../late.py` shorthand for a run of sibling modules."""
    if "/" not in token:
        return None
    *heads, tail = token.split("/")
    if len(heads) < 2 or any("." in head or not head for head in heads):
        return None
    return [f"{head}.py" for head in heads] + [tail]


def _hit(token: str, prefixes: Sequence[str], base: Path, want_dir: bool) -> str | None:
    """Return the prefix under which `token` resolves, trying the shorthand last."""
    for prefix in prefixes:
        target = base / f"{prefix}{token}"
        if target.is_dir() if want_dir else target.is_file():
            return f"{prefix}{token}"
    if want_dir:
        return None
    group = _sibling_group(token)
    if group is None:
        return None
    for prefix in prefixes:
        if all((base / f"{prefix}{member}").is_file() for member in group):
            return f"{prefix}{group[-1]}"
    return None


def _resolve_cell(cell: str, base: Path) -> list[tuple[str, str | None]]:
    """Resolve each path token in one table cell, left to right."""
    prefixes = [""]
    found: list[tuple[str, str | None]] = []
    for token in _TOKEN.findall(cell):
        if not _is_path_token(token):
            continue
        want_dir = token.endswith(("/", "/*"))
        cleaned = token[:-1] if token.endswith("*") else token
        hit = _hit(cleaned, prefixes, base, want_dir)
        found.append((token, hit))
        if hit is None:
            continue
        prefix = hit if want_dir else f"{PurePosixPath(hit).parent}/"
        if prefix not in prefixes:
            prefixes.append(prefix)
    return found


def implementation_tokens() -> list[tuple[str, str | None]]:
    return [item for impl, _ in _map_rows() for item in _resolve_cell(impl, SRC)]


def proof_tokens() -> list[tuple[str, str | None]]:
    return [item for _, proof in _map_rows() for item in _resolve_cell(proof, TESTS)]


def mapped_top_level_names() -> set[str]:
    """Return the first path segment of every implementation token the map resolves.

    A package contributes its directory name, a top-level module its filename, so one set
    answers both halves of the exhaustiveness check.
    """
    return {
        PurePosixPath(resolved).parts[0]
        for _, resolved in implementation_tokens()
        if resolved is not None
    }


def top_level_packages() -> list[str]:
    return sorted(
        path.name for path in SRC.iterdir() if path.is_dir() and (path / "__init__.py").is_file()
    )


def top_level_modules() -> list[str]:
    """Return the modules sitting directly in `src/fasterrag/`, excluding `__init__.py`."""
    return sorted(
        path.name
        for path in SRC.iterdir()
        if path.is_file() and path.suffix == ".py" and path.name != "__init__.py"
    )


def test_the_map_table_was_parsed() -> None:
    """A parser matching zero rows would report perfect parity over an empty set."""
    assert len(_map_rows()) >= 20


def test_the_map_names_modules() -> None:
    """Likewise vacuous if the token pattern stopped matching implementation paths."""
    assert len(implementation_tokens()) >= 50


def test_the_map_names_tests() -> None:
    """The proof column is the half most likely to rot, so it gets its own guard."""
    assert len(proof_tokens()) >= 25


def test_every_module_in_the_map_exists() -> None:
    missing = sorted({token for token, resolved in implementation_tokens() if resolved is None})

    assert missing == [], f"internals.md §2 names modules absent from src/fasterrag/: {missing}"


def test_every_proof_in_the_map_exists() -> None:
    """A row citing a deleted test claims proof that nobody can run."""
    missing = sorted({token for token, resolved in proof_tokens() if resolved is None})

    assert missing == [], f"internals.md §2 names tests absent from tests/: {missing}"


def test_every_top_level_package_has_a_row() -> None:
    """§1 sells the layer set as navigable; a package no row reaches breaks that promise."""
    mapped = mapped_top_level_names()
    unmapped = [name for name in top_level_packages() if name not in mapped]

    assert unmapped == [], f"packages under src/fasterrag/ unreached by internals.md §2: {unmapped}"


def test_every_top_level_module_has_a_row() -> None:
    """The other direction TASK-0255 asks for: a new root module must not be invisible."""
    mapped = mapped_top_level_names()
    unmapped = [
        name for name in top_level_modules() if name not in mapped and name not in EXEMPT_TOP_LEVEL
    ]

    assert unmapped == [], f"modules in src/fasterrag/ unreached by internals.md §2: {unmapped}"


def test_the_exemption_list_stays_minimal() -> None:
    """A shim that is deleted, or given a row after all, must not leave a standing hole."""
    stale = sorted(EXEMPT_TOP_LEVEL.difference(top_level_modules()))
    redundant = sorted(EXEMPT_TOP_LEVEL.intersection(mapped_top_level_names()))

    assert stale == [], f"EXEMPT_TOP_LEVEL names modules that no longer exist: {stale}"
    assert redundant == [], f"EXEMPT_TOP_LEVEL names modules §2 now reaches: {redundant}"


def test_the_sibling_shorthand_does_not_absolve_a_mistyped_path() -> None:
    """`errors/facade.py` names nothing, and two real siblings must not make it resolve."""
    assert _sibling_group("errors/facade.py") is None
    assert _hit("errors/facade.py", [""], SRC, want_dir=False) is None
    assert _sibling_group("fixed/recursive/semantic/layout/late.py") == [
        "fixed.py",
        "recursive.py",
        "semantic.py",
        "layout.py",
        "late.py",
    ]
