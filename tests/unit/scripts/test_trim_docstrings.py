"""Pin the merge contract of `scripts/trim_docstrings.py` against docs/code-notes.md.

The file is this repository's record of *why* code is the way it is, so the property that
matters is not what a run writes but what it leaves alone. See TASK-0343.
"""

from pathlib import Path

import pytest
from trim_docstrings import DEFAULT_HEADER, Note, _parse_notes, _write_notes, check

OVER_LONG = '''
def widen() -> None:
    """Widen the window.

    The window overlaps because a token at the edge sees context on one side only,
    and a chunk boundary that splits a sentence costs more at retrieval time than
    the duplicated tokens cost at write time.
    """
    return None
'''

HAND_WRITTEN = "\n".join(
    [
        *DEFAULT_HEADER,
        "## `src/fasterrag/kept.py`",
        "",
        "### `Kept.reason`",
        "",
        "> A note no run will ever regenerate.",
        "",
        "Its docstring is four lines long now, so the trimmer never visits it again.",
        "",
    ]
)


def body(lines: list[str]) -> str:
    """Join one entry's stored lines, ignoring the trailing blank the writer strips."""
    return "\n".join(lines).strip()


def build(root: Path, **modules: str) -> Path:
    """Write a fake tree under ``src/`` and return the root it was written to."""
    for relative, source in modules.items():
        path = root / "src" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


@pytest.fixture
def rooted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the script's repository root at a temporary tree."""
    monkeypatch.setattr("trim_docstrings.REPOSITORY_ROOT", tmp_path)
    return tmp_path


def test_apply_preserves_an_entry_it_did_not_generate(rooted: Path) -> None:
    """The deliverable of TASK-0343: a run must not delete what it did not write."""
    build(rooted, **{"a.py": OVER_LONG})
    notes = rooted / "code-notes.md"
    notes.write_text(HAND_WRITTEN, encoding="utf-8")

    result = check([rooted / "src"], apply=True, notes=notes)
    parsed = _parse_notes(notes.read_text(encoding="utf-8"))

    assert result.trimmed == 1
    assert body(parsed.entries["src/fasterrag/kept.py"]["Kept.reason"]) == (
        "> A note no run will ever regenerate.\n\n"
        "Its docstring is four lines long now, so the trimmer never visits it again."
    )
    assert "widen" in parsed.entries["src/a.py"]


def test_a_run_that_generates_nothing_leaves_the_file_byte_identical(rooted: Path) -> None:
    """The vacuity guard: an empty run is the case that used to truncate the file."""
    build(rooted, **{"a.py": '''def short() -> None:\n    """Fits."""\n'''})
    notes = rooted / "code-notes.md"
    notes.write_text(HAND_WRITTEN, encoding="utf-8")

    check([rooted / "src"], apply=True, notes=notes)

    assert notes.read_text(encoding="utf-8") == HAND_WRITTEN.rstrip() + "\n"


def test_a_regenerated_entry_is_updated_in_place(rooted: Path) -> None:
    """Merging by ``module::qualname`` means re-running replaces rather than duplicates."""
    notes = rooted / "code-notes.md"
    notes.write_text(HAND_WRITTEN, encoding="utf-8")
    stale = Note("src/a.py", "widen", "Widen the window.", "The old reasoning.")

    _write_notes([stale], path=notes)
    fresh = Note("src/a.py", "widen", "Widen the window.", "The new reasoning.")
    _write_notes([fresh], path=notes)
    parsed = _parse_notes(notes.read_text(encoding="utf-8"))

    assert parsed.entries["src/a.py"]["widen"].count("The new reasoning.") == 1
    assert "The old reasoning." not in parsed.entries["src/a.py"]["widen"]
    assert "Kept.reason" in parsed.entries["src/fasterrag/kept.py"]


def test_parsing_and_rewriting_the_real_notes_file_is_lossless() -> None:
    """Round-trip the committed 6,000-line file: the merge must be able to write it back."""
    real = Path(__file__).resolve().parents[3] / "docs" / "code-notes.md"
    source = real.read_text(encoding="utf-8")
    parsed = _parse_notes(source)

    lines = list(parsed.header)
    for module in sorted(parsed.entries):
        lines.append(f"## `{module}`")
        lines.extend(parsed.preambles[module])
        for qualname, body in parsed.entries[module].items():
            lines.append(f"### `{qualname}`")
            lines.extend(body)

    assert sum(len(entries) for entries in parsed.entries.values()) > 500
    assert "\n".join(lines).rstrip() + "\n" == source


def test_a_missing_notes_file_is_created_with_the_default_header(rooted: Path) -> None:
    """First run on a fresh checkout still has to produce a readable file."""
    notes = rooted / "code-notes.md"

    _write_notes([Note("src/a.py", "widen", "Widen the window.", "Because.")], path=notes)
    parsed = _parse_notes(notes.read_text(encoding="utf-8"))

    assert parsed.header == list(DEFAULT_HEADER)
    assert body(parsed.entries["src/a.py"]["widen"]) == "> Widen the window.\n\nBecause."
