from pathlib import Path

import pytest
from check_comments import (
    PROSE_COMMENT,
    RATCHET,
    REPOSITORY_ROOT,
    UNREADABLE,
    VACUOUS,
    Finding,
    check,
    over_ratchet,
)

# CRITICAL: every fixture needs at least one marker somewhere, or the third vacuity guard
# fires and the test reads a VACUOUS finding instead of the rule it meant to exercise.
ANCHOR = """
def anchor() -> None:
    \"\"\"Give every tree one permitted marker so the vacuity guard stays quiet.\"\"\"
    # TODO: this anchor exists so the scan recognises at least one marker.
    return None
"""


def build(root: Path, **modules: str) -> Path:
    """Write a fake package, mapping a dotted-free relative path to its source."""
    for relative, source in modules.items():
        path = root / "src" / "fasterrag" / relative.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return root


def prose(findings: list[Finding]) -> list[Finding]:
    return [finding for finding in findings if finding.rule == PROSE_COMMENT]


def test_an_explanatory_comment_is_a_violation(tmp_path: Path) -> None:
    """The canonical case: prose in a comment, where no gate could ever read it."""
    root = build(
        tmp_path,
        **{
            "a.py": ANCHOR
            + """
# The window overlaps because a token at the edge sees context on one side only.
VALUE = 1
"""
        },
    )

    found = prose(check(root, trees=("src",)))

    assert len(found) == 1
    assert found[0].path == "src/fasterrag/a.py"
    assert "1 line" in found[0].detail


def test_each_of_the_three_markers_is_permitted(tmp_path: Path) -> None:
    """CLAUDE.md names exactly these three, and the gate must accept all of them."""
    root = build(
        tmp_path,
        **{
            "a.py": """
# CRITICAL: this must not change.
FIRST = 1

# TODO: finish this.
SECOND = 2

# BLOCKED: waiting on a decision.
THIRD = 3
"""
        },
    )

    assert prose(check(root, trees=("src",))) == []


def test_a_marker_opens_a_block_of_continuation_lines(tmp_path: Path) -> None:
    """The form services/autopilot.py uses: one marker, then the paragraph it introduces."""
    root = build(
        tmp_path,
        **{
            "a.py": """
def drop() -> None:
    \"\"\"Drop a scratch collection.\"\"\"
    try:
        pass
    # CRITICAL: the net is wide here on purpose, and one failure never stops the loop.
    # This runs in the cleanup path of an operation that creates indexes, so an
    # unexpected error from one drop must not leave the remaining scratch collections
    # on a production database.
    except Exception:
        raise
"""
        },
    )

    assert prose(check(root, trees=("src",))) == []


def test_a_block_ends_at_a_blank_line(tmp_path: Path) -> None:
    """Continuation means *immediately* following; a gap starts ordinary prose again."""
    root = build(
        tmp_path,
        **{
            "a.py": ANCHOR
            + """
# CRITICAL: this must not change.

# But this paragraph is attached to nothing and is ordinary prose.
VALUE = 1
"""
        },
    )

    found = prose(check(root, trees=("src",)))

    assert len(found) == 1
    assert "attached to nothing" in found[0].detail


def test_a_block_ends_at_a_line_of_code(tmp_path: Path) -> None:
    """A statement between the marker and the next comment closes the block."""
    root = build(
        tmp_path,
        **{
            "a.py": ANCHOR
            + """
# CRITICAL: this must not change.
VALUE = 1
# And this one explains the next statement instead.
OTHER = 2
"""
        },
    )

    assert len(prose(check(root, trees=("src",)))) == 1


def test_a_trailing_marker_does_not_open_a_block(tmp_path: Path) -> None:
    """A marker sharing a line with code introduces nothing, so the next line is prose."""
    root = build(
        tmp_path,
        **{
            "a.py": ANCHOR
            + """
VALUE = 1  # CRITICAL: this must not change.
# A paragraph that only looks like it continues the line above.
OTHER = 2
"""
        },
    )

    assert len(prose(check(root, trees=("src",)))) == 1


def test_contiguous_lines_are_reported_as_one_finding(tmp_path: Path) -> None:
    """A four-line explanation is one defect to fix, not four."""
    root = build(
        tmp_path,
        **{
            "a.py": ANCHOR
            + """
# One line of reasoning,
# a second line of reasoning,
# and a third.
VALUE = 1
"""
        },
    )

    found = prose(check(root, trees=("src",)))

    assert len(found) == 1
    assert "3 lines" in found[0].detail


def test_a_hash_inside_a_string_is_not_a_comment(tmp_path: Path) -> None:
    """The gate reads tokens, so a literal containing ``#`` must never match."""
    root = build(
        tmp_path,
        **{
            "a.py": ANCHOR
            + '''
TEMPLATE = "# this is data, not a comment"
DOCUMENT = """
# nor is this
"""
'''
        },
    )

    assert prose(check(root, trees=("src",))) == []


@pytest.mark.parametrize(
    "directive",
    [
        "# noqa: E501",
        "# type: ignore[arg-type]",
        "# pragma: no cover",
        "# pragma: no cover - reached only without the extra",
        "# fmt: off",
        "# ruff: noqa",
    ],
)
def test_a_tool_directive_needs_no_marker(tmp_path: Path, directive: str) -> None:
    """These address a tool rather than a reader, and have nowhere else to live."""
    root = build(tmp_path, **{"a.py": ANCHOR + f"\nVALUE = 1  {directive}\n"})

    assert prose(check(root, trees=("src",))) == []


def test_a_shebang_and_an_encoding_line_are_not_comments(tmp_path: Path) -> None:
    """The interpreter reads both, so neither is prose a reader has to be spared."""
    root = build(
        tmp_path,
        **{"a.py": "#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n" + ANCHOR},
    )

    assert prose(check(root, trees=("src",))) == []


def test_a_hash_bang_further_down_the_file_is_still_prose(tmp_path: Path) -> None:
    """Only line one can be a shebang; the same text lower down explains something."""
    root = build(tmp_path, **{"a.py": ANCHOR + "\n#!/usr/bin/env python\nVALUE = 1\n"})

    assert len(prose(check(root, trees=("src",)))) == 1


def test_a_trailing_comment_is_a_violation(tmp_path: Path) -> None:
    """CLAUDE.md bans inline comments by name, not only standalone paragraphs."""
    root = build(tmp_path, **{"a.py": ANCHOR + "\nVALUE = 1  # the default nobody tuned\n"})

    assert len(prose(check(root, trees=("src",)))) == 1


def test_a_scan_that_matches_no_file_fails(tmp_path: Path) -> None:
    """A gate that cannot tell 'clean' from 'never ran' is worse than no gate."""
    findings = check(tmp_path, trees=("src",))

    assert len(findings) == 1
    assert findings[0].rule == VACUOUS
    assert "no Python module" in findings[0].detail


def test_a_scan_that_tokenizes_no_comment_fails(tmp_path: Path) -> None:
    """Zero comments across a whole tree means the tokenizer stopped seeing them."""
    root = build(tmp_path, **{"a.py": '"""A module with no comment anywhere."""\n\nVALUE = 1\n'})

    findings = check(root, trees=("src",))

    assert len(findings) == 1
    assert findings[0].rule == VACUOUS
    assert "no comment at all" in findings[0].detail


def test_a_file_that_does_not_tokenize_is_reported_rather_than_read_as_clean(
    tmp_path: Path,
) -> None:
    """TASK-0342: a partial read used to pass as a file with nothing left in it."""
    root = build(
        tmp_path,
        **{
            "a.py": ANCHOR,
            "broken.py": 'TEXT = """unterminated\n# this prose comment is never reached\n',
        },
    )

    findings = check(root, trees=("src",))
    unreadable = [finding for finding in findings if finding.rule == UNREADABLE]

    assert [finding.path for finding in unreadable] == ["src/fasterrag/broken.py"]
    assert over_ratchet(findings) == unreadable


def test_a_scan_that_recognises_no_marker_fails(tmp_path: Path) -> None:
    """The marker pattern is the whole rule, so matching none of them means it is broken."""
    root = build(tmp_path, **{"a.py": "VALUE = 1  # pragma: no cover\n"})

    findings = check(root, trees=("src",))

    assert len(findings) == 1
    assert findings[0].rule == VACUOUS
    assert "marker" in findings[0].detail


def test_the_ratchet_absorbs_a_known_violation_but_not_a_second_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ratcheted file may not grow; the entry records a backlog, not a permission."""
    monkeypatch.setattr("check_comments.RATCHET", {f"{PROSE_COMMENT}:src/fasterrag/legacy.py": 2})
    known = Finding(PROSE_COMMENT, "src/fasterrag/legacy.py", 146, "known")
    extra = Finding(PROSE_COMMENT, "src/fasterrag/legacy.py", 999, "new")

    assert over_ratchet([known, known]) == []
    assert over_ratchet([known, known, extra]) == [extra]


def test_an_unratcheted_file_fails_on_its_first_violation() -> None:
    fresh = Finding(PROSE_COMMENT, "src/fasterrag/brand_new.py", 10, "new")

    assert over_ratchet([fresh]) == [fresh]


def test_every_ratchet_entry_still_describes_the_real_tree() -> None:
    """A stale ratchet entry hides the next violation in that file behind a dead number."""
    findings = check(REPOSITORY_ROOT)
    counted: dict[str, int] = {}
    for finding in findings:
        counted[finding.key] = counted.get(finding.key, 0) + 1

    assert {key: counted.get(key, 0) for key in RATCHET} == RATCHET


def test_the_real_tree_carries_nothing_beyond_the_ratchet() -> None:
    """The backlog is what RATCHET names and nothing else."""
    assert over_ratchet(check(REPOSITORY_ROOT)) == []


def test_the_real_tree_is_scanned_rather_than_skipped() -> None:
    """The vacuity guard must never be what a green run actually reports."""
    findings = check(REPOSITORY_ROOT)

    assert [finding for finding in findings if finding.rule == VACUOUS] == []


def test_the_form_claude_md_cites_as_correct_stays_green() -> None:
    """services/autopilot.py is the worked example; the gate may never object to it."""
    findings = check(REPOSITORY_ROOT)

    assert [f for f in findings if f.path == "src/fasterrag/services/autopilot.py"] == []
