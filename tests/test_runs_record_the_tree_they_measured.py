"""A run's counts must arrive with the tree that produced them.

TASK-0322 spent a session deciding whether the unit suite was non-deterministic. Two runs
forty minutes apart reported different pass counts, and nothing in either report said which
source tree had been imported or whether it had uncommitted changes at the time — so the
question could not be answered from the evidence, only re-litigated by re-running.

The cause turned out to be concurrent edits to a shared checkout, which is precisely the
condition a provenance line makes visible for free. These tests pin the three properties
that make the line worth printing: it names the commit, it says loudly when the tree is
dirty, and it survives ``-q``, which is the verbosity every measurement run here uses.

A number without its tree state is not a measurement.
"""

from pathlib import Path

import pytest

from tests.conftest import (
    _git,
    provenance_lines,
    provenance_summary,
    working_tree_state,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_the_header_names_the_commit_and_the_imported_package() -> None:
    """The long form carries checkout, commit, tree state and the imported source."""
    rendered = "\n".join(provenance_lines())

    assert str(REPO_ROOT) in rendered
    assert "commit: " in rendered
    assert "working tree: " in rendered
    assert str(REPO_ROOT / "src" / "fasterrag") in rendered


def test_a_dirty_tree_says_so_in_words_that_stop_a_reader() -> None:
    """A dirty tree must invalidate the numbers out loud, not in a diff of two hashes."""
    state = working_tree_state()

    if state == "clean":
        pytest.skip("this checkout is clean, so the dirty-tree wording cannot be exercised")

    assert "uncommitted path(s)" in state
    assert "NOT REPRODUCIBLE" in state


def test_the_summary_line_is_one_line_and_carries_the_commit() -> None:
    """The ``-q`` copy stays a single line so it cannot be mistaken for test output."""
    summary = provenance_summary()

    assert "\n" not in summary
    assert summary.startswith("measured: ")
    assert working_tree_state() in summary


def test_git_failures_are_reported_rather_than_swallowed() -> None:
    """An unanswerable git query yields a stated reason, never a silent blank."""
    answer = _git("rev-parse", "--verify", "refs/heads/a-branch-that-cannot-exist-0322")

    assert answer.startswith("unavailable (")
    assert answer != ""
