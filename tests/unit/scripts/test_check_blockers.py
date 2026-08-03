from pathlib import Path

from check_blockers import check


def build(tmp_path: Path, todo: str, blockers: str) -> tuple[Path, Path]:
    todo_path = tmp_path / "todo.md"
    blockers_path = tmp_path / "blockers.md"
    todo_path.write_text(todo, encoding="utf-8")
    blockers_path.write_text(blockers, encoding="utf-8")
    return todo_path, blockers_path


def test_this_repository_has_a_faithful_view() -> None:
    """The gate runs against the real files in CI; it must be green here first."""
    assert check() == []


def test_an_open_blocker_is_accepted(tmp_path: Path) -> None:
    todo, blockers = build(tmp_path, "- [ ] TASK-0001: a thing\n", "TASK-0001 blocks release\n")

    assert check(todo, blockers) == []


def test_an_id_that_exists_nowhere_is_reported(tmp_path: Path) -> None:
    """A typo'd id is a blocker pointing at nothing."""
    todo, blockers = build(tmp_path, "- [ ] TASK-0001: a thing\n", "TASK-9999 blocks release\n")

    assert any("TASK-9999" in entry for entry in check(todo, blockers))


def test_a_resolved_blocker_left_behind_is_reported(tmp_path: Path) -> None:
    """Presenting a done task as outstanding wastes exactly the attention this file buys."""
    todo, blockers = build(
        tmp_path, "- [x] TASK-0001: a thing — ✅ 2026-01-01\n", "TASK-0001 blocks release\n"
    )

    drift = check(todo, blockers)

    assert any("already ticked" in entry for entry in drift)


def test_a_completed_task_cited_as_context_is_allowed(tmp_path: Path) -> None:
    """History is what a reader needs; the ✅ makes the claim visible rather than implicit."""
    todo, blockers = build(
        tmp_path,
        "- [x] TASK-0001: a thing — ✅ 2026-01-01\n- [ ] TASK-0002: decide\n",
        "TASK-0002 is open. TASK-0001 ✅ made it visible but did not decide it.\n",
    )

    assert check(todo, blockers) == []


def test_a_missing_view_is_not_a_failure(tmp_path: Path) -> None:
    """The view is optional; only an inaccurate one is a problem."""
    todo = tmp_path / "todo.md"
    todo.write_text("- [ ] TASK-0001: a thing\n", encoding="utf-8")

    assert check(todo, tmp_path / "absent.md") == []
