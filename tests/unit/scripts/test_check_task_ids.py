from pathlib import Path

from check_task_ids import check, main

LEDGER = (
    "- [ ] TASK-0001: still open, work remains\n"
    "- [x] TASK-0002: finished and ticked — ✅ 2026-11-01\n"
)


def build(root: Path, *, todo: str = LEDGER, files: dict[str, str] | None = None) -> Path:
    docs = root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "todo.md").write_text(todo, encoding="utf-8")
    for relative, text in (files or {}).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def test_a_docstring_citing_an_undefined_id_is_caught(tmp_path: Path) -> None:
    """A reader who greps the ledger for it finds nothing and cannot tell why."""
    root = build(
        tmp_path,
        files={"src/fasterrag/thing.py": '"""Shipped under TASK-0404."""\n'},
    )

    violations = check(root)

    assert len(violations) == 1
    assert "src/fasterrag/thing.py:1:" in violations[0]
    assert "TASK-0404" in violations[0]
    assert "does not define" in violations[0]


def test_a_comment_citing_an_undefined_id_is_caught(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        files={"src/fasterrag/thing.py": '"""Module."""\n\n# CRITICAL: see TASK-0404.\nX = 1\n'},
    )

    violations = check(root)

    assert len(violations) == 1
    assert ":3:" in violations[0]


def test_a_pending_marker_on_a_ticked_id_is_caught(tmp_path: Path) -> None:
    """The marker says work remains, the ledger says it is done; one of them is wrong."""
    root = build(
        tmp_path,
        files={"src/fasterrag/thing.py": '"""Module."""\n\n# TODO: finish TASK-0002.\nX = 1\n'},
    )

    violations = check(root)

    assert len(violations) == 1
    assert "TASK-0002" in violations[0]
    assert "ticks as done" in violations[0]


def test_a_blocked_marker_on_a_ticked_id_is_caught(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        files={"tests/unit/test_thing.py": '"""Module."""\n\n# BLOCKED: TASK-0002 2026-11-01\n'},
    )

    assert len(check(root)) == 1


def test_a_continuation_line_of_a_marker_block_counts_as_the_marker(tmp_path: Path) -> None:
    """CLAUDE.md lets a marker open a block, so the id often sits on a later line."""
    root = build(
        tmp_path,
        files={
            "src/fasterrag/thing.py": (
                '"""Module."""\n\n# TODO: the accounting is not surfaced yet;\n'
                "# see TASK-0002 for the follow-up.\nX = 1\n"
            )
        },
    )

    violations = check(root)

    assert len(violations) == 1
    assert ":4:" in violations[0]


def test_a_marker_citing_an_open_id_passes(tmp_path: Path) -> None:
    """The only pending marker the repository can honestly write."""
    root = build(
        tmp_path,
        files={"src/fasterrag/thing.py": '"""Module."""\n\n# TODO: finish TASK-0001.\nX = 1\n'},
    )

    assert check(root) == []


def test_a_docstring_citing_a_ticked_id_passes(tmp_path: Path) -> None:
    """Docstrings carry history on purpose; a finished task is the right citation for it."""
    root = build(
        tmp_path,
        files={"src/fasterrag/thing.py": '"""Extracted behind an ABC under TASK-0002."""\n'},
    )

    assert check(root) == []


def test_a_critical_comment_citing_a_ticked_id_passes(tmp_path: Path) -> None:
    """A CRITICAL block names the task that established the invariant, which is done."""
    root = build(
        tmp_path,
        files={
            "src/fasterrag/thing.py": (
                '"""Module."""\n\n# CRITICAL: the ordering here was fixed by TASK-0002.\nX = 1\n'
            )
        },
    )

    assert check(root) == []


def test_a_critical_marker_ends_a_preceding_pending_block(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        files={
            "src/fasterrag/thing.py": (
                '"""Module."""\n\n# TODO: something remains.\n'
                "# CRITICAL: established by TASK-0002.\nX = 1\n"
            )
        },
    )

    assert check(root) == []


def test_a_blank_line_ends_a_pending_block(tmp_path: Path) -> None:
    """Two comment groups separated by a blank line are two comments, not one marker."""
    root = build(
        tmp_path,
        files={
            "src/fasterrag/thing.py": (
                '"""Module."""\n\n# TODO: something remains.\n\n# TASK-0002 shipped.\nX = 1\n'
            )
        },
    )

    assert check(root) == []


def test_an_id_in_a_plain_string_literal_is_not_a_citation(tmp_path: Path) -> None:
    """Test fixtures hold ids no ledger defines; failing those is firing on correct code."""
    root = build(
        tmp_path,
        files={
            "tests/unit/test_thing.py": (
                '"""Covers TASK-0001."""\n\nLEDGER = "- [ ] TASK-0404: a fixture entry"\n'
            )
        },
    )

    assert check(root) == []


def test_every_violation_is_reported_at_once(tmp_path: Path) -> None:
    """Fixing one at a time means one CI round trip per citation."""
    root = build(
        tmp_path,
        files={
            "src/fasterrag/a.py": '"""Shipped under TASK-0404."""\n',
            "src/fasterrag/b.py": '"""Module."""\n\n# TODO: finish TASK-0002.\n',
            "scripts/c.py": '"""Shipped under TASK-0505."""\n',
        },
    )

    assert len(check(root)) == 3


def test_all_three_trees_are_scanned(tmp_path: Path) -> None:
    for tree in ("src/fasterrag", "tests/unit", "scripts"):
        root = build(
            tmp_path / tree.replace("/", "_"),
            files={f"{tree}/thing.py": '"""Shipped under TASK-0404."""\n'},
        )
        assert check(root), tree


def test_a_scan_matching_no_ids_at_all_fails(tmp_path: Path) -> None:
    """The vacuity guard: a scan that resolves nothing has proved nothing."""
    root = build(tmp_path, files={"src/fasterrag/thing.py": '"""A module with no ids."""\n'})

    violations = check(root)

    assert len(violations) == 1
    assert "matched no TASK- id at all" in violations[0]


def test_a_tree_with_no_python_files_fails(tmp_path: Path) -> None:
    root = build(tmp_path)

    violations = check(root)

    assert len(violations) == 1
    assert "proved nothing" in violations[0]


def test_a_ledger_defining_no_ids_fails(tmp_path: Path) -> None:
    """Without entries every citation would resolve against an empty set and look clean."""
    root = build(
        tmp_path,
        todo="# Ledger\n\nNothing here is a checklist entry.\n",
        files={"src/fasterrag/thing.py": '"""Shipped under TASK-0002."""\n'},
    )

    violations = check(root)

    assert len(violations) == 1
    assert "nothing to resolve" in violations[0]


def test_a_missing_ledger_fails(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "thing.py").write_text('"""Shipped under TASK-0002."""\n', "utf-8")

    assert check(tmp_path)


def test_pycache_is_not_scanned(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        files={
            "src/fasterrag/thing.py": '"""Shipped under TASK-0002."""\n',
            "src/fasterrag/__pycache__/stale.py": '"""Shipped under TASK-0404."""\n',
        },
    )

    assert check(root) == []


def test_a_file_that_does_not_parse_is_not_a_crash(tmp_path: Path) -> None:
    """A half-written file in a working tree must not take the gate down with it."""
    root = build(
        tmp_path,
        files={
            "src/fasterrag/good.py": '"""Shipped under TASK-0002."""\n',
            "src/fasterrag/broken.py": "def oops(\n",
        },
    )

    assert [violation for violation in check(root) if "broken.py" not in violation] == []


def test_a_file_that_does_not_parse_is_reported_rather_than_skipped(tmp_path: Path) -> None:
    """TASK-0342: a gate that reports a file clean because it could not read it is worse."""
    root = build(
        tmp_path,
        files={
            "src/fasterrag/good.py": '"""Shipped under TASK-0002."""\n',
            "src/fasterrag/broken.py": '"""Shipped under TASK-0404."""\ndef oops(\n',
        },
    )

    violations = check(root)

    assert violations != []
    assert any(
        "src/fasterrag/broken.py" in violation and "went unchecked" in violation
        for violation in violations
    )


def test_a_file_that_does_not_tokenize_is_reported_rather_than_skipped(tmp_path: Path) -> None:
    """An unterminated literal hides every comment after it, markers included."""
    root = build(
        tmp_path,
        files={
            "src/fasterrag/good.py": '"""Shipped under TASK-0002."""\n',
            "src/fasterrag/broken.py": 'TEXT = """unterminated\n# TODO: TASK-0002 remains\n',
        },
    )

    violations = check(root)

    assert any(
        "src/fasterrag/broken.py" in violation and "tokenizing stopped" in violation
        for violation in violations
    )


def test_an_unreadable_file_fails_the_gate_rather_than_reporting_clean(tmp_path: Path) -> None:
    """The proof that matters: the exit code, which is all CI reads."""
    root = build(tmp_path, files={"src/fasterrag/broken.py": "def oops(\n"})

    assert main(["--root", str(root)]) == 1


def test_this_repository_passes_its_own_gate() -> None:
    """The gate runs against the real tree in CI; it must be green here first."""
    assert check() == []
