from pathlib import Path

from check_claims import check

TICKED = "- [x] TASK-0242: enforce the token budgets — ✅ 2026-10-26\n"


def build(tmp_path: Path, *, todo: str = TICKED, **documents: str) -> Path:
    """Lay out a docs/ directory holding a ledger and the named markdown files."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "todo.md").write_text(todo, encoding="utf-8")
    for name, body in documents.items():
        (docs / f"{name}.md").write_text(body, encoding="utf-8")
    return docs


def test_this_repository_makes_no_stale_denial() -> None:
    """The gate runs against the real docs in CI; it must be green here first."""
    assert check() == []


def test_the_internals_cost_governor_wording_is_caught(tmp_path: Path) -> None:
    """Verbatim from internals.md before f7c027e; services/budget.py had shipped."""
    docs = build(
        tmp_path,
        internals=(
            "| Estimation D9 | `services/estimation.py`. The runtime governor does not "
            "exist — `BUDGET_EXCEEDED` is raised by nothing (TASK-0242) | tests |\n"
        ),
    )

    assert any("TASK-0242" in entry for entry in check(docs))


def test_the_watch_flag_wording_is_caught(tmp_path: Path) -> None:
    """Verbatim from differentiators.md before e8ea244; --watch had been implemented."""
    docs = build(
        tmp_path,
        todo="- [x] TASK-0196: follow a running job — ✅ 2026-10-26\n",
        differentiators=(
            "- **CLI / API surface.** `fasterrag ingest` (the `--watch` flag is declared "
            "but **not implemented** — it prints a notice; TASK-0196).\n"
        ),
    )

    assert any("TASK-0196" in entry for entry in check(docs))


def test_the_missing_suite_wording_is_caught(tmp_path: Path) -> None:
    """Verbatim from failure-modes.md before e8ea244; the partition suite existed."""
    docs = build(
        tmp_path,
        todo="- [x] TASK-0241: script the chaos scenarios — ✅ 2026-10-27\n",
        failure_modes=(
            "> - **Three rows name a test that does not exist.** Rows 16 and 35 cite "
            '"partition injection" — none of these suites has been written (TASK-0241 '
            "for the missing chaos scenarios).\n"
        ),
    )

    assert any("TASK-0241" in entry for entry in check(docs))


def test_a_denial_citing_open_work_is_accepted(tmp_path: Path) -> None:
    """The denial is never the failure. Honest prose about unbuilt things stays green."""
    docs = build(
        tmp_path,
        todo="- [ ] TASK-0159: build the cache_only rung\n",
        quickstart="The `cache_only` degradation rung is not built (TASK-0159).\n",
    )

    assert check(docs) == []


def test_a_ticked_id_in_a_neighbouring_sentence_is_not_paired(tmp_path: Path) -> None:
    """quickstart.md's real shape: a shipped-work sentence beside a truthful denial."""
    docs = build(
        tmp_path,
        todo=TICKED + "- [ ] TASK-0159: build the cache_only rung\n",
        quickstart=(
            "Both `cost.*_token_budget` keys are enforced (TASK-0242 ✅). Known gaps: the "
            "`cache_only` degradation rung is not built (TASK-0159).\n"
        ),
    )

    assert check(docs) == []


def test_a_ticked_id_in_a_neighbouring_table_cell_is_not_paired(tmp_path: Path) -> None:
    """Two cells of one row are two unrelated statements."""
    docs = build(
        tmp_path,
        todo=TICKED + "- [ ] TASK-0159: build the cache_only rung\n",
        internals=(
            "| Degradation | the `cache_only` rung is not built (TASK-0159) "
            "| TASK-0242 shipped the budgets |\n"
        ),
    )

    assert check(docs) == []


def test_a_denial_inside_a_fenced_block_is_ignored(tmp_path: Path) -> None:
    """A phrase inside a fence is sample output, not the document asserting anything."""
    docs = build(
        tmp_path,
        guide=(
            "The budgets are enforced.\n\n"
            "```\n$ fasterrag query\nerror: the budget does not exist (TASK-0242)\n```\n\n"
            "The watcher is not implemented (TASK-0999).\n"
        ),
    )

    assert check(docs) == []


def test_the_ledger_itself_is_not_scanned(tmp_path: Path) -> None:
    """todo.md ticks tasks on the same lines that describe what they will build."""
    docs = build(
        tmp_path,
        todo=(
            "- [x] TASK-0242: the governor does not exist yet — ✅ 2026-10-26\n"
            "- [ ] TASK-0159: the cache_only rung is not built\n"
        ),
        quickstart="The `cache_only` rung is not built (TASK-0159).\n",
    )

    assert check(docs) == []


def test_an_absent_docs_directory_fails(tmp_path: Path) -> None:
    """A scan that read nothing must never look like a clean repository."""
    assert check(tmp_path / "absent") != []


def test_a_scan_matching_no_denial_fails(tmp_path: Path) -> None:
    """Vacuity guard: silently matching nothing is how a gate becomes worthless."""
    docs = build(tmp_path, quickstart="Everything described here ships today.\n")

    problems = check(docs)

    assert any("matched no denial phrasing" in entry for entry in problems)


def test_an_empty_docs_directory_fails(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()

    assert any("no documents to scan" in entry for entry in check(docs))


def test_the_message_names_the_file_the_line_and_the_phrase(tmp_path: Path) -> None:
    """A gate that fires without saying where teaches people to ignore it."""
    docs = build(
        tmp_path,
        internals="A heading.\n\nThe runtime governor does not exist (TASK-0242).\n",
    )

    (message,) = check(docs)

    assert "docs/internals.md:3" in message
    assert "does not exist" in message
    assert "TASK-0242" in message
