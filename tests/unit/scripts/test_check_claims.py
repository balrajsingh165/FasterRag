from pathlib import Path

from check_claims import (
    ATTRIBUTED_DENIALS,
    EXEMPT_DOCUMENTS,
    REPOSITORY_ROOT,
    check,
    scanned_documents,
)

TICKED = "- [x] TASK-0242: enforce the token budgets — ✅ 2026-10-26\n"

ROOT_STUBS = {"README.md": "# fasterRag\n", "CLAUDE.md": "# CLAUDE.md\n"}


def build(
    tmp_path: Path, *, todo: str = TICKED, root: dict[str, str] | None = None, **documents: str
) -> Path:
    """Lay out a repository root holding a ledger, the named docs, and the root pages."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "todo.md").write_text(todo, encoding="utf-8")
    for name, body in documents.items():
        (docs / f"{name}.md").write_text(body, encoding="utf-8")
    for name, body in {**ROOT_STUBS, **(root or {})}.items():
        (tmp_path / name).write_text(body, encoding="utf-8")
    return tmp_path


def test_this_repository_makes_no_stale_denial() -> None:
    """The gate runs against the real docs in CI; it must be green here first."""
    assert check() == []


def test_the_internals_cost_governor_wording_is_caught(tmp_path: Path) -> None:
    """Verbatim from internals.md before f7c027e; services/budget.py had shipped."""
    root = build(
        tmp_path,
        internals=(
            "| Estimation D9 | `services/estimation.py`. The runtime governor does not "
            "exist — `BUDGET_EXCEEDED` is raised by nothing (TASK-0242) | tests |\n"
        ),
    )

    assert any("TASK-0242" in entry for entry in check(root))


def test_the_watch_flag_wording_is_caught(tmp_path: Path) -> None:
    """Verbatim from differentiators.md before e8ea244; --watch had been implemented."""
    root = build(
        tmp_path,
        todo="- [x] TASK-0196: follow a running job — ✅ 2026-10-26\n",
        differentiators=(
            "- **CLI / API surface.** `fasterrag ingest` (the `--watch` flag is declared "
            "but **not implemented** — it prints a notice; TASK-0196).\n"
        ),
    )

    assert any("TASK-0196" in entry for entry in check(root))


def test_the_missing_suite_wording_is_caught(tmp_path: Path) -> None:
    """Verbatim from failure-modes.md before e8ea244; the partition suite existed."""
    root = build(
        tmp_path,
        todo="- [x] TASK-0241: script the chaos scenarios — ✅ 2026-10-27\n",
        failure_modes=(
            "> - **Three rows name a test that does not exist.** Rows 16 and 35 cite "
            '"partition injection" — none of these suites has been written (TASK-0241 '
            "for the missing chaos scenarios).\n"
        ),
    )

    assert any("TASK-0241" in entry for entry in check(root))


def test_a_denial_citing_open_work_is_accepted(tmp_path: Path) -> None:
    """The denial is never the failure. Honest prose about unbuilt things stays green."""
    root = build(
        tmp_path,
        todo="- [ ] TASK-0159: build the cache_only rung\n",
        quickstart="The `cache_only` degradation rung is not built (TASK-0159).\n",
    )

    assert check(root) == []


def test_a_ticked_id_in_a_neighbouring_sentence_is_not_paired(tmp_path: Path) -> None:
    """quickstart.md's real shape: a shipped-work sentence beside a truthful denial."""
    root = build(
        tmp_path,
        todo=TICKED + "- [ ] TASK-0159: build the cache_only rung\n",
        quickstart=(
            "Both `cost.*_token_budget` keys are enforced (TASK-0242 ✅). Known gaps: the "
            "`cache_only` degradation rung is not built (TASK-0159).\n"
        ),
    )

    assert check(root) == []


def test_a_ticked_id_in_a_neighbouring_table_cell_is_not_paired(tmp_path: Path) -> None:
    """Two cells of one row are two unrelated statements."""
    root = build(
        tmp_path,
        todo=TICKED + "- [ ] TASK-0159: build the cache_only rung\n",
        internals=(
            "| Degradation | the `cache_only` rung is not built (TASK-0159) "
            "| TASK-0242 shipped the budgets |\n"
        ),
    )

    assert check(root) == []


def test_a_denial_inside_a_fenced_block_is_ignored(tmp_path: Path) -> None:
    """A phrase inside a fence is sample output, not the document asserting anything."""
    root = build(
        tmp_path,
        guide=(
            "The budgets are enforced.\n\n"
            "```\n$ fasterrag query\nerror: the budget does not exist (TASK-0242)\n```\n\n"
            "The watcher is not implemented (TASK-0999).\n"
        ),
    )

    assert check(root) == []


def test_the_ledger_itself_is_not_scanned(tmp_path: Path) -> None:
    """todo.md ticks tasks on the same lines that describe what they will build."""
    root = build(
        tmp_path,
        todo=(
            "- [x] TASK-0242: the governor does not exist yet — ✅ 2026-10-26\n"
            "- [ ] TASK-0159: the cache_only rung is not built\n"
        ),
        quickstart="The `cache_only` rung is not built (TASK-0159).\n",
    )

    assert check(root) == []


def test_an_absent_docs_directory_fails(tmp_path: Path) -> None:
    """A scan that read nothing must never look like a clean repository."""
    assert check(tmp_path / "absent") != []


def test_a_scan_matching_no_denial_fails(tmp_path: Path) -> None:
    """Vacuity guard: silently matching nothing is how a gate becomes worthless."""
    root = build(tmp_path, quickstart="Everything described here ships today.\n")

    problems = check(root)

    assert any("matched no denial phrasing" in entry for entry in problems)


def test_an_empty_docs_directory_fails(tmp_path: Path) -> None:
    root = tmp_path
    (root / "docs").mkdir()
    for name, body in ROOT_STUBS.items():
        (root / name).write_text(body, encoding="utf-8")

    assert any("no documents to scan" in entry for entry in check(root))


def test_the_message_names_the_file_the_line_and_the_phrase(tmp_path: Path) -> None:
    """A gate that fires without saying where teaches people to ignore it."""
    root = build(
        tmp_path,
        internals="A heading.\n\nThe runtime governor does not exist (TASK-0242).\n",
    )

    (message,) = check(root)

    assert "docs/internals.md:3" in message
    assert "does not exist" in message
    assert "TASK-0242" in message


def test_the_reported_line_survives_a_long_document(tmp_path: Path) -> None:
    """Line numbers were searched for, not counted, and drifted up to 76 lines forward."""
    filler = "| a | b (TASK-0159) | c |\n" * 40
    root = build(
        tmp_path,
        todo=TICKED + "- [ ] TASK-0159: build the cache_only rung\n",
        internals=f"# Title\n\n{filler}\nThe governor does not exist (TASK-0242).\n",
    )

    (message,) = check(root)

    assert "docs/internals.md:44" in message


def test_a_denial_cell_is_paired_with_the_task_named_in_its_row(tmp_path: Path) -> None:
    """The live false negative: the subject cell names the id, the next cell makes the claim."""
    root = build(
        tmp_path,
        todo=TICKED,
        internals=(
            "| **Cost governor (TASK-0242)** | five docs describe enforcement that does "
            "not exist | Engineering |\n"
        ),
    )

    drift = check(root)

    assert any("TASK-0242" in entry for entry in drift)


def test_the_readme_is_scanned(tmp_path: Path) -> None:
    """Gap (a) of TASK-0309: the page PyPI renders went unread for the gate's whole life."""
    root = build(
        tmp_path,
        todo="- [x] TASK-0196: follow a running job — ✅ 2026-10-26\n",
        quickstart="Everything else here ships today.\n",
        root={"README.md": "The `--watch` flag is not yet implemented (TASK-0196).\n"},
    )

    assert any("README.md:1" in entry for entry in check(root))


def test_the_instruction_file_is_scanned(tmp_path: Path) -> None:
    """CLAUDE.md is loaded into every session, so a stale denial there is read most often."""
    root = build(
        tmp_path,
        todo="- [x] TASK-0196: follow a running job — ✅ 2026-10-26\n",
        quickstart="Everything else here ships today.\n",
        root={"CLAUDE.md": "Job following is unbuilt (TASK-0196).\n"},
    )

    assert any("CLAUDE.md:1" in entry for entry in check(root))


def test_a_missing_root_document_fails_rather_than_narrowing_the_scan(tmp_path: Path) -> None:
    """Losing a scan root silently is the defect; it must not look like a clean tree."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "todo.md").write_text(TICKED, encoding="utf-8")
    (docs / "quickstart.md").write_text("The rung is not built (TASK-0159).\n", encoding="utf-8")

    problems = check(tmp_path)

    assert any("README.md" in entry and "absent" in entry for entry in problems)


def test_the_cookbook_unbuilt_wording_is_caught(tmp_path: Path) -> None:
    """Verbatim from cookbook.md before 650943b; the governor had shipped three months earlier."""
    root = build(
        tmp_path,
        cookbook=(
            "> ⚠️ And `cost.per_tenant_token_budget` is **not enforced by anything** — "
            "the runtime governor is unbuilt, so setting it raises `ConfigError` at "
            "startup (TASK-0242).\n"
        ),
    )

    assert any("TASK-0242" in entry for entry in check(root))


def test_the_refuse_startup_until_wording_is_caught(tmp_path: Path) -> None:
    """Verbatim from cookbook.md's capability table before the governor shipped."""
    root = build(
        tmp_path,
        cookbook=(
            "| Cost | `embeddings.tiering` | The runtime budget keys refuse startup "
            "until the governor ships (TASK-0242) |\n"
        ),
    )

    assert any("TASK-0242" in entry for entry in check(root))


def test_the_pending_task_wording_is_caught(tmp_path: Path) -> None:
    """README.md's shape: a denial with no denial verb at all."""
    root = build(
        tmp_path,
        integrations="Milvus/Weaviate/Pinecone adapters pending (TASK-0242).\n",
    )

    assert any("TASK-0242" in entry for entry in check(root))


def test_the_until_it_lands_wording_is_caught(tmp_path: Path) -> None:
    """verification.md's shape, stale for three months with the old pattern set."""
    root = build(
        tmp_path,
        verification="REST provisioning is TASK-0242 — verify via CLI until it lands.\n",
    )

    assert any("TASK-0242" in entry for entry in check(root))


def test_an_attributive_unbuilt_is_not_a_denial(tmp_path: Path) -> None:
    """release-readiness.md:46 calls a BUILT cap 'the most dangerous kind of unbuilt feature'."""
    root = build(
        tmp_path,
        quickstart="The rung is not built (TASK-0159).\n",
        todo=TICKED + "- [ ] TASK-0159: build the cache_only rung\n",
        release_readiness=(
            "| **Cost governor (TASK-0242) — ✅ built** | An unbuilt spend cap is the "
            "most dangerous kind of unbuilt feature | Engineering |\n"
        ),
    )

    assert check(root) == []


def test_refusing_to_start_without_an_until_is_not_a_denial(tmp_path: Path) -> None:
    """Four documents use 'refuses to start' for fail-fast behaviour that shipped."""
    root = build(
        tmp_path,
        quickstart="The rung is not built (TASK-0159).\n",
        todo=TICKED + "- [ ] TASK-0159: build the cache_only rung\n",
        security="Enabling auth with no usable key **refuses to start** (TASK-0242 ✅).\n",
    )

    assert check(root) == []


def test_the_changelog_is_not_scanned(tmp_path: Path) -> None:
    """A dated release entry is frozen history, true of the repository at its heading."""
    root = build(
        tmp_path,
        quickstart="The rung is not built (TASK-0159).\n",
        todo=TICKED + "- [ ] TASK-0159: build the cache_only rung\n",
        CHANGELOG="- Build slice S7: `redis` raises until TASK-0242 ships it.\n",
    )

    assert check(root) == []
    assert all(path.name != "CHANGELOG.md" for path in scanned_documents(root))


def test_the_attribution_allowlist_still_names_live_sentences() -> None:
    """An entry kept after its sentence is rewritten is a standing hole nobody sees."""
    stale = [
        f"{document}: {fragment!r}"
        for document, fragment in ATTRIBUTED_DENIALS
        if fragment not in (REPOSITORY_ROOT / document).read_text(encoding="utf-8")
    ]

    assert stale == [], f"ATTRIBUTED_DENIALS names prose that no longer exists: {stale}"


def test_the_exemptions_stay_small_and_real() -> None:
    """Both lists are escape hatches; growth is the thing worth noticing."""
    missing = sorted(name for name in EXEMPT_DOCUMENTS if not (REPOSITORY_ROOT / name).is_file())

    assert missing == [], f"EXEMPT_DOCUMENTS names absent files: {missing}"
    assert len(EXEMPT_DOCUMENTS) <= 2
    assert len(ATTRIBUTED_DENIALS) <= 3
