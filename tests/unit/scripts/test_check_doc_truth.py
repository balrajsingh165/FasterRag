from pathlib import Path

from check_doc_truth import IMPLEMENTATION_THRESHOLD, check


def build(root: Path, *, modules: int, claude: str = "", readme: str = "") -> Path:
    package = root / "src" / "fasterrag"
    package.mkdir(parents=True)
    for index in range(modules):
        (package / f"module_{index}.py").write_text("", encoding="utf-8")
    (root / "CLAUDE.md").write_text(claude or "Build phase. Code lives in src/.\n", "utf-8")
    (root / "README.md").write_text(readme or "A RAG framework.\n", "utf-8")
    return root


def test_the_audits_actual_inversion_is_caught(tmp_path: Path) -> None:
    """CLAUDE.md claimed a documentation-only repo while src/ held ~200 modules."""
    root = build(
        tmp_path,
        modules=IMPLEMENTATION_THRESHOLD + 5,
        claude="**Current repository state: documentation only.** No implementation code "
        "exists and none may be written.\n",
    )

    violations = check(root)

    assert violations
    assert "CLAUDE.md" in violations[0]


def test_truthful_docs_pass(tmp_path: Path) -> None:
    violations = check(build(tmp_path, modules=IMPLEMENTATION_THRESHOLD + 5))

    assert violations == []


def test_the_claim_is_fair_while_the_tree_is_still_a_scaffold(tmp_path: Path) -> None:
    """Before code exists the sentence is true, and the gate must not fire on it."""
    root = build(tmp_path, modules=2, claude="This is a documentation-only repository.\n")

    assert check(root) == []


def test_the_readme_is_guarded_too(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        modules=IMPLEMENTATION_THRESHOLD + 5,
        readme="Status: no implementation code exists yet.\n",
    )

    violations = check(root)

    assert violations
    assert "README.md" in violations[0]


def test_a_phrase_inside_a_code_block_is_not_a_claim(tmp_path: Path) -> None:
    """Quoted history and example output are not the document asserting anything."""
    root = build(
        tmp_path,
        modules=IMPLEMENTATION_THRESHOLD + 5,
        claude="Earlier the file said:\n\n```\ndocumentation only\n```\n\nIt no longer does.\n",
    )

    assert check(root) == []


def test_every_contradiction_is_reported_at_once(tmp_path: Path) -> None:
    """Fixing one at a time means one CI round trip per sentence."""
    root = build(
        tmp_path,
        modules=IMPLEMENTATION_THRESHOLD + 5,
        claude="documentation only\n",
        readme="no implementation code exists\n",
    )

    assert len(check(root)) == 2


def test_the_line_number_is_reported(tmp_path: Path) -> None:
    root = build(
        tmp_path,
        modules=IMPLEMENTATION_THRESHOLD + 5,
        claude="one\ntwo\nthis is a documentation-only repository\n",
    )

    assert ":3:" in check(root)[0]


def test_this_repository_passes_its_own_gate() -> None:
    """The gate runs against the real tree in CI; it must be green here first."""
    assert check() == []
