"""Pin `scripts/check_unbuilt_claims.py`: the gate that asks the tree instead of the ledger.

Every fixture below is a sentence somebody actually wrote in this repository, because the
gate exists to catch the umbrella-id hole that produced findings on three days in six.
"""

from pathlib import Path

import pytest
from check_unbuilt_claims import (
    ADAPTER_DENIAL,
    COLUMN_DENIAL,
    LABEL_DENIAL,
    LAYER_DENIAL,
    PROCESS_SURFACE,
    RATCHET,
    RUNG_DENIAL,
    STRUCTURE_ABSENT,
    STRUCTURE_PENDING,
    VACUOUS,
    Finding,
    adapter_findings,
    check,
    column_findings,
    denied_columns,
    over_ratchet,
    process_findings,
    prose_findings,
    registered_layers,
    registered_processes,
    registered_providers,
    registered_rungs,
    structure_findings,
)

FACTORY = """
from typing import Final

_BUILTIN_ADAPTERS: Final[dict[str, str]] = {
    "qdrant": "fasterrag.adapters.vectordb.qdrant:QdrantAdapter",
    "milvus": "fasterrag.adapters.vectordb.milvus:MilvusAdapter",
}
_PLANNED_ADAPTERS: Final[frozenset[str]] = frozenset({"pinecone"})
"""

LADDER = """
from typing import Final

_BUILTIN_RUNGS: Final[dict[str, str]] = {
    "hybrid_only": "fasterrag.services.querying:fused_shortlist",
}
_PLANNED_RUNGS: Final[frozenset[str]] = frozenset({"cache_only"})
"""

PROCESSES = """
from typing import Final

BUILTIN_PROCESSES: Final[dict[str, object]] = {"serve": run_serve}
PLANNED_PROCESSES: Final[frozenset[str]] = frozenset({"worker"})
"""

PROJECT = """
[tool.pytest.ini_options]
addopts = "-m 'not soak'"
markers = [
    "integration: requires containerized backends",
    "soak: long-running stability runs",
]
"""

# CRITICAL: every fake tree carries one table, one tier name in prose, one rung name in
# prose, one built rung and one declared-unbuilt rung, because a vacuity guard fires without
# each of them. A fixture that trips a guard tests the guard and not the rule it was written
# for.
FIXTURE = """# fixture.md

| Surface | How |
|---|---|
| CLI | shipped |

The integration suite runs on every push, and a reranker outage serves hybrid_only.
Start the API with `fasterrag serve`.
"""

MAP_HEADER = """# structure.md

> Paths marked *(pending)* do not exist yet; everything else is in the tree today.

```
repo/
├── src/
│   ├── real.py            # shipped
```
"""


def build(
    root: Path,
    *,
    factory: str = FACTORY,
    project: str = PROJECT,
    ladder: str = LADDER,
    processes: str = PROCESSES,
    docs: dict[str, str] | None = None,
) -> Path:
    """Write a fake tree: the three registries the gate reads, and the given docs."""
    path = root / "src" / "fasterrag" / "adapters" / "vectordb"
    path.mkdir(parents=True, exist_ok=True)
    (path / "factory.py").write_text(factory, encoding="utf-8")
    services = root / "src" / "fasterrag" / "services"
    services.mkdir(parents=True, exist_ok=True)
    (services / "degradation.py").write_text(ladder, encoding="utf-8")
    commands = root / "src" / "fasterrag" / "cli" / "commands"
    commands.mkdir(parents=True, exist_ok=True)
    (commands / "processes.py").write_text(processes, encoding="utf-8")
    (root / "pyproject.toml").write_text(project, encoding="utf-8")
    (root / "tests" / "integration").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "integration" / "test_live.py").write_text("", encoding="utf-8")
    (root / "docs").mkdir(exist_ok=True)
    (root / "docs" / "fixture.md").write_text(FIXTURE, encoding="utf-8")
    (root / "docs" / "todo.md").write_text("- [ ] TASK-0049: adapters\n", encoding="utf-8")
    (root / "docs" / "structure.md").write_text(MAP_HEADER, encoding="utf-8")
    for name in ("README.md", "CLAUDE.md"):
        (root / name).write_text("Qdrant ships.\n", encoding="utf-8")
    for name, body in (docs or {}).items():
        (root / "docs" / name).write_text(body, encoding="utf-8")
    return root


def rules(findings: list[Finding]) -> list[str]:
    return sorted(finding.rule for finding in findings)


def test_a_denial_of_a_registered_adapter_is_a_finding(tmp_path: Path) -> None:
    """The live sentence: an open umbrella id, a shipped adapter, every other gate green."""
    root = build(tmp_path, docs={"scope.md": "Milvus is specified but not built (TASK-0049).\n"})

    found, _ = adapter_findings(root)

    assert len(found) == 1
    assert found[0].rule == ADAPTER_DENIAL
    assert "milvus" in found[0].detail


def test_a_denial_of_a_planned_adapter_is_not_a_finding(tmp_path: Path) -> None:
    """Pinecone is genuinely unbuilt, and saying so must stay green forever."""
    root = build(tmp_path, docs={"scope.md": "Pinecone is specified but not built (TASK-0049).\n"})

    found, _ = adapter_findings(root)

    assert found == []


def test_the_shipped_half_of_a_mixed_sentence_does_not_fire(tmp_path: Path) -> None:
    """One sentence routinely says what shipped and what did not; only the lie is a finding."""
    root = build(
        tmp_path,
        docs={"scope.md": "Qdrant and Milvus ship; Pinecone is unbuilt (TASK-0049).\n"},
    )

    found, _ = adapter_findings(root)

    assert found == []


def test_an_affirmation_before_the_name_is_not_a_denial_of_it(tmp_path: Path) -> None:
    """deployment.md's live sentence, and the gate's only measured false positive."""
    root = build(
        tmp_path,
        docs={"a.md": "doctor-gating shipped for Qdrant and pending for Langfuse.\n"},
    )

    found, _ = adapter_findings(root)

    assert found == []


def test_an_enumeration_is_not_split_by_its_commas(tmp_path: Path) -> None:
    """architecture.md's shape: the denial sits after a list, and every name in it counts."""
    root = build(
        tmp_path,
        docs={"a.md": "**Milvus, Qdrant, and Pinecone are specified but not built**.\n"},
    )

    found, _ = adapter_findings(root)

    assert len(found) == 1
    assert "milvus" in found[0].detail
    assert "qdrant" in found[0].detail


def test_never_run_is_not_a_denial_that_a_thing_exists(tmp_path: Path) -> None:
    """Milvus is registered and has never reached a server, and that sentence is true."""
    root = build(
        tmp_path,
        docs={"a.md": "Milvus is shipped but never run against a server, and unverified.\n"},
    )

    found, _ = adapter_findings(root)

    assert found == []


COLUMN = """# status

| Shipped | Not yet |
|---|---|
| Qdrant | {item} |
"""


def test_a_column_header_denies_every_item_under_it(tmp_path: Path) -> None:
    """README.md:9 at 650943b: the denial is in the header and no cell carries a verb."""
    root = build(tmp_path, docs={"a.md": COLUMN.format(item="Milvus/Pinecone adapters")})

    found, _ = column_findings(root)

    assert len(found) == 1
    assert found[0].rule == COLUMN_DENIAL
    assert "milvus" in found[0].detail


def test_a_denied_column_item_that_merely_names_a_provider_is_left_alone(tmp_path: Path) -> None:
    """README's live item: the cell denies a contract run, which has genuinely never run."""
    item = "a Milvus or Pinecone contract run against a real server"
    root = build(tmp_path, docs={"a.md": COLUMN.format(item=item)})

    found, _ = column_findings(root)

    assert found == []


def test_a_planned_provider_under_a_denying_header_is_not_a_finding(tmp_path: Path) -> None:
    """An item reading "Pinecone adapter" is decided, and the denial of it is true."""
    root = build(tmp_path, docs={"a.md": COLUMN.format(item="Pinecone adapter")})

    found, _ = column_findings(root)

    assert found == []


def test_a_header_that_denies_nothing_lends_nothing(tmp_path: Path) -> None:
    """151 of 152 tables have a header like this one, and none of them denies anything."""
    table = COLUMN.replace("| Shipped | Not yet |", "| Backend | Notes |")
    root = build(tmp_path, docs={"a.md": table.format(item="Milvus adapter")})

    found, _ = column_findings(root)

    assert found == []


def test_a_blockquoted_table_is_still_a_table(tmp_path: Path) -> None:
    """README's table is blockquoted, so losing the quote prefix loses the whole rule."""
    quoted = "# status\n\n" + "\n".join(
        f"> {line}" for line in COLUMN.format(item="Milvus adapter").split("\n")[2:]
    )
    root = build(tmp_path, docs={"a.md": quoted})

    found, tables = column_findings(root)

    assert tables == 2
    assert [finding.rule for finding in found] == [COLUMN_DENIAL]


def test_an_item_carrying_its_own_denial_is_left_to_the_prose_rule(tmp_path: Path) -> None:
    """Two rules on one cell report it twice, and the prose rule has a guard this lacks."""
    root = build(tmp_path, docs={"a.md": COLUMN.format(item="Milvus is not built")})

    columns, _ = column_findings(root)
    adapters, _ = adapter_findings(root)

    assert columns == []
    assert [finding.rule for finding in adapters] == [ADAPTER_DENIAL]


def test_a_denial_of_a_tier_with_tests_is_a_finding(tmp_path: Path) -> None:
    """The TASK-0369 shape with no registry to decide it, decided by the marker registry."""
    root = build(tmp_path, docs={"a.md": "The integration harness is not yet built.\n"})

    found, _ = prose_findings(root)

    assert [finding.rule for finding in found] == [LAYER_DENIAL]
    assert "integration" in found[0].detail


def test_a_denial_of_a_tier_with_no_tests_is_not_a_finding(tmp_path: Path) -> None:
    """Nothing is marked soak and there is no tests/soak, so the denial is true."""
    root = build(tmp_path, docs={"a.md": "The soak harness is not yet built.\n"})

    found, _ = prose_findings(root)

    assert found == []


def test_a_denial_that_labels_a_list_reaches_the_list(tmp_path: Path) -> None:
    """reliability.md:56 at 650943b, the sentence TASK-0369 names: a header written as prose."""
    body = (
        "**Specified but not built** (TASK-0243): integration (testcontainers) "
        "and 24 h soak (no memory growth) — both waiting on hardware.\n"
    )
    root = build(tmp_path, docs={"a.md": body})

    found, _ = prose_findings(root)

    assert [finding.rule for finding in found] == [LABEL_DENIAL]
    assert found[0].detail.startswith("line 1 lists integration after a denial")


def test_a_label_whose_list_carries_its_own_denial_is_left_alone(tmp_path: Path) -> None:
    """An inheriting rule never covers a list that denies for itself; that is other work."""
    body = "Not built: the integration tier is unbuilt and the soak tier is unbuilt.\n"
    root = build(tmp_path, docs={"a.md": body})

    found, _ = prose_findings(root)

    assert [finding.rule for finding in found] == [LAYER_DENIAL]


def test_a_sentence_denying_a_result_rather_than_a_tier_is_left_alone(tmp_path: Path) -> None:
    """README's live wording: the tier exists, and a denial of its *result* is still true."""
    body = "Not built: soak test layer and any integration *result*.\n"
    root = build(tmp_path, docs={"a.md": body})

    found, _ = prose_findings(root)

    assert found == []


def test_a_denial_of_a_rung_the_ladder_serves_is_a_finding(tmp_path: Path) -> None:
    """The class TASK-0369 left ungated for want of a registry, decided by _BUILTIN_RUNGS."""
    root = build(tmp_path, docs={"a.md": "The hybrid_only rung is not yet built.\n"})

    found, _ = prose_findings(root)

    assert [finding.rule for finding in found] == [RUNG_DENIAL]
    assert "hybrid_only" in found[0].detail


def test_a_denial_of_a_declared_unbuilt_rung_is_not_a_finding(tmp_path: Path) -> None:
    """`cache_only` is genuinely unbuilt (TASK-0159), and saying so must stay green."""
    root = build(tmp_path, docs={"a.md": "The cache_only rung is specified but not built.\n"})

    found, _ = prose_findings(root)

    assert found == []


def test_the_live_wording_of_every_cache_only_denial_is_left_alone(tmp_path: Path) -> None:
    """Four live sentences, one per document, that the rule must never fire on today."""
    body = (
        "The `cache_only` rung is specified but not built (TASK-0159).\n\n"
        "As built, only `hybrid_only` and `extractive` are served; `cache_only` is "
        "specified but not implemented (TASK-0159), and a vector-DB outage currently "
        "returns a retryable `RETRIEVAL_FAILED` problem instead.\n\n"
        "Known gaps: the `cache_only` degradation rung is not built (TASK-0159).\n\n"
        "`hybrid_only` (reranker loss) and `extractive` (LLM loss) are implemented and "
        "chaos-verified; the `cache_only` rung (vector-DB loss) is not built yet.\n"
    )
    root = build(tmp_path, docs={"a.md": body})

    found, _ = prose_findings(root)

    assert found == []


def test_a_denial_of_what_a_rung_produces_is_left_alone(tmp_path: Path) -> None:
    """The item must BE the rung: a denial of its output denies no rung."""
    root = build(tmp_path, docs={"a.md": "An extractive answer for a 404 is not built.\n"})

    found, _ = prose_findings(root)

    assert found == []


def test_a_rung_under_a_denying_column_header_is_a_finding(tmp_path: Path) -> None:
    """README's third column is where a shipped rung would be listed as unbuilt."""
    body = "| Shipped | Not yet |\n|---|---|\n| CLI | hybrid_only rung |\n"
    root = build(tmp_path, docs={"a.md": body})

    found, _ = column_findings(root)

    assert [finding.rule for finding in found] == [COLUMN_DENIAL]
    assert "hybrid_only" in found[0].detail


def test_the_ladder_registry_decides_which_rungs_exist(tmp_path: Path) -> None:
    """_BUILTIN_RUNGS and _PLANNED_RUNGS are read as code, exactly as the factories are."""
    root = build(tmp_path)

    served, declared = registered_rungs(root)

    assert served == {"hybrid_only"}
    assert declared == {"cache_only"}


def test_the_marker_registry_decides_which_tiers_exist(tmp_path: Path) -> None:
    """--strict-markers makes the list executable truth: an unregistered marker errors."""
    root = build(tmp_path)

    built, unbuilt = registered_layers(root)

    assert built == {"integration"}
    assert unbuilt == {"soak"}


def test_a_tier_marked_on_a_test_outside_its_directory_counts(tmp_path: Path) -> None:
    """tests/load carries no marker and a marked test may own no directory; either counts."""
    root = build(tmp_path)
    (root / "tests" / "unit").mkdir(parents=True, exist_ok=True)
    (root / "tests" / "unit" / "test_x.py").write_text(
        "import pytest\n\n@pytest.mark.soak\ndef test_x() -> None: ...\n", encoding="utf-8"
    )

    built, unbuilt = registered_layers(root)

    assert built == {"integration", "soak"}
    assert unbuilt == set()


def test_a_stale_pending_marker_in_the_structure_map_is_a_finding(tmp_path: Path) -> None:
    """structure.md's own second line is the contract this half enforces."""
    root = build(tmp_path)
    (root / "src" / "real.py").write_text("", encoding="utf-8")
    (root / "docs" / "structure.md").write_text(
        MAP_HEADER.replace("# shipped", "(pending)"), encoding="utf-8"
    )

    found, entries = structure_findings(root)

    assert entries == 2
    assert rules(found) == [STRUCTURE_PENDING]


def test_a_listed_path_that_is_not_in_the_tree_is_a_finding(tmp_path: Path) -> None:
    """The other half of the same sentence: everything unmarked is in the tree today."""
    root = build(tmp_path)

    found, _ = structure_findings(root)

    assert rules(found) == [STRUCTURE_ABSENT]
    assert "src/real.py" in found[0].detail


def test_a_map_that_matches_the_tree_reports_nothing(tmp_path: Path) -> None:
    """The gate must be silent on a correct map, or it teaches people to ignore it."""
    root = build(tmp_path)
    (root / "src" / "real.py").write_text("", encoding="utf-8")

    found, entries = structure_findings(root)

    assert found == []
    assert entries == 2


def test_an_empty_registry_is_vacuous(tmp_path: Path) -> None:
    """The first guard: _BUILTIN_ADAPTERS is the entire source of truth for half the gate."""
    root = build(tmp_path, factory="EMPTY = 1\n")

    found = check(root)

    assert rules(found) == [VACUOUS]
    assert "no factory registers any provider" in found[0].detail


def test_a_registry_with_nothing_planned_is_vacuous(tmp_path: Path) -> None:
    """With nothing planned every denial is automatically false and the rule stops deciding."""
    root = build(tmp_path, factory=FACTORY.replace('frozenset({"pinecone"})', "frozenset()"))

    found = check(root)

    assert rules(found) == [VACUOUS]
    assert "planned provider" in found[0].detail


def test_documents_that_name_no_provider_are_vacuous(tmp_path: Path) -> None:
    """If prose stops spelling the registry keys, the join silently stops existing."""
    root = build(tmp_path)
    for name in ("README.md", "CLAUDE.md"):
        (root / name).write_text("Nothing to see.\n", encoding="utf-8")
    (root / "docs" / "structure.md").write_text(MAP_HEADER, encoding="utf-8")

    found = check(root)

    assert rules(found) == [VACUOUS]
    assert "no clause naming any of the" in found[0].detail


def test_a_missing_root_document_is_vacuous(tmp_path: Path) -> None:
    """TASK-0310's guard, inherited: losing a scan root silently is the defect."""
    root = build(tmp_path)
    (root / "CLAUDE.md").unlink()

    found = check(root)

    assert rules(found) == [VACUOUS]
    assert "CLAUDE.md" in found[0].path


def test_an_unparseable_structure_map_is_vacuous(tmp_path: Path) -> None:
    """A diagram with no box-drawing characters proves nothing and must say so."""
    root = build(tmp_path)
    (root / "docs" / "structure.md").write_text("Qdrant lives in src/.\n", encoding="utf-8")

    found = check(root)

    assert rules(found) == [VACUOUS]
    assert "no tree-diagram entries" in found[0].detail


def test_a_registry_with_no_markers_is_vacuous(tmp_path: Path) -> None:
    """[tool.pytest.ini_options].markers is the tier rule's whole vocabulary."""
    root = build(tmp_path, project="[tool.pytest.ini_options]\nasyncio_mode = 'auto'\n")

    found = check(root)

    assert rules(found) == [VACUOUS]
    assert "no pytest marker" in found[0].detail


def test_a_suite_with_every_tier_built_still_catches_a_false_denial(tmp_path: Path) -> None:
    """Finishing every tier is the goal, so it must not read as the rule going blind.

    This case asserted the opposite until 2026-11-11, when TASK-0243 built the soak layer
    and the guard short-circuited the whole gate — failing `main` for succeeding. The rule
    catches a *false* denial, so a vocabulary whose denials are all false is the case it
    catches every one of; `named[TIER] == 0` is what still proves the matcher ran.
    """
    root = build(tmp_path, docs={"scope.md": "The soak suite is not built (TASK-0243).\n"})
    (root / "tests" / "soak").mkdir(parents=True)
    (root / "tests" / "soak" / "test_long.py").write_text("", encoding="utf-8")

    found = check(root)

    assert VACUOUS not in rules(found)
    assert any("soak" in finding.detail for finding in found)


def test_a_corpus_with_no_table_at_all_is_vacuous(tmp_path: Path) -> None:
    """A blockquote prefix once hid the only table that matters; an empty parse must say so."""
    root = build(tmp_path)
    (root / "docs" / "fixture.md").write_text("The integration suite runs.\n", encoding="utf-8")

    found = check(root)

    assert rules(found) == [VACUOUS]
    assert "no markdown table" in found[0].detail


def test_documents_that_name_no_tier_are_vacuous(tmp_path: Path) -> None:
    """If prose stops spelling the marker names, the join silently stops existing."""
    root = build(tmp_path)
    (root / "docs" / "fixture.md").write_text(
        FIXTURE.replace("The integration suite runs on every push, and a", "A"),
        encoding="utf-8",
    )

    found = check(root)

    assert rules(found) == [VACUOUS]
    assert "tiers that have tests" in found[0].detail


def test_a_ladder_with_no_registered_rung_is_vacuous(tmp_path: Path) -> None:
    """_BUILTIN_RUNGS is the rung rule's whole vocabulary, as the factory is the adapters'."""
    root = build(tmp_path, ladder="EMPTY = 1\n")

    found = check(root)

    assert rules(found) == [VACUOUS]
    assert "no rung registers an implementation" in found[0].detail


def test_a_ladder_with_every_rung_implemented_is_vacuous(tmp_path: Path) -> None:
    """The nothing-planned guard's sibling: with every rung built, every denial is false."""
    root = build(tmp_path, ladder=LADDER.replace('frozenset({"cache_only"})', "frozenset()"))

    found = check(root)

    assert rules(found) == [VACUOUS]
    assert "every rung the ladder declares is implemented" in found[0].detail


def test_documents_that_name_no_rung_are_vacuous(tmp_path: Path) -> None:
    """Counted per vocabulary: one total passes while the rung half matches nothing."""
    root = build(tmp_path)
    (root / "docs" / "fixture.md").write_text(
        FIXTURE.replace(", and a reranker outage serves hybrid_only", ""), encoding="utf-8"
    )

    found = check(root)

    assert rules(found) == [VACUOUS]
    assert "rungs the ladder serves" in found[0].detail


def test_a_document_that_tells_an_operator_to_run_an_unbuilt_process_is_a_finding(
    tmp_path: Path,
) -> None:
    """python-api.md:218 said to run `fasterrag worker` processes; the command only waited."""
    root = build(
        tmp_path,
        docs={"a.md": "For heavy ingestion run dedicated `fasterrag worker` processes.\n"},
    )

    found, mentions = process_findings(root)

    assert rules(found) == [PROCESS_SURFACE]
    assert "worker" in found[0].detail
    assert mentions == 1


def test_an_unbuilt_process_in_a_code_block_is_a_finding(tmp_path: Path) -> None:
    """quickstart.md's Path C put it in a fence, the most direct instruction a page gives."""
    root = build(tmp_path, docs={"a.md": "Run:\n\n```bash\nfasterrag worker --pools cpu\n```\n"})

    found, _ = process_findings(root)

    assert rules(found) == [PROCESS_SURFACE]
    assert "line 4" in found[0].detail


@pytest.mark.parametrize(
    "sentence",
    [
        "The `fasterrag worker` command was removed on 2026-11-12.",
        "There is no `fasterrag worker` process yet (TASK-0130).",
        "A separate `fasterrag worker` is not built (TASK-0130).",
        "`fasterrag worker` no longer exists.",
    ],
)
def test_a_sentence_saying_the_process_is_absent_is_left_alone(
    tmp_path: Path, sentence: str
) -> None:
    """The documents must be able to say where the command went."""
    root = build(tmp_path, docs={"a.md": f"{sentence}\n"})

    assert process_findings(root)[0] == []


def test_a_built_process_and_an_unrelated_worker_are_left_alone(tmp_path: Path) -> None:
    """Only `fasterrag <planned name>` is decided: pools of workers are real."""
    root = build(
        tmp_path,
        docs={"a.md": "Run `fasterrag serve`; it starts embedding workers and the worker pool.\n"},
    )

    assert process_findings(root)[0] == []


def test_a_process_moved_into_the_registry_stops_being_decided(tmp_path: Path) -> None:
    """The day TASK-0130 ships a worker, the same sentence becomes true and must pass."""
    shipped = PROCESSES.replace('{"serve": run_serve}', '{"serve": run_serve, "worker": w}')
    root = build(
        tmp_path,
        processes=shipped,
        docs={"a.md": "Run `fasterrag worker` beside the API.\n"},
    )

    assert process_findings(root)[0] == []


def test_a_cli_with_no_registered_process_is_vacuous(tmp_path: Path) -> None:
    root = build(tmp_path, processes="BUILTIN_PROCESSES: dict[str, object] = {}\n")

    found = check(root)

    assert rules(found) == [VACUOUS]
    assert "BUILTIN_PROCESSES" in found[0].detail


def test_documents_that_name_no_built_process_are_vacuous(tmp_path: Path) -> None:
    root = build(tmp_path)
    fixture = root / "docs" / "fixture.md"
    fixture.write_text(
        fixture.read_text(encoding="utf-8").replace("Start the API with `fasterrag serve`.", ""),
        encoding="utf-8",
    )

    found = check(root)

    assert rules(found) == [VACUOUS]
    assert "fasterrag serve" in found[0].detail


def test_the_real_registry_runs_serve_and_names_the_removed_worker() -> None:
    """Pinned so a rename in processes.py cannot quietly empty the rule's vocabulary."""
    assert registered_processes() == ({"serve"}, {"worker"})


def test_the_ratchet_counts_by_path(tmp_path: Path) -> None:
    """Line numbers move under nine concurrent agents; paths do not."""
    findings = [
        Finding(ADAPTER_DENIAL, "docs/a.md", "x"),
        Finding(ADAPTER_DENIAL, "docs/a.md", "y"),
    ]

    assert over_ratchet(findings, {"docs/a.md": 2}) == []
    assert len(over_ratchet(findings, {"docs/a.md": 1})) == 1


def test_a_ratchet_entry_nothing_uses_is_itself_reported() -> None:
    """A licence for a fixed defect is how a ratchet stops being a ratchet."""
    messages = over_ratchet([], {"docs/gone.md": 3})

    assert len(messages) == 1
    assert "delete the entry" in messages[0]


def test_the_shipped_ratchet_is_empty() -> None:
    """Every finding the gate has made was fixed; an allowance would be a standing exception."""
    assert RATCHET == {}


def test_the_real_tree_is_clean() -> None:
    """The gate ships green, and the five documents it found are the proof it was not.

    The only findings left are the ratcheted worker sentences, and nothing goes past them.
    """
    found = check()

    assert over_ratchet(found) == []
    assert {finding.rule for finding in found} <= {PROCESS_SURFACE}
    assert {finding.path for finding in found} <= set(RATCHET)


@pytest.mark.parametrize("name", ["milvus", "weaviate", "chroma", "qdrant", "pgvector"])
def test_the_real_registry_names_the_adapters_that_shipped(name: str) -> None:
    """Pinned so a rename in factory.py cannot quietly empty the gate's vocabulary."""
    shipped, planned = registered_providers()

    assert name in shipped
    assert planned == {"pinecone"}


def test_the_real_ladder_serves_two_rungs_and_declares_a_third() -> None:
    """The pair the class turns on: `cache_only` is TASK-0159 and must stay undecided."""
    served, declared = registered_rungs()

    assert served == {"hybrid_only", "extractive"}
    assert declared == {"cache_only"}


def test_every_registered_tier_in_the_real_suite_now_has_test_code() -> None:
    """TASK-0369 turned on `tests/load` holding modules and `soak` holding none.

    TASK-0243 closed the second half on 2026-11-11, so the set this asserted is now
    empty — which is a milestone and not a regression, and the reason `check` no longer
    treats an empty unbuilt set as the rule going blind.
    """
    built, unbuilt = registered_layers()

    assert {"integration", "eval", "load", "chaos", "soak"} <= built
    assert unbuilt == set()


def test_the_only_denying_column_header_in_the_corpus_is_readmes() -> None:
    """Measured: one denying header over 152 tables, and its items are the rule's input."""
    items, tables = denied_columns()

    assert tables > 100
    assert {path for path, _, _ in items} == {"README.md"}
