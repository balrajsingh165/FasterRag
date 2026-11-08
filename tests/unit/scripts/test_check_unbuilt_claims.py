"""Pin `scripts/check_unbuilt_claims.py`: the gate that asks the tree instead of the ledger.

Every fixture below is a sentence somebody actually wrote in this repository, because the
gate exists to catch the umbrella-id hole that produced findings on three days in six.
"""

from pathlib import Path

import pytest
from check_unbuilt_claims import (
    ADAPTER_DENIAL,
    RATCHET,
    STRUCTURE_ABSENT,
    STRUCTURE_PENDING,
    VACUOUS,
    Finding,
    adapter_findings,
    check,
    over_ratchet,
    registered_providers,
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

MAP_HEADER = """# structure.md

> Paths marked *(pending)* do not exist yet; everything else is in the tree today.

```
repo/
├── src/
│   ├── real.py            # shipped
```
"""


def build(root: Path, *, factory: str = FACTORY, docs: dict[str, str] | None = None) -> Path:
    """Write a fake tree: one adapter factory, the root documents, and the given docs."""
    path = root / "src" / "fasterrag" / "adapters" / "vectordb"
    path.mkdir(parents=True, exist_ok=True)
    (path / "factory.py").write_text(factory, encoding="utf-8")
    (root / "docs").mkdir(exist_ok=True)
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
    """Every finding this gate has reported was real and was fixed in the same series."""
    assert RATCHET == {}


def test_the_real_tree_is_clean() -> None:
    """The gate ships green, and the five documents it found are the proof it was not."""
    assert check() == []


@pytest.mark.parametrize("name", ["milvus", "weaviate", "chroma", "qdrant", "pgvector"])
def test_the_real_registry_names_the_adapters_that_shipped(name: str) -> None:
    """Pinned so a rename in factory.py cannot quietly empty the gate's vocabulary."""
    shipped, planned = registered_providers()

    assert name in shipped
    assert planned == {"pinecone"}
