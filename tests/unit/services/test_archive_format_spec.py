"""Pin the archive the exporter writes against ``docs/archive-format.md``.

Every other archive test writes the format and then reads it back, which proves the two
halves agree with each other and nothing more — they would agree just as happily on a
format the specification never described. This module reads the spec tables and asserts
the real exporter satisfies them, so a field dropped from the writer or renamed in the
document fails here rather than at an operator's first cross-deployment restore.
"""

import json
import re
import tarfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from fasterrag.adapters.vectordb.base import Point, SparseVector
from fasterrag.config.schema import Settings
from fasterrag.services.archive import (
    ARCHIVE_SUFFIX,
    DENSE_ONLY_FORMAT_VERSION,
    FORMAT_VERSION,
    export_archive,
)

SPEC = Path(__file__).resolve().parents[3] / "docs" / "archive-format.md"

REQUIRED = "✅"
_MEMBER = re.compile(r"^[└├]── (\S+)\s+# (REQUIRED|OPTIONAL)")
_ROW = re.compile(r"^\|\s*(.+?)\s*\|(.*)$")
_TICKED = re.compile(r"`([^`]+)`")


def spec_text() -> str:
    """Return the specification document."""
    return SPEC.read_text(encoding="utf-8")


def declared_members() -> dict[str, str]:
    """Return every archive member the container diagram names, mapped to its obligation."""
    found: dict[str, str] = {}
    for line in spec_text().splitlines():
        match = _MEMBER.match(line.strip())
        if match:
            found[match.group(1)] = match.group(2)
    return found


def required_fields(heading: str) -> set[str]:
    """Return the field names a spec table under ``heading`` marks required."""
    body = spec_text().split(heading, 1)[1]
    fields: set[str] = set()
    for line in body.splitlines():
        if line.startswith("##") or (line.startswith("**") and fields):
            break
        if not _ROW.match(line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3 or REQUIRED not in cells[2]:
            continue
        fields.update(_TICKED.findall(cells[0]))
    return fields


def resolve(payload: dict[str, Any], path: str) -> Any:
    """Return the value at a dotted path, or ``None`` when any step is absent."""
    current: Any = payload
    for step in path.split("."):
        if not isinstance(current, dict) or step not in current:
            return None
        current = current[step]
    return current


def point(chunk_id: str, sparse: SparseVector | None = None) -> Point:
    """Return one point shaped the way the indexer stores them."""
    return Point(
        point_id=chunk_id,
        collection="docs",
        sparse=sparse,
        vector=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        payload={
            "document_id": "d_1",
            "source_uri": "docs/policy.md",
            "content_hash": "sha-abc",
            "text": f"body of {chunk_id}",
            "span": {"start": 0, "end": 10},
            "chunk_index": 0,
            "embedding_model": "m",
            "embedding_model_version": "1",
        },
    )


class FakeAdapter:
    """Serves a fixed set of points through the iteration contract."""

    def __init__(self, points: list[Point]) -> None:
        self.points = points

    async def iterate_points(
        self, collection: str, *, with_vectors: bool = False, batch_size: int = 256
    ) -> AsyncIterator[Point]:
        for item in self.points:
            yield item


@pytest.fixture
async def exported(tmp_path: Path) -> dict[str, bytes]:
    """Return the members of an archive written by the real exporter."""
    await export_archive(
        Settings.model_validate({}),
        FakeAdapter([point("c_1"), point("c_2")]),  # type: ignore[arg-type]
        collection="docs",
        destination=tmp_path / "backup",
        include_vectors=True,
    )
    with tarfile.open(tmp_path / f"backup{ARCHIVE_SUFFIX}", "r:gz") as tar:
        return {
            member.name: tar.extractfile(member).read()  # type: ignore[union-attr]
            for member in tar.getmembers()
        }


def rows(body: bytes) -> list[dict[str, Any]]:
    """Parse a JSONL member."""
    return [json.loads(line) for line in body.decode("utf-8").splitlines() if line.strip()]


def test_the_spec_names_both_versions_this_build_writes() -> None:
    """A spec that documents another version describes an archive nobody produces."""
    assert f"`{FORMAT_VERSION}`" in spec_text()
    assert f"`{DENSE_ONLY_FORMAT_VERSION}`" in spec_text()


def test_the_declared_members_are_the_ones_the_spec_lists() -> None:
    """Guards the parser itself: an unparsed diagram would make every check below vacuous."""
    assert declared_members() == {
        "manifest.json": "REQUIRED",
        "checksums.sha256": "REQUIRED",
        "documents.jsonl": "REQUIRED",
        "chunks.jsonl": "REQUIRED",
        "vectors.jsonl": "OPTIONAL",
        "index.lock": "REQUIRED",
        "golden-set.jsonl": "OPTIONAL",
    }


def test_every_required_member_is_written(exported: dict[str, bytes]) -> None:
    required = {name for name, duty in declared_members().items() if duty == "REQUIRED"}

    assert required <= set(exported)


def test_no_member_appears_that_the_spec_never_declared(exported: dict[str, bytes]) -> None:
    """A member outside the spec is a format change nobody else can read."""
    assert set(exported) <= set(declared_members())


def test_the_manifest_carries_every_required_field(exported: dict[str, bytes]) -> None:
    manifest = json.loads(exported["manifest.json"])
    fields = required_fields("## `manifest.json` schema")

    assert fields
    missing = [name for name in sorted(fields) if resolve(manifest, name) is None]
    assert not missing


def test_every_document_row_carries_its_required_fields(exported: dict[str, bytes]) -> None:
    fields = required_fields("**`documents.jsonl`**")

    assert fields == {"document_id", "source_uri", "content_hash", "version", "metadata"}
    for row in rows(exported["documents.jsonl"]):
        assert fields <= set(row)


def test_every_chunk_row_carries_its_required_fields(exported: dict[str, bytes]) -> None:
    fields = required_fields("**`chunks.jsonl`**")

    assert fields == {"chunk_id", "document_id", "text", "span", "metadata"}
    for row in rows(exported["chunks.jsonl"]):
        assert fields <= set(row)
        assert set(row["span"]) == {"start", "end"}


def test_every_vector_row_is_as_wide_as_the_manifest_declares(exported: dict[str, bytes]) -> None:
    """The spec states `length == embedding.dimensions`, and import creates the collection at it."""
    fields = required_fields("**`vectors.jsonl`**")
    declared = json.loads(exported["manifest.json"])["embedding"]["dimensions"]

    assert fields == {"chunk_id", "vector"}
    for row in rows(exported["vectors.jsonl"]):
        assert fields <= set(row)
        assert len(row["vector"]) == declared


def test_the_checksums_member_covers_every_other_member(exported: dict[str, bytes]) -> None:
    """The spec says SHA-256 of every other file; one left out is one nobody verifies."""
    listed = {
        line.split("  ", 1)[1]
        for line in exported["checksums.sha256"].decode("utf-8").splitlines()
        if line.strip()
    }

    assert listed == set(exported) - {"manifest.json", "checksums.sha256"}


@pytest.fixture
async def hybrid_exported(tmp_path: Path) -> dict[str, bytes]:
    """Return the members of an archive the real exporter took from a hybrid collection."""
    leg = SparseVector(indices=[4, 70], values=[0.75, 0.25])
    await export_archive(
        Settings.model_validate({}),
        FakeAdapter([point("c_1", leg), point("c_2", leg)]),  # type: ignore[arg-type]
        collection="docs",
        destination=tmp_path / "backup",
        include_vectors=True,
    )
    with tarfile.open(tmp_path / f"backup{ARCHIVE_SUFFIX}", "r:gz") as tar:
        return {
            member.name: tar.extractfile(member).read()  # type: ignore[union-attr]
            for member in tar.getmembers()
        }


def test_the_spec_keeps_the_keyword_leg_optional_on_a_vector_row() -> None:
    """An archive written before TASK-0302 has none, and must stay importable."""
    assert "sparse" not in required_fields("**`vectors.jsonl`**")
    assert "| `sparse` |" in spec_text()


def test_the_keyword_leg_is_written_in_the_shape_the_spec_names(
    hybrid_exported: dict[str, bytes],
) -> None:
    """The spec names two equal-length lists; anything else is a format nobody else reads."""
    for row in rows(hybrid_exported["vectors.jsonl"]):
        assert set(row["sparse"]) == {"indices", "values"}
        assert len(row["sparse"]["indices"]) == len(row["sparse"]["values"])
        assert row["sparse"]["indices"] == sorted(row["sparse"]["indices"])


def test_the_spec_names_includes_sparse_as_the_manifest_declaration(
    hybrid_exported: dict[str, bytes],
) -> None:
    """The reader refuses an archive whose rows and manifest disagree, so the two must agree."""
    assert "| `includes_sparse` |" in spec_text()
    assert json.loads(hybrid_exported["manifest.json"])["includes_sparse"] is True


def test_an_archive_with_a_keyword_leg_declares_the_major_the_spec_assigns_it(
    hybrid_exported: dict[str, bytes],
) -> None:
    """The version table is the whole backward-compatibility story; a writer off it is a trap."""
    declared = json.loads(hybrid_exported["manifest.json"])["format_version"]

    assert declared == FORMAT_VERSION
    assert f"| `{FORMAT_VERSION}` | any vector row carries `sparse` |" in spec_text()


def test_the_keyword_leg_adds_no_member_the_spec_never_declared(
    hybrid_exported: dict[str, bytes],
) -> None:
    """It extends `vectors.jsonl` rather than arriving as a member an old reader ignores."""
    assert set(hybrid_exported) <= set(declared_members())
