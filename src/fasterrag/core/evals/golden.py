"""Golden sets: the ground truth retrieval quality is measured against.

One JSONL record per query, in the schema of ``docs/testing-strategy.md`` §1.6. The eval
harness, the regression gate (D7), and Autopilot (D6) all read this one format, so the three
cannot drift apart in what they consider correct.

A record with no relevant chunks and no reference answer is an **adversarial** record: a
question the corpus cannot answer. Those exist to prove the system refuses rather than
invents (D5), so they are counted separately and never averaged into retrieval metrics —
scoring recall against an empty ground truth would be meaningless.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fasterrag.errors import ErrorCode, FasterRagError

__all__ = ["GoldenRecord", "load_golden_set", "write_golden_set"]

_REQUIRED = ("id", "query", "source", "created_at")


@dataclass(frozen=True, slots=True)
class GoldenRecord:
    """One evaluated query and what a correct answer would retrieve."""

    id: str
    query: str
    source: str
    created_at: str
    relevant_chunk_ids: tuple[str, ...] = ()
    relevant_document_ids: tuple[str, ...] = ()
    answer_reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def adversarial(self) -> bool:
        """Return whether this query is deliberately unanswerable from the corpus."""
        return not self.relevant_chunk_ids and not self.relevant_document_ids

    @property
    def generated(self) -> bool:
        """Return whether Autopilot produced this record rather than a human."""
        return self.source == "autopilot"

    def is_relevant(self, chunk_id: str, document_id: str | None) -> bool:
        """Return whether a retrieved chunk counts as a hit.

        Document-level ground truth is honored as well as chunk-level, because chunk ids
        change whenever the chunker configuration changes; a golden set pinned only to
        chunk ids would silently score zero after any re-chunking.
        """
        if chunk_id in self.relevant_chunk_ids:
            return True
        return document_id is not None and document_id in self.relevant_document_ids

    def as_dict(self) -> dict[str, Any]:
        """Return the JSONL form."""
        return {
            "id": self.id,
            "query": self.query,
            "relevant_chunk_ids": list(self.relevant_chunk_ids),
            "relevant_document_ids": list(self.relevant_document_ids),
            "answer_reference": self.answer_reference,
            "metadata": self.metadata,
            "source": self.source,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, where: str = "<memory>") -> GoldenRecord:
        """Build a record from its JSONL form.

        Raises:
            FasterRagError: With ``VALIDATION_FAILED`` if a required field is missing, so a
                malformed golden set fails loudly rather than quietly scoring nothing.
        """
        missing = [name for name in _REQUIRED if not payload.get(name)]
        if missing:
            raise FasterRagError(
                f"golden record in {where} is missing required fields: {', '.join(missing)}",
                code=ErrorCode.VALIDATION_FAILED,
            )

        return cls(
            id=str(payload["id"]),
            query=str(payload["query"]),
            source=str(payload["source"]),
            created_at=str(payload["created_at"]),
            relevant_chunk_ids=tuple(payload.get("relevant_chunk_ids") or ()),
            relevant_document_ids=tuple(payload.get("relevant_document_ids") or ()),
            answer_reference=payload.get("answer_reference"),
            metadata=dict(payload.get("metadata") or {}),
        )


def load_golden_set(path: str | Path) -> list[GoldenRecord]:
    """Read a golden set from JSONL.

    Raises:
        FasterRagError: If the file is missing, a line is not valid JSON, a record is
            malformed, or two records share an id.
    """
    source = Path(path)
    if not source.is_file():
        raise FasterRagError(f"golden set not found: {source}", code=ErrorCode.NOT_FOUND)

    records: list[GoldenRecord] = []
    seen: set[str] = set()

    for number, line in enumerate(_lines(source), start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise FasterRagError(
                f"{source} line {number} is not valid JSON: {exc.msg}",
                code=ErrorCode.VALIDATION_FAILED,
            ) from exc

        record = GoldenRecord.from_dict(payload, where=f"{source} line {number}")
        if record.id in seen:
            raise FasterRagError(
                f"{source} line {number} repeats the query id {record.id!r}; ids must be "
                "unique or scores double-count",
                code=ErrorCode.VALIDATION_FAILED,
            )

        seen.add(record.id)
        records.append(record)

    return records


def _lines(path: Path) -> Iterator[str]:
    """Yield the file's non-blank lines."""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield line


def write_golden_set(path: str | Path, records: Sequence[GoldenRecord]) -> None:
    """Write a golden set as JSONL.

    Golden sets are versioned files reviewed like code, so records are written in the order
    given and keys are stable — a regenerated set produces a readable diff rather than a
    reshuffle.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        json.dumps(record.as_dict(), sort_keys=True, separators=(",", ": ")) for record in records
    )
    target.write_text(f"{body}\n" if body else "", encoding="utf-8")
