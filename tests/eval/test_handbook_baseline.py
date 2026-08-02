"""The committed retrieval baseline over the handbook fixture corpus.

This is what the regression gate (D7) compares against: a fixed corpus, a hand-authored
golden set, and a recorded score. A gate needs all three — without the corpus the numbers
mean nothing, without the golden set there is nothing to score, and without a committed
baseline there is nothing to regress *from*.

The golden set is ``source: "human"`` on purpose. A generated set is exactly what
``fasterrag.core.evals.generator`` produces, but committing one as the CI baseline would make
the baseline move whenever the generating model does, and a gate whose reference drifts under
it detects noise rather than regressions.

Ground truth is recorded at **document** level. Chunk ids are a function of the chunker
configuration, so a chunk-level set would go stale the moment ``chunking.chunk_size`` changed
— which is one of the very things the gate exists to let people change safely.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fasterrag.core.evals import EvalReport, GoldenRecord, load_golden_set
from fasterrag.core.identity import document_id

pytestmark = pytest.mark.eval

DATASET = Path(__file__).parent / "datasets" / "handbook"
CORPUS = DATASET / "corpus"
GOLDEN = DATASET / "golden.jsonl"
BASELINE = DATASET / "baseline.json"

# CRITICAL: k is part of the measurement. recall@10 and recall@5 are different numbers, so a
# baseline recorded at one k cannot be compared against a run at another.
K = 5


def resolve_ground_truth(records: list[GoldenRecord], corpus: Path) -> list[GoldenRecord]:
    r"""Rewrite filename ground truth into the ids the pipeline actually produces.

    The committed set names files, because a document id is a hash of the absolute source
    path and would differ on every machine. Resolution happens here so the dataset stays
    portable and the translation is visible rather than implied.

    # CRITICAL: the path is *not* resolved. A document id hashes the source string it was
    # ingested under, so on Windows `d:\...` and `D:\...` — the same file — hash differently.
    # Building the id from a differently-normalised path than the ingest used produces zero
    # matches and a baseline of 0.0 that looks like a retrieval failure. See TASK-0141.
    """
    resolved: list[GoldenRecord] = []
    for record in records:
        ids = tuple(document_id(str(corpus / name)) for name in record.relevant_document_ids)
        resolved.append(
            GoldenRecord(
                id=record.id,
                query=record.query,
                source=record.source,
                created_at=record.created_at,
                relevant_chunk_ids=record.relevant_chunk_ids,
                relevant_document_ids=ids,
                answer_reference=record.answer_reference,
                metadata=record.metadata,
            )
        )
    return resolved


def test_the_fixture_corpus_is_committed() -> None:
    documents = sorted(path.name for path in CORPUS.glob("*.md"))

    assert len(documents) == 6
    assert "leave-policy.md" in documents


def test_the_golden_set_is_committed_and_loads() -> None:
    records = load_golden_set(GOLDEN)

    assert len(records) == 15
    assert len({record.id for record in records}) == 15


def test_the_golden_set_is_human_authored() -> None:
    """A generated set would move the baseline whenever the generating model moved."""
    assert {record.source for record in load_golden_set(GOLDEN)} == {"human"}


def test_the_golden_set_carries_adversarial_records() -> None:
    records = load_golden_set(GOLDEN)
    adversarial = [record for record in records if record.adversarial]

    assert len(adversarial) == 3
    for record in adversarial:
        assert record.relevant_document_ids == ()
        assert record.answer_reference is None


def test_every_answerable_record_names_a_document_that_exists() -> None:
    for record in load_golden_set(GOLDEN):
        for name in record.relevant_document_ids:
            assert (CORPUS / name).is_file(), f"{record.id} names a missing document: {name}"


def test_every_answerable_record_carries_a_reference_answer() -> None:
    for record in load_golden_set(GOLDEN):
        if not record.adversarial:
            assert record.answer_reference, f"{record.id} has ground truth but no reference"


def test_ground_truth_resolves_to_pipeline_document_ids() -> None:
    resolved = resolve_ground_truth(load_golden_set(GOLDEN), CORPUS)
    answerable = [record for record in resolved if not record.adversarial]

    assert answerable
    for record in answerable:
        assert all(identifier.startswith("d_") for identifier in record.relevant_document_ids)


def test_the_baseline_is_committed_and_well_formed() -> None:
    """A gate with no committed baseline blocks rather than passing, so one must exist."""
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))

    assert payload["k"] == K
    assert 0.0 <= payload["recall_at_k"] <= 1.0
    assert 0.0 <= payload["ndcg_at_k"] <= 1.0
    assert 0.0 <= payload["mrr"] <= 1.0
    assert payload["scored"] == 12
    assert payload["adversarial"] == 3
    assert payload["embedding_model"]
    assert payload["notes"]


def test_the_baseline_covers_every_measurable_record() -> None:
    """Scored plus adversarial must account for the whole set, or coverage silently shrank."""
    payload = json.loads(BASELINE.read_text(encoding="utf-8"))

    assert payload["scored"] + payload["adversarial"] == len(load_golden_set(GOLDEN))


def report_as_baseline(report: EvalReport, embedding_model: str, notes: str) -> dict[str, object]:
    """Render a run as the committed baseline shape."""
    return {
        "k": report.k,
        "scored": report.scored,
        "adversarial": report.adversarial,
        "recall_at_k": round(report.recall_at_k, 4),
        "mrr": round(report.mrr, 4),
        "ndcg_at_k": round(report.ndcg_at_k, 4),
        "embedding_model": embedding_model,
        "notes": notes,
    }


__all__ = ["CORPUS", "GOLDEN", "K", "report_as_baseline", "resolve_ground_truth"]
