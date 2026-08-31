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

from fasterrag.config.schema import Settings
from fasterrag.core.evals import EvalReport, GoldenRecord, load_golden_set
from fasterrag.core.identity import document_id
from fasterrag.errors import FasterRagError
from fasterrag.services.regression import load_baseline, write_baseline

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


def test_the_baseline_is_committed_and_loads_through_the_gate() -> None:
    """A gate with no committed baseline blocks rather than passing, so one must exist.

    Loaded through ``load_baseline`` rather than parsed as JSON: the gate is the only
    consumer that matters, and a baseline it cannot read is not a baseline.
    """
    baseline = load_baseline(BASELINE)

    assert baseline is not None
    assert baseline.k == K
    assert 0.0 <= baseline.recall_at_k <= 1.0
    assert 0.0 <= baseline.ndcg_at_k <= 1.0
    assert 0.0 <= baseline.mrr <= 1.0
    assert baseline.scored == 12
    assert baseline.embedding_model
    assert baseline.config_hash


def test_the_committed_baseline_has_drifted_from_the_canonical_config() -> None:
    """The gap this variant exists to expose, asserted so it cannot be forgotten.

    ``Baseline.comparable_to`` requires the embedding model *and* ``retrieval_config_hash``
    to match, and the handbook baseline recorded 2026-08-02 no longer matches: the hash
    covers the whole chunking and retrieval models, so the five tunables added since —
    ``token_counter``, ``chars_per_token``, ``semantic_percentile``, ``bm25_k1``, ``bm25_b``
    — retired it. The D7 gate would therefore report *blocked* rather than pass or fail, and
    a gate that cannot run protects nothing (TASK-0244).

    The drift is not merely cosmetic. ``token_counter`` defaults to ``auto``, which counts
    with the model's own tokenizer and so moves chunk boundaries, meaning the recorded
    metrics may genuinely no longer hold. Re-recording needs a live backend.

    Written as a failing promise, following the rule the disk-full case established (T13):
    when the baseline is re-recorded this case goes red, which is the point — the re-record
    must move this assertion and the CI wiring together, and cannot land silently.
    """
    baseline = load_baseline(BASELINE)

    assert baseline is not None
    assert baseline.comparable_to(Settings.model_validate({})) is False


def test_comparability_discriminates_rather_than_always_refusing(tmp_path: Path) -> None:
    """A baseline is comparable to the settings it was recorded under, and to nothing else.

    Both directions are asserted from one freshly written baseline, because the negative
    alone proves nothing: an implementation that always returned ``False`` would satisfy it,
    and would also make the drift case above pass for entirely the wrong reason. Verified by
    mutation — always-``False`` fails here.
    """
    settings = Settings.model_validate({})
    report = EvalReport(k=K, scored=1, adversarial=0, recall_at_k=1.0, mrr=1.0, ndcg_at_k=1.0)
    recorded = write_baseline(tmp_path / "baseline.json", report, settings)

    assert recorded.comparable_to(settings) is True
    assert recorded.comparable_to(Settings.model_validate({"retrieval": {"top_k": 99}})) is False
    assert (
        recorded.comparable_to(
            Settings.model_validate({"embeddings": {"model": "some/other-model"}})
        )
        is False
    )


def test_the_baseline_covers_every_answerable_record() -> None:
    """Scored plus adversarial must account for the whole set, or coverage silently shrank."""
    baseline = load_baseline(BASELINE)
    records = load_golden_set(GOLDEN)
    adversarial = sum(1 for record in records if record.adversarial)

    assert baseline is not None
    assert baseline.scored + adversarial == len(records)


def test_a_hand_edited_baseline_is_refused_rather_than_ignored(tmp_path: Path) -> None:
    """A silently-ignored baseline would let the gate pass; a missing one blocks it."""
    broken = tmp_path / "baseline.json"
    broken.write_text(json.dumps({"k": 5, "made_up_field": 1}), encoding="utf-8")

    with pytest.raises(FasterRagError):
        load_baseline(broken)


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
