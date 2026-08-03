import json
from pathlib import Path

FIXTURE = Path(__file__).resolve().parents[3] / "tests" / "eval" / "datasets" / "policies"


def records() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in (FIXTURE / "golden.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def baseline() -> dict[str, float]:
    payload: dict[str, float] = json.loads((FIXTURE / "baseline.json").read_text(encoding="utf-8"))
    return payload


def test_the_baseline_leaves_ranking_headroom() -> None:
    """The point of this fixture: recall saturates but ranking does not.

    The handbook fixture scores 1.0 on all three, so a subtle ranking regression has nowhere
    to show. Here recall is 1.0 while MRR and nDCG sit below it, which is what makes a change
    that merely reorders results visible.
    """
    recorded = baseline()

    assert recorded["recall_at_k"] == 1.0
    assert recorded["mrr"] < 1.0
    assert recorded["ndcg_at_k"] < 1.0


def test_the_headroom_exceeds_the_default_tolerance() -> None:
    """Headroom smaller than the gate's tolerance would be headroom the gate cannot use."""
    recorded = baseline()

    assert 1.0 - recorded["mrr"] > 0.02
    assert 1.0 - recorded["ndcg_at_k"] > 0.02


def test_every_answerable_query_names_exactly_one_document() -> None:
    """Ambiguous ground truth would make a ranking change unmeasurable rather than visible."""
    for record in records():
        relevant = record["relevant_document_ids"]
        assert isinstance(relevant, list)
        assert len(relevant) <= 1, record["id"]


def test_no_record_carries_metadata() -> None:
    """CRITICAL: golden metadata is passed to the retriever as a *filter*, not annotation.

    Descriptive keys here filter every candidate away and score a flat 0.0 that reads as
    total retrieval failure. This fixture scored exactly that until the keys were removed.
    """
    for record in records():
        assert record["metadata"] == {}, record["id"]


def test_the_corpus_has_near_identical_siblings() -> None:
    """Discrimination is the whole design: each document has look-alikes it must beat."""
    names = sorted(path.stem for path in (FIXTURE / "corpus").glob("*.md"))
    families = {name.rsplit("-", 2)[0] if name[-4:].isdigit() else "access" for name in names}

    assert len(names) == 12
    for family in families:
        siblings = [name for name in names if name.startswith(family)]
        assert len(siblings) >= 4, family


def test_the_set_includes_unanswerable_queries() -> None:
    """A retriever must decline to return something confident when nothing is relevant."""
    adversarial = [record for record in records() if not record["relevant_document_ids"]]

    assert len(adversarial) >= 3
