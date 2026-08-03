import inspect

from fasterrag.services import evaluation


def test_faithfulness_scoring_is_a_separate_entry_point() -> None:
    """Retrieval scoring is local and free; this is two provider calls per record."""
    assert hasattr(evaluation, "score_faithfulness")
    assert "score_faithfulness" in evaluation.__all__


def test_retrieval_scoring_does_not_call_it_implicitly() -> None:
    """A harness that silently started generating would bill for a metrics run."""
    source = inspect.getsource(evaluation.score_collection)

    assert "score_faithfulness" not in source
    assert "build_generation" not in source


def test_adversarial_records_are_excluded() -> None:
    """They are deliberately unanswerable, so a refusal is the correct behaviour."""
    source = inspect.getsource(evaluation.score_faithfulness)

    assert "not record.adversarial" in source


def test_a_refusal_is_counted_rather_than_scored_zero() -> None:
    """Withholding an answer is a correct outcome under D5, not a failure."""
    source = inspect.getsource(evaluation.score_faithfulness)

    assert "refused" in source
    assert "insufficient_evidence" in source


def test_the_generation_service_is_always_closed() -> None:
    source = inspect.getsource(evaluation.score_faithfulness)

    assert "finally:" in source
    assert "service.close()" in source
