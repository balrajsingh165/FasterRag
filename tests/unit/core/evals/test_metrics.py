import math

import pytest

from fasterrag.core.evals.metrics import dcg, ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_counts_hits_against_the_ground_truth() -> None:
    assert recall_at_k([True, False, True], total_relevant=4) == pytest.approx(0.5)


def test_recall_is_perfect_when_everything_relevant_is_found() -> None:
    assert recall_at_k([True, True], total_relevant=2) == pytest.approx(1.0)


def test_recall_only_counts_inside_the_window() -> None:
    hits = [False, False, True]

    assert recall_at_k(hits, total_relevant=1, k=2) == pytest.approx(0.0)
    assert recall_at_k(hits, total_relevant=1, k=3) == pytest.approx(1.0)


def test_recall_never_exceeds_one() -> None:
    assert recall_at_k([True, True, True], total_relevant=2) == pytest.approx(1.0)


def test_recall_of_an_empty_ground_truth_is_zero() -> None:
    assert recall_at_k([True], total_relevant=0) == 0.0


def test_reciprocal_rank_is_the_inverse_of_the_first_hit() -> None:
    assert reciprocal_rank([True, False]) == pytest.approx(1.0)
    assert reciprocal_rank([False, True]) == pytest.approx(0.5)
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)


def test_reciprocal_rank_ignores_hits_after_the_first() -> None:
    assert reciprocal_rank([False, True, True]) == pytest.approx(0.5)


def test_reciprocal_rank_is_zero_when_nothing_is_relevant() -> None:
    assert reciprocal_rank([False, False]) == 0.0
    assert reciprocal_rank([]) == 0.0


def test_dcg_discounts_by_position() -> None:
    assert dcg([True]) == pytest.approx(1.0)
    assert dcg([False, True]) == pytest.approx(1 / math.log2(3))
    assert dcg([True, True]) == pytest.approx(1.0 + 1 / math.log2(3))


def test_ndcg_is_one_for_a_perfect_ranking() -> None:
    assert ndcg_at_k([True, True, False], total_relevant=2) == pytest.approx(1.0)


def test_ndcg_penalizes_a_worse_order_of_the_same_hits() -> None:
    good = ndcg_at_k([True, False, True], total_relevant=2)
    bad = ndcg_at_k([False, True, True], total_relevant=2)

    assert good > bad
    assert good < 1.0


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved() -> None:
    assert ndcg_at_k([False, False], total_relevant=2) == 0.0


def test_ndcg_of_an_empty_ground_truth_is_zero() -> None:
    assert ndcg_at_k([True], total_relevant=0) == 0.0


def test_ndcg_does_not_penalize_a_window_smaller_than_the_ground_truth() -> None:
    """Two relevant chunks exist but only one fits in the window; finding it is perfect."""
    assert ndcg_at_k([True], total_relevant=2, k=1) == pytest.approx(1.0)


def test_ndcg_never_exceeds_one() -> None:
    assert ndcg_at_k([True, True, True], total_relevant=1) <= 1.0
