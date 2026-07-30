from collections.abc import Sequence

import pytest

from fasterrag.core.retrieval.fusion import DEFAULT_RRF_K, FusedResult, Ranking, rrf_fuse
from fasterrag.retrieval import rrf_fuse as public_rrf_fuse


def ids(results: Sequence[FusedResult]) -> list[str]:
    return [result.id for result in results]


def test_the_documented_constant_is_sixty() -> None:
    assert DEFAULT_RRF_K == 60


def test_a_single_ranking_is_returned_in_order() -> None:
    fused = rrf_fuse(["a", "b", "c"])

    assert [result.id for result in fused] == ["a", "b", "c"]


def test_scores_follow_the_documented_formula() -> None:
    fused = rrf_fuse(["a", "b"], k=60)

    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[1].score == pytest.approx(1 / 62)


def test_a_document_both_legs_rank_beats_one_leg_favourite() -> None:
    dense = ["only_dense", "agreed", "x"]
    sparse = ["only_sparse", "agreed", "y"]

    fused = rrf_fuse(dense, sparse)

    assert fused[0].id == "agreed"


def test_agreement_outranks_a_single_first_place() -> None:
    dense = ["solo", "shared"]
    sparse = ["other", "shared"]

    fused = rrf_fuse(dense, sparse)

    assert fused[0].id == "shared"
    assert fused[0].score > fused[1].score


def test_every_document_from_every_leg_survives() -> None:
    fused = rrf_fuse(["a", "b"], ["c", "d"])

    assert set(ids(fused)) == {"a", "b", "c", "d"}


def test_each_leg_rank_is_recorded() -> None:
    fused = rrf_fuse(
        Ranking(name="dense", ids=["a", "b"]),
        Ranking(name="bm25", ids=["b", "a"]),
    )
    by_id = {result.id: result for result in fused}

    assert by_id["a"].rank_in("dense") == 1
    assert by_id["a"].rank_in("bm25") == 2
    assert by_id["b"].rank_in("dense") == 2


def test_a_document_absent_from_a_leg_has_no_rank_there() -> None:
    fused = rrf_fuse(
        Ranking(name="dense", ids=["a"]),
        Ranking(name="bm25", ids=["b"]),
    )
    by_id = {result.id: result for result in fused}

    assert by_id["a"].rank_in("bm25") is None


def test_weights_tilt_the_balance_between_legs() -> None:
    dense = Ranking(name="dense", ids=["d1"], weight=1.0)

    light = rrf_fuse(dense, Ranking(name="bm25", ids=["s1"], weight=0.1))
    heavy = rrf_fuse(dense, Ranking(name="bm25", ids=["s1"], weight=5.0))

    assert light[0].id == "d1"
    assert heavy[0].id == "s1"


def test_agreement_still_beats_a_heavily_weighted_single_leg() -> None:
    fused = rrf_fuse(
        Ranking(name="dense", ids=["d1", "shared"], weight=1.0),
        Ranking(name="bm25", ids=["s1", "shared"], weight=0.1),
    )

    assert fused[0].id == "shared"


def test_a_zero_weight_leg_contributes_nothing() -> None:
    fused = rrf_fuse(
        Ranking(name="dense", ids=["a"], weight=1.0),
        Ranking(name="bm25", ids=["b"], weight=0.0),
    )
    by_id = {result.id: result for result in fused}

    assert by_id["b"].score == 0.0
    assert fused[0].id == "a"


def test_a_larger_constant_flattens_the_advantage_of_top_positions() -> None:
    tight = rrf_fuse(["a", "b"], k=1)
    loose = rrf_fuse(["a", "b"], k=1000)

    assert tight[0].score / tight[1].score > loose[0].score / loose[1].score


def test_a_non_positive_constant_is_refused() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        rrf_fuse(["a"], k=0)

    with pytest.raises(ValueError, match="must be positive"):
        rrf_fuse(["a"], k=-1)


def test_fusing_nothing_returns_nothing() -> None:
    assert rrf_fuse() == []
    assert rrf_fuse([], []) == []


def test_ties_break_deterministically() -> None:
    first = rrf_fuse(["b", "a"], ["a", "b"])
    second = rrf_fuse(["b", "a"], ["a", "b"])

    assert ids(first) == ids(second)


def test_the_public_surface_exports_the_documented_function() -> None:
    assert public_rrf_fuse is rrf_fuse
