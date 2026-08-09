"""Reciprocal Rank Fusion invariants, over generated legs rather than chosen ones.

Fusion is the last thing that touches the ranking before a chunk becomes context, so a
document it misplaces is a wrong answer with a plausible citation attached. The example
suite beside this one pinned each invariant with one hand-picked leg pair; the general
forms turned out to be false in two ways, both fixed under TASK-0228:

* ``rrf_fuse(["a", "b", "b"])`` returned ``["b", "a"]``. A repeated id was scored once per
  occurrence, so a single leg could reorder its own input, and ``rank_in`` reported the
  document's *worst* position because the later write won.
* A leg weighted ``0.0`` still contributed every document it ranked, at score ``0.0``.
  ``QueryService`` truncates the fused list to ``candidates`` and hands the whole shortlist
  to the reranker, which can promote any of them — so a leg an operator had weighted out of
  the fusion could still put chunks in front of the model.

Commutativity in the legs is asserted here too, but as a guarantee held rather than a bug
found: replacing :func:`math.fsum` with a running total leaves every test below green,
because the value ranges ``rrf_fuse`` permits are ~11 orders of magnitude too narrow for
reassociation to change a sum. The properties are kept because the implementation should
not depend on that coincidence, not because they once caught something.

What is deliberately *not* asserted here: that a larger ``rrf_k`` flattens a multi-leg
fusion. It does not. ``k`` damps position within a leg, and as it grows every position
converges to ``1/k``, so the score tends toward the weighted count of legs that ranked the
document — agreement is amplified, not flattened. The flattening property is single-leg.
"""

from collections.abc import Sequence
from itertools import pairwise

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from fasterrag.core.retrieval.fusion import FusedResult, Ranking, rrf_fuse

ID = st.text(alphabet="abcde", min_size=1, max_size=2)

# CRITICAL: no `unique=True`. Repeated ids inside one leg are the input that broke
# order preservation, and a strategy that filtered them out would have made every
# assertion below true of the buggy implementation as well.
LEG = st.lists(ID, max_size=8)

WEIGHT = st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False)
POSITIVE_WEIGHT = st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
K = st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False)


def ids(results: Sequence[FusedResult]) -> list[str]:
    """Return the fused order."""
    return [result.id for result in results]


def scores(results: Sequence[FusedResult]) -> dict[str, float]:
    """Return the fused score of every document."""
    return {result.id: result.score for result in results}


def legs_from(bodies: Sequence[Sequence[str]], weights: Sequence[float]) -> list[Ranking]:
    """Build uniquely named legs, so every leg's rank is recorded under its own key."""
    return [
        Ranking(name=f"leg_{index}", ids=list(body), weight=weight)
        for index, (body, weight) in enumerate(zip(bodies, weights, strict=True))
    ]


@st.composite
def leg_sets(draw: st.DrawFn, weights: st.SearchStrategy[float] = WEIGHT) -> list[Ranking]:
    """Draw between one and four named legs with independent weights."""
    bodies = draw(st.lists(LEG, min_size=1, max_size=4))
    drawn = draw(st.lists(weights, min_size=len(bodies), max_size=len(bodies)))
    return legs_from(bodies, drawn)


@st.composite
def leg_sets_with_a_silenced_leg(draw: st.DrawFn) -> list[Ranking]:
    """Draw legs of which at least one is weighted zero, by construction.

    ``WEIGHT`` reaches zero on its own, but only for two examples in five (measured). Reaching
    the silenced case by drawing freely and then filtering with ``assume`` therefore discards
    most of the budget, leaves the two tests below one unlucky seed away from Hypothesis's
    ``filter_too_much`` health check, and — worse — makes them *vacuous* on every example it
    does keep by accident rather than by design. Silencing a drawn leg instead means every
    example exercises the path the test exists to check.
    """
    legs = draw(leg_sets())
    silenced = draw(st.integers(min_value=0, max_value=len(legs) - 1))
    return [
        Ranking(name=leg.name, ids=leg.ids, weight=0.0 if index == silenced else leg.weight)
        for index, leg in enumerate(legs)
    ]


@given(leg=LEG, k=K, weight=POSITIVE_WEIGHT)
def test_a_single_leg_comes_back_in_its_own_order(leg: list[str], k: float, weight: float) -> None:
    """One leg fused alone is that leg, deduplicated at each id's best position.

    The general form of "a single ranking is returned in order". Scoring a repeat as a
    second vote made a twice-listed runner-up overtake the leg's own first result.
    """
    fused = rrf_fuse(Ranking(name="only", ids=leg, weight=weight), k=k)

    assert ids(fused) == list(dict.fromkeys(leg))


@given(leg=LEG, k=K)
def test_a_repeated_id_is_ranked_at_its_best_position(leg: list[str], k: float) -> None:
    """``rank_i(d)`` is one number, and it is the best one."""
    assume(leg)
    fused = rrf_fuse(Ranking(name="only", ids=leg), k=k)

    for result in fused:
        assert result.rank_in("only") == leg.index(result.id) + 1


@given(leg=LEG, other=LEG, k=K)
def test_a_repeat_is_not_a_second_vote(leg: list[str], other: list[str], k: float) -> None:
    """Re-listing a leg's results behind itself leaves every best position where it was."""
    fused = rrf_fuse(Ranking(name="a", ids=leg), Ranking(name="b", ids=other), k=k)
    doubled = rrf_fuse(
        Ranking(name="a", ids=[*leg, *leg, *leg]),
        Ranking(name="b", ids=other),
        k=k,
    )

    assert ids(fused) == ids(doubled)
    assert scores(fused) == scores(doubled)


@given(legs=leg_sets(), k=K, data=st.data())
def test_the_order_the_legs_arrive_in_changes_nothing(
    legs: list[Ranking], k: float, data: st.DataObject
) -> None:
    """Fusion is a sum, so it must be commutative in the legs — exactly, not nearly.

    Exact equality is the point: a score that shifts by one ULP when the legs are reordered
    is invisible until two documents that should have tied by id swap places. ``fsum`` makes
    that impossible by construction; see the module docstring for why a running total would
    also pass today, and why the guarantee is not left resting on that.
    """
    shuffled = data.draw(st.permutations(legs))
    assume([leg.name for leg in shuffled] != [leg.name for leg in legs])

    reference = rrf_fuse(*legs, k=k)
    permuted = rrf_fuse(*shuffled, k=k)

    assert ids(permuted) == ids(reference)
    assert scores(permuted) == scores(reference)
    assert [dict(result.ranks) for result in permuted] == [
        dict(result.ranks) for result in reference
    ]


@given(legs=leg_sets(), k=K)
def test_the_output_is_the_union_of_the_weighted_legs_exactly_once(
    legs: list[Ranking], k: float
) -> None:
    """Nothing invented, nothing lost, nothing twice — and nothing from a silenced leg."""
    fused = rrf_fuse(*legs, k=k)
    expected = {identifier for leg in legs if leg.weight != 0 for identifier in leg.ids}

    assert set(ids(fused)) == expected
    assert len(ids(fused)) == len(set(ids(fused)))


@given(legs=leg_sets_with_a_silenced_leg(), k=K)
def test_a_zero_weight_leg_leaves_every_other_document_untouched(
    legs: list[Ranking], k: float
) -> None:
    """Weight zero is the end of the weighting scale: the leg may as well not have run."""
    with_silenced = rrf_fuse(*legs, k=k)
    without = rrf_fuse(*(leg for leg in legs if leg.weight != 0), k=k)

    assert ids(with_silenced) == ids(without)
    assert scores(with_silenced) == scores(without)


@given(legs=leg_sets_with_a_silenced_leg(), k=K)
def test_a_zero_weight_leg_still_reports_where_it_ranked_a_surviving_document(
    legs: list[Ranking], k: float
) -> None:
    """Silencing a leg must not blind the trace to it — ``bm25_rank`` is diagnostic."""
    fused = rrf_fuse(*legs, k=k)
    by_id = {result.id: result for result in fused}

    for leg in legs:
        if leg.weight != 0:
            continue
        for position, identifier in enumerate(leg.ids, start=1):
            result = by_id.get(identifier)
            if result is not None and leg.ids.index(identifier) + 1 == position:
                assert result.rank_in(leg.name) == position


@given(legs=leg_sets(weights=POSITIVE_WEIGHT), k=K, data=st.data())
def test_a_document_no_leg_ranks_lower_never_scores_lower(
    legs: list[Ranking], k: float, data: st.DataObject
) -> None:
    """Dominance: rank at least as well everywhere, and appear in at least as many legs.

    Random legs almost never dominate one another, so the pair is constructed — ``alpha``
    is inserted immediately ahead of ``beta`` in every leg ``beta`` appears in, and into a
    generated subset of the rest.
    """
    extras = data.draw(st.lists(st.booleans(), min_size=len(legs), max_size=len(legs)))
    cuts = data.draw(st.lists(st.integers(min_value=0, max_value=8), min_size=len(legs)))
    carries = data.draw(st.lists(st.booleans(), min_size=len(legs), max_size=len(legs)))
    assume(any(carries))

    built: list[Ranking] = []
    for leg, carry, extra, cut in zip(legs, carries, extras, cuts, strict=False):
        body = list(leg.ids)
        at = min(cut, len(body))
        if carry:
            body[at:at] = ["alpha", "beta"]
        elif extra:
            body[at:at] = ["alpha"]
        built.append(Ranking(name=leg.name, ids=body, weight=leg.weight))

    fused = rrf_fuse(*built, k=k)
    by_id = {result.id: result for result in fused}

    assert by_id["alpha"].score > by_id["beta"].score
    assert ids(fused).index("alpha") < ids(fused).index("beta")


@given(
    position=st.integers(min_value=2, max_value=40),
    k=st.floats(min_value=40.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    weight=POSITIVE_WEIGHT,
)
def test_agreement_between_legs_beats_one_leg_s_favourite(
    position: int, k: float, weight: float
) -> None:
    """Two legs agreeing at rank ``p`` beat a single leg's rank-1 result while ``p < k+2``.

    That threshold is the whole reason ``k`` defaults to 60 rather than 0: it is derived
    from ``2w/(k+p) > w/(k+1)``, so the property is checked against the arithmetic rather
    than against one example that happened to work.
    """
    assume(position < k + 2)
    dense = ["solo", *(f"d{n}" for n in range(position - 2)), "agreed"]
    sparse = [*(f"s{n}" for n in range(position - 1)), "agreed"]

    fused = rrf_fuse(
        Ranking(name="dense", ids=dense, weight=weight),
        Ranking(name="sparse", ids=sparse, weight=weight),
        k=k,
    )

    assert fused[0].id == "agreed"
    assert fused[0].ranks == {"dense": position, "sparse": position}


@given(
    position=st.integers(min_value=2, max_value=400),
    k=st.floats(min_value=0.01, max_value=20.0, allow_nan=False, allow_infinity=False),
    weight=POSITIVE_WEIGHT,
)
def test_below_that_threshold_the_confident_leg_wins_instead(
    position: int, k: float, weight: float
) -> None:
    """The other side of the same inequality, so the test above cannot pass vacuously.

    Only ``solo`` against ``agreed`` is asserted: the padding that pushes ``agreed`` down
    occupies the good positions itself, so it legitimately outranks both.
    """
    assume(position > k + 2)
    dense = ["solo", *(f"d{n}" for n in range(position - 2)), "agreed"]
    sparse = [*(f"s{n}" for n in range(position - 1)), "agreed"]

    fused = rrf_fuse(
        Ranking(name="dense", ids=dense, weight=weight),
        Ranking(name="sparse", ids=sparse, weight=weight),
        k=k,
    )
    order = ids(fused)

    assert order.index("solo") < order.index("agreed")


@given(
    leg=st.lists(ID, min_size=2, max_size=8, unique=True),
    low=st.floats(min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False),
    gap=st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
)
def test_a_larger_constant_flattens_one_leg_s_positions(
    leg: list[str], low: float, gap: float
) -> None:
    """Within a leg, raising ``k`` compresses the ratio between every adjacent pair."""
    tight = rrf_fuse(leg, k=low)
    loose = rrf_fuse(leg, k=low + gap)

    assert ids(tight) == leg
    assert ids(loose) == leg
    for (better, worse), (flat_better, flat_worse) in zip(
        pairwise(tight), pairwise(loose), strict=True
    ):
        assert better.score / worse.score > flat_better.score / flat_worse.score


@st.composite
def mirrored_leg_pairs(draw: st.DrawFn) -> list[Ranking]:
    """Draw legs in mirror-image pairs, so ``alpha`` and ``beta`` must score identically.

    Each pair is one generated leg plus its twin with ``alpha`` and ``beta`` swapped and the
    same weight, so the two documents end up holding the *same multiset* of contributions —
    only in a different order. Two pairs are the minimum that can expose the difference:
    with a single pair each document sums two terms, and adding two floats is commutative.
    """
    pairs = draw(st.integers(min_value=2, max_value=3))
    legs: list[Ranking] = []
    for index in range(pairs):
        body = draw(st.lists(ID, max_size=4))
        cuts = sorted(draw(st.lists(st.integers(min_value=0, max_value=4), min_size=2, max_size=2)))
        forward = list(body)
        forward[cuts[0] : cuts[0]] = ["alpha"]
        forward[cuts[1] + 1 : cuts[1] + 1] = ["beta"]
        mirrored = [
            "beta" if name == "alpha" else "alpha" if name == "beta" else name for name in forward
        ]
        weight = draw(st.sampled_from([0.3, 0.7, 1.0, 1.3, 2.0]))
        legs.append(Ranking(name=f"leg_{index}a", ids=forward, weight=weight))
        legs.append(Ranking(name=f"leg_{index}b", ids=mirrored, weight=weight))
    return legs


@given(legs=mirrored_leg_pairs(), k=K)
def test_documents_holding_the_same_contributions_score_the_same(
    legs: list[Ranking], k: float
) -> None:
    """Equal sums must be equal floats, or the id tie-break silently never runs.

    ``sum`` is not associative in general, so the same multiset of contributions added in two
    orders can land one ULP apart — which looks harmless until the ordering of two documents
    that should have tied starts depending on which leg the caller listed first.

    This test does not currently distinguish ``sum`` from ``fsum``, and no input does: see the
    module docstring. It pins the invariant, not the arithmetic.
    """
    fused = rrf_fuse(*legs, k=k)
    by_id = scores(fused)

    assert by_id["alpha"] == by_id["beta"]
    assert ids(fused).index("alpha") < ids(fused).index("beta")


@given(
    first=ID,
    second=ID,
    k=K,
    weight=POSITIVE_WEIGHT,
)
def test_an_exact_tie_breaks_by_ascending_id(
    first: str, second: str, k: float, weight: float
) -> None:
    """Mirrored ranks tie exactly, and the id decides — never the arrival order.

    Constructed, because two documents landing on the same score by accident is not
    something uniform generation produces.
    """
    assume(first != second)
    fused = rrf_fuse(
        Ranking(name="dense", ids=[first, second], weight=weight),
        Ranking(name="sparse", ids=[second, first], weight=weight),
        k=k,
    )

    assert fused[0].score == fused[1].score
    assert ids(fused) == sorted([first, second])


@given(legs=leg_sets(), k=K)
def test_scores_follow_the_documented_formula(legs: list[Ranking], k: float) -> None:
    """An oracle written from the docstring, not from the implementation."""
    fused = rrf_fuse(*legs, k=k)

    for result in fused:
        expected = sorted(
            leg.weight / (k + list(leg.ids).index(result.id) + 1)
            for leg in legs
            if leg.weight != 0 and result.id in leg.ids
        )
        assert result.score == pytest.approx(sum(expected))


@given(
    legs=leg_sets(),
    k=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
)
def test_a_non_positive_constant_is_always_refused(legs: list[Ranking], k: float) -> None:
    """Including ``-0.0``, which is not positive however it prints."""
    with pytest.raises(ValueError, match="must be positive"):
        rrf_fuse(*legs, k=k)
