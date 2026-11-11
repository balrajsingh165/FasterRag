"""The soak detector's own tests: can it see a leak, and can it stay quiet without one.

These carry no ``soak`` marker and run in the ordinary unit lane, for the reason
``tests/load/test_load_harness.py`` gives: the ``soak`` marker means "needs a long run on
reference hardware", and nothing here needs either. What they assert is that the decision
rule in ``trend.py`` can fail — a soak test that cannot distinguish a seeded leak from a
flat series is green forever and proves nothing, which is the vacuity class this repository
keeps finding.

Every series here is synthetic and seeded, so each case is a statement about the rule and
never about the machine it ran on.
"""

from __future__ import annotations

import random

import pytest

from tests.soak.trend import (
    HANDLE_BUDGET,
    MINIMUM_SAMPLES,
    SoakError,
    assess,
    theil_sen_slope,
)

INTERVAL_S = 2.0
DURATION_SAMPLES = 60
BASELINE_BYTES = 120_000_000.0
NOISE_BYTES = 4_000_000.0
ITERATIONS = 400


def times(count: int = DURATION_SAMPLES) -> list[float]:
    """Return evenly spaced sample timestamps."""
    return [index * INTERVAL_S for index in range(count)]


def flat(seed: int, count: int = DURATION_SAMPLES) -> list[float]:
    """Return a noisy series with no trend."""
    rng = random.Random(seed)
    return [BASELINE_BYTES + rng.uniform(-NOISE_BYTES, NOISE_BYTES) for _ in range(count)]


def leaking(seed: int, per_second: float, count: int = DURATION_SAMPLES) -> list[float]:
    """Return a noisy series with a seeded linear leak."""
    rng = random.Random(seed)
    return [
        BASELINE_BYTES + per_second * index * INTERVAL_S + rng.uniform(-NOISE_BYTES, NOISE_BYTES)
        for index in range(count)
    ]


def sawtooth(count: int = DURATION_SAMPLES) -> list[float]:
    """Return an allocator-shaped series: climb, release, climb, release, no net growth."""
    return [BASELINE_BYTES + (index % 10) * NOISE_BYTES for index in range(count)]


def test_a_seeded_leak_is_detected() -> None:
    """A leak of 1 MB/s over a two-minute window must be called growth."""
    trend = assess(
        "rss",
        times(),
        leaking(1, per_second=1_000_000.0),
        budget=BASELINE_BYTES * 0.10,
        iterations=ITERATIONS,
    )

    assert trend.growing
    assert trend.significant
    assert trend.material
    assert trend.slope_per_s == pytest.approx(1_000_000.0, rel=0.2)


def test_a_flat_noisy_series_is_not_called_growth() -> None:
    """Noise many times the leak's per-sample step must not read as a trend."""
    for seed in range(5):
        trend = assess(
            "rss",
            times(),
            flat(seed),
            budget=BASELINE_BYTES * 0.10,
            iterations=ITERATIONS,
        )

        assert not trend.growing, trend.summary()


def test_the_false_positive_rate_over_flat_series_stays_near_the_significance_level() -> None:
    """The calibration that justifies alpha, and the reason the budget is over the run."""
    seeds = 40
    flagged = sum(
        assess("rss", times(20), flat(seed, 20), budget=0.0, iterations=200).growing
        for seed in range(seeds)
    )

    assert flagged <= seeds // 10, f"{flagged} of {seeds} flat series were called growth"


def test_a_sawtooth_is_not_called_growth() -> None:
    """Arenas taken and released in cycles are the normal shape, not a leak."""
    trend = assess(
        "rss",
        times(),
        sawtooth(),
        budget=BASELINE_BYTES * 0.10,
        iterations=ITERATIONS,
    )

    assert not trend.growing, trend.summary()


def test_comparing_the_two_endpoints_would_have_been_fooled_where_this_rule_is_not() -> None:
    """The reason the rule regresses rather than subtracting, stated as a test.

    A sawtooth sampled just after a release and just before the next one reports a large
    endpoint-to-endpoint rise with no growth in it at all.
    """
    series = sawtooth()
    endpoints = series[-1] - series[0]

    trend = assess("rss", times(), series, budget=BASELINE_BYTES * 0.10, iterations=ITERATIONS)

    assert endpoints > 0
    assert not trend.growing


def test_a_significant_but_immaterial_trend_is_not_a_failure() -> None:
    """A trend the run can see but whose growth stays under budget is reported, not failed."""
    trend = assess(
        "rss",
        times(),
        leaking(2, per_second=50_000.0),
        budget=BASELINE_BYTES * 0.10,
        iterations=ITERATIONS,
    )

    assert trend.significant
    assert not trend.material
    assert not trend.growing


def test_a_leak_smaller_than_the_run_could_detect_is_reported_rather_than_missed_silently() -> None:
    """The honest half: a short run cannot see a slow leak, and must say how slow.

    This is the property that makes a two-minute proving run honest about a 24 h claim.
    """
    trend = assess(
        "rss",
        times(),
        leaking(4, per_second=10.0),
        budget=BASELINE_BYTES * 0.10,
        iterations=ITERATIONS,
    )

    assert not trend.growing
    assert trend.detectable > trend.growth


def test_the_smallest_detectable_growth_rises_with_the_noise() -> None:
    """A noisier run can rule out less, and the report has to say so."""
    quiet = assess(
        "rss",
        times(),
        [BASELINE_BYTES + (index % 2) for index in range(DURATION_SAMPLES)],
        budget=0.0,
        iterations=ITERATIONS,
    )
    loud = assess("rss", times(), flat(3), budget=0.0, iterations=ITERATIONS)

    assert loud.detectable > quiet.detectable


def test_a_series_shorter_than_the_minimum_is_refused_rather_than_judged() -> None:
    """Too few samples is an error about the run, never a quiet pass."""
    short = MINIMUM_SAMPLES - 1

    with pytest.raises(SoakError, match="below the"):
        assess("rss", times(short), flat(0, short), budget=0.0, iterations=ITERATIONS)


def test_one_outlier_does_not_set_the_slope() -> None:
    """The property least squares does not have, which is why Theil-Sen is used."""
    series = [BASELINE_BYTES] * DURATION_SAMPLES
    series[0] = BASELINE_BYTES - NOISE_BYTES * 500

    assert theil_sen_slope(times(), series) == 0.0


def test_a_handle_leak_of_one_descriptor_per_cycle_is_detected() -> None:
    """Handles carry a zero budget, so the rule must see a slow integer climb."""
    series = [10.0 + index // 4 for index in range(DURATION_SAMPLES)]

    trend = assess("handles", times(), series, budget=HANDLE_BUDGET, iterations=ITERATIONS)

    assert trend.growing, trend.summary()


def test_a_constant_handle_count_is_not_growth() -> None:
    """The ordinary case: handles open once and stay open."""
    trend = assess(
        "handles",
        times(),
        [10.0] * DURATION_SAMPLES,
        budget=HANDLE_BUDGET,
        iterations=ITERATIONS,
    )

    assert not trend.growing, trend.summary()


def test_a_series_with_one_timestamp_is_refused() -> None:
    """A sampler that never advanced its clock has measured nothing."""
    with pytest.raises(SoakError, match="one timestamp"):
        theil_sen_slope([1.0] * DURATION_SAMPLES, flat(0))
