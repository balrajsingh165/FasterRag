"""Decide whether a resource series grew, from the series rather than from its endpoints.

A soak test's whole question is "did this leak?", and the obvious way to answer it — read
RSS at the start, read it at the end, subtract — cannot answer it at all. Two samples have
no notion of noise, so the difference between a leak and an allocator that happened to hold
a free list at the second reading is invisible; the comparison returns a number either way
and the number reads as evidence. Every function here exists to replace that subtraction.

**Theil-Sen, not least squares.** A process's RSS series is a sawtooth: arenas are taken in
jumps and released in jumps, and a garbage collection at the wrong moment is a single
sample far off the line. Least squares weights that sample by the square of its residual,
so one collection cycle can set the slope. Theil-Sen takes the median of the pairwise
slopes and needs about 29% of the samples to be corrupted before it moves, which is the
property the sawtooth demands.

**A permutation test, not a threshold.** Having a slope is not having a trend. The null
hypothesis a soak run has to reject is that the readings are exchangeable in time — that
the same numbers in a different order would have produced a slope just as steep. So the
observed slope is compared against the slope distribution of the *same samples* shuffled,
which needs no assumption about how RSS is distributed (it is not normal, not continuous,
and not independent between neighbours) and no absolute number chosen by a human. On this
project's hardware that matters twice over: TASK-0350 measured a hosted runner's own
variance at 2.07x on an identical commit, so any constant anybody wrote here would be a
constant about a machine rather than about fasterRag.

**Significance is not materiality, so both are required.** A run long enough will find a
statistically real slope in almost anything, including the import caches that fill once and
stop. A finding therefore has to clear two bars: the trend is distinguishable from a
reordering of itself, *and* the growth it accounts for exceeds a budget expressed as a
fraction of the process's own baseline, which is the only way the budget stays meaningful
on a machine with a different footprint.

**Growth is counted over the window that was observed, never over the one documented.**
The first version of this module multiplied the slope by the documented 24 h horizon, and
its own tests killed it: a two-minute run amplified by 720 turns every noise slope into
gigabytes, every series becomes material, and the materiality bar stops filtering anything
— a 5% false-positive rate straight through to red. So ``growth`` is ``slope * run_s``, the
rise the samples actually account for. The 24 h figure is still reported, as
``extrapolated``, and it is labelled an extrapolation because that is what it is: a
statement that assumes a linear leak continues unchanged for 720 times longer than anyone
watched.

**What the run could not have seen is reported with what it saw.** ``Trend.detectable`` is
the (1 - alpha) quantile of the null slope distribution over the same window: the smallest
growth this particular run had the power to call. A short run's ``detectable`` is large,
and printing it is the difference between "no leak" and "no leak this run could have
found" — only the second of which is true.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

ALPHA: Final = 0.05
"""One-sided significance level for the permutation test."""

ITERATIONS: Final = 1000
"""Permutations per assessment; the smallest reportable p-value is 1 / (ITERATIONS + 1)."""

MINIMUM_SAMPLES: Final = 12
"""Below this the pairwise slope set is too small for the median to mean anything."""

HORIZON_S: Final = 24 * 60 * 60
"""The documented soak duration, used only for the reported extrapolation."""

RSS_BUDGET_FRACTION: Final = 0.10
"""Growth over the observed window allowed as a fraction of the run's own baseline RSS."""

HANDLE_BUDGET: Final = 0.0
"""Handles do not scale with hardware, so any significant upward trend is a leak."""


class SoakError(Exception):
    """Raised when a series cannot support the question being asked of it."""


@dataclass(frozen=True, slots=True)
class Trend:
    """One resource series' verdict, carrying every number the verdict was made from."""

    metric: str
    samples: int
    run_s: float
    baseline: float
    slope_per_s: float
    p_value: float
    growth: float
    detectable: float
    budget: float

    @property
    def significant(self) -> bool:
        """Return whether the slope is steeper than a reordering of the same samples."""
        return self.p_value <= ALPHA

    @property
    def material(self) -> bool:
        """Return whether the growth over the observed window exceeds the budget."""
        return self.growth > self.budget

    @property
    def growing(self) -> bool:
        """Return whether this series grew: distinguishable from noise and over budget."""
        return self.significant and self.material

    @property
    def extrapolated(self) -> float:
        """Return the 24 h figure this slope implies, which no short run has observed."""
        return self.slope_per_s * HORIZON_S

    def summary(self) -> str:
        """Return one line stating the verdict, its numbers, and this run's blind spot."""
        verdict = "GROWING" if self.growing else "no growth detected"
        return (
            f"{self.metric}: {verdict} | n={self.samples} over {self.run_s:,.0f}s "
            f"baseline={self.baseline:,.0f} slope={self.slope_per_s:+.6g}/s "
            f"p={self.p_value:.4f} growth={self.growth:+,.0f} budget={self.budget:,.0f} "
            f"smallest detectable={self.detectable:,.0f} "
            f"extrapolated/24h={self.extrapolated:+,.0f}"
        )


def theil_sen_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Return the median of every pairwise slope, which one outlier cannot move."""
    if len(xs) != len(ys):
        raise SoakError(f"the series disagree in length: {len(xs)} against {len(ys)}")
    slopes = [
        (ys[j] - ys[i]) / (xs[j] - xs[i])
        for i in range(len(xs))
        for j in range(i + 1, len(xs))
        if xs[j] != xs[i]
    ]
    if not slopes:
        raise SoakError("every sample shares one timestamp, so no slope exists")
    return statistics.median(slopes)


def _null_slopes(
    xs: Sequence[float], ys: Sequence[float], *, iterations: int, seed: int
) -> list[float]:
    """Return the slopes of the same samples in random time orders."""
    rng = random.Random(seed)
    shuffled = list(ys)
    slopes: list[float] = []
    for _ in range(iterations):
        rng.shuffle(shuffled)
        slopes.append(theil_sen_slope(xs, shuffled))
    return slopes


def _quantile(values: Sequence[float], fraction: float) -> float:
    """Return the value below which ``fraction`` of a sorted sample lies."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def assess(
    metric: str,
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    budget: float,
    iterations: int = ITERATIONS,
    seed: int = 0,
) -> Trend:
    """Return the trend verdict for one series, refusing a series too short to judge."""
    if len(ys) < MINIMUM_SAMPLES:
        raise SoakError(
            f"{metric}: {len(ys)} samples is below the {MINIMUM_SAMPLES} this test needs; "
            "raise FASTERRAG_SOAK_SECONDS or lower FASTERRAG_SOAK_SAMPLE_INTERVAL_S"
        )
    observed = theil_sen_slope(xs, ys)
    null = _null_slopes(xs, ys, iterations=iterations, seed=seed)
    exceeded = sum(1 for slope in null if slope >= observed)
    window = max(xs) - min(xs)
    return Trend(
        metric=metric,
        samples=len(ys),
        run_s=window,
        baseline=float(statistics.median(ys[: max(3, len(ys) // 10)])),
        slope_per_s=observed,
        p_value=(1 + exceeded) / (iterations + 1),
        growth=observed * window,
        detectable=_quantile(null, 1 - ALPHA) * window,
        budget=budget,
    )
