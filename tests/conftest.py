"""Test-suite-wide configuration.

Currently one concern: Hypothesis's per-example deadline.
"""

from hypothesis import HealthCheck, settings

# CRITICAL: the deadline is disabled, not merely raised. Hypothesis fails an example that
# takes longer than 200 ms by default, which makes every property test in this repo
# timing-dependent — and they assert *behaviour*, never latency. Under load (a machine
# running the Docker fleet, or CI on a shared runner) that produced intermittent failures in
# a different property test on each run: tenant isolation once, name scoping the next, both
# passing in isolation moments later.
#
# A correctness suite that fails for reasons unrelated to correctness is worse than a slower
# one. It teaches whoever sees red to re-run rather than investigate, which is exactly the
# habit that lets a real isolation failure through. Performance belongs in the benchmark
# ledger (docs/benchmarks.md), measured on isolated hardware, not asserted as a side effect
# of a property test.
settings.register_profile(
    "fasterrag",
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("fasterrag")
