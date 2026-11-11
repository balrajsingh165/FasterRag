"""Unit tests for the cost-governor measurement script's pure logic.

A harness that prints a plausible wrong number is worse than no harness, and the two parts
that can do that here are the byte walk and the baseline transcription. The timings are out
of scope — they are what the script exists to report, and asserting on them would make this
suite timing-dependent, which `tests/conftest.py` explains at length is not allowed.
"""

from __future__ import annotations

import sys
from collections import deque

from measure_cost_governor import (
    DEFAULT_CHARGES,
    ExactWindow,
    Measurement,
    Report,
    deep_size,
    measure,
    parse_args,
    render,
    seed,
)

from fasterrag.services.budget import ROLLING_WINDOW_SECONDS, CostGovernor


def test_the_byte_walk_counts_the_container_and_everything_under_it() -> None:
    """A walk that stopped at the container would report the dict header and miss 99% of it."""
    held = {"acme": deque([(1.0, 2), (3.0, 4)])}

    assert deep_size(held) > sys.getsizeof(held)


def test_the_byte_walk_counts_a_shared_object_once() -> None:
    """Counting an interned integer per reference would invent memory the process never paid."""
    shared = [0.5] * 100

    assert deep_size([shared, shared]) < deep_size([shared, list(shared)])


def test_the_superseded_baseline_holds_one_entry_per_charge() -> None:
    """If the transcription bucketed anything, the comparison would measure nothing."""
    window = ExactWindow(10**9)
    for index in range(50):
        window.append(1, tenant="acme", now=float(index))

    assert len(window._spent["acme"]) == 50


def test_the_superseded_baseline_evicts_on_each_entrys_own_timestamp() -> None:
    """The exactness the shipped design trades away has to be present in the baseline."""
    window = ExactWindow(10**9)
    window.record(100, tenant="acme", now=0.0)
    window.record(100, tenant="acme", now=10.0)

    assert window.spent("acme", now=ROLLING_WINDOW_SECONDS) == 100
    assert window.spent("acme", now=ROLLING_WINDOW_SECONDS + 10) == 0


def test_seeding_spreads_charges_across_the_whole_window() -> None:
    """Piling every charge on one instant would flatter the bucketed design into one bucket."""
    governor = CostGovernor(per_tenant=10**12)

    seed(governor, charges=2_000, tenants=1, window_seconds=ROLLING_WINDOW_SECONDS)

    assert len(governor._spent["tenant-000000"]) > 1


def test_seeding_the_baseline_bypasses_its_quadratic_writer() -> None:
    """``append`` exists precisely because ``record`` sums the window on every charge."""
    window = ExactWindow(10**9)

    seed(window, charges=100, tenants=1, window_seconds=ROLLING_WINDOW_SECONDS)

    assert len(window._spent["tenant-000000"]) == 100


def test_both_designs_report_the_same_in_window_total() -> None:
    """The cross-check that makes the paired run a comparison rather than two numbers."""
    report = measure(charges=[1_000], tenants=1, samples=2)

    assert len({item.spent_tokens for item in report.measurements}) == 1


def test_a_measurement_divides_its_bytes_by_its_charges() -> None:
    measurement = Measurement(
        design="d",
        charges=4,
        tenants=1,
        seed_s=0.0,
        traced_current_bytes=0,
        traced_peak_bytes=0,
        structure_bytes=400,
        retained_entries=4,
        read_p50_us=0.0,
        read_p95_us=0.0,
        read_worst_us=0.0,
        write_p50_us=0.0,
        write_p95_us=0.0,
        expiry_walk_ms=0.0,
        spent_tokens=0,
    )

    assert measurement.bytes_per_charge == 100.0
    assert measurement.as_dict()["bytes_per_charge"] == 100.0


def test_no_charges_never_divides_by_zero() -> None:
    """An empty scale is a user error, not a crash in the reporting path."""
    empty = Measurement(
        design="d",
        charges=0,
        tenants=1,
        seed_s=0.0,
        traced_current_bytes=0,
        traced_peak_bytes=0,
        structure_bytes=0,
        retained_entries=0,
        read_p50_us=0.0,
        read_p95_us=0.0,
        read_worst_us=0.0,
        write_p50_us=0.0,
        write_p95_us=0.0,
        expiry_walk_ms=0.0,
        spent_tokens=0,
    )

    assert empty.bytes_per_charge == 0.0


def test_the_table_names_both_designs_and_the_window() -> None:
    """A table without the window is a number whose period the reader has to guess."""
    report = measure(charges=[200], tenants=1, samples=2)

    rendered = render(report)

    assert "86,400 s" in rendered
    assert "bucketed" in rendered
    assert "tuple per charge" in rendered


def test_the_default_scales_are_the_three_the_ledger_records() -> None:
    assert DEFAULT_CHARGES == (10_000, 100_000, 1_000_000)
    assert parse_args([]).charge_counts == [10_000, 100_000, 1_000_000]


def test_the_scales_can_be_overridden_from_the_command_line() -> None:
    assert parse_args(["--charges", "5,7"]).charge_counts == [5, 7]


def test_an_empty_report_still_renders_its_header() -> None:
    assert "window:" in render(Report())
