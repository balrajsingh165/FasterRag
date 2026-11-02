import json
from pathlib import Path
from typing import Any

import pytest
from compare_nightly import Verdict, compare, compare_files, main

MACHINE: dict[str, Any] = {
    "cpu": "x86_64",
    "cores": 4,
    "ram_gb": 16.0,
    "gpu": "none detected (no nvidia-smi)",
    "storage_gb": 100.0,
    "os": "Linux 6.5.0",
    "python": "3.12.3",
    "fasterrag": "0.1.0",
}


def document(
    *,
    p50: float | None = 80.0,
    p95: float | None = 100.0,
    p99: float | None = 150.0,
    throughput: float | None = 2.0,
    suite: str = "query",
    dataset: str = "demo",
    machine: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a document shaped exactly as `fasterrag benchmark --json` emits one."""
    median = {
        "operations": 20,
        "seconds": 10.0,
        "throughput_per_sec": throughput,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "max_ms": 200.0,
    }
    return {
        "hardware": machine or MACHINE,
        "suites": [
            {
                "suite": suite,
                "dataset": dataset,
                "repetitions": 3,
                "cold": {"p50_ms": 9000.0},
                "median": {name: value for name, value in median.items() if value is not None},
                "runs": [],
                "notes": "",
            }
        ],
    }


def write(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_an_unchanged_night_is_within_tolerance() -> None:
    report = compare(document(), document())

    assert report.verdict is Verdict.WITHIN_TOLERANCE
    assert len(report.comparisons) == 4


def test_a_move_inside_the_tolerance_passes() -> None:
    """A 5% p95 rise is noise on a shared runner, not a signal."""
    report = compare(document(p95=100.0), document(p95=105.0))

    assert report.verdict is Verdict.WITHIN_TOLERANCE


def test_a_p95_rise_beyond_the_tolerance_regresses() -> None:
    report = compare(document(p95=100.0), document(p95=130.0))

    assert report.verdict is Verdict.REGRESSED
    assert any("p95_ms" in entry.reason for entry in report.regressions)
    assert "30.0% worse" in report.summary()


def test_the_tolerance_boundary_is_not_a_regression() -> None:
    """Exactly at the tolerance is allowed; only a move beyond it fails."""
    report = compare(document(p95=100.0), document(p95=110.0), tolerance=0.10)

    assert report.verdict is Verdict.WITHIN_TOLERANCE


def test_an_improvement_never_fails() -> None:
    """The retrieval gate never blocks on a rise, and neither does this one."""
    report = compare(document(p95=100.0, throughput=2.0), document(p95=20.0, throughput=9.0))

    assert report.verdict is Verdict.WITHIN_TOLERANCE


def test_a_throughput_collapse_regresses() -> None:
    """Throughput is higher-is-better, so a fall is the regression, not a rise."""
    report = compare(document(throughput=2.0), document(throughput=1.0))

    assert report.verdict is Verdict.REGRESSED
    assert any("throughput_per_sec" in entry.reason for entry in report.regressions)


def test_a_throughput_rise_is_not_a_regression() -> None:
    report = compare(document(throughput=2.0), document(throughput=4.0))

    assert report.verdict is Verdict.WITHIN_TOLERANCE


def test_no_previous_run_is_not_comparable_rather_than_green(tmp_path: Path) -> None:
    """The defect this script exists to avoid: reporting a pass having compared nothing."""
    current = write(tmp_path / "current.json", document())

    report = compare_files(tmp_path / "absent.json", current)

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert report.comparisons == []
    assert "NOT COMPARABLE" in report.summary()
    assert "within tolerance" not in report.summary()


def test_an_unparseable_previous_run_is_not_comparable(tmp_path: Path) -> None:
    """The benchmark step runs under `|| true`, so an empty file is an ordinary outcome."""
    previous = tmp_path / "previous.json"
    previous.write_text("", encoding="utf-8")
    current = write(tmp_path / "current.json", document())

    report = compare_files(previous, current)

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert any("not a readable benchmark document" in reason for reason in report.obstacles)


def test_a_json_array_is_not_a_benchmark_document(tmp_path: Path) -> None:
    previous = write(tmp_path / "previous.json", [1, 2, 3])
    current = write(tmp_path / "current.json", document())

    report = compare_files(previous, current)

    assert report.verdict is Verdict.NOT_COMPARABLE


def test_a_different_machine_is_not_comparable() -> None:
    """Identical numbers from a different runner measure the runner, not the change."""
    other = {**MACHINE, "cpu": "aarch64", "cores": 8}

    report = compare(document(), document(machine=other))

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert any("different machines" in reason for reason in report.obstacles)


def test_a_version_change_still_compares() -> None:
    """The code changing is the thing being watched for; refusing to compare hides it."""
    newer = {**MACHINE, "fasterrag": "0.2.0", "python": "3.12.7"}

    report = compare(document(p95=100.0), document(p95=130.0, machine=newer))

    assert report.verdict is Verdict.REGRESSED


def test_a_missing_fingerprint_is_not_comparable() -> None:
    """A run that names no machine is refused for that reason, not for a field mismatch."""
    without = document()
    del without["hardware"]

    report = compare(document(), without)

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert any("no hardware fingerprint" in reason for reason in report.obstacles)


def test_two_runs_with_no_fingerprint_at_all_are_not_comparable() -> None:
    """Two documents that each name no machine agree about nothing; that is not evidence."""
    first = document()
    second = document()
    del first["hardware"]
    del second["hardware"]

    report = compare(first, second)

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert any("no hardware fingerprint" in reason for reason in report.obstacles)


def test_a_different_dataset_is_not_comparable() -> None:
    """A different corpus produces different numbers for reasons that are not regressions."""
    report = compare(document(dataset="demo"), document(dataset="other", p95=130.0))

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert report.comparisons == []


def test_a_suite_with_no_counterpart_is_not_comparable() -> None:
    report = compare(document(suite="ingest"), document(suite="query"))

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert any("no counterpart" in reason for reason in report.obstacles)


def test_a_suite_that_vanished_is_reported() -> None:
    previous = document()
    previous["suites"].append(document(suite="ingest")["suites"][0])

    report = compare(previous, document())

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert any("ran previously but not tonight" in reason for reason in report.obstacles)


def test_a_benchmark_that_produced_no_suites_is_not_comparable() -> None:
    report = compare(document(), {"hardware": MACHINE, "suites": []})

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert any("did not complete" in reason for reason in report.obstacles)


def test_a_missing_metric_is_not_comparable_rather_than_skipped() -> None:
    """An absent p95 must not leave a green verdict covering three metrics out of four."""
    report = compare(document(), document(p95=None))

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert any("p95_ms is absent" in reason for reason in report.obstacles)


def test_a_zero_baseline_is_not_comparable() -> None:
    """A percentage against zero is undefined, and a zero p95 means nothing was sampled."""
    report = compare(document(p95=0.0), document(p95=400.0))

    assert report.verdict is Verdict.NOT_COMPARABLE
    assert any("says nothing" in reason for reason in report.obstacles)


def test_a_boolean_is_not_read_as_a_number() -> None:
    """JSON true is an int in Python; treating it as 1 ms would invent a measurement."""
    report = compare(document(), document(p95=True))

    assert report.verdict is Verdict.NOT_COMPARABLE


def test_the_report_serializes_what_it_compared() -> None:
    payload = compare(document(), document(p95=130.0)).as_dict()

    assert payload["verdict"] == "regressed"
    assert payload["compared"] == 4
    assert len(payload["comparisons"]) == 4


def test_the_markdown_summary_leads_with_the_verdict(tmp_path: Path) -> None:
    current = write(tmp_path / "current.json", document())

    body = compare_files(tmp_path / "absent.json", current).markdown()

    assert "NOT COMPARABLE" in body
    assert "never citable" in body


@pytest.mark.parametrize(
    ("previous", "expected"),
    [(document(p95=100.0), 0), (document(p95=1.0), 1), (None, 2)],
)
def test_the_exit_code_distinguishes_all_three_outcomes(
    tmp_path: Path, previous: dict[str, Any] | None, expected: int
) -> None:
    """A shell step can only branch on the third outcome if it has its own code."""
    current = write(tmp_path / "current.json", document(p95=100.0))
    baseline = tmp_path / "previous.json"
    if previous is not None:
        write(baseline, previous)

    code = main(["--previous", str(baseline), "--current", str(current)])

    assert code == expected


def test_the_json_and_summary_outputs_are_written(tmp_path: Path) -> None:
    current = write(tmp_path / "current.json", document())
    previous = write(tmp_path / "previous.json", document())
    report = tmp_path / "nested" / "comparison.json"
    summary = tmp_path / "summary.md"

    code = main(
        [
            "--previous",
            str(previous),
            "--current",
            str(current),
            "--json-out",
            str(report),
            "--summary",
            str(summary),
        ]
    )

    assert code == 0
    assert json.loads(report.read_text(encoding="utf-8"))["verdict"] == "within-tolerance"
    assert "Nightly benchmark comparison" in summary.read_text(encoding="utf-8")


def test_a_tighter_tolerance_catches_a_smaller_move(tmp_path: Path) -> None:
    current = write(tmp_path / "current.json", document(p95=105.0))
    previous = write(tmp_path / "previous.json", document(p95=100.0))

    assert main(["--previous", str(previous), "--current", str(current)]) == 0
    assert (
        main(["--previous", str(previous), "--current", str(current), "--tolerance", "0.01"]) == 1
    )


def test_a_negative_tolerance_is_refused(tmp_path: Path) -> None:
    """It would fail a run that improved, which is the opposite of the gate's rule."""
    current = write(tmp_path / "current.json", document())
    previous = write(tmp_path / "previous.json", document())

    with pytest.raises(SystemExit):
        main(["--previous", str(previous), "--current", str(current), "--tolerance", "-0.5"])
