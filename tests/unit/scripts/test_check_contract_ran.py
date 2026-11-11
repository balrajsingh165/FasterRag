"""The JUnit guard, pinned against every report state a contract job can hand it.

The gate exists because `pytest` exits 0 when every selected case skips, so the cases below
are written as the four ways a green exit code can be a lie — a skipped case, an empty
report, a missing one, a corrupt one — plus the one case that must stay green.
"""

from pathlib import Path

import pytest
from check_contract_ran import (
    EMPTY,
    MISSING,
    SKIPPED,
    TOO_FEW,
    UNREADABLE,
    check,
    main,
)

PASSED = '<testcase classname="tests.integration.weaviate.test_c" name="test_upsert_round_trips"/>'
SKIP = (
    '<testcase classname="tests.integration.milvus.test_c" name="test_search_returns_hits">'
    '<skipped message="no Milvus at http://localhost:19530 (MilvusException: server unavailable)"'
    ' type="pytest.skip"/></testcase>'
)
FAILED = (
    '<testcase classname="tests.integration.milvus.test_c" name="test_delete_is_idempotent">'
    '<failure message="assert 1 == 0">boom</failure></testcase>'
)


def write(path: Path, body: str, *, tests: int = 1) -> Path:
    """Write a JUnit report of the shape `pytest --junitxml` emits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" '
        f'tests="{tests}">{body}</testsuite></testsuites>',
        encoding="utf-8",
    )
    return path


def rules(path: Path, *paths: Path) -> list[str]:
    """Return the rule names the gate raised over these reports."""
    _, findings = check([path, *paths])
    return [finding.rule for finding in findings]


def test_a_report_of_executed_cases_passes(tmp_path: Path) -> None:
    """The one state that must stay green, or the gate is just a red light."""
    report = write(tmp_path / "contract.xml", PASSED * 3, tests=3)

    tally, findings = check([report])

    assert findings == []
    assert (tally.cases, tally.executed, tally.skipped) == (3, 3, 0)


def test_one_skipped_case_fails_the_whole_report(tmp_path: Path) -> None:
    """The defect itself: 282 passing cases do not excuse the one that measured nothing."""
    report = write(tmp_path / "contract.xml", PASSED + SKIP, tests=2)

    assert rules(report) == [SKIPPED]


def test_the_skip_reason_is_printed_beside_the_verdict(tmp_path: Path) -> None:
    """A skip whose reason is not in the log sends its reader back to the raw report."""
    report = write(tmp_path / "contract.xml", SKIP, tests=1)

    _, findings = check([report])

    assert "no Milvus at http://localhost:19530" in findings[0].detail


def test_an_empty_report_fails_rather_than_passing_vacuously(tmp_path: Path) -> None:
    """Zero collected cases is what a bad path argument produces, and it exits 0 in pytest."""
    report = write(tmp_path / "contract.xml", "", tests=0)

    assert rules(report) == [EMPTY, TOO_FEW]


def test_a_missing_report_fails(tmp_path: Path) -> None:
    """Pytest writes the file even when it fails, so an absent one means it never ran."""
    assert rules(tmp_path / "never-written.xml") == [MISSING, TOO_FEW]


def test_no_path_at_all_fails(tmp_path: Path) -> None:
    """A gate invoked with nothing to read must not report that nothing was wrong."""
    _, findings = check([])

    assert [finding.rule for finding in findings] == [MISSING]


def test_a_truncated_report_fails_rather_than_being_skipped_over(tmp_path: Path) -> None:
    """A half-uploaded artifact must not read as a clean run."""
    report = tmp_path / "contract.xml"
    report.write_text('<testsuites><testsuite name="pytest"><testcase', encoding="utf-8")

    assert rules(report) == [UNREADABLE, TOO_FEW]


def test_a_file_that_is_not_junit_fails(tmp_path: Path) -> None:
    """Well-formed XML that is not a report proves nothing about any contract."""
    report = tmp_path / "contract.xml"
    report.write_text("<coverage line-rate='0.94'/>", encoding="utf-8")

    assert rules(report) == [UNREADABLE, TOO_FEW]


def test_a_bare_testsuite_root_is_accepted(tmp_path: Path) -> None:
    """Some writers omit the <testsuites> wrapper, and that is not a corrupt report."""
    report = tmp_path / "contract.xml"
    report.write_text(f'<testsuite name="pytest" tests="1">{PASSED}</testsuite>', encoding="utf-8")

    assert rules(report) == []


def test_a_failed_case_is_not_a_finding_because_it_executed(tmp_path: Path) -> None:
    """A contract case that ran and failed is a result; the pytest step is what goes red."""
    report = write(tmp_path / "contract.xml", FAILED, tests=1)

    tally, findings = check([report])

    assert findings == []
    assert (tally.failed, tally.executed) == (1, 1)


def test_several_reports_are_judged_together(tmp_path: Path) -> None:
    """One suite per report is the shape of a matrix job, and one skip in either is fatal."""
    clean = write(tmp_path / "weaviate.xml", PASSED, tests=1)
    dirty = write(tmp_path / "milvus.xml", SKIP, tests=1)

    tally, findings = check([clean, dirty])

    assert [finding.rule for finding in findings] == [SKIPPED]
    assert (tally.cases, tally.executed) == (2, 1)


def test_a_floor_catches_a_run_that_selected_almost_nothing(tmp_path: Path) -> None:
    """Every case passing proves little when the run selected one of ninety-eight."""
    report = write(tmp_path / "contract.xml", PASSED, tests=1)

    _, findings = check([report], expect_at_least=96)

    assert [finding.rule for finding in findings] == [TOO_FEW]


def test_the_floor_is_counted_over_executed_cases_not_collected_ones(tmp_path: Path) -> None:
    """A skipped case must not help a report clear the floor it exists to enforce."""
    report = write(tmp_path / "contract.xml", PASSED + SKIP, tests=2)

    _, findings = check([report], expect_at_least=2)

    assert [finding.rule for finding in findings] == [SKIPPED, TOO_FEW]


@pytest.mark.parametrize(("body", "expected"), [(PASSED, 0), (SKIP, 1)])
def test_main_exits_nonzero_exactly_when_something_skipped(
    tmp_path: Path, body: str, expected: int
) -> None:
    """The exit code is the whole interface a CI step consumes."""
    report = write(tmp_path / "contract.xml", body, tests=1)

    assert main([str(report)]) == expected
