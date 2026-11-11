import json
from pathlib import Path

from check_must_run import (
    EMPTY_ROSTER,
    MISSING_ROSTER,
    NO_EVIDENCE,
    NO_REPORTS,
    RAN_NOWHERE,
    UNKNOWN_TEST,
    UNREADABLE,
    WRONG_MODE,
    Finding,
    check,
    main,
)

SYMLINK = "tests/unit/services/test_sources.py::test_a_symlink_out_of_an_allowed_root_is_refused"
OPT_IN = (
    "tests/eval/test_live_provider_gating.py"
    "::test_an_unset_opt_in_skips_exactly_as_a_falsey_one_does"
)


def write(path: Path, *, mode: str, outcomes: dict[str, str], version: int = 1) -> Path:
    """Write one report document of the shape tests/conftest.py emits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "report_version": version,
                "mode": mode,
                "label": path.stem,
                "platform": "linux",
                "collected": sorted(outcomes),
                "outcomes": outcomes,
            }
        ),
        encoding="utf-8",
    )
    return path


def roster(root: Path, *nodeids: str) -> Path:
    """Write the collect-only document naming every test that claims the marker."""
    return write(root / "roster.json", mode="collect", outcomes=dict.fromkeys(nodeids, ""))


def lane(root: Path, name: str, **outcomes: str) -> Path:
    """Write one lane's run report under the reports directory."""
    return write(root / "reports" / f"{name}.json", mode="run", outcomes=dict(outcomes))


def rules(findings: list[Finding]) -> list[str]:
    return sorted({finding.rule for finding in findings})


def test_a_test_that_ran_on_one_platform_and_skipped_on_the_other_is_accepted(
    tmp_path: Path,
) -> None:
    """The symlink case: a Windows skip is correct, so flattening the platforms is wrong."""
    expected = roster(tmp_path, SYMLINK)
    lane(tmp_path, "ubuntu", **{SYMLINK: "ran"})
    lane(tmp_path, "windows", **{SYMLINK: "skipped"})

    assert check(expected, [tmp_path / "reports"]) == []


def test_a_test_skipped_on_every_platform_is_the_whole_point(tmp_path: Path) -> None:
    """The bug this gate exists for, and the finding names each lane's verdict."""
    expected = roster(tmp_path, SYMLINK)
    lane(tmp_path, "ubuntu", **{SYMLINK: "skipped"})
    lane(tmp_path, "windows", **{SYMLINK: "skipped"})

    found = check(expected, [tmp_path / "reports"])

    assert rules(found) == [RAN_NOWHERE]
    assert found[0].subject == SYMLINK
    assert "ubuntu: skipped" in found[0].detail
    assert "windows: skipped" in found[0].detail


def test_a_marked_test_no_lane_collected_is_not_a_pass(tmp_path: Path) -> None:
    """Absent from every report is 'nobody ran it', never 'nothing to check'."""
    expected = roster(tmp_path, SYMLINK, OPT_IN)
    lane(tmp_path, "ubuntu", **{OPT_IN: "ran"})
    lane(tmp_path, "windows", **{OPT_IN: "ran"})

    found = check(expected, [tmp_path / "reports"])

    assert rules(found) == [RAN_NOWHERE]
    assert found[0].subject == SYMLINK
    assert "not-collected" in found[0].detail


def test_a_failing_test_counts_as_having_run(tmp_path: Path) -> None:
    """This gate judges execution; a failure fails its own job and is not this one's news."""
    expected = roster(tmp_path, OPT_IN)
    lane(tmp_path, "ubuntu", **{OPT_IN: "ran"})

    assert check(expected, [tmp_path / "reports"]) == []


def test_a_missing_roster_fails_rather_than_passing_vacuously(tmp_path: Path) -> None:
    """No roster means the gate never learned what to check."""
    lane(tmp_path, "ubuntu", **{OPT_IN: "ran"})

    found = check(tmp_path / "absent.json", [tmp_path / "reports"])

    assert rules(found) == [MISSING_ROSTER]


def test_a_roster_with_no_marked_test_fails(tmp_path: Path) -> None:
    """A registered marker nothing carries is a gate examining nothing."""
    expected = roster(tmp_path)
    lane(tmp_path, "ubuntu", **{OPT_IN: "ran"})

    found = check(expected, [tmp_path / "reports"])

    assert rules(found) == [EMPTY_ROSTER]


def test_an_absent_reports_directory_fails(tmp_path: Path) -> None:
    """The artifact did not arrive, which is the likeliest way this gate goes hollow."""
    expected = roster(tmp_path, SYMLINK)

    found = check(expected, [tmp_path / "reports"])

    assert rules(found) == [NO_REPORTS]
    assert "no lane report was found" in found[0].detail


def test_an_empty_reports_directory_fails(tmp_path: Path) -> None:
    """An uploaded-but-empty artifact must not read as a clean run."""
    expected = roster(tmp_path, SYMLINK)
    (tmp_path / "reports").mkdir()

    assert rules(check(expected, [tmp_path / "reports"])) == [NO_REPORTS]


def test_lane_reports_that_record_nothing_at_all_fail(tmp_path: Path) -> None:
    """Reports exist, hold no outcome between them, and prove exactly nothing."""
    expected = roster(tmp_path, SYMLINK)
    lane(tmp_path, "ubuntu")
    lane(tmp_path, "windows")

    found = check(expected, [tmp_path / "reports"])

    assert rules(found) == [NO_EVIDENCE]


def test_a_collect_only_document_is_refused_as_evidence(tmp_path: Path) -> None:
    """A roster copied into the reports directory records collection, not execution."""
    expected = roster(tmp_path, SYMLINK)
    write(tmp_path / "reports" / "ubuntu.json", mode="collect", outcomes={SYMLINK: ""})

    assert rules(check(expected, [tmp_path / "reports"])) == [NO_REPORTS, WRONG_MODE]


def test_a_run_document_is_refused_as_a_roster(tmp_path: Path) -> None:
    """A lane only sees its own tests, so it can never be the authoritative set."""
    wrong = write(tmp_path / "roster.json", mode="run", outcomes={SYMLINK: "ran"})
    lane(tmp_path, "ubuntu", **{SYMLINK: "ran"})

    assert rules(check(wrong, [tmp_path / "reports"])) == [WRONG_MODE]


def test_an_unreadable_roster_fails(tmp_path: Path) -> None:
    """A truncated artifact is not a clean one."""
    broken = tmp_path / "roster.json"
    broken.write_text("{not json", encoding="utf-8")

    assert rules(check(broken, [tmp_path / "reports"])) == [UNREADABLE]


def test_a_report_from_a_future_writer_is_refused(tmp_path: Path) -> None:
    """Reader and writer drifting apart must fail loudly, not read as zero outcomes."""
    expected = roster(tmp_path, SYMLINK)
    write(tmp_path / "reports" / "ubuntu.json", mode="run", outcomes={SYMLINK: "ran"}, version=2)

    assert rules(check(expected, [tmp_path / "reports"])) == [NO_REPORTS, UNREADABLE]


def test_a_lane_reporting_a_test_the_roster_lacks_fails(tmp_path: Path) -> None:
    """Disagreement means the evidence and the roster describe different trees."""
    expected = roster(tmp_path, SYMLINK)
    lane(tmp_path, "ubuntu", **{SYMLINK: "ran", OPT_IN: "ran"})

    found = check(expected, [tmp_path / "reports"])

    assert rules(found) == [UNKNOWN_TEST]
    assert found[0].subject == OPT_IN


def test_main_exits_one_on_a_test_that_skipped_everywhere(tmp_path: Path) -> None:
    """The exit code is what CI reads, so it is asserted and not inferred."""
    expected = roster(tmp_path, SYMLINK)
    lane(tmp_path, "ubuntu", **{SYMLINK: "skipped"})

    assert main([str(tmp_path / "reports"), "--roster", str(expected)]) == 1


def test_main_exits_zero_when_every_marked_test_ran_somewhere(tmp_path: Path) -> None:
    """The green path, asserted so a gate that can only fail is not mistaken for one."""
    expected = roster(tmp_path, SYMLINK, OPT_IN)
    lane(tmp_path, "ubuntu", **{SYMLINK: "ran", OPT_IN: "ran"})
    lane(tmp_path, "windows", **{SYMLINK: "skipped", OPT_IN: "ran"})

    assert main([str(tmp_path / "reports"), "--roster", str(expected)]) == 0


def test_this_tree_actually_carries_the_marker(tmp_path: Path) -> None:
    """A registered marker nobody uses would make the CI gate green and hollow."""
    tests = Path(__file__).resolve().parents[2]
    marked = sorted(
        path.relative_to(tests).as_posix()
        for path in tests.rglob("test_*.py")
        if "@pytest.mark.must_run" in path.read_text(encoding="utf-8")
    )

    assert marked, (
        "no test carries @pytest.mark.must_run, so scripts/check_must_run.py guards nothing"
    )
