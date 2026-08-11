"""The `doctor` wrapper's own work: which report it prints, and which exit code it picks.

The repairs themselves are the service's, and are tested there. What is exercised here is
the contract an operator scripts against — that `--fix` reports the machine *after* the
repairs and still exits 4 when something is left broken, so wiring `--fix` into a setup
script cannot turn a failed preflight into a green one.
"""

import json

import pytest

from fasterrag.cli.commands import diagnostics
from fasterrag.cli.main import main
from fasterrag.cli.output import ExitCode
from fasterrag.services.doctor import (
    DoctorCheck,
    DoctorReport,
    FixAttempt,
    FixOutcome,
    FixStatus,
)

BROKEN = DoctorReport(
    checks=[DoctorCheck(name="api_port", passed=False, detail="held", fix="Free port 8000.")]
)
HEALTHY = DoctorReport(checks=[DoctorCheck(name="api_port", passed=True, detail="free")])


def outcome(after: DoctorReport, status: FixStatus) -> FixOutcome:
    """Return an outcome whose post-fix report is ``after``, whatever was attempted."""
    return FixOutcome(
        before=BROKEN,
        after=after,
        attempts=[FixAttempt(check="api_port", status=status, detail="a reason")],
        rechecked=True,
    )


def test_without_fix_nothing_is_repaired(
    config: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Plain `doctor` must stay a read-only diagnostic."""
    called = False

    async def repairing(*args: object, **kwargs: object) -> FixOutcome:
        nonlocal called
        called = True
        return outcome(HEALTHY, "fixed")

    async def reporting(*args: object, **kwargs: object) -> DoctorReport:
        return BROKEN

    monkeypatch.setattr(diagnostics, "diagnose_and_fix", repairing)
    monkeypatch.setattr(diagnostics, "diagnose", reporting)

    code = main(["doctor", "--config", config])

    assert called is False
    assert code == ExitCode.PREFLIGHT
    assert "fixes:" not in capsys.readouterr().out


def test_fix_exits_zero_once_the_repairs_hold(config: str, monkeypatch: pytest.MonkeyPatch) -> None:
    async def repairing(*args: object, **kwargs: object) -> FixOutcome:
        return outcome(HEALTHY, "fixed")

    monkeypatch.setattr(diagnostics, "diagnose_and_fix", repairing)

    assert main(["doctor", "--fix", "--config", config]) == ExitCode.SUCCESS


def test_fix_still_exits_four_when_a_repair_did_not_hold(
    config: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--fix` must never launder a failed preflight into a passing one."""

    async def repairing(*args: object, **kwargs: object) -> FixOutcome:
        return outcome(BROKEN, "failed")

    monkeypatch.setattr(diagnostics, "diagnose_and_fix", repairing)

    assert main(["doctor", "--fix", "--config", config]) == ExitCode.PREFLIGHT


def test_fix_json_stays_one_parseable_document(
    config: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def repairing(*args: object, **kwargs: object) -> FixOutcome:
        return outcome(HEALTHY, "fixed")

    monkeypatch.setattr(diagnostics, "diagnose_and_fix", repairing)

    main(["doctor", "--fix", "--json", "--config", config])

    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["fixes"]["attempts"][0]["check"] == "api_port"
