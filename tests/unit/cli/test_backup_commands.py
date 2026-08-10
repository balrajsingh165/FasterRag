"""The ``backup`` and ``restore`` wrappers: exit codes, flag translation, and cleanup."""

import json
from pathlib import Path
from typing import Any

import pytest

from fasterrag.cli.commands import backup as backup_command
from fasterrag.cli.main import main
from fasterrag.cli.output import ExitCode
from fasterrag.errors import FasterRagError
from fasterrag.services.backup import BackupManifest
from tests.unit.cli.conftest import Closeable


class Recorder:
    """Captures the keyword arguments the wrapper hands the service."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.report: dict[str, Any] = {}

    @property
    def last(self) -> dict[str, Any]:
        return self.calls[-1]

    @property
    def mismatches(self) -> list[dict[str, Any]]:
        mismatches: list[dict[str, Any]] = self.report["mismatches"]
        return mismatches


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> Closeable:
    built = Closeable()
    monkeypatch.setattr(backup_command, "create_vector_db_adapter", lambda settings: built)
    return built


def manifest() -> BackupManifest:
    return BackupManifest(
        created_at="2026-08-09T00:00:00Z",
        fasterrag="0.1.0",
        collections={"docs": "docs-snap"},
        vector_counts={"docs": 42},
        artifacts=["index.lock"],
        config="config.yaml",
    )


@pytest.fixture
def backing_up(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()

    async def fake(destination: Path, adapter: object, **kwargs: Any) -> BackupManifest:
        recorder.calls.append({"destination": destination, **kwargs})
        if recorder.error is not None:
            raise recorder.error
        return manifest()

    monkeypatch.setattr(backup_command, "back_up", fake)
    return recorder


@pytest.fixture
def restoring(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()
    recorder.report = {
        "created_at": "2026-08-09T00:00:00Z",
        "collections": ["docs"],
        "artifacts": ["index.lock"],
        "mismatches": [],
    }

    async def fake(source: Path, adapter: object, **kwargs: Any) -> dict[str, Any]:
        recorder.calls.append({"source": source, **kwargs})
        if recorder.error is not None:
            raise recorder.error
        return recorder.report

    monkeypatch.setattr(backup_command, "restore", fake)
    return recorder


def test_a_backup_reports_what_it_captured(
    config: str,
    tmp_path: Path,
    adapter: Closeable,
    backing_up: Recorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["backup", "--config", config, str(tmp_path / "out")])

    out = capsys.readouterr().out
    assert code == ExitCode.SUCCESS
    assert "docs: docs-snap (42 vectors)" in out
    assert ".env excluded by design" in out


def test_a_backup_as_json_is_one_document_with_no_prose(
    config: str,
    tmp_path: Path,
    adapter: Closeable,
    backing_up: Recorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["backup", "--config", config, str(tmp_path / "out"), "--json"])

    out = capsys.readouterr().out
    assert json.loads(out)["collections"] == {"docs": "docs-snap"}
    assert "excluded by design" not in out


def test_no_collections_flag_means_every_collection_rather_than_none(
    config: str, tmp_path: Path, adapter: Closeable, backing_up: Recorder
) -> None:
    """The wrapper turns argparse's empty list into ``None``; a literal [] would back up nothing."""
    main(["backup", "--config", config, str(tmp_path / "out")])

    assert backing_up.last["collections"] is None


def test_named_collections_are_passed_through(
    config: str, tmp_path: Path, adapter: Closeable, backing_up: Recorder
) -> None:
    main(["backup", "--config", config, str(tmp_path / "out"), "--collections", "a", "b"])

    assert backing_up.last["collections"] == ["a", "b"]


def test_the_retain_flag_reaches_the_service(
    config: str, tmp_path: Path, adapter: Closeable, backing_up: Recorder
) -> None:
    main(["backup", "--config", config, str(tmp_path / "out"), "--retain", "3"])

    assert backing_up.last["retain"] == 3


@pytest.mark.parametrize(
    ("retryable", "expected"),
    [(True, ExitCode.UNREACHABLE), (False, ExitCode.FAILURE)],
)
def test_a_failed_backup_maps_the_error_to_its_documented_code(
    config: str,
    tmp_path: Path,
    adapter: Closeable,
    backing_up: Recorder,
    capsys: pytest.CaptureFixture[str],
    retryable: bool,
    expected: ExitCode,
) -> None:
    backing_up.error = FasterRagError("the backend is not answering", retryable=retryable)

    code = main(["backup", "--config", config, str(tmp_path / "out")])

    assert code == expected
    assert "the backend is not answering" in capsys.readouterr().err


def test_a_failed_backup_still_closes_the_adapter(
    config: str, tmp_path: Path, adapter: Closeable, backing_up: Recorder
) -> None:
    backing_up.error = FasterRagError("boom", retryable=False)

    main(["backup", "--config", config, str(tmp_path / "out")])

    assert adapter.closed == 1


def test_a_failed_backup_prints_a_problem_rather_than_a_traceback(
    config: str,
    tmp_path: Path,
    adapter: Closeable,
    backing_up: Recorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    backing_up.error = FasterRagError("boom", retryable=False)

    main(["backup", "--config", config, str(tmp_path / "out")])

    err = capsys.readouterr().err
    assert err.startswith("error: [")
    assert "Traceback" not in err


def test_an_invalid_config_never_reaches_the_service(bad_config: str, tmp_path: Path) -> None:
    assert main(["backup", "--config", bad_config, str(tmp_path / "out")]) == ExitCode.USAGE


def test_a_clean_restore_says_the_counts_matched(
    config: str,
    tmp_path: Path,
    adapter: Closeable,
    restoring: Recorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["restore", "--config", config, str(tmp_path / "set")])

    assert code == ExitCode.SUCCESS
    assert "verified" in capsys.readouterr().out


def test_a_restore_with_a_mismatch_exits_non_zero(
    config: str,
    tmp_path: Path,
    adapter: Closeable,
    restoring: Recorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A drill signed off on a restore holding the wrong vector count is the failure mode."""
    restoring.mismatches.append({"collection": "docs", "expected": 42, "actual": 7})

    code = main(["restore", "--config", config, str(tmp_path / "set")])

    assert code == ExitCode.FAILURE
    assert "mismatches" in capsys.readouterr().err


def test_a_restore_with_a_mismatch_still_emits_its_json_document(
    config: str,
    tmp_path: Path,
    adapter: Closeable,
    restoring: Recorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    restoring.mismatches.append({"collection": "docs"})

    main(["restore", "--config", config, str(tmp_path / "set"), "--json"])

    assert json.loads(capsys.readouterr().out)["mismatches"]


def test_collections_only_skips_the_control_files(
    config: str, tmp_path: Path, adapter: Closeable, restoring: Recorder
) -> None:
    main(["restore", "--config", config, str(tmp_path / "set"), "--collections-only"])

    assert restoring.last["restore_files"] is False


def test_without_collections_only_the_control_files_are_restored(
    config: str, tmp_path: Path, adapter: Closeable, restoring: Recorder
) -> None:
    main(["restore", "--config", config, str(tmp_path / "set")])

    assert restoring.last["restore_files"] is True


@pytest.mark.parametrize(
    ("retryable", "expected"),
    [(True, ExitCode.UNREACHABLE), (False, ExitCode.FAILURE)],
)
def test_a_failed_restore_maps_the_error_to_its_documented_code(
    config: str,
    tmp_path: Path,
    adapter: Closeable,
    restoring: Recorder,
    retryable: bool,
    expected: ExitCode,
) -> None:
    restoring.error = FasterRagError("the backend is not answering", retryable=retryable)

    assert main(["restore", "--config", config, str(tmp_path / "set")]) == expected


def test_an_invalid_config_never_reaches_the_restore(bad_config: str, tmp_path: Path) -> None:
    assert main(["restore", "--config", bad_config, str(tmp_path / "set")]) == ExitCode.USAGE
