"""The ``export`` and ``import`` wrappers (D11): exit codes, name resolution, cleanup."""

import json
from pathlib import Path
from typing import Any

import pytest

from fasterrag.cli.commands import portability
from fasterrag.cli.main import main
from fasterrag.cli.output import ExitCode
from fasterrag.errors import FasterRagError
from fasterrag.services.archive import ArchiveCounts
from fasterrag.services.archive_import import ImportCounts
from tests.unit.cli.conftest import Closeable


class FakeLocks:
    """A lock store whose enablement the test controls."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self.read_for: list[str] = []

    def read(self, collection: str) -> str:
        self.read_for.append(collection)
        return f"lock-for-{collection}"


class FakeReader:
    """An opened archive, standing in for a verified ``.fragx``."""

    collection = "archived-docs"


class Recorder:
    """Captures what the wrapper handed the service."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.router = Closeable()

    @property
    def last(self) -> dict[str, Any]:
        return self.calls[-1]


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> Closeable:
    built = Closeable()
    monkeypatch.setattr(portability, "create_vector_db_adapter", lambda settings: built)
    return built


@pytest.fixture
def locks(monkeypatch: pytest.MonkeyPatch) -> FakeLocks:
    store = FakeLocks()
    monkeypatch.setattr(portability, "create_lock_store", lambda settings: store)
    return store


@pytest.fixture
def exporting(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()

    async def fake(settings: object, adapter: object, **kwargs: Any) -> ArchiveCounts:
        recorder.calls.append(dict(kwargs))
        if recorder.error is not None:
            raise recorder.error
        return ArchiveCounts(documents=3, chunks=9, vectors=9)

    monkeypatch.setattr(portability, "export_archive", fake)
    return recorder


@pytest.fixture
def importing(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()

    async def fake(
        settings: object, adapter: object, reader: object, **kwargs: Any
    ) -> ImportCounts:
        recorder.calls.append(dict(kwargs))
        if recorder.error is not None:
            raise recorder.error
        return ImportCounts(documents=3, chunks=9)

    monkeypatch.setattr(portability, "open_archive", lambda path: FakeReader())
    monkeypatch.setattr(portability, "import_archive", fake)
    monkeypatch.setattr(portability, "create_embedding_router", lambda settings: recorder.router)
    return recorder


def test_export_reports_the_counts_it_wrote(
    config: str,
    tmp_path: Path,
    adapter: Closeable,
    locks: FakeLocks,
    exporting: Recorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(["export", "--config", config, "--out", str(tmp_path / "a.fragx")])

    out = capsys.readouterr().out
    assert code == ExitCode.SUCCESS
    assert "documents       3" in out


def test_export_without_vectors_warns_that_import_must_reembed(
    config: str,
    tmp_path: Path,
    adapter: Closeable,
    locks: FakeLocks,
    exporting: Recorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["export", "--config", config, "--out", str(tmp_path / "a.fragx")])

    assert "will need --reembed" in capsys.readouterr().out


def test_export_with_vectors_omits_the_warning(
    config: str,
    tmp_path: Path,
    adapter: Closeable,
    locks: FakeLocks,
    exporting: Recorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "export",
            "--config",
            config,
            "--out",
            str(tmp_path / "a.fragx"),
            "--include-vectors",
        ]
    )

    assert "will need --reembed" not in capsys.readouterr().out
    assert exporting.last["include_vectors"] is True


def test_export_falls_back_to_the_configured_collection(
    config: str, tmp_path: Path, adapter: Closeable, locks: FakeLocks, exporting: Recorder
) -> None:
    main(["export", "--config", config, "--out", str(tmp_path / "a.fragx")])

    assert exporting.last["collection"] == "default"


def test_export_prefers_the_named_collection(
    config: str, tmp_path: Path, adapter: Closeable, locks: FakeLocks, exporting: Recorder
) -> None:
    main(
        ["export", "--config", config, "--collection", "legal", "--out", str(tmp_path / "a.fragx")]
    )

    assert exporting.last["collection"] == "legal"


def test_export_carries_the_lockfile_when_locking_is_on(
    config: str, tmp_path: Path, adapter: Closeable, locks: FakeLocks, exporting: Recorder
) -> None:
    main(["export", "--config", config, "--out", str(tmp_path / "a.fragx")])

    assert exporting.last["lock"] == "lock-for-default"


def test_export_carries_no_lockfile_when_locking_is_off(
    config: str,
    tmp_path: Path,
    adapter: Closeable,
    exporting: Recorder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(portability, "create_lock_store", lambda settings: FakeLocks(enabled=False))

    main(["export", "--config", config, "--out", str(tmp_path / "a.fragx")])

    assert exporting.last["lock"] is None


@pytest.mark.parametrize(
    ("retryable", "expected"),
    [(True, ExitCode.UNREACHABLE), (False, ExitCode.FAILURE)],
)
def test_a_failed_export_uses_the_documented_code_for_its_error(
    config: str,
    tmp_path: Path,
    adapter: Closeable,
    locks: FakeLocks,
    exporting: Recorder,
    retryable: bool,
    expected: ExitCode,
) -> None:
    """Exit 1 for an unreachable backend made export the one command retries could not read."""
    exporting.error = FasterRagError("qdrant is not answering", retryable=retryable)

    code = main(["export", "--config", config, "--out", str(tmp_path / "a.fragx")])

    assert code == expected


def test_a_failed_export_still_closes_the_adapter(
    config: str, tmp_path: Path, adapter: Closeable, locks: FakeLocks, exporting: Recorder
) -> None:
    exporting.error = FasterRagError("boom", retryable=False)

    main(["export", "--config", config, "--out", str(tmp_path / "a.fragx")])

    assert adapter.closed == 1


def test_export_refuses_an_invalid_config(bad_config: str, tmp_path: Path) -> None:
    assert (
        main(["export", "--config", bad_config, "--out", str(tmp_path / "a.fragx")])
        == ExitCode.USAGE
    )


def test_import_defaults_to_the_collection_the_archive_names(
    config: str, tmp_path: Path, adapter: Closeable, importing: Recorder
) -> None:
    main(["import", "--config", config, str(tmp_path / "a.fragx")])

    assert importing.last["collection"] == "archived-docs"


def test_import_prefers_target_collection_over_the_global_flag(
    config: str, tmp_path: Path, adapter: Closeable, importing: Recorder
) -> None:
    """Both name a collection; ``--target-collection`` is the one that means "write here"."""
    main(
        [
            "import",
            "--config",
            config,
            str(tmp_path / "a.fragx"),
            "--collection",
            "global",
            "--target-collection",
            "explicit",
        ]
    )

    assert importing.last["collection"] == "explicit"


def test_import_uses_the_global_collection_when_no_target_is_given(
    config: str, tmp_path: Path, adapter: Closeable, importing: Recorder
) -> None:
    main(["import", "--config", config, str(tmp_path / "a.fragx"), "--collection", "global"])

    assert importing.last["collection"] == "global"


def test_import_builds_no_embedding_router_for_a_vector_copy(
    config: str, tmp_path: Path, adapter: Closeable, importing: Recorder
) -> None:
    """A router loads a model; building one for a copy that never embeds is pure startup cost."""
    main(["import", "--config", config, str(tmp_path / "a.fragx")])

    assert importing.last["router"] is None
    assert importing.router.closed == 0


def test_import_builds_and_closes_a_router_when_reembedding(
    config: str, tmp_path: Path, adapter: Closeable, importing: Recorder
) -> None:
    main(["import", "--config", config, str(tmp_path / "a.fragx"), "--reembed"])

    assert importing.last["reembed"] is True
    assert importing.router.closed == 1


def test_import_reports_which_mode_it_ran_in(
    config: str,
    tmp_path: Path,
    adapter: Closeable,
    importing: Recorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["import", "--config", config, str(tmp_path / "a.fragx")])

    assert "vector copy" in capsys.readouterr().out


def test_import_as_json_is_one_document(
    config: str,
    tmp_path: Path,
    adapter: Closeable,
    importing: Recorder,
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["import", "--config", config, str(tmp_path / "a.fragx"), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "collection": "archived-docs",
        "reembed": False,
        "documents": 3,
        "chunks": 9,
    }


def test_an_unreadable_archive_is_a_usage_error(
    config: str, tmp_path: Path, adapter: Closeable, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refused archive is the caller naming the wrong file, not the system failing."""

    def refuse(path: Path) -> FakeReader:
        raise FasterRagError("the archive is truncated", retryable=False)

    monkeypatch.setattr(portability, "open_archive", refuse)

    assert main(["import", "--config", config, str(tmp_path / "a.fragx")]) == ExitCode.USAGE


@pytest.mark.parametrize(
    ("retryable", "expected"),
    [(True, ExitCode.UNREACHABLE), (False, ExitCode.FAILURE)],
)
def test_a_failed_import_uses_the_documented_code_for_its_error(
    config: str,
    tmp_path: Path,
    adapter: Closeable,
    importing: Recorder,
    retryable: bool,
    expected: ExitCode,
) -> None:
    importing.error = FasterRagError("qdrant is not answering", retryable=retryable)

    assert main(["import", "--config", config, str(tmp_path / "a.fragx")]) == expected


def test_a_failed_import_closes_both_the_adapter_and_the_router(
    config: str, tmp_path: Path, adapter: Closeable, importing: Recorder
) -> None:
    importing.error = FasterRagError("boom", retryable=False)

    main(["import", "--config", config, str(tmp_path / "a.fragx"), "--reembed"])

    assert (adapter.closed, importing.router.closed) == (1, 1)


def test_import_refuses_an_invalid_config(bad_config: str, tmp_path: Path) -> None:
    assert main(["import", "--config", bad_config, str(tmp_path / "a.fragx")]) == ExitCode.USAGE
