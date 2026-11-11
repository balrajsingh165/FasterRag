"""The ``export`` and ``import`` wrappers (D11): exit codes, name resolution, cleanup."""

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

from fasterrag.cli.commands import portability
from fasterrag.cli.main import main
from fasterrag.cli.output import ExitCode
from fasterrag.errors import FasterRagError
from fasterrag.services.archive import (
    CHECKSUMS_NAME,
    CHUNKS_NAME,
    DENSE_ONLY_FORMAT_VERSION,
    DOCUMENTS_NAME,
    LOCK_NAME,
    MANIFEST_NAME,
    ArchiveCounts,
)
from fasterrag.services.archive_import import ArchiveReader, ImportCounts, open_archive
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

    def __enter__(self) -> "FakeReader":
        """Enter the scope that owns the staging directory, as the real reader does."""
        return self

    def __exit__(self, *details: object) -> None:
        """Leave that scope however the block ended."""


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


def built(tmp_path: Path) -> Path:
    """Write a minimal well-formed ``.fragx``, enough for a real reader to stage it."""
    members: dict[str, bytes] = {
        DOCUMENTS_NAME: b'{"document_id": "d_1"}\n',
        CHUNKS_NAME: b'{"chunk_id": "c_1", "document_id": "d_1"}\n',
        LOCK_NAME: b"{}\n",
    }
    members[MANIFEST_NAME] = json.dumps(
        {
            "format_version": DENSE_ONLY_FORMAT_VERSION,
            "collection": {"name": "archived-docs"},
            "embedding": {"model": "m", "model_version": "1", "dimensions": 2},
            "counts": {"documents": 1, "chunks": 1, "vectors": 0},
            "includes_vectors": False,
        },
        sort_keys=True,
    ).encode("utf-8")
    members[CHECKSUMS_NAME] = "".join(
        f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in members.items()
    ).encode("utf-8")

    path = tmp_path / "a.fragx"
    with tarfile.open(path, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    return path


@pytest.fixture
def staged(monkeypatch: pytest.MonkeyPatch, importing: Recorder) -> list[ArchiveReader]:
    """Open real archives, keeping every reader so its staging directory can be checked."""
    readers: list[ArchiveReader] = []

    def spy(path: Path) -> ArchiveReader:
        reader = open_archive(path)
        readers.append(reader)
        return reader

    monkeypatch.setattr(portability, "open_archive", spy)
    return readers


def test_a_successful_import_releases_the_staging_directory(
    config: str,
    tmp_path: Path,
    adapter: Closeable,
    importing: Recorder,
    staged: list[ArchiveReader],
) -> None:
    """Relying on the finalizer left 61 abandoned directories behind (TASK-0301)."""
    code = main(["import", "--config", config, str(built(tmp_path))])

    assert code == ExitCode.SUCCESS
    assert [reader.staging.exists() for reader in staged] == [False]


def test_a_failed_import_releases_the_staging_directory(
    config: str,
    tmp_path: Path,
    adapter: Closeable,
    importing: Recorder,
    staged: list[ArchiveReader],
) -> None:
    """A refusal is exactly the path that used to retain the handle keeping staging alive."""
    importing.error = FasterRagError("qdrant is not answering", retryable=False)

    code = main(["import", "--config", config, str(built(tmp_path))])

    assert code == ExitCode.FAILURE
    assert [reader.staging.exists() for reader in staged] == [False]
