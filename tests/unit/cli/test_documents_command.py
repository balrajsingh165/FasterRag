"""``fasterrag documents delete``: the real service behind the wrapper, a faked backend.

The command is run through ``main`` with only the backend, the journal location and the
cache replaced, so the exit code, the ``--json`` document and what is left in the index are
all produced by the code a user runs (TASK-0443).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fasterrag.adapters.vectordb.base import CollectionSpec
from fasterrag.cli.commands import documents as documents_command
from fasterrag.cli.main import main
from fasterrag.cli.output import ExitCode
from fasterrag.config.schema import Settings
from fasterrag.errors import RetrievalError
from fasterrag.services.journal import Journal
from tests.unit.services.test_documents import DIMENSIONS, MemoryAdapter, RecordingCache, chunk


class ClosingAdapter(MemoryAdapter):
    """Counts closes, because the command is obliged to release what it built."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1


class ClosingCache(RecordingCache):
    """Counts closes as well as invalidations."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = 0

    async def close(self) -> None:
        self.closed += 1


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> ClosingAdapter:
    """Return the backend the command will build, holding two collections."""
    store = ClosingAdapter(Settings())
    for name in ("default", "contracts"):
        store.specs.append(CollectionSpec(name=name, dimensions=DIMENSIONS))
    store.points = [
        chunk("d_1", 0, collection="default", tenant="acme"),
        chunk("d_1", 1, collection="default", tenant="globex"),
        chunk("d_2", 0, collection="default"),
        chunk("d_1", 0, collection="contracts"),
    ]
    monkeypatch.setattr(documents_command, "create_vector_db_adapter", lambda settings: store)
    return store


@pytest.fixture
def journal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Journal:
    """Return the journal the command will open, knowing ``d_1`` in ``default``."""
    opened = Journal(tmp_path / "journal")
    opened.remember_content_blocking("default", "hash-1", "d_1")
    monkeypatch.setattr(documents_command, "create_journal", lambda settings: opened)
    monkeypatch.setattr(documents_command, "create_lock_store", lambda settings: None)
    return opened


@pytest.fixture
def cache(monkeypatch: pytest.MonkeyPatch) -> ClosingCache:
    """Return the cache the command will build."""
    built = ClosingCache()
    monkeypatch.setattr(documents_command, "SemanticCache", lambda settings, store: built)
    return built


@pytest.fixture
def wired(backend: ClosingAdapter, journal: Journal, cache: ClosingCache) -> ClosingAdapter:
    """Return the backend, with the journal and cache replaced too."""
    return backend


def test_a_document_is_deleted_from_the_default_collection(
    config: str, wired: ClosingAdapter
) -> None:
    """There was no command that removed a document."""
    code = main(["--config", config, "documents", "delete", "d_1"])

    assert code == ExitCode.SUCCESS
    assert wired.chunks_of("default", "d_1") == []
    assert wired.chunks_of("default", "d_2") == [0]
    assert wired.chunks_of("contracts", "d_1") == [0]


def test_collection_names_the_collection_to_delete_from(config: str, wired: ClosingAdapter) -> None:
    code = main(["--config", config, "--collection", "contracts", "documents", "delete", "d_1"])

    assert code == ExitCode.SUCCESS
    assert wired.chunks_of("contracts", "d_1") == []
    assert wired.chunks_of("default", "d_1") == [0, 1]


def test_tenant_limits_the_delete_to_that_tenants_chunks(
    config: str, wired: ClosingAdapter
) -> None:
    code = main(["--config", config, "documents", "delete", "d_1", "--tenant", "acme"])

    assert code == ExitCode.SUCCESS
    assert [point.payload.get("tenant") for point in wired.held("default")] == ["globex", None]


def test_json_output_is_one_document(
    config: str, wired: ClosingAdapter, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--config", config, "--json", "documents", "delete", "d_1"])

    printed = json.loads(capsys.readouterr().out)
    assert code == ExitCode.SUCCESS
    assert set(printed) == {"collection", "document_id", "known", "deleted_at"}
    assert printed["collection"] == "default"
    assert printed["known"] is True


def test_the_dedup_record_is_dropped_so_the_document_can_return(
    config: str, wired: ClosingAdapter, journal: Journal
) -> None:
    main(["--config", config, "documents", "delete", "d_1"])

    assert journal.is_duplicate_blocking("default", "hash-1") is False


def test_a_document_that_does_not_exist_exits_zero_and_says_so(
    config: str, wired: ClosingAdapter, capsys: pytest.CaptureFixture[str]
) -> None:
    """A delete is idempotent; a script retrying it must not see a failure."""
    code = main(["--config", config, "documents", "delete", "d_never_ingested"])

    assert code == ExitCode.SUCCESS
    assert "no record" in capsys.readouterr().out
    assert len(wired.points) == 4


def test_a_missing_collection_is_a_failure_with_its_code(
    config: str, wired: ClosingAdapter, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--config", config, "--collection", "absent", "documents", "delete", "d_1"])

    assert code == ExitCode.FAILURE
    assert "[NOT_FOUND]" in capsys.readouterr().err


def test_an_unreachable_backend_exits_with_the_retryable_code(
    config: str, wired: ClosingAdapter
) -> None:
    wired.failures.append(RetrievalError("backend unreachable", retryable=True))

    code = main(["--config", config, "documents", "delete", "d_1"])

    assert code == ExitCode.UNREACHABLE


def test_the_cache_is_invalidated(config: str, wired: ClosingAdapter, cache: ClosingCache) -> None:
    main(["--config", config, "documents", "delete", "d_1"])

    assert len(cache.reasons) == 1


def test_what_the_command_built_is_closed_even_when_it_fails(
    config: str, wired: ClosingAdapter, cache: ClosingCache
) -> None:
    wired.failures.append(RetrievalError("backend unreachable", retryable=True))

    main(["--config", config, "documents", "delete", "d_1"])

    assert wired.closed == 1
    assert cache.closed == 1


def test_an_invalid_config_is_a_usage_error(bad_config: str) -> None:
    assert main(["--config", bad_config, "documents", "delete", "d_1"]) == ExitCode.USAGE
