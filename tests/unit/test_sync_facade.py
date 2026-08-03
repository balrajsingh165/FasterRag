import asyncio
import inspect
from pathlib import Path

import pytest

from fasterrag import facade, sync
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError, FasterRagError


def settings() -> Settings:
    return Settings.model_validate({})


def test_it_is_importable_at_the_documented_path() -> None:
    """python-api.md documents `from fasterrag.sync import FasterRag`."""
    assert sync.FasterRag.__module__ == "fasterrag.sync"
    assert facade.FasterRag.__module__ == "fasterrag.facade"


def test_it_mirrors_the_async_surface() -> None:
    """A method missing here silently sends a sync caller back to writing async code."""
    async_members = {
        name
        for name, value in inspect.getmembers(facade.FasterRag, callable)
        if not name.startswith("_") or name in {"__aenter__", "__aexit__"}
    }
    expected = {name for name in async_members if not name.startswith("__")}

    for name in expected:
        assert hasattr(sync.FasterRag, name), f"sync facade is missing {name}"


def test_no_method_is_a_coroutine() -> None:
    """The whole point is that a caller never awaits; one stray async def breaks that."""
    for name, member in inspect.getmembers(sync.FasterRag, inspect.isfunction):
        assert not inspect.iscoroutinefunction(member), name


def test_it_is_a_blocking_context_manager_not_an_async_one() -> None:
    assert hasattr(sync.FasterRag, "__enter__")
    assert not hasattr(sync.FasterRag, "__aenter__")


def test_from_config_reports_a_missing_file_the_same_way(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        sync.FasterRag.from_config(tmp_path / "absent.yaml")


def test_using_it_unstarted_names_the_context_manager() -> None:
    rag = sync.FasterRag.from_settings(settings())

    with pytest.raises(FasterRagError, match="not been started") as caught:
        rag.query("anything")

    assert "with FasterRag.from_config" in caught.value.detail


def test_an_unstarted_call_leaves_no_unawaited_coroutine(
    recwarn: pytest.WarningsRecorder,
) -> None:
    """The RuntimeWarning would print after the real error and bury it."""
    rag = sync.FasterRag.from_settings(settings())

    with pytest.raises(FasterRagError):
        rag.query("anything")

    assert not [w for w in recwarn if "never awaited" in str(w.message)]


async def test_starting_inside_a_running_loop_is_refused_not_deadlocked() -> None:
    """Blocking on a future from the thread driving the loop hangs with no message."""
    rag = sync.FasterRag.from_settings(settings())

    with pytest.raises(FasterRagError, match="running event loop") as caught:
        rag.__enter__()

    assert "async with" in caught.value.detail


def test_exiting_without_entering_is_a_no_op() -> None:
    """A `with` body that raised during __enter__ still calls __exit__."""
    sync.FasterRag.from_settings(settings()).__exit__(None, None, None)


def test_the_loop_is_closed_even_when_shutdown_fails() -> None:
    """A leaked event loop is silent — the process just holds its selector until it exits."""

    class Failing:
        settings = None

        async def __aenter__(self) -> "Failing":
            return self

        async def __aexit__(self, *exc: object) -> None:
            raise RuntimeError("shutdown failed")

    rag = sync.FasterRag(Failing())  # type: ignore[arg-type]
    rag.__enter__()
    loop = rag._loop
    assert loop is not None

    with pytest.raises(RuntimeError, match="shutdown failed"):
        rag.__exit__(None, None, None)

    assert loop.is_closed()


def test_estimate_needs_no_running_loop(tmp_path: Path) -> None:
    """It is synchronous on the async facade already, so it must work unentered."""
    document = tmp_path / "note.md"
    document.write_text("# Title\n\nBody text.\n", encoding="utf-8")

    estimate = sync.FasterRag.from_settings(settings()).estimate([str(document)])

    assert estimate.documents == 1


def test_streaming_returns_an_ordinary_iterator() -> None:
    """An async iterator here would defeat the purpose of the blocking facade."""
    assert not inspect.isasyncgenfunction(sync.FasterRag.query_stream)


def test_the_owned_loop_survives_across_calls() -> None:
    """asyncio.run per call would tear down the connection pools between calls.

    `doctor` is the one deliberate exception: it must work on an installation that cannot
    start, so it falls back to a short-lived loop when the facade was never entered. Every
    method that touches a backend goes through the owned loop.
    """
    source = Path(inspect.getfile(sync)).read_text(encoding="utf-8")
    methods = source.split("    def ")

    assert "new_event_loop()" in source
    for body in methods:
        if "asyncio.run(" in body:
            assert body.startswith("doctor("), f"asyncio.run outside doctor: {body[:40]}"


def test_it_adds_no_behavior_of_its_own() -> None:
    """A thin adapter, not a second implementation: every method delegates to _inner."""
    source = Path(inspect.getfile(sync)).read_text(encoding="utf-8")

    for name in ("ingest", "query", "retrieve", "index_lock"):
        body = source.split(f"def {name}(", 1)[1].split("\n    def ", 1)[0]
        assert "self._inner." in body, name


def test_asyncio_is_not_left_with_a_stray_current_loop() -> None:
    """Setting a global current loop would change behavior for unrelated code."""
    source = Path(inspect.getfile(sync)).read_text(encoding="utf-8")

    assert "set_event_loop(" not in source


def test_a_fresh_instance_has_no_loop_before_entering() -> None:
    assert sync.FasterRag.from_settings(settings())._loop is None


def test_asyncio_still_works_normally_after_the_facade_is_used() -> None:
    """A closed private loop must not leave the interpreter unable to start another."""
    rag = sync.FasterRag.from_settings(settings())
    rag.__exit__(None, None, None)

    assert asyncio.run(_answer()) == 42


async def _answer() -> int:
    return 42
