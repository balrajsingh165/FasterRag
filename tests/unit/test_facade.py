import inspect
from pathlib import Path

import pytest

import fasterrag
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError, FasterRagError
from fasterrag.facade import FasterRag
from fasterrag.services import doctor
from fasterrag.services.provisioning import ContainerState


def settings() -> Settings:
    return Settings.model_validate({})


class StubAdapter:
    """Stands in for the vector-db adapter so doctor reaches no server."""

    def __init__(self) -> None:
        """Start unclosed, so a test can assert the adapter was released."""
        self.closed = False

    async def health(self) -> HealthStatus:
        """Report an unreachable backend without opening a socket."""
        return HealthStatus(healthy=False, detail="stubbed", latency_ms=1.0)

    async def close(self) -> None:
        """Record that doctor released the adapter it built."""
        self.closed = True


async def _false() -> bool:
    """Stand in for a probe that must report absence without touching the machine."""
    return False


async def _volume_absent(volume: str) -> bool:
    """Report no Docker volume without shelling out to a daemon."""
    return False


async def _no_container(name: str = "fasterrag-qdrant") -> ContainerState:
    """Report no container without shelling out to a daemon."""
    return ContainerState(name=name, exists=False)


def test_the_facade_is_exported_from_the_package_root() -> None:
    """python-api.md documents `from fasterrag import FasterRag` as the entry point."""
    assert fasterrag.FasterRag is FasterRag
    assert "FasterRag" in fasterrag.__all__


def test_reading_the_version_does_not_import_the_facade() -> None:
    """The facade costs seconds to import; a packaging script reading __version__ must not pay."""
    source = Path(fasterrag.__file__).read_text(encoding="utf-8")
    top_level = [
        line
        for line in source.splitlines()
        if line.startswith(("import ", "from ")) and "." in line
    ]

    assert not [line for line in top_level if "facade" in line]


def test_an_unknown_attribute_still_raises_attribute_error() -> None:
    """A lazy __getattr__ that swallowed unknown names would hide typos."""
    with pytest.raises(AttributeError, match="no attribute"):
        _ = fasterrag.not_a_real_export


def test_from_settings_touches_no_backend() -> None:
    """Construction validates; entering the context is what connects."""
    rag = FasterRag.from_settings(settings())

    assert rag.settings is not None
    assert not rag._started


def test_from_config_reports_a_missing_file_as_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        FasterRag.from_config(tmp_path / "absent.yaml")


def test_using_it_unstarted_says_how_to_start_it() -> None:
    """Otherwise this surfaces as an AttributeError on None, three frames away."""
    rag = FasterRag.from_settings(settings())

    with pytest.raises(FasterRagError, match="async context manager") as caught:
        _ = rag.vector_db

    assert "async with" in caught.value.detail


@pytest.mark.parametrize("name", ["ingest", "query", "retrieve"])
async def test_every_pipeline_entry_point_checks_it_was_started(name: str) -> None:
    rag = FasterRag.from_settings(settings())

    with pytest.raises(FasterRagError, match="not been started"):
        await getattr(rag, name)("anything")


async def test_streaming_checks_it_was_started_too() -> None:
    """An async generator body does not run until iterated, so this is easy to miss."""
    rag = FasterRag.from_settings(settings())

    with pytest.raises(FasterRagError, match="not been started"):
        async for _ in rag.query_stream("anything"):
            pass


def test_estimate_works_without_starting(tmp_path: Path) -> None:
    """A preflight estimate that required a live backend would not be a preflight."""
    document = tmp_path / "note.md"
    document.write_text("# Title\n\nSome body text to chunk.\n", encoding="utf-8")

    estimate = FasterRag.from_settings(settings()).estimate([str(document)])

    assert estimate.documents == 1


def test_estimate_obeys_the_setting_that_claims_to_control_it(tmp_path: Path) -> None:
    """The embedded surface honours `cost.estimator` too (TASK-0200).

    Three control planes read the same configuration; a switch two of them respected would
    still leave the third reporting costs an operator had turned off.
    """
    document = tmp_path / "note.md"
    document.write_text("# Title\n\nSome body text to chunk.\n", encoding="utf-8")
    rag = FasterRag.from_settings(Settings.model_validate({"cost": {"estimator": False}}))

    with pytest.raises(FasterRagError, match=r"cost\.estimator"):
        rag.estimate([str(document)])


def test_the_documented_surface_exists() -> None:
    """python-api.md's table is the beta contract; a missing member breaks it silently."""
    for member in (
        "from_config",
        "from_settings",
        "ingest",
        "query",
        "query_stream",
        "retrieve",
        "estimate",
        "index_lock",
    ):
        assert hasattr(FasterRag, member), member


def test_it_is_an_async_context_manager() -> None:
    assert hasattr(FasterRag, "__aenter__")
    assert hasattr(FasterRag, "__aexit__")


async def test_shutdown_continues_past_a_resource_that_will_not_close() -> None:
    """One backend refusing to close must not leave the others open."""
    closed: list[str] = []

    class Stubborn:
        async def close(self) -> None:
            raise RuntimeError("refuses to close")

    class Willing:
        def __init__(self, name: str) -> None:
            self.name = name

        async def close(self) -> None:
            closed.append(self.name)

    rag = FasterRag.from_settings(settings())
    rag._started = True
    rag._cache = Stubborn()  # type: ignore[assignment]
    rag._embeddings = Willing("embeddings")  # type: ignore[assignment]
    rag._vector_db = Willing("vector_db")  # type: ignore[assignment]

    await rag.__aexit__(None, None, None)

    assert closed == ["embeddings", "vector_db"]


async def test_shutdown_never_raises_over_a_propagating_exception() -> None:
    """Raising here would replace the original failure and lose the cause."""

    class Stubborn:
        async def close(self) -> None:
            raise RuntimeError("refuses to close")

    rag = FasterRag.from_settings(settings())
    rag._started = True
    rag._vector_db = Stubborn()  # type: ignore[assignment]

    await rag.__aexit__(ValueError, ValueError("original"), None)


def test_the_facade_holds_no_pipeline_logic() -> None:
    """It composes services; behavior here would be behavior the REST API does not have."""
    source = Path(inspect.getfile(FasterRag)).read_text(encoding="utf-8")

    for forbidden in ("def _rrf", "def _score", "cosine", "def _chunk"):
        assert forbidden not in source, forbidden


def test_the_remaining_documented_surface_exists() -> None:
    """python-api.md's table is the beta contract for these too."""
    for member in ("doctor", "collections", "create_collection", "drop_collection", "replay"):
        assert hasattr(FasterRag, member), member


async def test_doctor_runs_without_starting_the_facade(monkeypatch: pytest.MonkeyPatch) -> None:
    """Doctor diagnoses environments that may not work; needing a working one defeats it.

    Every probe is stubbed because this asserts the facade's wiring, not the machine's.
    """
    adapter = StubAdapter()
    monkeypatch.setattr(doctor, "create_vector_db_adapter", lambda settings: adapter)
    # CRITICAL: TASK-0353. Unstubbed, `create_vector_db_adapter` built a real
    # AsyncQdrantClient, whose constructor starts a daemon thread that GETs the server
    # version — a unit test on the network, and a UserWarning raised on no test's thread
    # that surfaced on stderr after the run finished, in 1 of 6 runs. The Docker and port
    # probes beside it are the same class of defect as TASK-0324: unstubbed, this result is
    # a function of what happens to be running on the developer's machine.
    monkeypatch.setattr(doctor, "docker_available", _false)
    monkeypatch.setattr(doctor, "volume_exists", _volume_absent)
    monkeypatch.setattr(doctor, "container_state", _no_container)
    monkeypatch.setattr(doctor, "port_is_free", lambda host, port: True)
    monkeypatch.setattr(doctor, "port_is_reachable", lambda host, port: False)

    report = await FasterRag.from_settings(settings()).doctor()

    assert report.checks
    assert adapter.closed


@pytest.mark.parametrize("name", ["collections", "drop_collection", "replay"])
async def test_the_new_entry_points_check_they_were_started(name: str) -> None:
    rag = FasterRag.from_settings(settings())
    method = getattr(rag, name)

    with pytest.raises(FasterRagError, match="not been started"):
        await (method() if name == "collections" else method("anything"))
