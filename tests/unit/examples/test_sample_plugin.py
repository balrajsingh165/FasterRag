"""The sample plugin in ``examples/plugin-vectordb`` passes the contract suite in every lane.

CI's ``examples`` job installs the sample with pip and runs the same suite against the real
installation. This module is the cheaper half that runs wherever the unit tier runs: it puts
the sample's ``src`` on the path and registers its entry point from the sample's own
``pyproject.toml`` (``tests/support/plugins.py``), so the class is still reached only through
``vector_db.provider: memory`` and the factory, never imported by name.
"""

from __future__ import annotations

import tomllib
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from fasterrag.adapters.vectordb.base import ADAPTER_API_VERSION, CollectionSpec, VectorDBAdapter
from fasterrag.adapters.vectordb.factory import ENTRY_POINT_GROUP, create_vector_db_adapter
from fasterrag.config.schema import Settings
from tests.contract.vectordb import DIMENSIONS, VectorDBContract
from tests.support.plugins import install_distribution

SAMPLE = Path(__file__).resolve().parents[3] / "examples" / "plugin-vectordb"


def declared_entry_points() -> dict[str, str]:
    """Return the sample's registrations under the vector-database group."""
    project = tomllib.loads((SAMPLE / "pyproject.toml").read_text(encoding="utf-8"))
    entries: dict[str, str] = project["project"]["entry-points"][ENTRY_POINT_GROUP]
    return entries


@pytest.fixture
def installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the sample as a distribution whose entry points are the ones it declares."""
    monkeypatch.syspath_prepend(str(SAMPLE / "src"))
    install_distribution(
        tmp_path,
        monkeypatch,
        {ENTRY_POINT_GROUP: declared_entry_points()},
        name="fasterrag-memory-vectordb",
    )


def test_the_sample_registers_memory_under_the_vector_database_group() -> None:
    """The guide tells plugin authors to copy this file; it must say what the guide says."""
    assert declared_entry_points() == {"memory": "fasterrag_memory_vectordb:MemoryVectorDB"}


def test_the_sample_declares_the_adapter_api_this_tree_provides() -> None:
    """A sample that declared a stale version would be refused before the contract ran."""
    source = (SAMPLE / "src" / "fasterrag_memory_vectordb" / "adapter.py").read_text("utf-8")

    assert f'adapter_api_version: ClassVar[str] = "{ADAPTER_API_VERSION}"' in source


@pytest.mark.usefixtures("installed")
class TestSamplePluginContract(VectorDBContract):
    """The unmodified contract suite against the sample, selected by configuration."""

    @pytest.fixture
    async def adapter(self) -> AsyncIterator[VectorDBAdapter]:
        settings = Settings.model_validate(
            {"vector_db": {"provider": "memory", "api_key_env": None}}
        )
        built = create_vector_db_adapter(settings)
        assert type(built).__module__ == "fasterrag_memory_vectordb.adapter"
        yield built
        await built.close()

    @pytest.fixture
    async def collection(self, adapter: VectorDBAdapter) -> AsyncIterator[str]:
        name = f"contract-{uuid.uuid4().hex[:8]}"
        await adapter.create_collection(CollectionSpec(name=name, dimensions=DIMENSIONS))
        yield name
        await adapter.drop_collection(name)

    @pytest.fixture
    async def hybrid_collection(self, adapter: VectorDBAdapter) -> AsyncIterator[str]:
        name = f"contract-hybrid-{uuid.uuid4().hex[:8]}"
        await adapter.create_collection(
            CollectionSpec(name=name, dimensions=DIMENSIONS, sparse=True)
        )
        yield name
        await adapter.drop_collection(name)
