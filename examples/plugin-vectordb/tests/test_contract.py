"""Run the shared vector-database contract suite against the sample plugin, as installed.

The adapter is resolved through the factory by its registered name, never imported directly:
a pass here proves the installed entry point, the adapter API version check and the contract
together. Run it from a fasterRag checkout, where ``tests/contract/vectordb.py`` lives::

    pip install ./examples/plugin-vectordb
    python -m pytest examples/plugin-vectordb/tests
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
from tests.contract.vectordb import DIMENSIONS, VectorDBContract

from fasterrag.adapters.vectordb.base import CollectionSpec, VectorDBAdapter
from fasterrag.adapters.vectordb.factory import available_providers, create_vector_db_adapter
from fasterrag.config.schema import Settings


def test_the_plugin_is_installed_under_its_entry_point() -> None:
    """Without this, a missing install would fail every case below for a less obvious reason."""
    assert available_providers().get("memory", "").startswith("plugin")


class TestMemoryVectorDBContract(VectorDBContract):
    """The unmodified contract suite, against ``vector_db.provider: memory``."""

    @pytest.fixture
    async def adapter(self) -> AsyncIterator[VectorDBAdapter]:
        settings = Settings.model_validate(
            {"vector_db": {"provider": "memory", "api_key_env": None}}
        )
        built = create_vector_db_adapter(settings)
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
