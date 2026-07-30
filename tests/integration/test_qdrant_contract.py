"""The adapter contract suite run against a real Qdrant in every supported mode.

``docker`` mode uses the container fasterRag manages itself; ``external`` mode connects
to that same instance the way a user-run deployment is reached. Remote mode is external
mode with a non-local host — the same client path, differing only in the address — so it
is exercised here as external mode and verified against a second machine outside CI.
"""

from collections.abc import AsyncIterator

import pytest
from qdrant_client import AsyncQdrantClient

from fasterrag.adapters.vectordb.base import CollectionSpec, VectorDBAdapter
from fasterrag.adapters.vectordb.qdrant import QdrantAdapter
from fasterrag.config.schema import Settings
from tests.contract.vectordb import DIMENSIONS, VectorDBContract
from tests.integration.conftest import TEST_API_KEY, TEST_VOLUME

pytestmark = pytest.mark.integration


def external_settings(*, prefer_grpc: bool = False) -> Settings:
    return Settings.model_validate(
        {
            "vector_db": {
                "mode": "external",
                "host": "localhost",
                "prefer_grpc": prefer_grpc,
                "docker": {"volume": TEST_VOLUME},
            }
        }
    )


class QdrantContract(VectorDBContract):
    """Shared wiring for every Qdrant mode under test."""

    settings: Settings

    @pytest.fixture
    async def adapter(self, qdrant: Settings) -> AsyncIterator[VectorDBAdapter]:
        built = QdrantAdapter(self.settings)
        yield built
        await built.close()

    @pytest.fixture
    async def collection(
        self, adapter: VectorDBAdapter, collection_name: str
    ) -> AsyncIterator[str]:
        await adapter.create_collection(CollectionSpec(name=collection_name, dimensions=DIMENSIONS))
        yield collection_name

        client = AsyncQdrantClient(host="localhost", port=6333, api_key=TEST_API_KEY, https=False)
        await client.delete_collection(collection_name)
        await client.close()

    @pytest.fixture
    async def hybrid_collection(
        self, adapter: VectorDBAdapter, collection_name: str
    ) -> AsyncIterator[str]:
        name = f"{collection_name}-hybrid"
        await adapter.create_collection(
            CollectionSpec(name=name, dimensions=DIMENSIONS, sparse=True)
        )
        yield name

        client = AsyncQdrantClient(host="localhost", port=6333, api_key=TEST_API_KEY, https=False)
        await client.delete_collection(name)
        await client.close()

    @pytest.fixture
    async def misconfigured_adapter(self, qdrant: Settings) -> AsyncIterator[VectorDBAdapter]:
        wrong = QdrantAdapter(self.settings)
        wrong._api_key_env = "FASTERRAG_WRONG_KEY_VAR"
        yield wrong
        await wrong.close()


class TestDockerMode(QdrantContract):
    """System-managed container: fasterRag launches and owns the instance."""

    settings = Settings.model_validate(
        {"vector_db": {"mode": "docker", "docker": {"volume": TEST_VOLUME}}}
    )


class TestExternalMode(QdrantContract):
    """User-run instance reached over REST, the no-Docker and remote-host path."""

    settings = external_settings()


class TestExternalModeOverGrpc(QdrantContract):
    """Same instance reached with prefer_grpc, which requires port 6334 to be open."""

    settings = external_settings(prefer_grpc=True)
