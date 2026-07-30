import pytest

from fasterrag.adapters.embeddings import huggingface
from fasterrag.adapters.embeddings.huggingface import HuggingFaceEmbedder
from fasterrag.adapters.embeddings.sync import SentenceEmbedderBridge
from fasterrag.errors import ConfigError, EmbedError
from tests.unit.adapters.embeddings.conftest import DIMENSIONS, FakeModel, local_settings


def test_construction_loads_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[str] = []
    monkeypatch.setattr(huggingface, "load_model", lambda name: loaded.append(name))

    HuggingFaceEmbedder(local_settings())

    assert loaded == []


async def test_the_model_is_loaded_once_and_reused(model: FakeModel) -> None:
    adapter = HuggingFaceEmbedder(local_settings())

    await adapter.embed_documents(["one"])
    await adapter.embed_documents(["two"])

    assert adapter._model is model
    assert len(model.calls) == 2


async def test_documents_are_embedded_in_configured_batches(model: FakeModel) -> None:
    adapter = HuggingFaceEmbedder(local_settings(batch_size=2))

    result = await adapter.embed_documents(["a", "b", "c", "d", "e"])

    assert len(result.vectors) == 5
    assert [len(call) for call in model.calls] == [2, 2, 1]


async def test_the_result_records_the_model_and_version(model: FakeModel) -> None:
    adapter = HuggingFaceEmbedder(local_settings(model="BAAI/bge-small-en-v1.5"))

    result = await adapter.embed_documents(["a"])

    assert result.model == "BAAI/bge-small-en-v1.5"
    assert result.model_version == "BAAI/bge-small-en-v1.5"


async def test_dimensions_are_discovered_from_the_model(model: FakeModel) -> None:
    adapter = HuggingFaceEmbedder(local_settings())
    assert adapter.dimensions is None

    await adapter.embed_documents(["a"])

    assert adapter.dimensions == DIMENSIONS


async def test_a_configured_dimension_override_is_kept(model: FakeModel) -> None:
    adapter = HuggingFaceEmbedder(local_settings(dimensions=16))

    assert adapter.dimensions == 16


async def test_a_query_returns_one_vector(model: FakeModel) -> None:
    vector = await HuggingFaceEmbedder(local_settings()).embed_query("what is this")

    assert len(vector) == DIMENSIONS


async def test_an_encoding_failure_is_a_non_retryable_embed_error(model: FakeModel) -> None:
    model.raises = RuntimeError("out of memory")

    with pytest.raises(EmbedError, match="failed to encode") as caught:
        await HuggingFaceEmbedder(local_settings()).embed_documents(["a"])

    assert caught.value.retryable is False


async def test_health_reports_a_working_model(model: FakeModel) -> None:
    status = await HuggingFaceEmbedder(local_settings()).health()

    assert status.healthy is True


async def test_health_reports_a_broken_model_without_raising(model: FakeModel) -> None:
    model.raises = RuntimeError("no such model")

    status = await HuggingFaceEmbedder(local_settings()).health()

    assert status.healthy is False


async def test_closing_releases_the_model(model: FakeModel) -> None:
    adapter = HuggingFaceEmbedder(local_settings())
    await adapter.embed_documents(["a"])

    await adapter.close()

    assert adapter._model is None


def test_a_missing_package_names_the_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(name: str) -> object:
        raise ConfigError("embeddings.provider is 'huggingface', which needs sentence-transformers")

    monkeypatch.setattr(huggingface, "load_model", unavailable)

    with pytest.raises(ConfigError, match="sentence-transformers"):
        HuggingFaceEmbedder(local_settings()).encode(["a"])


def test_the_sync_bridge_calls_a_local_model_directly(model: FakeModel) -> None:
    bridge = SentenceEmbedderBridge(HuggingFaceEmbedder(local_settings()))

    vectors = bridge.embed(["one", "two"])

    assert len(vectors) == 2
    assert model.calls == [["one", "two"]]
