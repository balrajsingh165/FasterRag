import importlib.util

import pytest

from fasterrag.adapters.embeddings import (
    CohereEmbedder,
    EmbeddingAdapter,
    HuggingFaceEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    available_providers,
    create_embedding_adapter,
    resolve_adapter_class,
)
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError

PROVIDERS = [
    ("huggingface", HuggingFaceEmbedder),
    ("openai", OpenAIEmbedder),
    ("cohere", CohereEmbedder),
    ("ollama", OllamaEmbedder),
]


def settings_for(provider: str, **overrides: object) -> Settings:
    embeddings: dict[str, object] = {"provider": provider, **overrides}
    if provider in {"openai", "cohere"}:
        embeddings.setdefault("api_key_env", f"{provider.upper()}_API_KEY")
    return Settings.model_validate({"embeddings": embeddings})


@pytest.mark.parametrize(("provider", "expected"), PROVIDERS)
def test_every_documented_provider_resolves(provider: str, expected: type) -> None:
    assert resolve_adapter_class(provider) is expected
    assert available_providers()[provider] == "built-in"


@pytest.mark.parametrize(("provider", "expected"), PROVIDERS)
def test_the_factory_builds_each_provider(provider: str, expected: type) -> None:
    adapter = create_embedding_adapter(settings_for(provider))

    assert isinstance(adapter, expected)
    assert isinstance(adapter, EmbeddingAdapter)


@pytest.mark.parametrize(("provider", "expected"), PROVIDERS)
def test_construction_opens_nothing(provider: str, expected: type) -> None:
    adapter = create_embedding_adapter(settings_for(provider, model="pinned-model"))

    assert adapter.model == "pinned-model"
    assert adapter.model_version == "pinned-model"


def test_an_unknown_provider_lists_what_is_available() -> None:
    with pytest.raises(ConfigError, match="not registered") as caught:
        resolve_adapter_class("nonexistent")

    assert "huggingface" in caught.value.detail


@pytest.mark.parametrize(
    ("provider", "package", "extra"),
    [
        ("openai", "openai", "openai"),
        ("cohere", "cohere", "cohere"),
        ("ollama", "ollama", "ollama"),
    ],
)
async def test_a_provider_without_its_package_names_the_install_command(
    provider: str, package: str, extra: str
) -> None:
    if importlib.util.find_spec(package) is not None:
        pytest.skip(f"{package} is installed, so the missing-package path cannot be exercised")

    adapter = create_embedding_adapter(settings_for(provider))

    with pytest.raises(ConfigError, match=f"fasterrag\\[{extra}\\]"):
        await adapter.embed_query("anything")


@pytest.mark.parametrize("provider", ["openai", "cohere", "ollama"])
async def test_health_reports_a_missing_package_instead_of_raising(provider: str) -> None:
    if importlib.util.find_spec(provider) is not None:
        pytest.skip(f"{provider} is installed, so the missing-package path cannot be exercised")

    status = await create_embedding_adapter(settings_for(provider)).health()

    assert status.healthy is False
    assert status.detail is not None


def test_batching_splits_by_the_configured_size() -> None:
    adapter = create_embedding_adapter(settings_for("huggingface", batch_size=3))

    batches = adapter.batches(["a", "b", "c", "d", "e", "f", "g"])

    assert [len(batch) for batch in batches] == [3, 3, 1]


def test_batching_of_an_empty_input_produces_no_requests() -> None:
    adapter = create_embedding_adapter(settings_for("huggingface"))

    assert adapter.batches([]) == []


def test_a_dimension_override_is_reported() -> None:
    adapter = create_embedding_adapter(settings_for("openai", dimensions=256))

    assert adapter.dimensions == 256
