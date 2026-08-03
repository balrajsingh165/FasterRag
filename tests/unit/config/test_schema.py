import pytest
from pydantic import ValidationError

from fasterrag.config.schema import (
    ChunkingSettings,
    DockerSettings,
    EmbeddingsSettings,
    GenerationSettings,
    LlmSettings,
    ObservabilitySettings,
    RetrievalSettings,
    SecuritySettings,
    Settings,
    TieringSettings,
    VectorDbSettings,
)


def test_defaults_match_the_documented_reference() -> None:
    settings = Settings()

    assert settings.app.port == 8000
    assert settings.vector_db.provider == "qdrant"
    assert settings.vector_db.port == 6333
    assert settings.vector_db.grpc_port == 6334
    assert settings.vector_db.prefer_grpc is False
    assert settings.embeddings.provider == "huggingface"
    assert settings.embeddings.batch_size == 64
    assert settings.chunking.strategy == "recursive"
    assert settings.chunking.chunk_size == 768
    assert settings.retrieval.rrf_k == 60
    assert settings.retrieval.rerank is True
    assert settings.retrieval.rerank_top_n == 100
    assert settings.reliability.timeouts.llm_ms == 120000
    assert settings.index.reindex.strategy == "blue_green"


def test_every_integration_toggle_defaults_to_false() -> None:
    settings = Settings()

    assert settings.observability.dashboard is False
    assert settings.observability.otel is False
    assert settings.observability.langfuse is False
    assert settings.observability.grafana is False
    assert settings.autopilot.enabled is False
    assert settings.eval.regression_gate is False
    assert settings.security.auth is False
    assert settings.security.multi_tenancy is False
    assert settings.generation.grounded_or_refuse is False
    assert settings.cache.semantic is False
    assert settings.embeddings.tiering.enabled is False


def test_settings_are_immutable() -> None:
    settings = Settings()
    with pytest.raises(ValidationError):
        settings.app.port = 9000


def test_unknown_top_level_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="retrival"):
        Settings.model_validate({"retrival": {}})


def test_unknown_nested_keys_are_rejected() -> None:
    with pytest.raises(ValidationError, match="top_kk"):
        Settings.model_validate({"retrieval": {"top_kk": 5}})


def test_rule_1_top_k_must_not_exceed_rerank_top_n() -> None:
    with pytest.raises(ValidationError, match="less than or equal to"):
        RetrievalSettings(top_k=50, rerank_top_n=20)


def test_rule_2_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="must be less than"):
        ChunkingSettings(chunk_size=512, overlap=512)


def test_rule_3_hosted_embedding_providers_require_a_key_reference() -> None:
    with pytest.raises(ValidationError, match=r"embeddings\.api_key_env is required"):
        EmbeddingsSettings(provider="openai", api_key_env=None)

    assert EmbeddingsSettings(provider="huggingface", api_key_env=None).provider == "huggingface"


def test_rule_3_llm_providers_require_a_key_reference_except_local() -> None:
    with pytest.raises(ValidationError, match=r"llm\.api_key_env is required"):
        LlmSettings(provider="anthropic", api_key_env=None)

    assert LlmSettings(provider="ollama", api_key_env=None).provider == "ollama"


def test_openai_compatible_requires_a_base_url() -> None:
    with pytest.raises(ValidationError, match=r"llm\.base_url is required"):
        LlmSettings(provider="openai_compatible", api_key_env="CUSTOM_KEY")


def test_rule_4_grpc_port_must_differ_from_rest_port() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        VectorDbSettings(port=6333, grpc_port=6333)


def test_rule_5_citations_cannot_be_disabled_while_refusing() -> None:
    with pytest.raises(ValidationError, match="citations cannot be false"):
        GenerationSettings(grounded_or_refuse=True, citations=False)


def test_rule_6_otel_requires_an_endpoint() -> None:
    with pytest.raises(ValidationError, match="otel_endpoint is required"):
        ObservabilitySettings(otel=True)

    assert ObservabilitySettings(otel=True, otel_endpoint="http://localhost:4318").otel is True


def test_rule_7_named_docker_volume_is_required_on_windows_or_wsl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("fasterrag.config.schema.is_windows_or_wsl", lambda: True)
    with pytest.raises(ValidationError, match="named Docker volume"):
        DockerSettings(volume="/var/lib/qdrant")

    monkeypatch.setattr("fasterrag.config.schema.is_windows_or_wsl", lambda: False)
    assert DockerSettings(volume="/var/lib/qdrant").volume == "/var/lib/qdrant"


def test_rule_8_tiering_requires_rules() -> None:
    with pytest.raises(ValidationError, match="rules must be non-empty"):
        TieringSettings(enabled=True)

    tiering = TieringSettings.model_validate(
        {
            "enabled": True,
            "rules": [
                {"match": {"department": "legal"}, "provider": "openai", "model": "text-embed"}
            ],
        }
    )
    assert tiering.rules[0].provider == "openai"


def test_docker_image_tag_must_be_pinned() -> None:
    with pytest.raises(ValidationError, match="not allowed"):
        DockerSettings(image="qdrant/qdrant:latest")

    with pytest.raises(ValidationError, match="must pin a tag"):
        DockerSettings(image="qdrant/qdrant")

    assert DockerSettings(image="qdrant/qdrant:v1.9.0").image == "qdrant/qdrant:v1.9.0"


def test_reranker_model_is_required_when_reranking() -> None:
    with pytest.raises(ValidationError, match="reranker_model is required"):
        RetrievalSettings(rerank=True, reranker_model="  ")


def test_retrieval_leg_weights_must_sum_above_zero() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        RetrievalSettings(bm25_weight=0.0, dense_weight=0.0)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_size", 32),
        ("chunk_size", 4000),
        ("context_tokens", 10),
        ("context_tokens", 200),
        ("overlap", -1),
    ],
)
def test_documented_bounds_are_enforced(field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        ChunkingSettings.model_validate({field: value})


def test_api_key_env_rejects_a_value_that_is_not_a_variable_name() -> None:
    with pytest.raises(ValidationError, match="environment variable NAME"):
        SecuritySettings(api_key_env="sk-proj-not-a-variable-name")


def test_referenced_env_vars_are_discovered_with_their_config_paths() -> None:
    referenced = Settings().referenced_env_vars()

    assert referenced["QDRANT_API_KEY"] == "vector_db.api_key_env"
    assert referenced["OPENAI_API_KEY"] == "llm.api_key_env"
    assert referenced["FASTERRAG_API_KEY"] == "security.api_key_env"


def test_unreferenced_env_vars_are_not_required() -> None:
    settings = Settings.model_validate(
        {
            "vector_db": {"api_key_env": None},
            "llm": {
                "provider": "ollama",
                "api_key_env": None,
                "base_url": "http://localhost:11434",
            },
        }
    )
    assert set(settings.referenced_env_vars()) == {"FASTERRAG_API_KEY"}


def test_the_semantic_cache_accepts_a_disk_backend() -> None:
    """A memory cache dies with each CLI process, so it can never hit across invocations."""
    settings = Settings.model_validate({"cache": {"backend": "disk"}})

    assert settings.cache.backend == "disk"
