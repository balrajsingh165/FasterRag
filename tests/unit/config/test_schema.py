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


@pytest.mark.parametrize("value", [4, 1, None, "four"])
def test_the_removed_app_workers_key_is_refused_by_name_whatever_its_value(value: object) -> None:
    """A library caller building ``Settings`` directly gets the same reason the loader gives."""
    with pytest.raises(ValidationError, match=r"app\.workers was removed"):
        Settings.model_validate({"app": {"workers": value}})


def test_the_pipeline_worker_section_is_not_the_removed_key() -> None:
    """``workers.*`` sizes the real pools and must keep loading."""
    settings = Settings.model_validate({"workers": {"cpu_pool_size": 2}})

    assert settings.workers.cpu_pool_size == 2
    assert not hasattr(settings.app, "workers")


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


def test_the_api_key_is_required_only_when_auth_is_on() -> None:
    """With auth off, nothing reads FASTERRAG_API_KEY, yet a load demanded it (TASK-0444)."""
    assert "FASTERRAG_API_KEY" not in Settings().required_env_vars()

    secured = Settings.model_validate({"security": {"auth": True}})

    assert secured.required_env_vars()["FASTERRAG_API_KEY"] == "security.api_key_env"


@pytest.mark.parametrize("provider", ["chroma", "pgvector"])
def test_a_vector_database_that_reads_no_key_needs_none(provider: str) -> None:
    """Chroma and pgvector never read vector_db.api_key_env, yet QDRANT_API_KEY was demanded."""
    settings = Settings.model_validate(
        {"vector_db": {"provider": provider, "pgvector": {"dsn_env": "PGVECTOR_DSN"}}}
    )

    assert "QDRANT_API_KEY" not in settings.required_env_vars()


@pytest.mark.parametrize("provider", ["qdrant", "milvus", "weaviate"])
def test_a_vector_database_that_reads_its_key_still_requires_it(provider: str) -> None:
    settings = Settings.model_validate({"vector_db": {"provider": provider}})

    assert settings.required_env_vars()["QDRANT_API_KEY"] == "vector_db.api_key_env"


def test_the_pgvector_dsn_is_required_only_for_pgvector() -> None:
    elsewhere = Settings.model_validate({"vector_db": {"pgvector": {"dsn_env": "PGVECTOR_DSN"}}})
    pgvector = Settings.model_validate(
        {"vector_db": {"provider": "pgvector", "pgvector": {"dsn_env": "PGVECTOR_DSN"}}}
    )

    assert "PGVECTOR_DSN" not in elsewhere.required_env_vars()
    assert pgvector.required_env_vars()["PGVECTOR_DSN"] == "vector_db.pgvector.dsn_env"


def test_the_llm_key_stays_required_because_every_configuration_can_generate() -> None:
    """No setting turns generation off, so a keyed LLM provider is always in use."""
    assert Settings().required_env_vars()["OPENAI_API_KEY"] == "llm.api_key_env"


def test_a_keyless_provider_does_not_require_a_key_it_was_given_a_name_for() -> None:
    settings = Settings.model_validate(
        {
            "embeddings": {"provider": "huggingface", "api_key_env": "HF_KEY"},
            "llm": {"provider": "ollama", "api_key_env": "OLLAMA_KEY"},
        }
    )

    assert {"HF_KEY", "OLLAMA_KEY"}.isdisjoint(settings.required_env_vars())


def test_a_keyed_embedding_provider_still_requires_its_key() -> None:
    settings = Settings.model_validate(
        {"embeddings": {"provider": "openai", "api_key_env": "OPENAI_EMBED_KEY"}}
    )

    assert settings.required_env_vars()["OPENAI_EMBED_KEY"] == "embeddings.api_key_env"


def test_a_variable_one_unused_key_names_is_still_required_by_a_used_one() -> None:
    """Two keys may share a variable; skipping the unused one must not hide the other."""
    settings = Settings.model_validate(
        {
            "vector_db": {"provider": "chroma", "api_key_env": "SHARED_KEY"},
            "llm": {"api_key_env": "SHARED_KEY"},
        }
    )

    assert settings.required_env_vars()["SHARED_KEY"] == "llm.api_key_env"


def test_the_semantic_cache_accepts_a_disk_backend() -> None:
    """A memory cache dies with each CLI process, so it can never hit across invocations."""
    settings = Settings.model_validate({"cache": {"backend": "disk"}})

    assert settings.cache.backend == "disk"
