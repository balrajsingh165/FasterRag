import logging
from pathlib import Path

import pytest

from fasterrag.config.loader import load_settings
from fasterrag.errors import ConfigError, ErrorCode


def write_config(directory: Path, body: str) -> Path:
    path = directory / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


@pytest.mark.usefixtures("env")
def test_canonical_config_loads(canonical_config: Path) -> None:
    settings = load_settings(canonical_config, env_file=None)
    assert settings.vector_db.provider == "qdrant"
    assert settings.retrieval.rrf_k == 60


def test_missing_file_names_the_path() -> None:
    with pytest.raises(ConfigError, match="configuration file not found") as caught:
        load_settings("does-not-exist.yaml", env_file=None)
    assert caught.value.code is ErrorCode.CONFIG_INVALID


def test_malformed_yaml_is_reported(tmp_path: Path) -> None:
    path = write_config(tmp_path, "app:\n  port: [unclosed\n")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_settings(path, env_file=None)


def test_non_mapping_document_is_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path, "- one\n- two\n")
    with pytest.raises(ConfigError, match="mapping of configuration sections"):
        load_settings(path, env_file=None)


@pytest.mark.usefixtures("env")
def test_empty_file_falls_back_to_defaults(tmp_path: Path) -> None:
    path = write_config(tmp_path, "")
    assert load_settings(path, env_file=None).app.port == 8000


@pytest.mark.usefixtures("env")
def test_invalid_value_names_the_offending_key(tmp_path: Path) -> None:
    path = write_config(tmp_path, "app:\n  port: 70000\n")
    with pytest.raises(ConfigError, match=r"app\.port") as caught:
        load_settings(path, env_file=None)
    assert "is invalid" in caught.value.detail


@pytest.mark.usefixtures("env")
def test_cross_field_violation_names_the_section(tmp_path: Path) -> None:
    path = write_config(tmp_path, "retrieval:\n  top_k: 50\n  rerank_top_n: 20\n")
    with pytest.raises(ConfigError, match="retrieval"):
        load_settings(path, env_file=None)


@pytest.mark.usefixtures("env")
def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path, "retrieval:\n  top_kk: 5\n")
    with pytest.raises(ConfigError, match="top_kk"):
        load_settings(path, env_file=None)


@pytest.mark.usefixtures("env")
def test_validation_error_never_echoes_the_offending_value(tmp_path: Path) -> None:
    secret = "sk-proj-supersecretvalue"
    path = write_config(tmp_path, f"security:\n  api_key_env: {secret}\n")

    with pytest.raises(ConfigError) as caught:
        load_settings(path, env_file=None)

    rendered = f"{caught.value}{caught.value.__cause__}{caught.value.__context__}"
    assert secret not in rendered
    assert "security.api_key_env" in caught.value.detail


def test_missing_referenced_env_var_is_fatal(tmp_path: Path, env: dict[str, str]) -> None:
    del env["OPENAI_API_KEY"]
    path = write_config(tmp_path, "llm:\n  api_key_env: OPENAI_API_KEY\n")

    with pytest.raises(ConfigError, match="OPENAI_API_KEY") as caught:
        load_settings(path, env_file=None)
    assert "llm.api_key_env" in caught.value.detail


def test_blank_referenced_env_var_is_fatal(tmp_path: Path, env: dict[str, str]) -> None:
    env["OPENAI_API_KEY"] = "   "
    path = write_config(tmp_path, "llm:\n  api_key_env: OPENAI_API_KEY\n")

    with pytest.raises(ConfigError, match="missing or blank"):
        load_settings(path, env_file=None)


def test_env_error_never_echoes_a_secret_value(tmp_path: Path, env: dict[str, str]) -> None:
    env["QDRANT_API_KEY"] = "super-secret-qdrant-key"
    del env["FASTERRAG_API_KEY"]
    path = write_config(tmp_path, "security:\n  api_key_env: FASTERRAG_API_KEY\n")

    with pytest.raises(ConfigError) as caught:
        load_settings(path, env_file=None)
    assert "super-secret-qdrant-key" not in str(caught.value)


def test_unreferenced_env_vars_are_not_required(tmp_path: Path, env: dict[str, str]) -> None:
    del env["OPENAI_API_KEY"]
    del env["QDRANT_API_KEY"]
    path = write_config(
        tmp_path,
        "vector_db:\n"
        "  api_key_env: null\n"
        "llm:\n"
        "  provider: ollama\n"
        "  api_key_env: null\n"
        "  base_url: http://localhost:11434\n",
    )

    assert load_settings(path, env_file=None).llm.provider == "ollama"


def test_dotenv_file_supplies_missing_variables(tmp_path: Path, env: dict[str, str]) -> None:
    del env["OPENAI_API_KEY"]
    (tmp_path / ".env").write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")
    path = write_config(tmp_path, "llm:\n  api_key_env: OPENAI_API_KEY\n")

    assert load_settings(path, env_file=tmp_path / ".env").llm.api_key_env == "OPENAI_API_KEY"
    assert env["OPENAI_API_KEY"] == "from-dotenv"


def test_process_environment_wins_over_dotenv(tmp_path: Path, env: dict[str, str]) -> None:
    env["OPENAI_API_KEY"] = "from-process"
    (tmp_path / ".env").write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")
    path = write_config(tmp_path, "llm:\n  api_key_env: OPENAI_API_KEY\n")

    load_settings(path, env_file=tmp_path / ".env")
    assert env["OPENAI_API_KEY"] == "from-process"


@pytest.mark.usefixtures("env")
def test_missing_dotenv_file_is_not_an_error(tmp_path: Path) -> None:
    path = write_config(tmp_path, "")
    assert load_settings(path, env_file=tmp_path / "absent.env").app.port == 8000


@pytest.mark.usefixtures("env")
def test_large_chunk_size_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = write_config(tmp_path, "chunking:\n  chunk_size: 2000\n  overlap: 64\n")
    with caplog.at_level(logging.WARNING):
        load_settings(path, env_file=None)
    assert "context cliff" in caplog.text


@pytest.mark.usefixtures("env")
def test_in_place_reindex_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = write_config(tmp_path, "index:\n  reindex:\n    strategy: in_place\n")
    with caplog.at_level(logging.WARNING):
        load_settings(path, env_file=None)
    assert "development-only" in caplog.text


@pytest.mark.usefixtures("env")
def test_default_settings_do_not_warn(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = write_config(tmp_path, "")
    with caplog.at_level(logging.WARNING):
        load_settings(path, env_file=None)
    assert caplog.text == ""


def test_enabling_auth_that_nothing_enforces_refuses_to_start(tmp_path: Path) -> None:
    """An open API reporting itself authenticated is worse than a startup failure."""
    path = write_config(tmp_path, "security:\n  auth: true\n")

    with pytest.raises(ConfigError, match="enforced by nothing yet") as caught:
        load_settings(path, env_file=None)

    assert "security.auth" in caught.value.detail
    assert caught.value.code is ErrorCode.CONFIG_INVALID


def test_the_failure_names_the_slice_that_will_enforce_it(tmp_path: Path) -> None:
    path = write_config(tmp_path, "security:\n  multi_tenancy: true\n")

    with pytest.raises(ConfigError, match="TASK-0046"):
        load_settings(path, env_file=None)


def test_every_unenforced_setting_is_listed_at_once(tmp_path: Path) -> None:
    """Reporting one at a time makes an operator restart four times to learn four things."""
    path = write_config(
        tmp_path,
        "security:\n  auth: true\n  multi_tenancy: true\n"
        "cost:\n  per_query_token_budget: 1000\n  per_tenant_token_budget: 5000\n",
    )

    with pytest.raises(ConfigError) as caught:
        load_settings(path, env_file=None)

    for key in (
        "security.auth",
        "security.multi_tenancy",
        "cost.per_query_token_budget",
        "cost.per_tenant_token_budget",
    ):
        assert key in caught.value.detail


@pytest.mark.usefixtures("env")
def test_a_zero_token_budget_means_unlimited_and_is_not_rejected(tmp_path: Path) -> None:
    """Zero is the documented default for "no budget", not a budget of nothing."""
    path = write_config(tmp_path, "cost:\n  per_query_token_budget: 0\n")

    settings = load_settings(path, env_file=None)

    assert settings.cost.per_query_token_budget == 0


@pytest.mark.usefixtures("env")
def test_the_canonical_config_does_not_trip_the_check(canonical_config: Path) -> None:
    """The shipped defaults must start; a check that rejects them is a broken check."""
    settings = load_settings(canonical_config, env_file=None)

    assert not settings.security.auth
    assert not settings.security.multi_tenancy
