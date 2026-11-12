"""What ``api_key_env`` means for a provider the schema has never heard of (TASK-0439).

Cross-field rule 3 decides key requirements by provider *name*: ``openai`` and ``cohere``
embeddings need one, every LLM but ``ollama`` needs one. A plugin's name is in neither
list, so before this rule existed it fell wherever the phrasing happened to put it — a
keyless local LLM plugin was told to invent a key, and a config that said only
``provider: acme`` was handed ``OPENAI_API_KEY`` or ``QDRANT_API_KEY`` by default.

The rule these cases pin: a plugin never inherits a built-in's default key name. Where the
default names one (``llm``, ``vector_db``) the field must be written, as a variable name or
as ``null``; where the default is already ``null`` (``embeddings``) nothing is asked. A
named variable is still required to exist, by rule 9, like any other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fasterrag.config import plugins
from fasterrag.config.loader import load_settings
from fasterrag.config.schema import (
    PROVIDERS_NOT_REQUIRING_LLM_KEY,
    PROVIDERS_REQUIRING_EMBEDDING_KEY,
    Settings,
)
from fasterrag.errors import ConfigError
from tests.support.plugins import install_distribution

DEFAULTED = [
    ("llm", plugins.LLM_GROUP, "OPENAI_API_KEY"),
    ("vector_db", plugins.VECTOR_DB_GROUP, "QDRANT_API_KEY"),
]


@pytest.fixture
def acme(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Install one distribution registering ``acme`` as all three kinds of provider."""
    install_distribution(
        tmp_path,
        monkeypatch,
        {
            plugins.VECTOR_DB_GROUP: {"acme": "acme_pkg:VectorDB"},
            plugins.EMBEDDINGS_GROUP: {"acme": "acme_pkg:Embedder"},
            plugins.LLM_GROUP: {"acme": "acme_pkg:LLM"},
        },
    )


def validate(section: str, **body: Any) -> Settings:
    """Validate a configuration whose one written section names the ``acme`` plugin."""
    return Settings.model_validate({section: {"provider": "acme", **body}})


@pytest.mark.usefixtures("acme")
@pytest.mark.parametrize(("section", "group", "default"), DEFAULTED)
def test_a_plugin_is_never_handed_a_built_in_default_key_name(
    section: str, group: str, default: str
) -> None:
    """The defect: ``provider: acme`` alone gave third-party code another vendor's key name."""
    with pytest.raises(ValidationError) as caught:
        validate(section)

    message = caught.value.errors()[0]["msg"]
    assert f"{section}.api_key_env must be written out for plugin provider 'acme'" in message
    assert repr(default) in message
    assert "never handed to a plugin" in message


@pytest.mark.usefixtures("acme")
@pytest.mark.parametrize(("section", "group", "default"), DEFAULTED)
def test_a_plugin_may_say_it_needs_no_key(section: str, group: str, default: str) -> None:
    """A local plugin has no credential; before this an LLM one was told to invent one."""
    settings = validate(section, api_key_env=None)

    assert getattr(settings, section).api_key_env is None
    assert default not in settings.referenced_env_vars()


@pytest.mark.usefixtures("acme")
@pytest.mark.parametrize(("section", "group", "default"), DEFAULTED)
def test_a_plugin_may_name_its_own_key(section: str, group: str, default: str) -> None:
    settings = validate(section, api_key_env="ACME_API_KEY")

    assert settings.referenced_env_vars()["ACME_API_KEY"] == f"{section}.api_key_env"


@pytest.mark.usefixtures("acme")
def test_an_embedding_plugin_is_asked_for_nothing() -> None:
    """``embeddings.api_key_env`` already defaults to null, so there is nothing to inherit."""
    settings = validate("embeddings")

    assert settings.embeddings.api_key_env is None


@pytest.mark.usefixtures("acme")
def test_a_key_a_plugin_names_must_exist_like_any_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rule 9 is not relaxed for a plugin: a named variable that is absent stops the load."""
    monkeypatch.delenv("ACME_API_KEY", raising=False)
    path = tmp_path / "config.yaml"
    path.write_text(
        "vector_db:\n  api_key_env: null\nllm:\n  provider: acme\n  api_key_env: ACME_API_KEY\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="ACME_API_KEY") as caught:
        load_settings(path, env_file=None)

    assert "referenced by llm.api_key_env" in caught.value.detail


def test_the_built_in_rules_are_unchanged() -> None:
    """The plugin rule is an addition; rule 3 still decides every built-in by name."""
    assert frozenset({"openai", "cohere"}) == PROVIDERS_REQUIRING_EMBEDDING_KEY
    assert frozenset({"ollama"}) == PROVIDERS_NOT_REQUIRING_LLM_KEY

    with pytest.raises(ValidationError, match=r"embeddings.api_key_env is required"):
        Settings.model_validate({"embeddings": {"provider": "openai"}})
    with pytest.raises(ValidationError, match=r"llm.api_key_env is required"):
        Settings.model_validate({"llm": {"provider": "anthropic", "api_key_env": None}})

    assert Settings.model_validate({"llm": {"provider": "ollama", "api_key_env": None}})
    assert Settings.model_validate({"vector_db": {"provider": "chroma"}})
