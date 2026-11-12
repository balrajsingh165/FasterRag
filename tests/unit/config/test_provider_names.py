"""Provider names: built-ins, installed plugins, and everything else refused at load.

Until TASK-0439 ``vector_db.provider``, ``embeddings.provider``, ``llm.provider`` and a
tier rule's ``provider`` were closed ``Literal`` types, so a plugin the adapter factories
could already resolve was refused by the schema before any factory ran. These cases pin
both halves of the replacement: a registered name is accepted, and fail-fast did not
weaken — an unknown name still stops the load, now naming what is installed.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError

import fasterrag.config
from fasterrag.adapters.embeddings import factory as embeddings_factory
from fasterrag.adapters.llm import factory as llm_factory
from fasterrag.adapters.vectordb import factory as vectordb_factory
from fasterrag.config import plugins
from fasterrag.config.loader import load_settings
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError
from tests.support.plugins import install_distribution

KINDS: list[tuple[ModuleType, frozenset[str], str, str]] = [
    (
        vectordb_factory,
        plugins.BUILTIN_VECTOR_DB_PROVIDERS,
        plugins.VECTOR_DB_GROUP,
        "vector_db",
    ),
    (
        embeddings_factory,
        plugins.BUILTIN_EMBEDDING_PROVIDERS,
        plugins.EMBEDDINGS_GROUP,
        "embeddings",
    ),
    (llm_factory, plugins.BUILTIN_LLM_PROVIDERS, plugins.LLM_GROUP, "llm"),
]

GROUPS = [(group, section) for _, _, group, section in KINDS]


def section(name: str, provider: str) -> dict[str, Any]:
    """Return a section naming ``provider`` that passes every rule but the name's own."""
    body: dict[str, Any] = {"provider": provider, "api_key_env": "SOME_PROVIDER_KEY"}
    if provider == "openai_compatible":
        body["base_url"] = "https://llm.example.test/v1"
    if provider == "pgvector":
        body["pgvector"] = {"dsn_env": "PGVECTOR_DSN"}
    return {name: body}


def tiered(provider: str) -> dict[str, Any]:
    """Return an embeddings section whose one tier rule names ``provider``."""
    rule = {"match": {"department": "legal"}, "provider": provider, "model": "any-model"}
    return {"embeddings": {"tiering": {"enabled": True, "rules": [rule]}}}


def messages(error: ValidationError) -> str:
    """Return every violation as ``dotted.key: message``, the way the loader prints them."""
    return "\n".join(
        f"{'.'.join(str(part) for part in item['loc'])}: "
        f"{item['msg'].removeprefix('Value error, ')}"
        for item in error.errors()
    )


@pytest.mark.parametrize(("factory", "builtin", "group", "name"), KINDS)
def test_the_names_configuration_accepts_are_the_ones_the_factory_builds(
    factory: ModuleType, builtin: frozenset[str], group: str, name: str
) -> None:
    """The two lists are separate so ``config/`` need not import ``adapters/``; they must agree."""
    assert builtin == frozenset(factory._BUILTIN_ADAPTERS)
    assert group == factory.ENTRY_POINT_GROUP


@pytest.mark.parametrize(("factory", "builtin", "group", "name"), KINDS)
def test_every_built_in_name_is_accepted(
    factory: ModuleType, builtin: frozenset[str], group: str, name: str
) -> None:
    for provider in sorted(builtin):
        settings = Settings.model_validate(section(name, provider))

        assert getattr(settings, name).provider == provider


@pytest.mark.parametrize(("group", "name"), GROUPS)
def test_a_registered_plugin_name_is_accepted(
    group: str, name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect: ``provider: acme`` was refused by a closed list before any factory ran."""
    install_distribution(tmp_path, monkeypatch, {group: {"acme": "acme_pkg.adapter:Adapter"}})

    settings = Settings.model_validate(section(name, "acme"))

    assert getattr(settings, name).provider == "acme"


def test_a_tier_rule_may_name_a_registered_plugin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tier builds through the same factory as the default embedder, so it selects alike."""
    install_distribution(
        tmp_path, monkeypatch, {plugins.EMBEDDINGS_GROUP: {"acme": "acme_pkg:Embedder"}}
    )

    settings = Settings.model_validate(tiered("acme"))

    assert settings.embeddings.tiering.rules[0].provider == "acme"


@pytest.mark.parametrize(("factory", "builtin", "group", "name"), KINDS)
def test_an_unknown_name_fails_and_names_what_is_installed(
    factory: ModuleType,
    builtin: frozenset[str],
    group: str,
    name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-fast must not weaken: a typo stops the load and says what would have worked."""
    install_distribution(tmp_path, monkeypatch, {group: {"acme": "acme_pkg.adapter:Adapter"}})

    with pytest.raises(ValidationError) as caught:
        Settings.model_validate(section(name, "mydb"))

    text = messages(caught.value)
    assert f"{name}.provider: " in text
    assert "'mydb' is not an installed provider" in text
    assert repr(group) in text
    assert "acme" in text
    assert f"built-in providers: {', '.join(sorted(builtin))}; " in text


def test_an_unknown_name_with_no_plugin_installed_says_none_are() -> None:
    with pytest.raises(ValidationError) as caught:
        Settings.model_validate(section("vector_db", "mydb"))

    assert "plugins installed under the 'fasterrag.vectordb' entry-point group: none" in messages(
        caught.value
    )


def test_an_unknown_tier_rule_provider_fails_naming_the_rule() -> None:
    with pytest.raises(ValidationError) as caught:
        Settings.model_validate(tiered("mydb"))

    assert "embeddings.tiering.rules.0.provider: " in messages(caught.value)


def test_a_plugin_for_one_kind_is_not_selectable_as_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A vector database registered as ``acme`` must not validate as ``llm.provider: acme``."""
    install_distribution(
        tmp_path, monkeypatch, {plugins.VECTOR_DB_GROUP: {"acme": "acme_pkg.adapter:Adapter"}}
    )

    with pytest.raises(ValidationError) as caught:
        Settings.model_validate(section("llm", "acme"))

    assert "llm.provider: 'acme' is not an installed provider" in messages(caught.value)


def test_built_in_names_are_accepted_without_looking_at_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plugin cannot take over a built-in name, and a built-in config pays for no scan."""

    def refuse(group: str) -> frozenset[str]:
        raise AssertionError(f"a built-in name scanned the {group} entry points")

    monkeypatch.setattr(plugins, "registered_names", refuse)

    assert Settings.model_validate({}).vector_db.provider == "qdrant"


def test_the_loader_reports_an_unknown_provider_against_its_key(tmp_path: Path) -> None:
    """The same refusal through the real loader, as an operator would meet it."""
    path = tmp_path / "config.yaml"
    path.write_text("embeddings:\n  provider: mydb\n", encoding="utf-8")

    with pytest.raises(ConfigError) as caught:
        load_settings(path, env_file=None, require_env=False)

    assert "embeddings.provider: 'mydb' is not an installed provider" in caught.value.detail
    assert "cohere, huggingface, ollama, openai" in caught.value.detail


def test_nothing_in_the_config_package_imports_an_adapter() -> None:
    """Validating through ``adapters/`` would close a cycle: every adapter imports the schema."""
    offenders: dict[str, set[str]] = {}
    for path in sorted(Path(fasterrag.config.__file__).parent.glob("*.py")):
        imported: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        adapters = {name for name in imported if name.startswith("fasterrag.adapters")}
        if adapters:
            offenders[path.name] = adapters

    assert not offenders
