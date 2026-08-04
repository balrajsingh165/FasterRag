"""``--set`` overrides, which must be held to exactly the rules a file value is.

An override that skipped validation would be a way to put the system into a state the
config file cannot express — the opposite of what a fail-fast loader is for.
"""

from pathlib import Path

import pytest

from fasterrag.cli.main import main
from fasterrag.cli.output import ExitCode
from fasterrag.config.loader import apply_overrides, load_settings
from fasterrag.config.template import canonical_config_text
from fasterrag.errors import ConfigError


@pytest.fixture
def config(tmp_path: Path) -> Path:
    destination = tmp_path / "config.yaml"
    destination.write_text(canonical_config_text(), encoding="utf-8")
    return destination


def test_an_override_replaces_the_file_value(config: Path) -> None:
    settings = load_settings(config, require_env=False, overrides=["chunking.chunk_size=512"])

    assert settings.chunking.chunk_size == 512


def test_a_nested_key_is_reached(config: Path) -> None:
    settings = load_settings(
        config, require_env=False, overrides=["vector_db.collection.shard_number=3"]
    )

    assert settings.vector_db.collection.shard_number == 3


def test_values_arrive_as_types_rather_than_strings(config: Path) -> None:
    """`rerank=false` must become the boolean, not the truthy string "false"."""
    settings = load_settings(
        config,
        require_env=False,
        overrides=["retrieval.rerank=false", "retrieval.bm25_k1=0.9"],
    )

    assert settings.retrieval.rerank is False
    assert settings.retrieval.bm25_k1 == 0.9


def test_a_string_value_still_works(config: Path) -> None:
    settings = load_settings(
        config, require_env=False, overrides=["embeddings.model=intfloat/e5-base"]
    )

    assert settings.embeddings.model == "intfloat/e5-base"


def test_overrides_apply_in_order(config: Path) -> None:
    settings = load_settings(
        config, require_env=False, overrides=["chunking.chunk_size=512", "chunking.chunk_size=256"]
    )

    assert settings.chunking.chunk_size == 256


def test_an_out_of_range_override_is_refused(config: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        load_settings(config, require_env=False, overrides=["chunking.chunk_size=99999"])

    assert "chunk_size" in caught.value.detail


def test_an_unknown_key_is_refused(config: Path) -> None:
    """Otherwise a typo is silently accepted and the setting the operator meant never moves."""
    with pytest.raises(ConfigError):
        load_settings(config, require_env=False, overrides=["chunking.chunk_sise=512"])


def test_cross_field_rules_still_run(config: Path) -> None:
    """The rules run at construction, so an override must be merged before validation."""
    with pytest.raises(ConfigError) as caught:
        load_settings(
            config,
            require_env=False,
            overrides=["retrieval.rerank_top_n=20", "retrieval.top_k=50"],
        )

    assert "rerank_top_n" in caught.value.detail


def test_a_missing_equals_names_the_form(config: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        load_settings(config, require_env=False, overrides=["chunking.chunk_size"])

    assert "dotted.key=value" in caught.value.detail


def test_an_empty_key_is_refused(config: Path) -> None:
    with pytest.raises(ConfigError):
        load_settings(config, require_env=False, overrides=["=512"])


def test_descending_through_a_scalar_is_refused() -> None:
    """`--set app.port.deeper=1` must say why rather than raise an AttributeError."""
    with pytest.raises(ConfigError) as caught:
        apply_overrides({"app": {"port": 8000}}, ["app.port.deeper=1"])

    assert "port" in caught.value.detail


def test_the_source_mapping_is_not_mutated() -> None:
    """The caller's parsed YAML is reused elsewhere; an in-place edit would leak."""
    raw = {"chunking": {"chunk_size": 768}}

    apply_overrides(raw, ["chunking.chunk_size=512"])

    assert raw == {"chunking": {"chunk_size": 768}}


def test_no_overrides_leaves_the_file_untouched(config: Path) -> None:
    assert load_settings(config, require_env=False, overrides=[]).chunking.chunk_size == 768


def test_the_flag_works_from_the_command_line(
    config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["config", "show", "--changed", "--config", str(config), "--set", "app.port=9001"])

    assert code == ExitCode.SUCCESS
    assert "app.port" in capsys.readouterr().out


def test_the_flag_is_accepted_before_the_subcommand(
    config: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Operators write global flags in either position and neither reading is wrong."""
    code = main(["--set", "app.port=9001", "config", "show", "--changed", "--config", str(config)])

    assert code == ExitCode.SUCCESS
    assert "app.port" in capsys.readouterr().out
