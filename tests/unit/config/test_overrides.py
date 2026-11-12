"""``--set`` overrides, which must be held to exactly the rules a file value is.

An override that skipped validation would be a way to put the system into a state the
config file cannot express — the opposite of what a fail-fast loader is for.
"""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from fasterrag.cli.main import main
from fasterrag.cli.output import ExitCode
from fasterrag.config.loader import apply_overrides, load_settings
from fasterrag.config.template import canonical_config_text
from fasterrag.errors import ConfigError
from tests.unit.config.conftest import REPO_ROOT


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


def test_the_environment_can_carry_overrides(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A container has no command line to add flags to."""
    monkeypatch.setenv("FASTERRAG_SET", "chunking.chunk_size=512")

    assert load_settings(config, require_env=False).chunking.chunk_size == 512


def test_several_environment_overrides_are_comma_separated(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FASTERRAG_SET", "chunking.chunk_size=512, retrieval.top_k=5")

    settings = load_settings(config, require_env=False)

    assert settings.chunking.chunk_size == 512
    assert settings.retrieval.top_k == 5


def test_an_explicit_flag_beats_the_environment(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Losing to a variable invisible from the command line would be silent."""
    monkeypatch.setenv("FASTERRAG_SET", "chunking.chunk_size=512")

    settings = load_settings(config, require_env=False, overrides=["chunking.chunk_size=256"])

    assert settings.chunking.chunk_size == 256


def test_an_environment_override_is_validated_like_any_other(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FASTERRAG_SET", "chunking.chunk_size=99999")

    with pytest.raises(ConfigError):
        load_settings(config, require_env=False)


def test_a_blank_environment_override_changes_nothing(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FASTERRAG_SET", "   ")

    assert load_settings(config, require_env=False).chunking.chunk_size == 768


COMPOSE_FILES = [
    REPO_ROOT / "docker-compose.yml",
    *(REPO_ROOT / "deploy" / f"compose.{size}.yml" for size in ("small", "medium", "large")),
]
COMPOSE_OVERRIDE = re.compile(r"^\s*FASTERRAG_SET:\s*(?P<value>\S.*?)\s*$", re.MULTILINE)


def test_every_override_a_shipped_compose_file_sets_loads(
    config: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sizing presets set ``app.workers`` after nothing read it, and it would now refuse."""
    found = [
        (path.name, match.group("value"))
        for path in COMPOSE_FILES
        for match in COMPOSE_OVERRIDE.finditer(path.read_text(encoding="utf-8"))
    ]
    assert len(found) >= 4

    for name, value in found:
        monkeypatch.setenv("FASTERRAG_SET", value)
        try:
            load_settings(config, require_env=False)
        except ConfigError as exc:
            pytest.fail(f"{name} sets FASTERRAG_SET={value}, which does not load: {exc.detail}")


STOP_MARGIN_SECONDS = 10
SERVING_SERVICES = ("api", "dashboard")


COMPOSE_TAG = re.compile(r"!(reset|override)\s")


def _compose(path: Path) -> dict[str, Any]:
    """Return a Compose file as data, reading a Compose-only tag such as ``!reset`` as absent."""
    text = COMPOSE_TAG.sub("", path.read_text(encoding="utf-8"))
    loaded: dict[str, Any] = yaml.safe_load(text)
    return loaded


def _seconds(period: object, where: str) -> int:
    """Return a Compose ``stop_grace_period`` written as whole seconds, such as ``30s``."""
    matched = re.fullmatch(r"(\d+)s", str(period))
    assert matched, f"{where}: stop_grace_period {period!r} is not written as whole seconds"
    return int(matched.group(1))


def test_every_compose_file_gives_the_api_longer_to_stop_than_its_drain_takes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No stop period was set, so Docker's 10 s default cut the 20 s drain short."""
    base = _compose(REPO_ROOT / "docker-compose.yml")
    for path in COMPOSE_FILES:
        layer = _compose(path)
        for service in SERVING_SERVICES:
            merged = {**base["services"].get(service, {}), **layer["services"].get(service, {})}
            if not merged:
                continue
            override = (merged.get("environment") or {}).get("FASTERRAG_SET", "")
            monkeypatch.setenv("FASTERRAG_SET", override)
            drain = load_settings(
                REPO_ROOT / "config.yaml", require_env=False
            ).reliability.shutdown_grace_seconds
            where = f"{path.name} service {service}"
            assert "stop_grace_period" in merged, f"{where} sets no stop_grace_period"
            stop = _seconds(merged["stop_grace_period"], where)
            assert stop >= drain + STOP_MARGIN_SECONDS, (
                f"{where} is stopped after {stop}s, but reliability.shutdown_grace_seconds is "
                f"{drain}s and cancelling and settling what remains needs a margin of "
                f"{STOP_MARGIN_SECONDS}s"
            )


def test_trailing_separators_are_tolerated(config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A compose file with a trailing comma is a typo, not a reason to refuse to start."""
    monkeypatch.setenv("FASTERRAG_SET", "chunking.chunk_size=512,")

    assert load_settings(config, require_env=False).chunking.chunk_size == 512
