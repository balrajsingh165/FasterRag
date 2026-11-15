"""Configuration carries a schema version, and a mismatched one names its upgrade.

TASK-0502: nothing in ``config.yaml`` said which release's layout it followed, so a release
that renamed a key could only fail on the unknown key, never say what had changed.
"""

from pathlib import Path

import pytest

from fasterrag.config.loader import ConfigMigration, check_schema_version, load_settings
from fasterrag.config.schema import CONFIG_SCHEMA_VERSION, Settings
from fasterrag.errors import ConfigError, ErrorCode


def write_config(directory: Path, body: str) -> Path:
    path = directory / "config.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_the_current_schema_version_is_one() -> None:
    assert CONFIG_SCHEMA_VERSION == 1
    assert Settings().schema_version == 1


def test_the_canonical_config_states_its_schema_version(canonical_config: Path) -> None:
    """The canonical file must say which layout it follows, at the top."""
    first_key = next(
        line
        for line in canonical_config.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )
    assert first_key == f"schema_version: {CONFIG_SCHEMA_VERSION}"


@pytest.mark.usefixtures("env")
def test_a_config_written_before_schema_version_existed_loads_as_version_one(
    tmp_path: Path, canonical_config: Path
) -> None:
    """0.1.0b1 and 0.1.0b2 configs carry no schema_version and are public; they must load."""
    published = "\n".join(
        line
        for line in canonical_config.read_text(encoding="utf-8").splitlines()
        if not line.startswith("schema_version")
    )
    assert "schema_version" not in published
    settings = load_settings(write_config(tmp_path, published), env_file=None)
    assert settings.schema_version == 1


@pytest.mark.usefixtures("env")
def test_a_config_from_a_newer_release_is_refused_naming_the_upgrade(tmp_path: Path) -> None:
    """A newer layout would otherwise fail on whichever renamed key it reached first."""
    path = write_config(tmp_path, f"schema_version: {CONFIG_SCHEMA_VERSION + 1}\n")
    with pytest.raises(ConfigError) as caught:
        load_settings(path, env_file=None)
    detail = caught.value.detail
    assert caught.value.code is ErrorCode.CONFIG_INVALID
    assert f"schema_version {CONFIG_SCHEMA_VERSION + 1}" in detail
    assert "pip install --upgrade fasterrag" in detail


@pytest.mark.usefixtures("env")
@pytest.mark.parametrize("value", ["'one'", "1.5", "true", "0", "-3"])
def test_a_schema_version_that_is_not_a_positive_whole_number_is_refused(
    tmp_path: Path, value: str
) -> None:
    path = write_config(tmp_path, f"schema_version: {value}\n")
    with pytest.raises(ConfigError, match="schema_version"):
        load_settings(path, env_file=None)


def test_an_older_config_is_refused_with_each_step_that_upgrades_it(tmp_path: Path) -> None:
    """With a fake release at version 3, a version-1 file is told both steps, in order."""
    migrations = (
        ConfigMigration(1, "rename retrieval.top_k to retrieval.limit"),
        ConfigMigration(2, "move app.log_level to observability.log_level"),
    )
    with pytest.raises(ConfigError) as caught:
        check_schema_version({}, tmp_path / "config.yaml", current=3, migrations=migrations)
    detail = caught.value.detail
    assert "schema_version 1" in detail
    first = detail.index("1 -> 2: rename retrieval.top_k to retrieval.limit")
    second = detail.index("2 -> 3: move app.log_level to observability.log_level")
    assert first < second
    assert "schema_version: 3" in detail


def test_an_older_config_with_no_recorded_step_says_so(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no upgrade from schema_version 1 is recorded"):
        check_schema_version(
            {"schema_version": 1}, tmp_path / "config.yaml", current=2, migrations=()
        )


def test_the_current_version_passes(tmp_path: Path) -> None:
    check_schema_version(
        {"schema_version": CONFIG_SCHEMA_VERSION}, tmp_path / "config.yaml", migrations=()
    )


def test_every_recorded_migration_starts_below_the_current_version() -> None:
    from fasterrag.config.loader import CONFIG_MIGRATIONS

    assert all(step.from_version < CONFIG_SCHEMA_VERSION for step in CONFIG_MIGRATIONS)
    assert [step.from_version for step in CONFIG_MIGRATIONS] == list(
        range(1, CONFIG_SCHEMA_VERSION)
    )


def test_settings_built_directly_refuse_another_version() -> None:
    with pytest.raises(ValueError, match="schema_version"):
        Settings(schema_version=CONFIG_SCHEMA_VERSION + 1)
