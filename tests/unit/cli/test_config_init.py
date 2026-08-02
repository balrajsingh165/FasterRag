from pathlib import Path

import pytest

from fasterrag.cli.main import main
from fasterrag.cli.output import ExitCode
from fasterrag.config.loader import load_settings
from fasterrag.config.template import canonical_config_text


def run(arguments: list[str]) -> int:
    return main(arguments)


def test_init_writes_a_config_where_none_exists(tmp_path: Path) -> None:
    destination = tmp_path / "config.yaml"

    code = run(["config", "init", "--path", str(destination)])

    assert code == ExitCode.SUCCESS
    assert destination.is_file()


def test_what_it_writes_is_the_canonical_config_byte_for_byte(tmp_path: Path) -> None:
    """A template that has drifted from the documented config teaches keys that moved."""
    destination = tmp_path / "config.yaml"

    run(["config", "init", "--path", str(destination)])

    assert destination.read_text(encoding="utf-8") == canonical_config_text()


def test_the_packaged_template_matches_the_repository_config() -> None:
    """One file, force-included into the wheel — a second copy could drift silently."""
    repository = Path(__file__).resolve().parents[3] / "config.yaml"

    assert canonical_config_text() == repository.read_text(encoding="utf-8")


def test_the_written_config_actually_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A template that does not validate would send a new user straight into an error."""
    for name in ("OPENAI_API_KEY", "QDRANT_API_KEY", "FASTERRAG_API_KEY"):
        monkeypatch.setenv(name, "test-value")
    destination = tmp_path / "config.yaml"
    run(["config", "init", "--path", str(destination)])

    settings = load_settings(destination, env_file=None)

    assert settings.vector_db.provider == "qdrant"


def test_an_existing_config_is_never_overwritten_by_accident(tmp_path: Path) -> None:
    """A config.yaml is hand-edited the moment it exists; replacing one discards that work."""
    destination = tmp_path / "config.yaml"
    destination.write_text("app:\n  port: 9999\n", encoding="utf-8")

    code = run(["config", "init", "--path", str(destination)])

    assert code == ExitCode.USAGE
    assert destination.read_text(encoding="utf-8") == "app:\n  port: 9999\n"


def test_force_overwrites_deliberately(tmp_path: Path) -> None:
    destination = tmp_path / "config.yaml"
    destination.write_text("app:\n  port: 9999\n", encoding="utf-8")

    code = run(["config", "init", "--path", str(destination), "--force"])

    assert code == ExitCode.SUCCESS
    assert destination.read_text(encoding="utf-8") == canonical_config_text()


def test_a_missing_parent_directory_is_created(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "deeper" / "config.yaml"

    code = run(["config", "init", "--path", str(destination)])

    assert code == ExitCode.SUCCESS
    assert destination.is_file()


def test_the_missing_config_error_names_the_command_not_a_repository(tmp_path: Path) -> None:
    """This is the first error a `pip install` user sees; a checkout is not a fix for them."""
    from fasterrag.errors import ConfigError

    with pytest.raises(ConfigError) as caught:
        load_settings(tmp_path / "absent.yaml", env_file=None)

    assert "fasterrag config init" in caught.value.detail
    assert "repository" not in caught.value.detail


def test_init_also_writes_the_secrets_template(tmp_path: Path) -> None:
    """Without it the very next error names three variables and no file to put them in."""
    run(["config", "init", "--path", str(tmp_path / "config.yaml")])

    assert (tmp_path / ".env.example").is_file()


def test_the_secrets_template_is_never_written_as_dot_env(tmp_path: Path) -> None:
    """The loader reads .env; writing placeholders there could overwrite real credentials."""
    run(["config", "init", "--path", str(tmp_path / "config.yaml")])

    assert not (tmp_path / ".env").exists()


def test_an_existing_secrets_template_is_left_alone(tmp_path: Path) -> None:
    example = tmp_path / ".env.example"
    example.write_text("MY_KEY=mine\n", encoding="utf-8")

    run(["config", "init", "--path", str(tmp_path / "config.yaml")])

    assert example.read_text(encoding="utf-8") == "MY_KEY=mine\n"


def test_the_shipped_secrets_template_carries_no_real_value(tmp_path: Path) -> None:
    """It ships inside the wheel, so anything resembling a credential would ship with it."""
    run(["config", "init", "--path", str(tmp_path / "config.yaml")])

    for line in (tmp_path / ".env.example").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        _, _, value = stripped.partition("=")
        assert value in ("", "change-me"), stripped


def test_every_variable_the_canonical_config_references_is_in_the_template(
    tmp_path: Path,
) -> None:
    """A referenced variable missing from the template is a startup failure with no fix."""
    run(["config", "init", "--path", str(tmp_path / "config.yaml")])
    template = (tmp_path / ".env.example").read_text(encoding="utf-8")

    for name in ("OPENAI_API_KEY", "QDRANT_API_KEY", "FASTERRAG_API_KEY"):
        assert f"{name}=" in template, name
