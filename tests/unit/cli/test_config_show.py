from pathlib import Path

import pytest

from fasterrag.cli.main import main
from fasterrag.cli.output import ExitCode
from fasterrag.config.template import canonical_config_text


def write_config(tmp_path: Path, old: str = "", new: str = "") -> Path:
    body = canonical_config_text()
    if old:
        body = body.replace(old, new)
    destination = tmp_path / "config.yaml"
    destination.write_text(body, encoding="utf-8")
    return destination


def test_it_lists_settings_with_their_defaults(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = write_config(tmp_path)

    code = main(["config", "show", "--config", str(config)])

    out = capsys.readouterr().out
    assert code == ExitCode.SUCCESS
    assert "chunking.chunk_size" in out
    assert "embeddings.model" in out


def test_a_nested_section_is_walked_rather_than_printed_whole(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A section printed as one object hides the fields an operator came to find."""
    config = write_config(tmp_path)

    main(["config", "show", "--config", str(config)])

    out = capsys.readouterr().out
    assert "vector_db.collection.default_name" in out
    assert "reliability.retries.max_attempts" in out


def test_changed_lists_only_overrides(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = write_config(tmp_path, "chunk_size: 768", "chunk_size: 512")

    main(["config", "show", "--changed", "--config", str(config)])

    out = capsys.readouterr().out
    assert "chunking.chunk_size" in out
    assert "chunking.overlap" not in out


def test_the_shipped_config_reports_no_overrides(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The template and the schema defaults are two copies of one set of values."""
    config = write_config(tmp_path)

    main(["config", "show", "--changed", "--config", str(config)])

    assert "matches every default" in capsys.readouterr().out


def test_missing_environment_variables_do_not_stop_the_listing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """This command is most useful on the installation `config validate` refuses."""
    for name in ("OPENAI_API_KEY", "QDRANT_API_KEY", "FASTERRAG_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    config = write_config(tmp_path)

    code = main(["config", "show", "--config", str(config), "--json"])

    assert code == ExitCode.SUCCESS
    assert "chunking.chunk_size" in capsys.readouterr().out


def test_invalid_configuration_still_fails(tmp_path: Path) -> None:
    """Showing settings that did not validate would print values nothing will use."""
    config = tmp_path / "config.yaml"
    config.write_text("chunking:\n  chunk_size: 9999999\n", encoding="utf-8")

    assert main(["config", "show", "--config", str(config)]) == ExitCode.USAGE


def test_json_output_is_machine_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    config = write_config(tmp_path)

    main(["config", "show", "--config", str(config), "--json"])

    document = json.loads(capsys.readouterr().out)
    names = [row["setting"] for row in document["settings"]]
    assert "chunking.token_counter" in names
    assert "embeddings.dimensions" in names
