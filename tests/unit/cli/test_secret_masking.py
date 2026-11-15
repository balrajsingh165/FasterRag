"""The CLI masks every credential its configuration names, not only credential-named ones.

TASK-0541: the CLI installs its log handler before it loads configuration, so until the
handler learned the configuration's ``*_env`` names it masked by the name rule alone.
"""

import logging
from pathlib import Path

import pytest

from fasterrag.cli.main import main
from fasterrag.observability.logging import REDACTED
from tests.unit.cli.conftest import write_config

CUSTOM_VALUE = "custom-credential-0123456789"


def test_a_credential_under_a_custom_name_is_masked_once_the_cli_loads_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI masked only `_KEY`-style names because logging was set up before config."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "test-key")
    monkeypatch.setenv("LLM_CREDENTIAL_CUSTOM", CUSTOM_VALUE)
    config = write_config(tmp_path)
    path = Path(config)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "llm:\n  provider: ollama\n  api_key_env: null",
            "llm:\n  provider: ollama\n  api_key_env: LLM_CREDENTIAL_CUSTOM",
        ),
        encoding="utf-8",
    )
    assert "LLM_CREDENTIAL_CUSTOM" in path.read_text(encoding="utf-8")

    main(["config", "show", "--config", config])
    capsys.readouterr()
    logging.getLogger("fasterrag.test").warning("leaking %s", CUSTOM_VALUE)

    written = capsys.readouterr().err
    assert f"leaking {REDACTED}" in written
    assert CUSTOM_VALUE not in written
