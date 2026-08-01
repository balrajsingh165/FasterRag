import json
from pathlib import Path

import pytest

from fasterrag.cli.main import main
from fasterrag.cli.output import Console, ExitCode
from fasterrag.cli.parser import PENDING_COMMANDS, build_parser

VALID_CONFIG = """
app:
  host: 127.0.0.1
  port: 8000
vector_db:
  provider: qdrant
  mode: external
  api_key_env: null
embeddings:
  provider: huggingface
llm:
  provider: ollama
  api_key_env: null
"""


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("FASTERRAG_API_KEY", "test-key")
    path = tmp_path / "config.yaml"
    path.write_text(VALID_CONFIG, encoding="utf-8")
    return str(path)


def run(argv: list[str]) -> int:
    return main(argv)


def test_the_documented_exit_codes_are_the_ones_used() -> None:
    assert [code.value for code in ExitCode] == [0, 1, 2, 3, 4, 5]
    assert ExitCode.SUCCESS.value == 0
    assert ExitCode.USAGE.value == 2
    assert ExitCode.UNREACHABLE.value == 3
    assert ExitCode.PREFLIGHT.value == 4
    assert ExitCode.REGRESSION.value == 5


def test_every_documented_command_is_registered() -> None:
    parser = build_parser()
    actions = [
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    ]
    commands = set(actions[0].choices or [])

    assert {
        "serve",
        "worker",
        "ingest",
        "query",
        "index",
        "provision",
        "status",
        "doctor",
        "estimate",
        "config",
    } <= commands


def test_the_pending_commands_are_registered_so_help_lists_them() -> None:
    parser = build_parser()
    actions = [
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    ]

    assert set(PENDING_COMMANDS) <= set(actions[0].choices or [])


def test_a_global_flag_is_accepted_before_the_command() -> None:
    args = build_parser().parse_args(["--json", "doctor"])

    assert args.as_json is True


def test_a_global_flag_is_accepted_after_the_command() -> None:
    args = build_parser().parse_args(["doctor", "--json"])

    assert args.as_json is True


def test_no_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args([])

    assert exit_info.value.code == 2


def test_an_unknown_command_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exit_info:
        build_parser().parse_args(["nonsense"])

    assert exit_info.value.code == 2


def test_a_pending_command_explains_which_slice_ships_it(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = run(["export"])

    assert code == ExitCode.USAGE
    assert "TASK-0079" in capsys.readouterr().err


def test_valid_config_validates(config: str, capsys: pytest.CaptureFixture[str]) -> None:
    code = run(["config", "--config", config, "validate"])

    assert code == ExitCode.SUCCESS
    assert "is valid" in capsys.readouterr().out


def test_valid_config_reports_the_selected_providers_as_json(
    config: str, capsys: pytest.CaptureFixture[str]
) -> None:
    run(["config", "--config", config, "validate", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["valid"] is True
    assert payload["vector_db"] == "qdrant"
    assert "collection" in payload


def test_an_invalid_config_exits_with_the_usage_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("retrieval:\n  top_k: 9999\n", encoding="utf-8")

    code = run(["config", "--config", str(path), "validate"])

    assert code == ExitCode.USAGE
    assert "error:" in capsys.readouterr().err


def test_a_missing_config_exits_with_the_usage_code(tmp_path: Path) -> None:
    assert run(["config", "--config", str(tmp_path / "absent.yaml"), "validate"]) == ExitCode.USAGE


def test_an_invalid_config_still_produces_a_json_document(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("retrieval:\n  top_k: 9999\n", encoding="utf-8")

    run(["config", "--config", str(path), "validate", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["valid"] is False
    assert payload["detail"]


def test_provision_refuses_to_report_and_change_at_once(
    config: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(["provision", "--config", config, "qdrant", "--status", "--down"])

    assert code == ExitCode.USAGE
    assert "cannot be combined" in capsys.readouterr().err


def test_estimate_reports_an_empty_corpus_rather_than_failing(
    config: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    code = run(["estimate", "--config", config, str(empty), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert code == ExitCode.SUCCESS
    assert payload["documents"] == 0


def test_estimate_counts_a_real_document(
    config: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "note.txt").write_text("Either party may terminate on thirty days notice.")

    run(["estimate", "--config", config, str(tmp_path / "note.txt"), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["documents"] == 1
    assert payload["chunks"] >= 1
    assert payload["tokens"] > 0


def test_estimate_never_reports_a_price_it_does_not_know(
    config: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "note.txt").write_text("text")

    run(["estimate", "--config", config, str(tmp_path / "note.txt"), "--json"])
    payload = json.loads(capsys.readouterr().out)

    for provider in payload["providers"]:
        assert provider["cost_usd"] is None or provider["basis"]


def test_json_mode_writes_exactly_one_document(
    config: str, capsys: pytest.CaptureFixture[str]
) -> None:
    run(["config", "--config", config, "validate", "--json"])

    json.loads(capsys.readouterr().out)


def test_json_mode_suppresses_human_prose(config: str, capsys: pytest.CaptureFixture[str]) -> None:
    run(["config", "--config", config, "validate", "--json"])

    assert "is valid" not in capsys.readouterr().out


def test_quiet_mode_suppresses_success_output(
    config: str, capsys: pytest.CaptureFixture[str]
) -> None:
    run(["config", "--config", config, "validate", "--quiet"])

    assert capsys.readouterr().out == ""


def test_quiet_mode_still_reports_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("retrieval:\n  top_k: 9999\n", encoding="utf-8")

    run(["config", "--config", str(path), "validate", "--quiet"])

    assert "error:" in capsys.readouterr().err


def test_the_console_writes_nothing_to_stdout_in_json_mode(
    capsys: pytest.CaptureFixture[str],
) -> None:
    console = Console(as_json=True)
    console.emit("human prose")

    assert capsys.readouterr().out == ""


def test_the_console_writes_a_fix_when_one_is_known(capsys: pytest.CaptureFixture[str]) -> None:
    Console().problem("DOCKER_UNAVAILABLE", "docker is not running", "start Docker Desktop")

    captured = capsys.readouterr().err
    assert "DOCKER_UNAVAILABLE" in captured
    assert "fix: start Docker Desktop" in captured


def test_the_console_omits_the_fix_line_when_there_is_none(
    capsys: pytest.CaptureFixture[str],
) -> None:
    Console().problem("INTERNAL", "something went wrong")

    assert "fix:" not in capsys.readouterr().err


def test_ingest_requires_at_least_one_source() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ingest"])


def test_query_requires_a_question() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["query"])


def test_repeatable_filters_accumulate() -> None:
    args = build_parser().parse_args(["query", "q", "--filter", "a=1", "--filter", "b=2"])

    assert args.filter == ["a=1", "b=2"]


def test_repeatable_metadata_accumulates() -> None:
    args = build_parser().parse_args(["ingest", "a.txt", "--metadata", "x=1", "--metadata", "y=2"])

    assert args.metadata == ["x=1", "y=2"]


def test_index_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["index"])


def test_config_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["config"])
