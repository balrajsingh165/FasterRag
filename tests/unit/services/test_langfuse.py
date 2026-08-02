import re
from pathlib import Path

from fasterrag.services.langfuse import (
    GENERATED_SECRETS,
    PRESERVED_SECRETS,
    LangfusePlan,
    compose_arguments,
    compose_file,
    ensure_secrets,
    explain_failure,
    read_env,
    write_env,
)


def plan(tmp_path: Path) -> LangfusePlan:
    return LangfusePlan(root=tmp_path, env_file=tmp_path / ".env")


def test_missing_secrets_are_generated(tmp_path: Path) -> None:
    values, created = ensure_secrets(tmp_path / ".env")

    assert set(created) == set(GENERATED_SECRETS)
    assert all(values[name] for name in GENERATED_SECRETS)


def test_generation_only_reports_names_never_values(tmp_path: Path) -> None:
    """A secret that reaches a log is a secret that reaches a log aggregator."""
    _, created = ensure_secrets(tmp_path / ".env")

    assert created == sorted(GENERATED_SECRETS)


def test_existing_secrets_are_never_regenerated(tmp_path: Path) -> None:
    """Rotating SALT or ENCRYPTION_KEY invalidates every password and API key issued."""
    env = tmp_path / ".env"
    first, _ = ensure_secrets(env)

    second, created = ensure_secrets(env)

    assert created == []
    for name in PRESERVED_SECRETS:
        assert second[name] == first[name]


def test_re_running_does_not_duplicate_variables(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    ensure_secrets(env)
    ensure_secrets(env)

    body = env.read_text(encoding="utf-8")

    for name in GENERATED_SECRETS:
        assert body.count(f"{name}=") == 1


def test_hand_written_variables_survive_generation(tmp_path: Path) -> None:
    """Rewriting the file rather than appending would silently drop an operator's own keys."""
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=set-by-hand\n", encoding="utf-8")

    ensure_secrets(env)

    assert "OPENAI_API_KEY=set-by-hand" in env.read_text(encoding="utf-8")


def test_an_operator_supplied_secret_is_honoured(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("LANGFUSE_SALT=chosen-by-the-operator\n", encoding="utf-8")

    values, created = ensure_secrets(env)

    assert values["LANGFUSE_SALT"] == "chosen-by-the-operator"
    assert "LANGFUSE_SALT" not in created


def test_no_provider_key_is_handed_to_the_compose_subprocess(tmp_path: Path) -> None:
    """`.env` also holds LLM credentials the Langfuse stack has no business seeing."""
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")

    values, _ = ensure_secrets(env)

    assert set(values) == set(GENERATED_SECRETS)


def test_the_encryption_key_is_256_bits_of_hex(tmp_path: Path) -> None:
    """Langfuse rejects anything else, and does so only after the stack is already up."""
    values, _ = ensure_secrets(tmp_path / ".env")

    assert re.fullmatch(r"[0-9a-f]{64}", values["LANGFUSE_ENCRYPTION_KEY"])


def test_write_env_reports_nothing_when_nothing_is_new(tmp_path: Path) -> None:
    env = tmp_path / ".env"

    assert write_env(env, {"A": "1"}, {"A": "1"}) == []
    assert not env.exists()


def test_reading_tolerates_comments_blanks_and_a_bom(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("﻿# a comment\n\nA=1\nnot-a-pair\n", encoding="utf-8")

    assert read_env(env) == {"A": "1"}


def test_the_compose_file_carries_no_secret_value(tmp_path: Path) -> None:
    values, _ = ensure_secrets(tmp_path / ".env")

    body = compose_file(plan(tmp_path))

    for value in values.values():
        assert value not in body


def test_the_worker_shares_salt_and_encryption_key_with_the_web_container(tmp_path: Path) -> None:
    """Two halves disagreeing about the key produces rows the other cannot read."""
    body = compose_file(plan(tmp_path))

    assert body.count("SALT: ${LANGFUSE_SALT}") == 2
    assert body.count("ENCRYPTION_KEY: ${LANGFUSE_ENCRYPTION_KEY}") == 2


def test_no_bootstrap_value_is_quoted(tmp_path: Path) -> None:
    """Compose takes a quoted value literally, so the bootstrap silently misbehaves."""
    for line in compose_file(plan(tmp_path)).splitlines():
        stripped = line.strip()
        if not stripped.startswith("LANGFUSE_INIT_"):
            continue
        _, _, value = stripped.partition(":")
        assert '"' not in value, stripped
        assert "'" not in value, stripped


def test_the_bootstrap_declares_the_org_that_the_rest_depends_on(tmp_path: Path) -> None:
    """Without ORG_*, the PROJECT_* and USER_* variables are ignored entirely."""
    body = compose_file(plan(tmp_path))

    for name in (
        "LANGFUSE_INIT_ORG_ID",
        "LANGFUSE_INIT_ORG_NAME",
        "LANGFUSE_INIT_PROJECT_ID",
        "LANGFUSE_INIT_PROJECT_PUBLIC_KEY",
        "LANGFUSE_INIT_PROJECT_SECRET_KEY",
        "LANGFUSE_INIT_USER_EMAIL",
        "LANGFUSE_INIT_USER_PASSWORD",
    ):
        assert f"{name}:" in body


def test_the_stack_declares_every_documented_service(tmp_path: Path) -> None:
    body = compose_file(plan(tmp_path))

    for service in (
        "langfuse-web",
        "langfuse-worker",
        "langfuse-postgres",
        "langfuse-clickhouse",
        "langfuse-redis",
        "langfuse-minio",
    ):
        assert f"  {service}:" in body


def test_generating_the_compose_file_is_idempotent(tmp_path: Path) -> None:
    assert compose_file(plan(tmp_path)) == compose_file(plan(tmp_path))


def test_the_plan_reports_the_documented_url(tmp_path: Path) -> None:
    assert plan(tmp_path).url == "http://localhost:3000"


def test_compose_is_always_told_where_the_env_file_is(tmp_path: Path) -> None:
    """Compose looks beside the compose file; the secrets live at the project root instead.

    Omitting the flag interpolates every secret to the empty string, and the stack starts
    with blank passwords — which surfaces as a Postgres authentication error, not as a
    missing-variable one.
    """
    arguments = compose_arguments(plan(tmp_path), "up", "-d")

    assert "--env-file" in arguments
    assert arguments[arguments.index("--env-file") + 1] == str(tmp_path / ".env")


def test_the_generated_file_records_the_command_that_runs_it(tmp_path: Path) -> None:
    """An operator running the file by hand without --env-file gets the blank-password stack."""
    assert "--env-file" in compose_file(plan(tmp_path))


def test_a_failure_report_keeps_the_error_not_the_pull_progress() -> None:
    """Compose streams layer progress to stderr, so the head of a failed run is always noise."""
    stderr = "Pulling fs layer\n" * 200 + "Error: the thing that actually went wrong"

    assert "the thing that actually went wrong" in explain_failure(stderr)


def test_a_taken_port_is_reported_as_a_taken_port() -> None:
    stderr = "Bind for 0.0.0.0:9090 failed: port is already allocated"

    explained = explain_failure(stderr)

    assert "already taken" in explained
    assert "9090" in explained
