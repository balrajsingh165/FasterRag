"""Proves the live-provider gating logic, without contacting any provider.

``test_live_providers.py`` is credential-gated end to end, so on a machine with no keys every
one of its probes skips and the file reports nothing but green. That is trap T12 — a check
reporting PASS on a probe that never ran — and the only defence against it is evidence that
the gate itself is correct, produced by a test that does not depend on a credential.

So this module carries **no** ``eval`` marker. CI runs it under
``pytest -m "not integration and not eval"``, which is the job that must pass on every
commit, while ``pytest -m eval`` is the job that may legitimately skip everything. The gate
is therefore verified by a run that cannot skip, and the probes are verified only by a run
that reaches a provider.

Nothing here reads a real credential: every case sets its own environment variable to a
value this module chose, so a failure message can never carry a secret.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError

import pytest

from fasterrag.config.schema import Settings
from tests.eval.test_live_providers import (
    ALL_CHECKS,
    EMBEDDING_CHECKS,
    LIVE_TESTS_VAR,
    LLM_CHECKS,
    LiveCheck,
    LiveLedger,
    _credential,
    _endpoint,
    _gate,
    _model,
    _require_opt_in,
    embedding_settings,
    llm_settings,
)

OPENAI = LLM_CHECKS[0]
OLLAMA = next(check for check in LLM_CHECKS if check.provider == "ollama")
GATEWAY = next(check for check in LLM_CHECKS if check.provider == "openai_compatible")

SENTINEL = "sk-not-a-real-key-this-value-is-invented-by-the-test"


@pytest.fixture
def ledger(monkeypatch: pytest.MonkeyPatch) -> LiveLedger:
    """Swap in a throwaway ledger so these cases never pollute the live run's report."""
    replacement = LiveLedger()
    monkeypatch.setattr("tests.eval.test_live_providers.LEDGER", replacement)
    return replacement


@pytest.mark.parametrize("absent", ["", "   ", "\n\t "])
def test_an_absent_or_blank_credential_skips_and_names_the_variable(
    monkeypatch: pytest.MonkeyPatch, ledger: LiveLedger, absent: str
) -> None:
    monkeypatch.setenv(OPENAI.env_var, absent)

    with pytest.raises(pytest.skip.Exception) as skipped:
        _credential(OPENAI)

    assert str(skipped.value) == f"{OPENAI.env_var} is not set, so {OPENAI.name} was not contacted"
    assert ledger.unverified[OPENAI.name] == str(skipped.value)
    assert ledger.verified() == ()


def test_an_unset_credential_skips_exactly_as_a_blank_one_does(
    monkeypatch: pytest.MonkeyPatch, ledger: LiveLedger
) -> None:
    monkeypatch.delenv(OPENAI.env_var, raising=False)

    with pytest.raises(pytest.skip.Exception) as skipped:
        _credential(OPENAI)

    assert OPENAI.env_var in str(skipped.value)
    assert OPENAI.name in ledger.unverified


def test_a_present_credential_runs_the_probe_and_is_recorded_nowhere(
    monkeypatch: pytest.MonkeyPatch, ledger: LiveLedger
) -> None:
    monkeypatch.setenv(OPENAI.env_var, f"  {SENTINEL}  ")

    assert _credential(OPENAI) == SENTINEL
    assert ledger.unverified == {}
    assert ledger.verified() == ()


def test_the_skip_path_never_puts_a_credential_in_the_message(
    monkeypatch: pytest.MonkeyPatch, ledger: LiveLedger
) -> None:
    """A blank-looking key is still a key; neither the skip nor the ledger may echo one."""
    monkeypatch.setenv(GATEWAY.env_var, SENTINEL)
    monkeypatch.delenv(GATEWAY.base_url_env or "", raising=False)

    with pytest.raises(pytest.skip.Exception) as skipped:
        _endpoint(GATEWAY)

    assert SENTINEL not in str(skipped.value)
    assert SENTINEL not in "".join(ledger.unverified.values())


def test_a_hosted_provider_needs_no_endpoint(ledger: LiveLedger) -> None:
    assert _endpoint(OPENAI) is None
    assert ledger.unverified == {}


@pytest.mark.parametrize("scheme", ["localhost:11434", "ftp://host", "/var/run/ollama.sock"])
def test_an_endpoint_without_an_http_scheme_skips_rather_than_failing(
    monkeypatch: pytest.MonkeyPatch, ledger: LiveLedger, scheme: str
) -> None:
    monkeypatch.setenv(OLLAMA.base_url_env or "", scheme)

    with pytest.raises(pytest.skip.Exception) as skipped:
        _endpoint(OLLAMA)

    assert "http:// or https:// scheme" in str(skipped.value)
    assert OLLAMA.name in ledger.unverified


@pytest.mark.parametrize("url", ["http://localhost:11434", "https://gw.example.invalid/v1"])
def test_an_absolute_endpoint_is_returned_unchanged(
    monkeypatch: pytest.MonkeyPatch, ledger: LiveLedger, url: str
) -> None:
    monkeypatch.setenv(OLLAMA.base_url_env or "", url)

    assert _endpoint(OLLAMA) == url
    assert ledger.unverified == {}


def test_a_gateway_with_no_model_id_skips_instead_of_guessing_one(
    monkeypatch: pytest.MonkeyPatch, ledger: LiveLedger
) -> None:
    monkeypatch.delenv(GATEWAY.model_env, raising=False)

    with pytest.raises(pytest.skip.Exception) as skipped:
        _model(GATEWAY)

    assert GATEWAY.model_env in str(skipped.value)
    assert GATEWAY.name in ledger.unverified


def test_an_override_replaces_a_providers_default_model(
    monkeypatch: pytest.MonkeyPatch, ledger: LiveLedger
) -> None:
    monkeypatch.setenv(OPENAI.model_env, "  gpt-4.1-nano  ")

    assert _model(OPENAI) == "gpt-4.1-nano"
    assert ledger.unverified == {}


def test_a_default_model_is_used_when_no_override_is_set(
    monkeypatch: pytest.MonkeyPatch, ledger: LiveLedger
) -> None:
    monkeypatch.delenv(OPENAI.model_env, raising=False)

    assert _model(OPENAI) == OPENAI.default_model
    assert ledger.unverified == {}


def test_the_gate_reads_the_variable_it_was_handed(
    monkeypatch: pytest.MonkeyPatch, ledger: LiveLedger
) -> None:
    """``_gate`` must not quietly fall back to the check's credential variable."""
    monkeypatch.setenv(OPENAI.env_var, SENTINEL)
    monkeypatch.delenv("FASTERRAG_LIVE_GATE_PROBE", raising=False)

    with pytest.raises(pytest.skip.Exception) as skipped:
        _gate(OPENAI, "FASTERRAG_LIVE_GATE_PROBE")

    assert "FASTERRAG_LIVE_GATE_PROBE" in str(skipped.value)
    assert OPENAI.env_var not in str(skipped.value)


def test_every_advertised_provider_has_a_check_with_a_distinct_gate() -> None:
    """A provider added to a factory without a live check would otherwise go unnoticed."""
    from fasterrag.adapters.embeddings.factory import (
        available_providers as embedding_providers,
    )
    from fasterrag.adapters.llm.factory import available_providers as llm_providers

    covered_llm = {check.provider for check in LLM_CHECKS}
    covered_embeddings = {check.provider for check in EMBEDDING_CHECKS}

    assert covered_llm == set(llm_providers())
    assert set(embedding_providers()) - covered_embeddings == {"huggingface"}
    assert len({check.name for check in ALL_CHECKS}) == len(ALL_CHECKS)


def test_every_check_names_a_gate_rather_than_carrying_a_value() -> None:
    """A gate is an environment variable name; a committed literal would be a leaked key."""
    for check in ALL_CHECKS:
        assert check.env_var == check.env_var.upper()
        assert check.env_var.replace("_", "").isalnum()
        if check.base_url_env is not None:
            assert check.base_url_env == check.base_url_env.upper()
        assert not check.default_model.startswith(("sk-", "key-"))


def test_a_keyless_provider_is_exactly_the_one_the_schema_allows_to_omit_a_key() -> None:
    from fasterrag.config.schema import PROVIDERS_NOT_REQUIRING_LLM_KEY

    keyless = {check.provider for check in LLM_CHECKS if check.keyless}

    assert keyless == set(PROVIDERS_NOT_REQUIRING_LLM_KEY)


def test_the_ledger_refuses_to_record_a_provider_both_ways() -> None:
    led = LiveLedger()
    led.record_verified(OPENAI.name, "answered")

    with pytest.raises(ValueError, match="VERIFIED"):
        led.record_unverified(OPENAI.name, "OPENAI_API_KEY is not set")
    with pytest.raises(ValueError, match="without a reason"):
        led.record_unverified(OLLAMA.name, " ")

    assert led.verified() == (OPENAI.name,)


def test_an_empty_run_reports_nothing_reached() -> None:
    report = LiveLedger().report(ALL_CHECKS)

    assert f"0 of {len(ALL_CHECKS)} providers reached" in report
    assert "VERIFIED" not in report
    for check in ALL_CHECKS:
        assert f"{check.name}: NOT RUN" in report


@pytest.mark.parametrize("absent", ["", "   ", "0", "false", "no", "off", "maybe"])
def test_a_credential_alone_does_not_authorise_a_billed_call(
    monkeypatch: pytest.MonkeyPatch, ledger: LiveLedger, absent: str
) -> None:
    """TASK-0341: having a key exported is not the same as asking to spend money."""
    monkeypatch.setenv(OPENAI.env_var, SENTINEL)
    monkeypatch.setenv(LIVE_TESTS_VAR, absent)

    with pytest.raises(pytest.skip.Exception) as skipped:
        _require_opt_in(OPENAI)

    assert LIVE_TESTS_VAR in str(skipped.value)
    assert f"{LIVE_TESTS_VAR}=1" in str(skipped.value)
    assert SENTINEL not in str(skipped.value)
    assert ledger.unverified[OPENAI.name] == str(skipped.value)


def test_an_unset_opt_in_skips_exactly_as_a_falsey_one_does(
    monkeypatch: pytest.MonkeyPatch, ledger: LiveLedger
) -> None:
    monkeypatch.delenv(LIVE_TESTS_VAR, raising=False)

    with pytest.raises(pytest.skip.Exception):
        _require_opt_in(OPENAI)

    assert OPENAI.name in ledger.unverified


@pytest.mark.parametrize("present", ["1", "true", "TRUE", " yes ", "on"])
def test_an_explicit_opt_in_lets_the_probe_through(
    monkeypatch: pytest.MonkeyPatch, ledger: LiveLedger, present: str
) -> None:
    monkeypatch.setenv(LIVE_TESTS_VAR, present)

    _require_opt_in(OPENAI)

    assert ledger.unverified == {}
    assert ledger.verified() == ()


@pytest.mark.parametrize("build", [llm_settings, embedding_settings])
def test_no_probe_can_reach_a_provider_without_the_opt_in(
    monkeypatch: pytest.MonkeyPatch,
    ledger: LiveLedger,
    build: Callable[[LiveCheck], Settings],
) -> None:
    """The opt-in is checked before the credential, the SDK and the endpoint, not after."""
    monkeypatch.setenv(OPENAI.env_var, SENTINEL)
    monkeypatch.delenv(LIVE_TESTS_VAR, raising=False)

    with pytest.raises(pytest.skip.Exception) as skipped:
        build(OPENAI)

    assert LIVE_TESTS_VAR in str(skipped.value)


def test_a_check_cannot_be_rewritten_by_the_test_that_uses_it() -> None:
    """The checks are module-level singletons shared across cases, so they must be frozen."""
    with pytest.raises(FrozenInstanceError):
        # CRITICAL: mypy is right that this assignment is illegal, and that is the assertion.
        # The checks are module-level singletons shared by every case, so one case mutating a
        # check would silently change what later cases test. This proves the dataclass refuses
        # at runtime, not only at type-check time, so the ignore cannot be dropped.
        OPENAI.env_var = "SOMETHING_ELSE"  # type: ignore[misc]

    assert OPENAI.env_var == "OPENAI_API_KEY"
