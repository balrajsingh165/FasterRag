"""Credential-gated checks that each advertised provider really accepts our calls.

Every other provider test in this suite substitutes a fake client, which proves the adapter
issues the request it intends to and proves nothing about whether the vendor accepts it.
``docs/integrations.md`` and the README advertise OpenAI, Anthropic, Cohere and Ollama; a
provider list is a promise, so each entry gets the smallest call that can keep it — one tiny
generation, or one tiny embedding — asserting a well-formed typed response and the token
accounting the adapter claims to populate.

Cost of one full run with every credential present: **seven requests**, each of a few dozen
tokens — four generations (OpenAI, Anthropic, Cohere, Ollama) and three embeddings (OpenAI,
Cohere, Ollama). The two Ollama checks go to a local server and cost nothing.

What a green run does and does not establish:

* A provider is verified **only** when its own check ran and recorded evidence of a
  response. An absent credential skips that provider and nothing else; the suite never
  fails for a missing key, so the maintainer can run with one key or with none.
* A run with no credentials present proves nothing about any provider. That misreading —
  a skipped probe counted as a passed one — is trap T12, and it is the shape of the failure
  that hit the Redis suite on 2026-10-26. Two things make it structurally hard here: every
  provider has its own test, so pytest's own SKIPPED line is the record and no aggregate can
  launder it; and the summary test at the bottom is green only when *every* provider was
  reached, skipping with the per-provider report otherwise.

Run with ``pytest -m eval tests/eval/test_live_providers.py -rs`` — ``-rs`` prints the skip
reasons, which is where the report lives.

Ollama has no credential, so its gate is an explicit opt-in: ``OLLAMA_HOST`` pointing at a
running server. Gating it on nothing would mean a connection attempt on every run and a
failure, not a skip, on every machine without Ollama. The HuggingFace embedder is local and
already exercised against real weights in ``test_real_models.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Literal

import pytest

from fasterrag.adapters.embeddings.base import EmbeddingResult
from fasterrag.adapters.embeddings.factory import create_embedding_adapter
from fasterrag.adapters.llm.base import Completion
from fasterrag.adapters.llm.factory import create_llm_adapter
from fasterrag.config.schema import Settings

pytestmark = pytest.mark.eval

PROMPT = "Reply with the single word: ok"
SYSTEM = "You are a connectivity probe. Answer in one word."
PASSAGE = "fasterRag live provider check."
MAX_TOKENS = 16

VERIFIED = "VERIFIED"
NOT_CHECKED = "NOT CHECKED"
NOT_RUN = "NOT RUN"


@dataclass(frozen=True, slots=True)
class LiveCheck:
    """One advertised provider and what it takes to reach it."""

    kind: Literal["llm", "embeddings"]
    provider: str
    env_var: str
    package: str
    extra: str
    default_model: str
    reports_token_usage: bool

    @property
    def name(self) -> str:
        """Return the identifier this provider is recorded and reported under."""
        return f"{self.provider} {self.kind}"

    @property
    def model(self) -> str:
        """Return the model to probe, overridable so a retired id can be replaced."""
        override = f"FASTERRAG_LIVE_MODEL_{self.provider.upper()}_{self.kind.upper()}"
        configured = os.environ.get(override, "").strip()
        return configured or self.default_model


LLM_CHECKS: tuple[LiveCheck, ...] = (
    LiveCheck(
        kind="llm",
        provider="openai",
        env_var="OPENAI_API_KEY",
        package="openai",
        extra="openai",
        default_model="gpt-4o-mini",
        reports_token_usage=True,
    ),
    LiveCheck(
        kind="llm",
        provider="anthropic",
        env_var="ANTHROPIC_API_KEY",
        package="anthropic",
        extra="anthropic",
        default_model="claude-3-5-haiku-latest",
        reports_token_usage=True,
    ),
    LiveCheck(
        kind="llm",
        provider="cohere",
        env_var="COHERE_API_KEY",
        package="cohere",
        extra="cohere",
        default_model="command-r",
        reports_token_usage=True,
    ),
    LiveCheck(
        kind="llm",
        provider="ollama",
        env_var="OLLAMA_HOST",
        package="ollama",
        extra="ollama",
        default_model="llama3.2",
        reports_token_usage=True,
    ),
)

EMBEDDING_CHECKS: tuple[LiveCheck, ...] = (
    LiveCheck(
        kind="embeddings",
        provider="openai",
        env_var="OPENAI_API_KEY",
        package="openai",
        extra="openai",
        default_model="text-embedding-3-small",
        reports_token_usage=True,
    ),
    # TODO: cohere and ollama embeddings report no token usage because neither adapter
    # surfaces it — Cohere returns billed units in response.meta that CohereEmbedder
    # discards. The assertion below pins the current behaviour rather than asserting an
    # accounting that does not exist; see TASK-0205 follow-up.
    LiveCheck(
        kind="embeddings",
        provider="cohere",
        env_var="COHERE_API_KEY",
        package="cohere",
        extra="cohere",
        default_model="embed-english-v3.0",
        reports_token_usage=False,
    ),
    LiveCheck(
        kind="embeddings",
        provider="ollama",
        env_var="OLLAMA_HOST",
        package="ollama",
        extra="ollama",
        default_model="nomic-embed-text",
        reports_token_usage=False,
    ),
)

ALL_CHECKS: tuple[LiveCheck, ...] = LLM_CHECKS + EMBEDDING_CHECKS


@dataclass(slots=True)
class LiveLedger:
    """What this run actually reached, and what it did not.

    Verification is evidence-keyed on purpose: ``verified`` is derived from the recorded
    responses rather than tracked alongside them, so there is no counter that can be
    incremented by a provider that was never called.
    """

    evidence: dict[str, str] = field(default_factory=dict)
    unverified: dict[str, str] = field(default_factory=dict)

    def record_verified(self, name: str, evidence: str) -> None:
        """Record that ``name`` returned a live response.

        Raises:
            ValueError: If no evidence is supplied, or the provider was already recorded
                as unverified in this run.
        """
        if not evidence.strip():
            raise ValueError(f"{name} cannot be recorded verified without evidence")
        if name in self.unverified:
            raise ValueError(f"{name} was already recorded as {NOT_CHECKED}")
        self.evidence[name] = evidence

    def record_unverified(self, name: str, reason: str) -> None:
        """Record why ``name`` was not reached.

        Raises:
            ValueError: If no reason is supplied, or the provider already returned a
                live response in this run.
        """
        if not reason.strip():
            raise ValueError(f"{name} cannot be recorded unverified without a reason")
        if name in self.evidence:
            raise ValueError(f"{name} was already recorded as {VERIFIED}")
        self.unverified[name] = reason

    def verified(self) -> tuple[str, ...]:
        """Return the providers that returned a live response, in recorded order."""
        return tuple(self.evidence)

    def report(self, checks: tuple[LiveCheck, ...]) -> str:
        """Render one line per provider, naming every one that was not reached."""
        reached = [check for check in checks if check.name in self.evidence]
        headline = f"live provider verification: {len(reached)} of {len(checks)} providers reached"
        if len(reached) < len(checks):
            headline += "; every other provider below is unverified by this run"
        lines = [headline]
        for check in checks:
            if check.name in self.evidence:
                lines.append(f"  {check.name}: {VERIFIED} - {self.evidence[check.name]}")
            elif check.name in self.unverified:
                lines.append(f"  {check.name}: {NOT_CHECKED} - {self.unverified[check.name]}")
            else:
                lines.append(f"  {check.name}: {NOT_RUN} - not collected in this run")
        return "\n".join(lines)


LEDGER = LiveLedger()


def _credential(check: LiveCheck) -> str:
    """Return the gating environment value, recording and skipping when it is absent.

    Raises:
        Skipped: If the variable is unset or blank, naming the provider and the variable.
    """
    value = os.environ.get(check.env_var, "").strip()
    if not value:
        reason = f"{check.env_var} is not set, so {check.name} was not contacted"
        LEDGER.record_unverified(check.name, reason)
        pytest.skip(reason)
    return value


def _require_package(check: LiveCheck) -> None:
    """Record and skip when the provider SDK is not installed.

    Raises:
        Skipped: If the SDK package is missing.
    """
    if find_spec(check.package) is None:
        reason = (
            f"the {check.package} package is not installed, so {check.name} was not "
            f"contacted; install it with 'pip install \"fasterrag[{check.extra}]\"'"
        )
        LEDGER.record_unverified(check.name, reason)
        pytest.skip(reason)


def _endpoint(check: LiveCheck) -> str | None:
    """Return the Ollama endpoint for a local provider, or None for a hosted one.

    Raises:
        Skipped: If the endpoint is set to something that is not an absolute URL.
    """
    if check.provider != "ollama":
        return None

    host = _credential(check)
    if not host.startswith(("http://", "https://")):
        reason = (
            f"{check.env_var} is set to a value with no http:// or https:// scheme, which "
            f"the configuration contract rejects, so {check.name} was not contacted"
        )
        LEDGER.record_unverified(check.name, reason)
        pytest.skip(reason)
    return host


def llm_settings(check: LiveCheck) -> Settings:
    """Build configuration pointing generation at the provider under check."""
    _require_package(check)
    base_url = _endpoint(check)
    if check.provider != "ollama":
        _credential(check)

    return Settings.model_validate(
        {
            "llm": {
                "provider": check.provider,
                "model": check.model,
                "api_key_env": None if check.provider == "ollama" else check.env_var,
                "base_url": base_url,
                "temperature": 0.0,
                "max_tokens": MAX_TOKENS,
            }
        }
    )


def embedding_settings(check: LiveCheck) -> Settings:
    """Build configuration pointing embedding at the provider under check.

    ``OllamaEmbedder`` reads its host from ``llm.base_url``, which is why the endpoint is
    set there rather than under ``embeddings``.
    """
    _require_package(check)
    base_url = _endpoint(check)
    if check.provider != "ollama":
        _credential(check)

    payload: dict[str, object] = {
        "embeddings": {
            "provider": check.provider,
            "model": check.model,
            "api_key_env": None if check.provider == "ollama" else check.env_var,
        }
    }
    if base_url is not None:
        payload["llm"] = {"provider": "ollama", "api_key_env": None, "base_url": base_url}
    return Settings.model_validate(payload)


@pytest.mark.parametrize("check", LLM_CHECKS, ids=lambda check: check.name)
async def test_the_provider_accepts_a_real_generation(check: LiveCheck) -> None:
    settings = llm_settings(check)
    adapter = create_llm_adapter(settings)
    try:
        completion = await adapter.complete(PROMPT, system=SYSTEM)
    finally:
        await adapter.close()

    assert isinstance(completion, Completion)
    assert completion.text.strip()
    assert completion.model == check.model
    assert completion.prompt_tokens > 0
    assert completion.completion_tokens > 0

    LEDGER.record_verified(
        check.name,
        f"{check.model} answered with {completion.prompt_tokens} prompt and "
        f"{completion.completion_tokens} completion tokens",
    )


@pytest.mark.parametrize("check", EMBEDDING_CHECKS, ids=lambda check: check.name)
async def test_the_provider_accepts_a_real_embedding(check: LiveCheck) -> None:
    settings = embedding_settings(check)
    adapter = create_embedding_adapter(settings)
    try:
        result = await adapter.embed_documents([PASSAGE])
    finally:
        await adapter.close()

    assert isinstance(result, EmbeddingResult)
    assert len(result.vectors) == 1
    assert len(result.vectors[0]) > 0
    assert all(isinstance(value, float) for value in result.vectors[0])
    assert result.model == check.model
    assert result.model_version

    if check.reports_token_usage:
        assert result.total_tokens is not None
        assert result.total_tokens > 0
    else:
        assert result.total_tokens is None

    LEDGER.record_verified(
        check.name,
        f"{check.model} returned a {len(result.vectors[0])}-dimensional vector, "
        f"total_tokens={result.total_tokens}",
    )


def test_a_missing_credential_skips_the_provider_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    check = LLM_CHECKS[0]
    ledger = LiveLedger()
    monkeypatch.setattr("tests.eval.test_live_providers.LEDGER", ledger)
    monkeypatch.delenv(check.env_var, raising=False)

    with pytest.raises(pytest.skip.Exception) as skipped:
        _credential(check)

    assert check.env_var in str(skipped.value)
    assert check.name in str(skipped.value)
    assert ledger.verified() == ()
    assert check.name in ledger.unverified


def test_a_missing_sdk_skips_the_provider_naming_the_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    check = LLM_CHECKS[0]
    ledger = LiveLedger()
    monkeypatch.setattr("tests.eval.test_live_providers.LEDGER", ledger)
    monkeypatch.setattr("tests.eval.test_live_providers.find_spec", lambda name: None)

    with pytest.raises(pytest.skip.Exception) as skipped:
        _require_package(check)

    assert check.package in str(skipped.value)
    assert f'fasterrag[{check.extra}]"' in str(skipped.value)
    assert check.name in str(skipped.value)
    assert ledger.verified() == ()
    assert check.name in ledger.unverified


def test_an_unverified_provider_can_never_be_counted_as_verified() -> None:
    ledger = LiveLedger()
    ledger.record_unverified(LLM_CHECKS[0].name, "OPENAI_API_KEY is not set")

    with pytest.raises(ValueError, match=NOT_CHECKED):
        ledger.record_verified(LLM_CHECKS[0].name, "answered")
    with pytest.raises(ValueError, match="without evidence"):
        ledger.record_verified(LLM_CHECKS[1].name, "   ")

    assert ledger.verified() == ()


def test_the_report_names_every_provider_the_ledger_never_reached() -> None:
    ledger = LiveLedger()
    ledger.record_verified(LLM_CHECKS[0].name, "answered with 9 tokens")
    ledger.record_unverified(LLM_CHECKS[1].name, "ANTHROPIC_API_KEY is not set")

    report = ledger.report(ALL_CHECKS)

    assert report.count(VERIFIED) == 1
    assert f"1 of {len(ALL_CHECKS)} providers reached" in report
    for check in ALL_CHECKS:
        assert check.name in report
    assert f"{LLM_CHECKS[1].name}: {NOT_CHECKED}" in report
    assert f"{EMBEDDING_CHECKS[0].name}: {NOT_RUN}" in report


def test_this_run_names_every_provider_it_did_not_reach() -> None:
    """Pass only when every advertised provider was reached; otherwise skip with the report.

    A run that reached nothing must not leave a green test behind that could be read as
    coverage, so the honest outcome of a keyless run is a skip carrying the full table.
    """
    report = LEDGER.report(ALL_CHECKS)
    reached = LEDGER.verified()

    assert len(reached) == len(LEDGER.evidence)
    assert set(reached) <= {check.name for check in ALL_CHECKS}
    assert set(reached).isdisjoint(LEDGER.unverified)

    if len(reached) < len(ALL_CHECKS):
        pytest.skip(report)
