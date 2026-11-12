"""Credential-gated checks that each advertised provider really accepts our calls.

Every other provider test in this suite substitutes a fake client, which proves the adapter
issues the request it intends to and proves nothing about whether the vendor accepts it.
``docs/integrations.md`` advertises five generation providers and four embedding providers;
a provider list is a promise, so each entry gets the smallest call that can keep it — one
tiny generation, or one tiny embedding — asserting a well-formed typed response and the
token accounting the adapter claims to populate.

Cost of one full run with every credential present: **eight requests**, each of a few dozen
tokens — five generations (OpenAI, Anthropic, Cohere, Ollama, ``openai_compatible``) and
three embeddings (OpenAI, Cohere, Ollama). The two Ollama checks go to a local server and
cost nothing, and ``openai_compatible`` costs whatever the gateway it is pointed at charges.

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

Run with ``FASTERRAG_LIVE_PROVIDER_TESTS=1 pytest -m eval tests/eval/test_live_providers.py
-rs`` — ``-rs`` prints the skip reasons, which is where the report lives.

**The opt-in variable is the gate; a credential is not.** Until TASK-0341 these probes ran
whenever a key happened to be in the environment, and the environment is not a statement of
intent: a developer with ``OPENAI_API_KEY`` exported for unrelated work, or a process that
loaded a ``.env`` on the way past, spent money by typing a test command. Every probe now
skips unless ``FASTERRAG_LIVE_PROVIDER_TESTS`` is one of ``1``, ``true``, ``yes`` or ``on``,
and the skip reason says how to turn it on. The marker filter in ``pyproject.toml`` keeps
these out of a bare ``pytest``; this gate is what stands behind that filter when someone
runs ``pytest -m eval``.

Ollama has no credential, so its gate is an explicit opt-in: ``OLLAMA_HOST`` pointing at a
running server. Gating it on nothing would mean a connection attempt on every run and a
failure, not a skip, on every machine without Ollama. ``openai_compatible`` names no vendor
at all, so it is gated on all three of the things only its operator knows —
``FASTERRAG_LIVE_OPENAI_COMPATIBLE_URL``, ``FASTERRAG_LIVE_OPENAI_COMPATIBLE_KEY`` and
``FASTERRAG_LIVE_MODEL_OPENAI_COMPATIBLE_LLM`` — because a gateway has no default model id
anyone else could guess. The HuggingFace embedder is local and already exercised against
real weights in ``test_real_models.py``; it is the one advertised provider with no entry
here, and it needs none.

The gating helpers themselves are proved by ``test_live_provider_gating.py``, which carries
no ``eval`` marker and therefore runs in the ordinary CI job: without it, the only evidence
that a skip is a real gate rather than a silent bug would come from a job that skips.

``reports_token_usage`` is a claim about the vendor, not about our code. A ``False`` must
carry a ``usage_note`` saying what the provider's documented response omits, so the flag can
never quietly record an unimplemented extraction as a vendor limitation; the constructor
refuses one without a reason. The flags are checked against the adapters by
``tests/unit/adapters/embeddings/test_usage.py``, which runs in ordinary CI and needs no
credential, while the live probes here are what can falsify the claim against the vendor.
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

LIVE_TESTS_VAR = "FASTERRAG_LIVE_PROVIDER_TESTS"
LIVE_TESTS_ON = frozenset({"1", "true", "yes", "on"})

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
    base_url_env: str | None = None
    keyless: bool = False
    usage_note: str = ""
    """Why this provider states no billed usage. Mandatory when the flag is ``False``.

    A ``False`` that nobody had to justify is how an unimplemented extraction gets recorded
    as a provider limitation, which is the defect this field exists to make impossible.
    """

    def __post_init__(self) -> None:
        """Refuse a silent ``reports_token_usage=False``."""
        if not self.reports_token_usage and not self.usage_note.strip():
            raise ValueError(f"{self.name} must say why it reports no token usage")
        if self.reports_token_usage and self.usage_note.strip():
            raise ValueError(f"{self.name} reports usage, so it needs no usage_note")

    @property
    def name(self) -> str:
        """Return the identifier this provider is recorded and reported under."""
        return f"{self.provider} {self.kind}"

    @property
    def model_env(self) -> str:
        """Return the variable that overrides the model id this provider is probed with."""
        return f"FASTERRAG_LIVE_MODEL_{self.provider.upper()}_{self.kind.upper()}"

    @property
    def model(self) -> str:
        """Return the model to probe, overridable so a retired id can be replaced."""
        configured = os.environ.get(self.model_env, "").strip()
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
        base_url_env="OLLAMA_HOST",
        keyless=True,
    ),
    LiveCheck(
        kind="llm",
        provider="openai_compatible",
        env_var="FASTERRAG_LIVE_OPENAI_COMPATIBLE_KEY",
        package="openai",
        extra="openai",
        default_model="",
        reports_token_usage=True,
        base_url_env="FASTERRAG_LIVE_OPENAI_COMPATIBLE_URL",
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
    LiveCheck(
        kind="embeddings",
        provider="cohere",
        env_var="COHERE_API_KEY",
        package="cohere",
        extra="cohere",
        default_model="embed-english-v3.0",
        reports_token_usage=True,
    ),
    LiveCheck(
        kind="embeddings",
        provider="ollama",
        env_var="OLLAMA_HOST",
        package="ollama",
        extra="ollama",
        default_model="nomic-embed-text",
        reports_token_usage=True,
        base_url_env="OLLAMA_HOST",
        keyless=True,
    ),
)

ALL_CHECKS: tuple[LiveCheck, ...] = LLM_CHECKS + EMBEDDING_CHECKS


@dataclass(slots=True)
class LiveLedger:
    """What this run actually reached, and what it did not."""

    evidence: dict[str, str] = field(default_factory=dict)
    unverified: dict[str, str] = field(default_factory=dict)

    def record_verified(self, name: str, evidence: str) -> None:
        """Record that ``name`` returned a live response."""
        if not evidence.strip():
            raise ValueError(f"{name} cannot be recorded verified without evidence")
        if name in self.unverified:
            raise ValueError(f"{name} was already recorded as {NOT_CHECKED}")
        self.evidence[name] = evidence

    def record_unverified(self, name: str, reason: str) -> None:
        """Record why ``name`` was not reached."""
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


def _gate(check: LiveCheck, variable: str) -> str:
    """Return a gating environment value, recording and skipping when it is absent.

    Raises:
        Skipped: If the variable is unset or blank, naming the provider and the variable.
    """
    value = os.environ.get(variable, "").strip()
    if not value:
        reason = f"{variable} is not set, so {check.name} was not contacted"
        LEDGER.record_unverified(check.name, reason)
        pytest.skip(reason)
    return value


def _require_opt_in(check: LiveCheck) -> None:
    """Record and skip unless the run explicitly asked for billed live calls.

    Raises:
        Skipped: Unless ``FASTERRAG_LIVE_PROVIDER_TESTS`` is set to a truthy value.
    """
    if os.environ.get(LIVE_TESTS_VAR, "").strip().lower() in LIVE_TESTS_ON:
        return
    reason = (
        f"{LIVE_TESTS_VAR} is not set, so {check.name} was not contacted; these probes "
        f"spend real money, so a credential alone does not authorise them. To opt in, run "
        f"{LIVE_TESTS_VAR}=1 pytest -m eval -rs with the provider's credential set"
    )
    LEDGER.record_unverified(check.name, reason)
    pytest.skip(reason)


def _credential(check: LiveCheck) -> str:
    """Return the provider's credential, recording and skipping when it is absent.

    Raises:
        Skipped: If the credential variable is unset or blank. The value is never logged.
    """
    return _gate(check, check.env_var)


def _model(check: LiveCheck) -> str:
    """Return the model id to probe, recording and skipping when none is configured.

    Raises:
        Skipped: If the provider has no default model and no override is set.
    """
    model = check.model
    if not model:
        reason = (
            f"{check.model_env} is not set and {check.name} has no default model, because a "
            f"gateway names its own models, so {check.name} was not contacted"
        )
        LEDGER.record_unverified(check.name, reason)
        pytest.skip(reason)
    return model


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
    """Return the base URL a self-hosted provider is reached at, or None for a hosted one.

    Raises:
        Skipped: If the endpoint is unset, or set to something that is not an absolute URL.
    """
    if check.base_url_env is None:
        return None

    host = _gate(check, check.base_url_env)
    if not host.startswith(("http://", "https://")):
        reason = (
            f"{check.base_url_env} is set to a value with no http:// or https:// scheme, "
            f"which the configuration contract rejects, so {check.name} was not contacted"
        )
        LEDGER.record_unverified(check.name, reason)
        pytest.skip(reason)
    return host


def llm_settings(check: LiveCheck) -> Settings:
    """Build configuration pointing generation at the provider under check."""
    _require_opt_in(check)
    _require_package(check)
    base_url = _endpoint(check)
    if not check.keyless:
        _credential(check)

    return Settings.model_validate(
        {
            "llm": {
                "provider": check.provider,
                "model": _model(check),
                "api_key_env": None if check.keyless else check.env_var,
                "base_url": base_url,
                "temperature": 0.0,
                "max_tokens": MAX_TOKENS,
            }
        }
    )


def embedding_settings(check: LiveCheck) -> Settings:
    """Build configuration pointing embedding at the provider under check.

    ``OllamaEmbedder`` takes its host from ``llm.base_url`` when the LLM is Ollama too,
    which is why the endpoint is set there rather than under ``embeddings``.
    """
    _require_opt_in(check)
    _require_package(check)
    base_url = _endpoint(check)
    if not check.keyless:
        _credential(check)

    payload: dict[str, object] = {
        "embeddings": {
            "provider": check.provider,
            "model": _model(check),
            "api_key_env": None if check.keyless else check.env_var,
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
        # A provider that quietly started reporting usage must fail here rather than keep
        # its flag: the flag claims the vendor states nothing, and the live call is the only
        # thing that can falsify that claim.
        assert result.total_tokens is None, (
            f"{check.name} now reports {result.total_tokens} tokens, so reports_token_usage "
            f"is stale; its recorded reason was: {check.usage_note}"
        )

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
