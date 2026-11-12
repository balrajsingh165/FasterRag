import importlib.util
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from typing import Any

import httpx
import pytest

from fasterrag.adapters.llm import (
    AnthropicGenerator,
    CohereGenerator,
    Completion,
    LLMAdapter,
    OllamaGenerator,
    OpenAICompatibleGenerator,
    OpenAIGenerator,
    available_providers,
    create_llm_adapter,
    resolve_adapter_class,
)
from fasterrag.adapters.llm.base import classify_llm_failure, require_llm_extra, require_llm_key
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError, ErrorCode

PROVIDERS = [
    ("openai", OpenAIGenerator),
    ("openai_compatible", OpenAICompatibleGenerator),
    ("anthropic", AnthropicGenerator),
    ("cohere", CohereGenerator),
    ("ollama", OllamaGenerator),
]

PACKAGES = {
    "openai": "openai",
    "openai_compatible": "openai",
    "anthropic": "anthropic",
    "cohere": "cohere",
    "ollama": "ollama",
}


class FakeStatusError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def settings_for(provider: str, **overrides: Any) -> Settings:
    llm: dict[str, Any] = {"provider": provider, **overrides}
    if provider == "ollama":
        llm.setdefault("api_key_env", None)
        llm.setdefault("base_url", "http://localhost:11434")
    else:
        llm.setdefault("api_key_env", f"{provider.upper()}_API_KEY")
    if provider == "openai_compatible":
        llm.setdefault("base_url", "http://localhost:8000/v1")
    return Settings.model_validate({"llm": llm})


@pytest.mark.parametrize(("provider", "expected"), PROVIDERS)
def test_every_documented_provider_resolves(provider: str, expected: type) -> None:
    assert resolve_adapter_class(provider) is expected
    assert available_providers()[provider] == "built-in"


@pytest.mark.parametrize(("provider", "expected"), PROVIDERS)
def test_the_factory_builds_each_provider(provider: str, expected: type) -> None:
    adapter = create_llm_adapter(settings_for(provider))

    assert isinstance(adapter, expected)
    assert isinstance(adapter, LLMAdapter)


@pytest.mark.parametrize(("provider", "expected"), PROVIDERS)
def test_construction_opens_nothing(provider: str, expected: type) -> None:
    adapter = create_llm_adapter(settings_for(provider, model="pinned-model"))

    assert adapter.model == "pinned-model"
    assert isinstance(adapter, expected)
    assert adapter._client is None


def test_an_openai_compatible_endpoint_reuses_the_openai_wire_protocol() -> None:
    assert issubclass(OpenAICompatibleGenerator, OpenAIGenerator)
    assert OpenAICompatibleGenerator.provider == "openai_compatible"


def test_an_unknown_provider_lists_what_is_available() -> None:
    with pytest.raises(ConfigError, match="not registered") as caught:
        resolve_adapter_class("nonexistent")

    assert "anthropic" in caught.value.detail


@pytest.mark.parametrize(("provider", "expected"), PROVIDERS)
async def test_a_provider_without_its_package_names_the_install_command(
    provider: str, expected: type
) -> None:
    package = PACKAGES[provider]
    if importlib.util.find_spec(package) is not None:
        pytest.skip(f"{package} is installed, so the missing-package path cannot be exercised")

    adapter = create_llm_adapter(settings_for(provider))

    with pytest.raises(ConfigError, match="fasterrag\\["):
        await adapter.complete("anything")


@pytest.mark.parametrize(("provider", "expected"), PROVIDERS)
async def test_health_reports_a_missing_package_instead_of_raising(
    provider: str, expected: type
) -> None:
    if importlib.util.find_spec(PACKAGES[provider]) is not None:
        pytest.skip("package installed, so the missing-package path cannot be exercised")

    status = await create_llm_adapter(settings_for(provider)).health()

    assert status.healthy is False


def test_the_timeout_comes_from_the_reliability_settings() -> None:
    configured = Settings.model_validate(
        {
            "llm": {"provider": "ollama", "api_key_env": None},
            "reliability": {"timeouts": {"llm_ms": 45000}},
        }
    )

    assert create_llm_adapter(configured).timeout == pytest.approx(45.0)


def test_a_missing_package_names_the_install_command() -> None:
    error = require_llm_extra("anthropic", "anthropic", "anthropic")

    assert 'pip install "fasterrag[anthropic]"' in error.detail


def test_a_missing_key_names_the_variable_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROVIDER_KEY", raising=False)

    with pytest.raises(ConfigError, match="PROVIDER_KEY"):
        require_llm_key("PROVIDER_KEY", "openai")


def classify(exc: BaseException) -> Any:
    return classify_llm_failure(
        exc, provider="openai", operation="complete", key_env="OPENAI_API_KEY"
    )


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_credentials_are_never_retried(status: int) -> None:
    error = classify(FakeStatusError(status))

    assert error.retryable is False
    assert "OPENAI_API_KEY" in error.detail
    assert error.code is ErrorCode.GENERATION_FAILED


@pytest.mark.parametrize("status", [429, 500, 503])
def test_rate_limits_and_server_faults_are_retryable(status: int) -> None:
    assert classify(FakeStatusError(status)).retryable is True


@pytest.mark.parametrize("status", [400, 404, 422])
def test_client_mistakes_are_not_retryable(status: int) -> None:
    assert classify(FakeStatusError(status)).retryable is False


def test_a_timeout_is_retryable() -> None:
    assert classify(TimeoutError("too slow")).retryable is True


def test_an_unclassified_transport_failure_is_retryable() -> None:
    error = classify(OSError("connection reset"))

    assert error.retryable is True
    assert "unreachable" in error.detail


@pytest.mark.parametrize(
    "mistake",
    [
        TypeError("AsyncBaseCohere.chat() got an unexpected keyword argument 'messages'"),
        AttributeError("'ChatResponse' object has no attribute 'choices'"),
    ],
)
def test_a_programming_error_is_not_mistaken_for_an_outage(mistake: Exception) -> None:
    """It has no status, so it read as "unreachable, retryable" and was retried as an outage."""
    error = classify(mistake)

    assert error.code is ErrorCode.INTERNAL
    assert error.retryable is False
    assert "unreachable" not in error.detail
    assert type(mistake).__name__ in error.detail


def test_a_programming_error_does_not_quote_the_failing_call() -> None:
    """The exception text can carry argument values; only its type is safe to publish."""
    error = classify(TypeError("chat() got an unexpected keyword argument 'sk-secret'"))

    assert "sk-secret" not in error.detail


def test_a_status_outranks_the_exception_type() -> None:
    """A vendor error that subclasses TypeError still says what the provider answered."""

    class StatusTypeError(TypeError):
        status_code = 429

    error = classify(StatusTypeError("rate limited"))

    assert error.retryable is True
    assert error.code is ErrorCode.GENERATION_FAILED


def test_a_secret_never_appears_in_a_classified_error() -> None:
    assert "sk-" not in classify(FakeStatusError(401)).detail


class HeaderedStatusError(FakeStatusError):
    """Carries headers the way Cohere's ``ApiError`` does: as a plain attribute."""

    def __init__(self, status_code: int, headers: dict[str, str]) -> None:
        super().__init__(status_code)
        self.headers = headers


class ResponseStatusError(FakeStatusError):
    """Carries headers the way OpenAI's and Anthropic's ``APIStatusError`` do: on a response."""

    def __init__(self, status_code: int, headers: dict[str, str]) -> None:
        super().__init__(status_code)
        self.response = httpx.Response(status_code, headers=headers)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (HeaderedStatusError(429, {"retry-after": "7"}), 7.0),
        (ResponseStatusError(429, {"Retry-After": "2.5"}), 2.5),
        (ResponseStatusError(503, {"retry-after-ms": "1500", "retry-after": "9"}), 1.5),
        (HeaderedStatusError(429, {"Retry-After": "a fortnight"}), None),
        (HeaderedStatusError(429, {"Retry-After": "-3"}), None),
        (FakeStatusError(429), None),
    ],
)
def test_a_provider_s_retry_after_is_carried_on_the_error(
    error: Exception, expected: float | None
) -> None:
    """The provider's ``Retry-After`` was dropped, so a retry could arrive before it allowed."""
    assert classify(error).retry_after == expected


def test_a_retry_after_given_as_a_date_becomes_seconds_from_now() -> None:
    moment = datetime.now(UTC) + timedelta(seconds=30)
    error = classify(HeaderedStatusError(429, {"Retry-After": format_datetime(moment, True)}))

    assert error.retry_after is not None
    assert 25 <= error.retry_after <= 31


def test_a_refused_credential_carries_no_retry_after() -> None:
    """Only a failure worth retrying says when to retry."""
    assert classify(HeaderedStatusError(401, {"Retry-After": "7"})).retry_after is None


def test_a_completion_reports_whether_it_was_cut_short() -> None:
    assert Completion(text="x", model="m", finish_reason="length").truncated is True
    assert Completion(text="x", model="m", finish_reason="stop").truncated is False
    assert Completion(text="x", model="m").truncated is False
