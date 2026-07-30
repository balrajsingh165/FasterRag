import pytest

from fasterrag.adapters.embeddings.base import (
    classify_provider_failure,
    require_extra,
    require_key,
)
from fasterrag.errors import ConfigError, EmbedError, ErrorCode
from tests.unit.adapters.embeddings.conftest import FakeStatusError


def test_a_missing_package_names_the_install_command() -> None:
    error = require_extra("openai", "openai", "openai")

    assert isinstance(error, ConfigError)
    assert 'pip install "fasterrag[openai]"' in error.detail


def test_a_present_key_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROVIDER_KEY", "value")

    assert require_key("PROVIDER_KEY", "openai") == "value"


def test_a_missing_key_names_the_variable_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROVIDER_KEY", raising=False)

    with pytest.raises(ConfigError, match="PROVIDER_KEY"):
        require_key("PROVIDER_KEY", "openai")


def test_a_blank_key_is_treated_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROVIDER_KEY", "   ")

    with pytest.raises(ConfigError, match="is not set"):
        require_key("PROVIDER_KEY", "openai")


def test_an_unset_key_env_name_is_rejected() -> None:
    with pytest.raises(ConfigError, match="must name a variable"):
        require_key(None, "openai")


def classify(exc: BaseException) -> EmbedError:
    return classify_provider_failure(
        exc, provider="openai", operation="embed", key_env="OPENAI_API_KEY"
    )


@pytest.mark.parametrize("status", [401, 403])
def test_rejected_credentials_are_never_retried(status: int) -> None:
    error = classify(FakeStatusError(status))

    assert error.retryable is False
    assert "OPENAI_API_KEY" in error.detail


@pytest.mark.parametrize("status", [408, 504])
def test_provider_timeouts_get_the_timeout_code(status: int) -> None:
    error = classify(FakeStatusError(status))

    assert error.code is ErrorCode.EMBED_PROVIDER_TIMEOUT
    assert error.retryable is True


@pytest.mark.parametrize("status", [429, 500, 503])
def test_rate_limits_and_server_faults_are_retryable(status: int) -> None:
    assert classify(FakeStatusError(status)).retryable is True


@pytest.mark.parametrize("status", [400, 404, 422])
def test_client_mistakes_are_not_retryable(status: int) -> None:
    assert classify(FakeStatusError(status)).retryable is False


def test_a_bare_timeout_is_classified_as_a_timeout() -> None:
    error = classify(TimeoutError("too slow"))

    assert error.code is ErrorCode.EMBED_PROVIDER_TIMEOUT
    assert error.retryable is True


def test_an_unclassified_transport_failure_is_retryable() -> None:
    error = classify(OSError("connection reset"))

    assert error.retryable is True
    assert "unreachable" in error.detail


def test_a_secret_value_never_appears_in_a_classified_error() -> None:
    error = classify(FakeStatusError(401))

    assert "sk-" not in error.detail
