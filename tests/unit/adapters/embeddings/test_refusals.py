"""What a provider refusal says beyond its status: send less, or wait this long (TASK-0478).

A refusal for size and a ``Retry-After`` used to be classified like any other status, so the
worker pool retried an oversized batch unchanged until its attempts ran out and ignored the
pause the provider asked for.
"""

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import pytest

from fasterrag.adapters.embeddings.base import (
    EmbedThrottledError,
    EmbedTooLargeError,
    classify_provider_failure,
    retry_after_seconds,
)
from fasterrag.errors import EmbedError


class Response:
    """The part of an ``httpx.Response`` a vendor error carries."""

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers


class VendorError(Exception):
    """A provider error with a status, a message and response headers."""

    def __init__(
        self,
        status_code: int,
        message: str = "",
        *,
        headers: dict[str, str] | None = None,
        on_response: bool = True,
    ) -> None:
        super().__init__(message or f"status {status_code}")
        self.status_code = status_code
        if on_response:
            self.response = Response(headers or {})
        else:
            self.headers = headers or {}


def classify(exc: BaseException) -> EmbedError:
    return classify_provider_failure(exc, provider="openai", operation="embed", key_env="KEY")


def test_payload_too_large_is_a_size_refusal() -> None:
    """A 413 was "status 413, not retryable", so the batch failed instead of being split."""
    error = classify(VendorError(413, "Request Entity Too Large"))

    assert isinstance(error, EmbedTooLargeError)
    assert error.retryable is False


@pytest.mark.parametrize(
    "message",
    [
        "This model's maximum context length is 8192 tokens, however you requested 9000 tokens",
        "invalid request: total number of texts must be at most 96 - received 128",
        "too many tokens in the input",
        "the input length exceeds the context length",
    ],
)
def test_a_token_or_batch_limit_is_a_size_refusal(message: str) -> None:
    assert isinstance(classify(VendorError(400, message)), EmbedTooLargeError)


def test_a_size_refusal_does_not_quote_the_input() -> None:
    error = classify(VendorError(400, "maximum context length exceeded for 'sk-secret text'"))

    assert "sk-secret" not in error.detail


def test_an_ordinary_bad_request_is_not_a_size_refusal() -> None:
    error = classify(VendorError(400, "unknown model 'text-embedding-4'"))

    assert not isinstance(error, EmbedTooLargeError)
    assert error.retryable is False


def test_a_rate_limit_carries_the_pause_the_provider_asked_for() -> None:
    error = classify(VendorError(429, headers={"retry-after": "7"}))

    assert isinstance(error, EmbedThrottledError)
    assert error.retry_after_seconds == 7.0
    assert error.retryable is True


def test_headers_on_the_error_itself_are_read_too() -> None:
    """Cohere's ``ApiError`` carries ``headers`` directly rather than a response object."""
    error = classify(VendorError(503, headers={"Retry-After": "2"}, on_response=False))

    assert isinstance(error, EmbedThrottledError)
    assert error.retry_after_seconds == 2.0


def test_a_rate_limit_without_a_pause_is_an_ordinary_retryable_error() -> None:
    error = classify(VendorError(429))

    assert not isinstance(error, EmbedThrottledError)
    assert error.retryable is True


def test_milliseconds_are_preferred_when_the_provider_states_them() -> None:
    headers = {"retry-after-ms": "1500", "retry-after": "2"}

    assert retry_after_seconds(VendorError(429, headers=headers)) == 1.5


def test_an_http_date_is_turned_into_a_delay() -> None:
    when = datetime.now(UTC) + timedelta(seconds=30)
    delay = retry_after_seconds(VendorError(429, headers={"retry-after": format_datetime(when)}))

    assert delay is not None
    assert 25 <= delay <= 31


@pytest.mark.parametrize("value", ["soon", "-3", "nan", ""])
def test_an_unreadable_pause_is_ignored(value: str) -> None:
    assert retry_after_seconds(VendorError(429, headers={"retry-after": value})) is None


def test_a_date_in_the_past_means_no_pause() -> None:
    when = datetime.now(UTC) - timedelta(minutes=5)

    assert (
        retry_after_seconds(VendorError(429, headers={"retry-after": format_datetime(when)})) == 0
    )
