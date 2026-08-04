from typing import Any

import pytest

from fasterrag.config.schema import Settings
from fasterrag.core.chunking import tokenizer as tokenizer_module
from fasterrag.core.chunking.models import CHARS_PER_TOKEN, EstimatingTokenCounter
from fasterrag.core.chunking.tokenizer import ModelTokenCounter, create_token_counter

DENSE = "def _resolve(self, x: int) -> dict[str, int]: return {'k': x}"


class StubTokenizer:
    """Counts one token per whitespace-separated word, recording every call."""

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        self.calls += 1
        return list(range(len(text.split())))


@pytest.fixture(autouse=True)
def _clear_cache() -> Any:
    """The loader caches per process, which would leak a stub between tests."""
    tokenizer_module.load_tokenizer.cache_clear()
    yield
    tokenizer_module.load_tokenizer.cache_clear()


def settings(**embeddings: Any) -> Settings:
    return Settings.model_validate({"embeddings": embeddings} if embeddings else {})


def test_a_local_provider_gets_the_real_tokenizer() -> None:
    counter = create_token_counter(settings(provider="huggingface", model="some/model"))

    assert isinstance(counter, ModelTokenCounter)


def test_a_hosted_provider_keeps_the_estimate() -> None:
    """Counting through a hosted API would mean a network round trip per chunk."""
    counter = create_token_counter(
        settings(provider="openai", model="text-embedding-3-small", api_key_env="OPENAI_API_KEY")
    )

    assert isinstance(counter, EstimatingTokenCounter)


def test_the_real_count_replaces_the_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tokenizer_module, "load_tokenizer", lambda model: StubTokenizer())

    counter = ModelTokenCounter("some/model")

    assert counter.count(DENSE) == len(DENSE.split())
    assert EstimatingTokenCounter().count(DENSE) != len(DENSE.split())


def test_an_unloadable_tokenizer_falls_back_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing optional dependency must not fail a chunking run."""
    monkeypatch.setattr(tokenizer_module, "load_tokenizer", lambda model: None)

    counter = ModelTokenCounter("some/model")

    assert counter.count(DENSE) == EstimatingTokenCounter().count(DENSE)


def test_a_tokenizer_that_cannot_encode_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken:
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            raise ValueError("input is not valid for this vocabulary")

    monkeypatch.setattr(tokenizer_module, "load_tokenizer", lambda model: Broken())

    assert ModelTokenCounter("some/model").count(DENSE) == EstimatingTokenCounter().count(DENSE)


def test_empty_text_costs_no_call(monkeypatch: pytest.MonkeyPatch) -> None:
    stub = StubTokenizer()
    monkeypatch.setattr(tokenizer_module, "load_tokenizer", lambda model: stub)

    assert ModelTokenCounter("some/model").count("   \n  ") == 0
    assert stub.calls == 0


def test_chars_per_token_stays_the_estimate() -> None:
    """It sizes a search window, not a count; a per-model value would only add variance."""
    assert ModelTokenCounter("some/model").chars_per_token == CHARS_PER_TOKEN


def test_the_load_happens_once_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """A counter is built per document; an uncached load would re-read the vocabulary."""
    loads = []

    class Recording:
        @staticmethod
        def from_pretrained(model: str, **_kwargs: Any) -> StubTokenizer:
            loads.append(model)
            return StubTokenizer()

    monkeypatch.setitem(
        __import__("sys").modules, "transformers", type("m", (), {"AutoTokenizer": Recording})
    )

    for _ in range(5):
        ModelTokenCounter("some/model").count("one two")

    assert loads == ["some/model"]


def test_a_failed_load_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retrying per document would warn per document and re-pay the import each time."""
    attempts = []

    class Failing:
        @staticmethod
        def from_pretrained(model: str, **_kwargs: Any) -> Any:
            attempts.append(model)
            raise OSError("no local snapshot")

    monkeypatch.setitem(
        __import__("sys").modules, "transformers", type("m", (), {"AutoTokenizer": Failing})
    )

    for _ in range(5):
        ModelTokenCounter("some/model").count("one two")

    assert attempts == ["some/model"]


def test_the_load_never_reaches_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cold cache inside a worker pool would fetch once per process, on the hot path."""
    seen: dict[str, Any] = {}

    class Recording:
        @staticmethod
        def from_pretrained(_model: str, **kwargs: Any) -> StubTokenizer:
            seen.update(kwargs)
            return StubTokenizer()

    monkeypatch.setitem(
        __import__("sys").modules, "transformers", type("m", (), {"AutoTokenizer": Recording})
    )

    tokenizer_module.load_tokenizer("some/model")

    assert seen["local_files_only"] is True
