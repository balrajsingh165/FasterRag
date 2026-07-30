from collections.abc import Sequence
from typing import Any

import pytest

from fasterrag.adapters.embeddings import huggingface
from fasterrag.config.schema import Settings

DIMENSIONS = 4


class FakeModel:
    """Stands in for a loaded sentence-transformers model."""

    def __init__(self, dimensions: int = DIMENSIONS) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []
        self.batch_sizes: list[int] = []
        self.raises: Exception | None = None

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if self.raises is not None:
            raise self.raises
        self.calls.append(list(texts))
        self.batch_sizes.append(batch_size)
        return [[float(index)] * self.dimensions for index in range(len(texts))]

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimensions


class FakeStatusError(Exception):
    """A provider error carrying an HTTP status, as the SDKs raise."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def local_settings(**embeddings: Any) -> Settings:
    """Return settings using the local provider with no credential required."""
    return Settings.model_validate({"embeddings": {"provider": "huggingface", **embeddings}})


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> FakeModel:
    """Patch model loading so no weights are ever downloaded."""
    fake = FakeModel()
    monkeypatch.setattr(huggingface, "load_model", lambda name: fake)
    return fake


def vectors_of(count: int, dimensions: int = DIMENSIONS) -> list[list[float]]:
    """Return placeholder vectors."""
    return [[float(index)] * dimensions for index in range(count)]


class FakeClient:
    """Records embedding calls made through a provider SDK."""

    def __init__(self, dimensions: int = DIMENSIONS) -> None:
        self.dimensions = dimensions
        self.calls: list[dict[str, Any]] = []
        self.raises: Exception | None = None
        self.closed = False

    def _vectors(self, count: int) -> list[list[float]]:
        return vectors_of(count, self.dimensions)

    def record(self, **kwargs: Any) -> Sequence[str]:
        if self.raises is not None:
            raise self.raises
        self.calls.append(kwargs)
        texts = kwargs.get("input") or kwargs.get("texts") or [kwargs.get("prompt", "")]
        return list(texts)

    async def close(self) -> None:
        self.closed = True
