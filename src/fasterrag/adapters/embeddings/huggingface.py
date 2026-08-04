"""Local sentence-transformers embedding provider — the default.

Nothing leaves the host: the model runs in the embedding worker pool on CPU or GPU. The
model is loaded once per adapter instance and reused for every batch, because reloading a
model per task is the single largest avoidable cost in a naive pipeline and is prohibited
by design (``docs/architecture.md`` §2).

Encoding is CPU- or GPU-bound rather than I/O-bound, so it runs in a worker thread to keep
the event loop free when the adapter is used from an async context.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from fasterrag.adapters.embeddings.base import EmbeddingAdapter, EmbeddingResult
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError, EmbedError
from fasterrag.observability.logging import get_logger

__all__ = ["HuggingFaceEmbedder"]

_logger = get_logger(__name__)


def load_model(name: str) -> Any:
    """Load a sentence-transformers model.

    Imported lazily so that importing fasterRag does not pull a deep-learning stack into
    processes that never embed anything.

    Raises:
        ConfigError: If sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ConfigError(
            "embeddings.provider is 'huggingface', which needs sentence-transformers; "
            "install it with 'pip install sentence-transformers'"
        ) from exc

    return SentenceTransformer(name)


def _embedding_dimension(model: Any) -> int | None:
    """Return a loaded model's vector size across sentence-transformers versions.

    The accessor was renamed in sentence-transformers 5.x; calling the old name still works
    but emits a deprecation warning, and it will eventually stop working. Both names are
    tried so the adapter spans versions rather than pinning one.
    """
    for accessor in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        method = getattr(model, accessor, None)
        if callable(method):
            size = method()
            if isinstance(size, int):
                return size
    return None


def _dimension_mismatch(model: str, configured: int, emitted: int) -> ConfigError:
    """Return the error for a configured vector size the local model does not produce.

    Raised at load time rather than left to the vector database, because the failure would
    otherwise surface as a rejected upsert after a collection had already been created at
    the wrong width — long after the setting that caused it, and with nothing naming it.

    Truncating to the configured size is deliberately not offered. Shortening a vector is
    only lossless for a model trained for it (Matryoshka representation learning), and
    silently truncating one that was not degrades retrieval in a way no error reports.
    """
    return ConfigError(
        f"embeddings.dimensions is {configured} but the local model {model!r} emits "
        f"{emitted}-dimensional vectors. Remove embeddings.dimensions to accept the "
        f"model's own size, or choose a model that emits {configured}. Shortening the "
        "output is only offered by hosted providers whose models are trained for it "
        "(for example OpenAI's text-embedding-3 family), where the API does it server-side"
    )


class HuggingFaceEmbedder(EmbeddingAdapter):
    """Embeds locally with a sentence-transformers model."""

    provider = "huggingface"

    def __init__(self, settings: Settings) -> None:
        """Build the adapter without loading the model."""
        super().__init__(settings)
        self._model: Any | None = None
        self._dimensions = settings.embeddings.dimensions

    @property
    def model(self) -> str:
        """Return the configured model id."""
        return self.config.model

    @property
    def model_version(self) -> str:
        """Return the loaded model's version.

        Local models carry no version field, so the identifier is the model id itself.
        Drift is still detected, because changing ``embeddings.model`` changes this value.
        """
        return self.config.model

    @property
    def dimensions(self) -> int | None:
        """Return the vector size, known once the model is loaded."""
        return self._dimensions

    def _loaded(self) -> Any:
        """Return the model, loading it on first use.

        Raises:
            ConfigError: If ``embeddings.dimensions`` disagrees with what the model emits.
        """
        if self._model is None:
            _logger.info("loading embedding model", extra={"model": self.config.model})
            self._model = load_model(self.config.model)
            emitted = _embedding_dimension(self._model)
            if self._dimensions is None:
                self._dimensions = emitted
            elif emitted is not None and emitted != self._dimensions:
                self._model = None
                raise _dimension_mismatch(self.config.model, self._dimensions, emitted)
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode texts synchronously.

        The path the CPU worker pool and the semantic chunker use, where there is no event
        loop to protect.

        Raises:
            EmbedError: If the model fails to encode the batch.
        """
        model = self._loaded()
        try:
            vectors = model.encode(list(texts), batch_size=self.config.batch_size)
        except (RuntimeError, ValueError, OSError) as exc:
            raise EmbedError(
                f"the local embedding model failed to encode a batch: {type(exc).__name__}",
                retryable=False,
            ) from exc

        return [[float(value) for value in vector] for vector in vectors]

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed passages, keeping the event loop free while the model runs."""
        vectors: list[list[float]] = []
        for batch in self.batches(texts):
            vectors.extend(await asyncio.to_thread(self.encode, batch))

        return EmbeddingResult(
            vectors=vectors,
            model=self.model,
            model_version=self.model_version,
        )

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        vectors = await asyncio.to_thread(self.encode, [text])
        return vectors[0]

    async def health(self) -> HealthStatus:
        """Report whether the model can be loaded and used."""
        try:
            await asyncio.to_thread(self.encode, ["health check"])
        except (ConfigError, EmbedError) as exc:
            return HealthStatus(healthy=False, detail=exc.detail)
        return HealthStatus(healthy=True, detail=f"local model {self.model} loaded")

    async def close(self) -> None:
        """Release the loaded model."""
        self._model = None
