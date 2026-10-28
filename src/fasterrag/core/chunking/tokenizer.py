"""The embedding model's own tokenizer, used to make ``chunking.chunk_size`` mean tokens.

The estimating counter assumes four characters per token. That is close enough for English
prose and wrong for everything else: code, tables, CJK text, and long identifiers all tokenize
far denser than the assumption, so a "512-token" chunk can be well past the model's limit and
get silently truncated at embed time — losing the tail of a chunk with nothing reporting it.

This counter asks the model that will actually embed the text. It is optional and lazily
loaded, so the estimate remains the fallback: a deployment with no local tokenizer, or one
running a hosted provider that exposes none, keeps working exactly as before.

**It loads a tokenizer, not a model.** A tokenizer is a vocabulary file measured in megabytes,
not the weights measured in gigabytes — the distinction is what makes this affordable inside
the CPU worker pool where chunking runs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Final

from fasterrag.config.schema import Settings
from fasterrag.core.chunking.models import CHARS_PER_TOKEN, EstimatingTokenCounter, TokenCounter
from fasterrag.observability.logging import get_logger

__all__ = ["ModelTokenCounter", "create_token_counter", "load_tokenizer"]

# Providers whose tokenizer is a local artifact we can load. A hosted provider is deliberately
# absent: counting its tokens would mean a network round trip per chunk, which turns chunking
# from a local CPU pass into a rate-limited one.
_LOCAL_PROVIDERS: Final[frozenset[str]] = frozenset({"huggingface"})

_logger = get_logger(__name__)


@lru_cache(maxsize=4)
def load_tokenizer(model: str) -> Any | None:
    """Load a tokenizer once per process, returning ``None`` when it is unavailable."""
    try:
        from transformers import AutoTokenizer

        # CRITICAL: local files only. Chunking must never trigger a download — it runs
        # inside the CPU worker pool, once per worker, and a cold cache would turn a
        # local CPU pass into a network fetch repeated across every process. If the
        # model is in use its tokenizer is already cached; if it is not, the estimate
        # is the correct answer rather than a reason to reach for the network.
        loaded: Any = AutoTokenizer.from_pretrained(model, local_files_only=True)
    except Exception as exc:
        # Deliberately broad: this traverses transformers, huggingface_hub, and the
        # filesystem, each raising its own types, and none of them should be able to
        # fail a chunking run that the estimate can complete perfectly well.
        _logger.warning(
            "the embedding tokenizer could not be loaded; falling back to the estimate, "
            "so chunk_size counts approximate tokens",
            extra={"model": model, "error": type(exc).__name__},
        )
        return None
    return loaded


class ModelTokenCounter:
    """Counts tokens with the embedding model's own tokenizer."""

    def __init__(self, model: str, chars_per_token: int = CHARS_PER_TOKEN) -> None:
        """Record the model without loading anything yet."""
        self.model = model
        self._estimate = EstimatingTokenCounter(chars_per_token)

    def count(self, text: str) -> int:
        """Return the token count, falling back to the estimate when unavailable."""
        stripped = text.strip()
        if not stripped:
            return 0

        tokenizer = load_tokenizer(self.model)
        if tokenizer is None:
            return self._estimate.count(stripped)

        try:
            return len(tokenizer.encode(stripped, add_special_tokens=False))
        except Exception:
            # A tokenizer that loaded but cannot encode this particular string must not
            # abort the document; the estimate is a worse answer, not no answer.
            return self._estimate.count(stripped)

    @property
    def chars_per_token(self) -> int:
        """Return the assumed characters per token, used to size the first split pass."""
        return self._estimate.chars_per_token


def create_token_counter(settings: Settings) -> TokenCounter:
    """Return the token counter selected by ``chunking.token_counter``."""
    mode = settings.chunking.token_counter
    estimate = EstimatingTokenCounter(settings.chunking.chars_per_token)

    if mode == "estimate":
        return estimate
    if mode == "model":
        return ModelTokenCounter(settings.embeddings.model, settings.chunking.chars_per_token)
    if settings.embeddings.provider not in _LOCAL_PROVIDERS:
        return estimate
    return ModelTokenCounter(settings.embeddings.model, settings.chunking.chars_per_token)
