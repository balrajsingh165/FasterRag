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
    """Load a tokenizer once per process, returning ``None`` when it is unavailable.

    # CRITICAL: this must stay cached. A counter is built per document inside the CPU
    # pool, and an uncached load would re-read the vocabulary from disk for every file —
    # turning a millisecond-scale count into a hundred-millisecond one on the pipeline's
    # hottest path. The cache also stores the failure, so a missing dependency is
    # diagnosed and warned about once per process rather than once per document.
    """
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
        """Record the model without loading anything yet.

        Loading is deferred because a chunker is constructed for every document, and
        paying the load at construction would cost it even for a run that never chunks.

        Args:
            model: The embedding model whose tokenizer to load.
            chars_per_token: Ratio for the fallback estimate and the first split pass.
        """
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
        """Return the assumed characters per token, used to size the first split pass.

        Left as the estimate even when a real tokenizer is loaded. A chunker splits on
        characters — that is what keeps chunk offsets exact — and then re-splits whatever
        this ratio got wrong by counting for real. A per-model average would move the
        starting guess without making it right for any particular text, because the ratio
        varies far more between prose and CJK within one model than it does between models.
        Tune it through ``chunking.chars_per_token`` when a corpus is uniformly dense.
        """
        return self._estimate.chars_per_token


def create_token_counter(settings: Settings) -> TokenCounter:
    """Return the token counter selected by ``chunking.token_counter``.

    ``auto`` uses the model's own tokenizer when the provider ships a local one and the
    estimate otherwise. ``estimate`` forces the ratio-based counter, which is the faster
    choice for a corpus of English prose where the ratio is already close. ``model`` forces
    the real tokenizer for any provider, which is how a deployment using a hosted embedding
    model still gets exact counts — it needs the matching tokenizer cached locally, and
    falls back to the estimate with a warning when it is not.

    Returns:
        The counter to chunk with. Never raises: a counter that could fail configuration
        would make chunking depend on a model download.
    """
    mode = settings.chunking.token_counter
    estimate = EstimatingTokenCounter(settings.chunking.chars_per_token)

    if mode == "estimate":
        return estimate
    if mode == "model":
        return ModelTokenCounter(settings.embeddings.model, settings.chunking.chars_per_token)
    if settings.embeddings.provider not in _LOCAL_PROVIDERS:
        return estimate
    return ModelTokenCounter(settings.embeddings.model, settings.chunking.chars_per_token)
