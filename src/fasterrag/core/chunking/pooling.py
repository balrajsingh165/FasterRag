"""The pooling half of late chunking (D-late, ``chunking.strategy: late``).

Ordinary chunking embeds each chunk alone, so a chunk that says "It was raised from 35
pounds" carries no trace of what "it" is — the pronoun's referent lives in a neighbouring
chunk and is simply gone. Late chunking fixes this by inverting the order: run the model
over the *whole document* once, then average the token representations that fall inside
each chunk's span. The boundaries are identical; only the vectors change, and each one now
carries context from beyond its own edges.

This module owns that pooling. It needs token-level output from the model, which an
embedding API does not expose and a locally loaded transformer does — which is why late
chunking is available for local models only, and why the caller is expected to fall back
to ordinary embedding rather than fail when the model cannot supply it.

Documents longer than the model's context are processed in overlapping windows and each
token takes its representation from the window that gave it the most surrounding context.
Late chunking is most valuable with a long-context embedding model, where a whole document
fits in one pass; the windowing keeps it correct rather than truncated on a short one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from fasterrag.core.chunking.models import Segment
from fasterrag.errors import EmbedError

__all__ = ["WINDOW_OVERLAP_FRACTION", "pool_spans", "supports_pooling"]

# How much of each window repeats the one before it. A token at a window's edge sees
# context on one side only, so windows overlap and every token is scored by how centred it
# is — a quarter is enough for the edges to lose to a neighbouring window's middle.
WINDOW_OVERLAP_FRACTION = 4

_SPECIAL_TOKEN_BUDGET = 2


def supports_pooling(model: Any) -> bool:
    """Return whether ``model`` exposes the token-level output pooling needs.

    Checked by duck typing rather than by class, because the caller holds whatever the
    embedding adapter loaded and the useful question is what it can do, not what it is.
    """
    if getattr(model, "tokenizer", None) is None:
        return False
    try:
        return getattr(model[0], "auto_model", None) is not None
    except (TypeError, KeyError, IndexError):
        return False


def _windows(total: int, size: int) -> list[tuple[int, int]]:
    """Return overlapping ``[start, end)`` token windows covering ``total`` tokens."""
    if total <= size:
        return [(0, total)]

    step = max(size - size // WINDOW_OVERLAP_FRACTION, 1)
    bounds = []
    start = 0
    while start < total:
        end = min(start + size, total)
        bounds.append((start, end))
        if end == total:
            break
        start += step
    return bounds


def _sentinels(tokenizer: Any) -> tuple[list[int], list[int]]:
    """Return the leading and trailing special-token ids a window should be wrapped in.

    Built from the tokenizer's own ids rather than through a helper method, because the
    convenience helpers for this differ between transformers major versions while
    ``cls_token_id`` and ``sep_token_id`` have been stable throughout. A tokenizer that
    defines neither is wrapped in nothing, which is correct for the families that use no
    sentinels at all.
    """
    start = getattr(tokenizer, "cls_token_id", None)
    end = getattr(tokenizer, "sep_token_id", None)
    return ([start] if start is not None else [], [end] if end is not None else [])


def _token_states(model: Any, ids: Sequence[int], size: int) -> Any:
    """Return one representation per token, taken from its most contextual window.

    A token appearing in several windows is kept from the window where it sits furthest
    from an edge: that is the copy that saw context on both sides, which is the entire
    reason the document is embedded whole rather than in pieces.
    """
    import torch

    tokenizer = model.tokenizer
    network = model[0].auto_model
    device = next(network.parameters()).device

    width = int(network.config.hidden_size)
    states = torch.zeros(len(ids), width)
    centrality = torch.full((len(ids),), -1.0)

    prefix, suffix = _sentinels(tokenizer)

    for start, end in _windows(len(ids), size):
        window = list(ids[start:end])
        input_ids = torch.tensor([prefix + window + suffix], device=device)
        attention = torch.ones_like(input_ids)

        with torch.no_grad():
            hidden = network(input_ids=input_ids, attention_mask=attention).last_hidden_state[0]

        # Drop the sentinels so positions line up with the document's own token indices.
        body = hidden[len(prefix) : len(prefix) + len(window)].to("cpu")

        positions = torch.arange(len(window))
        edge_distance = torch.minimum(positions, len(window) - 1 - positions).float()

        better = edge_distance > centrality[start:end]
        states[start:end][better] = body[better]
        centrality[start:end][better] = edge_distance[better]

    return states


def pool_spans(model: Any, text: str, spans: Sequence[Segment]) -> list[list[float]]:
    """Return one pooled, normalized vector per span of ``text``.

    Args:
        model: A loaded local embedding model exposing ``tokenizer`` and a transformer.
        text: The whole document, embedded in one pass (or in overlapping windows).
        spans: Character ``(start, end)`` ranges, one per chunk, in document order.

    Returns:
        One unit-length vector per span, in the same order.

    Raises:
        EmbedError: If the model cannot supply token-level output, or the document
            produces no tokens at all.
    """
    if not supports_pooling(model):
        raise EmbedError(
            "late chunking needs token-level output, which this embedding model does not "
            "expose; use a local model or choose another chunking strategy",
            retryable=False,
        )
    if not spans:
        return []

    import torch

    tokenizer = model.tokenizer
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets: list[tuple[int, int]] = [tuple(pair) for pair in encoded["offset_mapping"]]
    ids: list[int] = list(encoded["input_ids"])

    if not ids:
        raise EmbedError("late chunking produced no tokens for this document", retryable=False)

    size = max(int(getattr(model, "max_seq_length", 512)) - _SPECIAL_TOKEN_BUDGET, 1)
    states = _token_states(model, ids, size)

    vectors: list[list[float]] = []
    for start, end in spans:
        selected = [
            index
            for index, (begin, finish) in enumerate(offsets)
            if finish > begin and begin < end and finish > start
        ]
        if not selected:
            # A span holding only whitespace or characters the tokenizer dropped. Its
            # chunk still needs a vector, and a zero one would rank against every query,
            # so it falls back to the document's own mean.
            selected = list(range(len(ids)))

        pooled = states[selected].mean(dim=0)
        vectors.append(torch.nn.functional.normalize(pooled, dim=0).tolist())

    return vectors
