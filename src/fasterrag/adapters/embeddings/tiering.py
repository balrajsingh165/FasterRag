"""Tiered embedding: route document classes to different models.

High-volume, low-priority material embeds with a cheap model while material where
retrieval precision matters gets an expensive one, which is one of the levers that makes
embedding cost controllable (D9, ``docs/integrations.md`` §2).

Rules are ordered and the first match wins, so a specific rule placed above a general one
takes effect. Matching reuses the same filter grammar the vector database adapters accept,
so operators mean the same thing everywhere rather than differing per subsystem.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from fasterrag.adapters.embeddings.base import EmbeddingAdapter
from fasterrag.adapters.embeddings.factory import create_embedding_adapter
from fasterrag.adapters.vectordb.base import Filter, validate_filter
from fasterrag.config.schema import Settings, TierRule
from fasterrag.observability.logging import get_logger

__all__ = ["TieringRouter", "create_embedding_router", "matches"]

_logger = get_logger(__name__)


def _compare(operator: str, actual: Any, expected: Any) -> bool:
    """Evaluate one filter operator against a metadata value."""
    if operator == "$eq":
        return bool(actual == expected)
    if operator == "$ne":
        return bool(actual != expected)
    if operator == "$in":
        return actual in expected
    if operator == "$nin":
        return actual not in expected

    if actual is None or isinstance(actual, bool) or not isinstance(actual, int | float):
        return False
    if operator == "$gt":
        return bool(actual > expected)
    if operator == "$gte":
        return bool(actual >= expected)
    if operator == "$lt":
        return bool(actual < expected)
    return bool(actual <= expected)


def matches(filters: Filter, metadata: Mapping[str, Any]) -> bool:
    """Return whether ``metadata`` satisfies every condition in ``filters``.

    An absent key never matches: a rule can only route material it can actually
    identify, so unlabelled documents fall through to the default model.
    """
    for key, condition in filters.items():
        if key not in metadata:
            return False

        actual = metadata[key]
        if not isinstance(condition, Mapping):
            if actual != condition:
                return False
            continue

        operator = next(iter(condition))
        if not _compare(operator, actual, condition[operator]):
            return False

    return True


class TieringRouter:
    """Chooses the embedding adapter for a document, by its metadata."""

    def __init__(
        self,
        default: EmbeddingAdapter,
        tiers: Sequence[tuple[Filter, EmbeddingAdapter]] = (),
    ) -> None:
        """Build the router.

        Args:
            default: Adapter used when no rule matches.
            tiers: Ordered ``(filter, adapter)`` pairs; the first match wins.
        """
        self.default = default
        self.tiers = list(tiers)

    @property
    def enabled(self) -> bool:
        """Return whether any routing rule is configured."""
        return bool(self.tiers)

    def select(self, metadata: Mapping[str, Any] | None = None) -> EmbeddingAdapter:
        """Return the adapter that should embed a document with this metadata."""
        if metadata:
            for filters, adapter in self.tiers:
                if matches(filters, metadata):
                    return adapter
        return self.default

    def adapters(self) -> list[EmbeddingAdapter]:
        """Return every adapter the router owns, the default first."""
        seen: list[EmbeddingAdapter] = [self.default]
        for _, adapter in self.tiers:
            if adapter not in seen:
                seen.append(adapter)
        return seen

    async def close(self) -> None:
        """Close every adapter the router owns."""
        for adapter in self.adapters():
            await adapter.close()


def _settings_for(settings: Settings, rule: TierRule) -> Settings:
    """Return settings with the embedding provider and model a rule selects."""
    embeddings = settings.embeddings.model_copy(
        update={"provider": rule.provider, "model": rule.model}
    )
    return settings.model_copy(update={"embeddings": embeddings})


def create_embedding_router(settings: Settings) -> TieringRouter:
    """Build the router described by ``embeddings.tiering``.

    Returns:
        A router holding one adapter per distinct rule plus the default. When tiering is
        disabled the router simply always returns the default adapter, so callers need no
        conditional.

    Raises:
        ConfigError: If a rule names an unresolvable provider, or a rule filter uses an
            unsupported operator.
    """
    default = create_embedding_adapter(settings)
    if not settings.embeddings.tiering.enabled:
        return TieringRouter(default)

    tiers: list[tuple[Filter, EmbeddingAdapter]] = []
    for rule in settings.embeddings.tiering.rules:
        validate_filter(rule.match)
        tiers.append((rule.match, create_embedding_adapter(_settings_for(settings, rule))))

    _logger.info(
        "tiered embedding enabled",
        extra={"rules": len(tiers), "default_model": default.model},
    )
    return TieringRouter(default, tiers)
