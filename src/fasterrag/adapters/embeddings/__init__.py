"""Embedding provider adapters, the factory, and the tiered-embedding router."""

from fasterrag.adapters.embeddings.base import EmbeddingAdapter, EmbeddingResult
from fasterrag.adapters.embeddings.cohere import CohereEmbedder
from fasterrag.adapters.embeddings.factory import (
    ENTRY_POINT_GROUP,
    available_providers,
    create_embedding_adapter,
    resolve_adapter_class,
)
from fasterrag.adapters.embeddings.huggingface import HuggingFaceEmbedder
from fasterrag.adapters.embeddings.ollama import OllamaEmbedder
from fasterrag.adapters.embeddings.openai import OpenAIEmbedder
from fasterrag.adapters.embeddings.sync import SentenceEmbedderBridge
from fasterrag.adapters.embeddings.tiering import (
    TieringRouter,
    create_embedding_router,
    matches,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "CohereEmbedder",
    "EmbeddingAdapter",
    "EmbeddingResult",
    "HuggingFaceEmbedder",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "SentenceEmbedderBridge",
    "TieringRouter",
    "available_providers",
    "create_embedding_adapter",
    "create_embedding_router",
    "matches",
    "resolve_adapter_class",
]
