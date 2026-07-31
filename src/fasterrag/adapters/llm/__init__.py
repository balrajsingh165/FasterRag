"""LLM provider adapters and the factory that selects one from configuration."""

from fasterrag.adapters.llm.anthropic import AnthropicGenerator
from fasterrag.adapters.llm.base import Completion, LLMAdapter
from fasterrag.adapters.llm.cohere import CohereGenerator
from fasterrag.adapters.llm.factory import (
    ENTRY_POINT_GROUP,
    available_providers,
    create_llm_adapter,
    resolve_adapter_class,
)
from fasterrag.adapters.llm.ollama import OllamaGenerator
from fasterrag.adapters.llm.openai import OpenAICompatibleGenerator, OpenAIGenerator

__all__ = [
    "ENTRY_POINT_GROUP",
    "AnthropicGenerator",
    "CohereGenerator",
    "Completion",
    "LLMAdapter",
    "OllamaGenerator",
    "OpenAICompatibleGenerator",
    "OpenAIGenerator",
    "available_providers",
    "create_llm_adapter",
    "resolve_adapter_class",
]
