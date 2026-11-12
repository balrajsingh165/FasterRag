"""An extractive "LLM" that answers from the context it is given, with no model and no network.

The examples' generation scripts need an answer to print, and CI has neither an API key nor a
running Ollama. This adapter reads the P1 prompt fasterRag builds (``docs/prompts.md``): each
context passage starts with its citation marker, ``[^chunk-id]``, on a line of its own. It
answers with the first sentence of the first passage and cites that passage, so the answer,
its citation and its trace are all real outputs of the pipeline; only the wording is not a
model's. It is selected by configuration like any other provider; nothing in fasterRag knows
it exists.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from typing import ClassVar, Final

from fasterrag.adapters.llm.base import Completion, LLMAdapter
from fasterrag.adapters.vectordb.base import HealthStatus

__all__ = ["NO_CONTEXT_ANSWER", "ScriptedLLM", "scripted_answer"]

NO_CONTEXT_ANSWER: Final = "The provided context does not contain an answer."

_MARKER: Final = re.compile(r"^\[\^([A-Za-z0-9_\-]+)\]")
_SENTENCE: Final = re.compile(r"(.+?[.!?])(?:\s|$)", re.S)


def scripted_answer(prompt: str) -> str:
    """Return the first sentence of the first context passage, citing it."""
    chunk_id: str | None = None
    prose: list[str] = []
    for line in prompt.splitlines():
        marker = _MARKER.match(line)
        if marker is not None:
            if chunk_id is not None:
                break
            chunk_id = marker.group(1)
        elif line.startswith("</context-"):
            break
        elif chunk_id is not None and not line.lstrip().startswith("#"):
            prose.append(line)

    text = " ".join(" ".join(prose).split())
    if chunk_id is None or not text:
        return NO_CONTEXT_ANSWER
    sentence = _SENTENCE.match(text)
    return f"{(sentence.group(1) if sentence else text).strip()} [^{chunk_id}]"


class ScriptedLLM(LLMAdapter):
    """Answers extractively from the first context passage of the prompt."""

    provider: ClassVar[str] = "scripted"
    adapter_api_version: ClassVar[str] = "1.0"

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        """Return the scripted answer whole."""
        text = scripted_answer(prompt)
        return Completion(
            text=text,
            model=self.model,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            finish_reason="stop",
        )

    async def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        """Yield the scripted answer a word at a time, as a model streams tokens."""
        words = scripted_answer(prompt).split(" ")
        for position, word in enumerate(words):
            yield word if position == 0 else f" {word}"

    async def health(self) -> HealthStatus:
        """Report healthy: there is nothing to reach."""
        return HealthStatus(healthy=True, detail="scripted, offline", latency_ms=0.0)

    async def close(self) -> None:
        """Release nothing."""
