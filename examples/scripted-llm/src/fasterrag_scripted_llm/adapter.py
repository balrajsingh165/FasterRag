"""An extractive "LLM" that answers from the context it is given, with no model and no network.

The examples' generation scripts need an answer to print, and CI has neither an API key nor a
running Ollama. This adapter reads the P1 prompt fasterRag builds (``docs/prompts.md``): each
context passage starts with its citation marker, ``[^chunk-id]``, on a line of its own. It
answers with the first sentence of the first passage and cites that passage, so the answer,
its citation and its trace are all real outputs of the pipeline; only the wording is not a
model's. It is selected by configuration like any other provider; nothing in fasterRag knows
it exists.

It also grades. Called with the P3 system prompt (``docs/prompts.md``), it scores the answer
the way the contract asks, deterministically: each sentence of the answer is a claim, and a
claim is supported when its words occur, in order and contiguously, in the context passages.
That makes ``fasterrag benchmark --suite eval --faithfulness`` a reproducible gate in CI: a
change that stops the right context reaching the prompt makes the scripted answer
unsupported, and the score falls. It measures the pipeline, never a model's judgement.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from typing import ClassVar, Final

from fasterrag.adapters.llm.base import Completion, LLMAdapter
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.core.faithfulness import P3_SYSTEM_PROMPT

__all__ = ["NO_CONTEXT_ANSWER", "ScriptedLLM", "scripted_answer", "scripted_verdict"]

NO_CONTEXT_ANSWER: Final = "The provided context does not contain an answer."

_MARKER: Final = re.compile(r"^\[\^([A-Za-z0-9_\-]+)\]")
_SENTENCE: Final = re.compile(r"(.+?[.!?])(?:\s|$)", re.S)
_SECTION: Final = re.compile(r"<(context|answer)-([0-9a-f]+)>\n(.*?)\n</\1-\2>", re.S)
_CITATION: Final = re.compile(r"\[[^\]]*\]")
_WORD: Final = re.compile(r"[a-z0-9]+")


def _words(text: str) -> str:
    """Return ``text`` as its lower-cased words, space-separated and space-padded."""
    return f" {' '.join(_WORD.findall(_CITATION.sub(' ', text).lower()))} "


def scripted_verdict(prompt: str) -> str:
    """Grade a P3 prompt: the share of the answer's sentences found verbatim in the context."""
    sections = {match.group(1): match.group(3) for match in _SECTION.finditer(prompt)}
    passages = [
        line
        for line in sections.get("context", "").splitlines()
        if not _MARKER.match(line) and not line.lstrip().startswith("#")
    ]
    context = _words(" ".join(passages))
    answer = " ".join(sections.get("answer", "").split())
    claims = [
        claim
        for claim in (match.group(1) for match in _SENTENCE.finditer(f"{answer} "))
        if _words(claim).strip()
    ]
    if answer == NO_CONTEXT_ANSWER or not claims:
        unsupported: list[str] = []
        score = 1.0
    else:
        unsupported = [claim for claim in claims if _words(claim) not in context]
        score = 1 - len(unsupported) / len(claims)
    return json.dumps(
        {
            "score": round(score, 4),
            "unsupported_claims": unsupported,
            "reasoning": "scripted: a claim is supported when its words occur in the context",
        }
    )


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
        """Return the scripted answer whole, or a scripted P3 verdict when asked to grade."""
        text = scripted_verdict(prompt) if system == P3_SYSTEM_PROMPT else scripted_answer(prompt)
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
