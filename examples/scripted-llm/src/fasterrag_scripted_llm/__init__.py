"""A deterministic stand-in LLM, registered as ``scripted``, for running examples offline."""

from fasterrag_scripted_llm.adapter import ScriptedLLM, scripted_answer

__all__ = ["ScriptedLLM", "scripted_answer"]
