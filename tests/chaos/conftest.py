"""Shared fixtures for the chaos suite (D12).

Every scenario here injects a *real* fault and asserts the documented behavior — a suite that
mocked the failure would prove only that the mock behaves as written.

Scenarios that need a live backend are marked ``integration`` so they run in the job that
has one; those that inject faults purely inside the pipeline run everywhere, because a
reliability guarantee nobody can re-run on a laptop is not one anybody checks.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fasterrag.config.schema import Settings
from fasterrag.services.journal import Journal


@pytest.fixture
def chaos_settings(tmp_path: Path) -> Settings:
    """Return settings pointed at a throwaway workspace with retries turned down.

    Retries are reduced so a scenario asserting dead-lettering does not spend a minute in
    exponential backoff proving something the first attempt already showed.
    """
    return Settings.model_validate(
        {
            "vector_db": {"provider": "qdrant", "mode": "external", "api_key_env": None},
            "embeddings": {"provider": "huggingface"},
            "llm": {"provider": "ollama", "api_key_env": None},
            "security": {"api_key_env": "FASTERRAG_API_KEY"},
            "reliability": {"retries": {"max_attempts": 2, "backoff_base_ms": 1}},
        }
    )


@pytest.fixture
def chaos_journal(tmp_path: Path) -> Journal:
    """Return a journal rooted in a temporary directory."""
    return Journal(tmp_path / "journal")
