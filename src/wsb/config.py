"""Project-level configuration and path defaults."""

from __future__ import annotations

import os
from pathlib import Path


def _find_project_root() -> Path:
    """Walk up from CWD to find the project root (contains README.md)."""
    return next(
        (p for p in Path.cwd().resolve().parents if (p / "README.md").exists()),
        Path.cwd().resolve(),
    )


PROJECT_ROOT = _find_project_root()
EVAL_DIR = PROJECT_ROOT / "Evaluations"

DEFAULT_JUDGE_MODEL = os.getenv("BWD_LLM_JUDGE_MODEL", "gpt-5-mini")
DEFAULT_TEMPERATURE = 1.0
DEFAULT_N_RUNS = 10
DEFAULT_SEED = 27

# Gemini (Google) OpenAI-compatible endpoint
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
