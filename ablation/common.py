"""Shared utilities for weak-signal ablation experiments."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ABLATION_ROOT = REPO_ROOT / "ablation"
DATA_ROOT = ABLATION_ROOT / "data"
OUTPUTS_ROOT = ABLATION_ROOT / "outputs"
RESULTS_ROOT = ABLATION_ROOT / "results"

CONSTRUCTION_ROOT = REPO_ROOT / "construction" / "outputs"
PREDICTION_ROOT = REPO_ROOT / "prediction" / "python" / "outputs"
PREDICTION_PY_ROOT = REPO_ROOT / "prediction" / "python"
EVALUATION_ROOT = REPO_ROOT / "evaluation"
EVALUATION_RESULTS_ROOT = EVALUATION_ROOT / "results"
SRC_ROOT = REPO_ROOT / "src"

DEFAULT_MODELS = [
    "deepseek_r1_0528",
    "dr_tulu",
    "gpt_5_3_chat",
    "qwen3.5_397b",
    "qwen3_30b_awq_rag",
    "qwen3_8b_rag",
    "tongyi",
]
DEFAULT_DIRECTIONS = ["problem", "solution"]
YEAR_SLUG = "2023_2024"


def ensure_repo_layout() -> None:
    """Validate that the benchmark folders expected by ablation scripts exist."""
    required = [
        CONSTRUCTION_ROOT,
        PREDICTION_PY_ROOT,
        EVALUATION_ROOT,
        EVALUATION_RESULTS_ROOT,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required benchmark paths: {missing}")


def ensure_import_paths() -> None:
    """Add original benchmark module roots to `sys.path` if needed."""
    for path in (PREDICTION_PY_ROOT, EVALUATION_ROOT, SRC_ROOT):
        raw = str(path)
        if raw not in sys.path:
            sys.path.insert(0, raw)


def import_module_from_path(module_name: str, file_path: Path):
    """Import a Python module from an explicit file path."""
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def slugify(value: str) -> str:
    """Convert a string to a lowercase filesystem-friendly slug."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def utc_now_tag() -> str:
    """Return current UTC timestamp in compact tag format."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> Any:
    """Read a UTF-8 JSON file and return the parsed object."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    """Write a JSON payload with UTF-8 and stable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize_title(title: str) -> str:
    """Normalize a paper or signal title for string comparisons."""
    normalized = re.sub(r"\s+", " ", title.lower()).strip()
    normalized = re.sub(r"[^a-z0-9 ]+", "", normalized)
    return normalized


ensure_repo_layout()
ensure_import_paths()
