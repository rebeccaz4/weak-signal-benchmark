"""Signal loading from JSON files."""

from __future__ import annotations

import json
from pathlib import Path


def load_signals(path: str | Path) -> list[str]:
    """Load a JSON array of signal strings from *path*.

    Raises FileNotFoundError if the file does not exist, and ValueError
    if the content is not a list of strings.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Signal file not found: {path}")
    with open(path) as f:
        signals = json.load(f)
    if not isinstance(signals, list) or not all(isinstance(s, str) for s in signals):
        raise ValueError(f"Signal file must contain a JSON list of strings: {path}")
    return signals
