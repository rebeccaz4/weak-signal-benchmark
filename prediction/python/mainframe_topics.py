"""
Mainframe topics loaded from weak_signals_by_domain.json.

Provides:
    TOPICS_BY_DOMAIN  – dict[domain_name, list[topic_name]]
    ALL_DOMAINS       – list of all domain names
    make_domain_slug  – convert domain name to filesystem-safe slug
"""
import json
import re
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_JSON_PATH = _SCRIPT_DIR / "../../construction/weak_signals_by_domain.json"


def _load() -> dict[str, list[str]]:
    with open(_JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


TOPICS_BY_DOMAIN: dict[str, list[str]] = _load()
ALL_DOMAINS: list[str] = list(TOPICS_BY_DOMAIN.keys())


def make_domain_slug(domain: str) -> str:
    """Convert a domain name to a filesystem-safe slug."""
    return re.sub(r"[^a-z0-9]+", "_", domain.lower()).strip("_")
