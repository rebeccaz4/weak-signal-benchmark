"""Data loaders for GT and prediction signals."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
CONSTRUCTION_ROOT = REPO_ROOT / "construction" / "outputs"
PREDICTION_ROOT = REPO_ROOT / "prediction" / "python" / "outputs"
YEAR_SLUG = "2023_2024"
DIRECTIONS = ("problem", "solution")


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_gt_signals(
    domain_slug: str,
    topic_slug: str,
    direction: str,
    construction_root: Path = CONSTRUCTION_ROOT,
) -> list[str]:
    """Load GT weak signals (just the 'signal' text) from result_latest.json."""
    path = construction_root / domain_slug / topic_slug / direction / "result_latest.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ws["signal"] for ws in data["result"]["weak_signals"]]


def load_pred_signals(
    model: str,
    domain_slug: str,
    topic_slug: str,
    direction: str,
    prediction_root: Path = PREDICTION_ROOT,
    year_slug: str = YEAR_SLUG,
) -> list[str]:
    """Load predicted signals from signals_latest.json."""
    path = (
        prediction_root / model / domain_slug / topic_slug / direction / year_slug / "signals_latest.json"
    )
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("signals", [])


@dataclass
class EvalItem:
    model: str
    domain_slug: str
    topic_slug: str
    direction: str
    gt_signals: list[str]
    pred_signals: list[str]

    @property
    def key(self) -> str:
        return f"{self.model}/{self.domain_slug}/{self.topic_slug}/{self.direction}"

    def is_valid(self) -> bool:
        return len(self.gt_signals) > 0 and len(self.pred_signals) > 0


def iter_eval_items(
    models: list[str] | None = None,
    construction_root: Path = CONSTRUCTION_ROOT,
    prediction_root: Path = PREDICTION_ROOT,
) -> Iterator[EvalItem]:
    """Yield EvalItem for every (model, domain, topic, direction) combination
    where both GT and predictions exist."""
    available_models = models or [d.name for d in sorted(prediction_root.iterdir()) if d.is_dir()]

    for domain_dir in sorted(construction_root.iterdir()):
        if not domain_dir.is_dir():
            continue
        for topic_dir in sorted(domain_dir.iterdir()):
            if not topic_dir.is_dir():
                continue
            for direction in DIRECTIONS:
                gt = load_gt_signals(domain_dir.name, topic_dir.name, direction, construction_root)
                if not gt:
                    continue
                for model in available_models:
                    pred = load_pred_signals(
                        model, domain_dir.name, topic_dir.name, direction, prediction_root
                    )
                    yield EvalItem(
                        model=model,
                        domain_slug=domain_dir.name,
                        topic_slug=topic_dir.name,
                        direction=direction,
                        gt_signals=gt,
                        pred_signals=pred,
                    )
