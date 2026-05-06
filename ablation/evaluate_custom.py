#!/usr/bin/env python3
"""Custom ablation evaluation runner for the sampled benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from types import SimpleNamespace

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
AB_DIR = Path(__file__).resolve().parent
EVAL_DIR = REPO_ROOT / "evaluation"
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from loaders import CONSTRUCTION_ROOT  # type: ignore[attr-defined]

AB_OUTPUTS_ROOT = AB_DIR / "outputs"
RESULTS_ROOT = AB_DIR / "results" / "evaluation"
MASTER_CSV = RESULTS_ROOT / "eval_all.csv"
SUMMARY_MD = RESULTS_ROOT / "eval_summary.md"
TOPICS_JSON = AB_DIR / "data" / "topic_benchmark_tongyi_20.json"
YEAR_SLUG = "2023_2024"
DIRECTIONS = ("problem", "solution")
RAW_FIELDS = [
    "model",
    "domain",
    "topic",
    "direction",
    "setting",
    "precision",
    "recall",
    "f1",
    "precision_std",
    "recall_std",
    "f1_std",
    "n_pred",
    "n_gt",
    "n_runs",
]


def _load_env() -> None:
    env_files = [
        REPO_ROOT / ".env",
        REPO_ROOT / "prediction" / "python" / ".env",
        AB_DIR / ".env",
    ]
    for path in env_files:
        if path.exists():
            load_dotenv(path, override=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ablation outputs on the sampled-20 topics.")
    parser.add_argument("--settings", nargs="+", type=int, choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gpt-5.4"))
    parser.add_argument("--api-key", default=os.getenv("IKUNCODE_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("JUDGE_BASE_URL", "https://api.ikuncode.cc/v1"))
    parser.add_argument("--user-agent", default="Mozilla/5.0")
    parser.add_argument("--n-runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


def _load_topics(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("topics", [])


def _load_pred_signals(
    model: str,
    domain_slug: str,
    topic_slug: str,
    direction: str,
    prediction_root: Path = AB_OUTPUTS_ROOT,
    year_slug: str = YEAR_SLUG,
) -> list[str]:
    path = prediction_root / model / domain_slug / topic_slug / direction / year_slug / "signals_latest.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("signals", [])


@dataclass
class EvalItem:
    model: str
    domain_slug: str
    topic_slug: str
    direction: str
    gt_signals: list[str]
    pred_signals: list[str]

    def key(self) -> tuple[str, str, str, str, str]:
        settings = ("set_bertscore", "set_llm", "signal_bertscore", "signal_llm")
        return (self.model, self.domain_slug, self.topic_slug, self.direction, settings[0])


def iter_items(
    models: list[str] | None,
    topics: list[dict[str, str]],
    prediction_root: Path = AB_OUTPUTS_ROOT,
) -> Iterator[EvalItem]:
    available_models = models or sorted([p.name for p in prediction_root.iterdir() if p.is_dir()])
    for topic in topics:
        domain = topic["domain"]
        topic_slug = topic["topic"]
        for direction in DIRECTIONS:
            gt_path = CONSTRUCTION_ROOT / domain / topic_slug / direction / "result_latest.json"
            if not gt_path.exists():
                continue
            gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
            gt_signals = [entry["signal"] for entry in gt_data["result"]["weak_signals"]]
            if not gt_signals:
                continue
            for model in available_models:
                pred = _load_pred_signals(model, domain, topic_slug, direction)
                if not pred:
                    continue
                yield EvalItem(
                    model=model,
                    domain_slug=domain,
                    topic_slug=topic_slug,
                    direction=direction,
                    gt_signals=gt_signals,
                    pred_signals=pred,
                )


def _result_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (row["model"], row["domain"], row["topic"], row["direction"], row["setting"])


def _load_existing_keys(path: Path) -> set[tuple[str, str, str, str, str]]:
    if not path.exists():
        return set()
    keys = set()
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            keys.add(_result_key(row))
    return keys


def evaluate_row(
    setting: int,
    gt_signals: list[str],
    pred_signals: list[str],
    llm_client: object | None,
    args: argparse.Namespace,
    eval_modules: dict[str, object],
) -> dict[str, float | int]:
    if setting == 1:
        return eval_modules["bertscore"].eval_set_bertscore(gt_signals, pred_signals)
    if setting == 2:
        return eval_modules["llm_set"].eval_set_llm(
            gt_signals,
            pred_signals,
            client=llm_client,
            judge_model=args.judge_model,
            n_runs=args.n_runs,
        )
    if setting == 3:
        return eval_modules["bertscore"].eval_signal_bertscore(gt_signals, pred_signals)
    if setting == 4:
        return eval_modules["llm_signal"].eval_signal_llm(
            gt_signals,
            pred_signals,
            api_key=args.api_key,
            base_url=args.base_url,
            judge_model=args.judge_model,
            user_agent=args.user_agent,
            n_runs=args.n_runs,
        )
    raise ValueError("Unsupported setting")


def summarize_csv(csv_path: Path) -> str:
    grouped: dict[tuple[str, str], list[dict[str, float]]] = {}
    with csv_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row["model"], row["setting"])
            entry = grouped.setdefault(key, [])
            entry.append({field: float(row[field]) for field in ("precision", "recall", "f1")})
    lines = ["# Ablation Evaluation Summary", "", "| Model | Setting | N | P | R | F1 |", "|---|---|---:|---:|---:|---:|"]
    for (model, setting), rows in sorted(grouped.items()):
        n = len(rows)
        p = sum(r["precision"] for r in rows) / n
        r = sum(r["recall"] for r in rows) / n
        f = sum(r["f1"] for r in rows) / n
        lines.append(f"| `{model}` | `{setting}` | {n} | {p:.4f} | {r:.4f} | {f:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    _load_env()
    args = parse_args()
    need_llm = 2 in args.settings or 4 in args.settings
    if need_llm and not args.api_key:
        raise SystemExit("LLM settings require --api-key or IKUNCODE_API_KEY.")

    topics = _load_topics(TOPICS_JSON)
    models = args.models or sorted([p.name for p in AB_OUTPUTS_ROOT.iterdir() if p.is_dir()])

    eval_modules: dict[str, object] = {}
    if 1 in args.settings or 3 in args.settings:
        from bertscore_eval import eval_set_bertscore, eval_signal_bertscore  # noqa: E402

        eval_modules["bertscore"] = SimpleNamespace(
            eval_set_bertscore=eval_set_bertscore,
            eval_signal_bertscore=eval_signal_bertscore,
        )
    if 2 in args.settings:
        from llm_set_eval import eval_set_llm  # noqa: E402
        from openai import OpenAI  # noqa: E402

        eval_modules["llm_set"] = SimpleNamespace(eval_set_llm=eval_set_llm)
        llm_client = OpenAI(
            api_key=args.api_key,
            base_url=args.base_url,
            default_headers={"User-Agent": args.user_agent},
        )
    else:
        llm_client = None
    if 4 in args.settings:
        from llm_signal_eval import eval_signal_llm  # noqa: E402

        eval_modules["llm_signal"] = SimpleNamespace(eval_signal_llm=eval_signal_llm)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    if MASTER_CSV.exists():
        existing_keys = _load_existing_keys(MASTER_CSV) if args.skip_existing else set()
    else:
        existing_keys = set()
    is_new = not MASTER_CSV.exists() or MASTER_CSV.stat().st_size == 0
    fh = MASTER_CSV.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=RAW_FIELDS)
    if is_new:
        writer.writeheader()

    setting_names = {1: "set_bertscore", 2: "set_llm", 3: "signal_bertscore", 4: "signal_llm"}
    total = 0
    skipped = 0
    failed = 0

    for item in iter_items(models=models, topics=topics):
        for setting in args.settings:
            key = (item.model, item.domain_slug, item.topic_slug, item.direction, setting_names[setting])
            if args.skip_existing and key in existing_keys:
                print("  [jump existing]", key)
                skipped += 1
                continue
            try:
                result = evaluate_row(
                    setting,
                    item.gt_signals,
                    item.pred_signals,
                    llm_client,
                    args,
                    eval_modules,
                )
            except Exception as exc:
                failed += 1
                print("  [error]", key, exc, file=sys.stderr)
                continue
            row = {
                "model": item.model,
                "domain": item.domain_slug,
                "topic": item.topic_slug,
                "direction": item.direction,
                "setting": setting_names[setting],
                "n_pred": len(item.pred_signals),
                "n_gt": len(item.gt_signals),
                "n_runs": result.get("n_runs", args.n_runs),
                "precision": result.get("precision", ""),
                "recall": result.get("recall", ""),
                "f1": result.get("f1", ""),
                "precision_std": result.get("precision_std", ""),
                "recall_std": result.get("recall_std", ""),
                "f1_std": result.get("f1_std", ""),
            }
            writer.writerow(row)
            existing_keys.add(key)
            total += 1
    fh.close()

    summary = summarize_csv(MASTER_CSV)
    SUMMARY_MD.write_text(summary, encoding="utf-8")
    print(f"Wrote {MASTER_CSV} (+{total} new rows, skipped {skipped}, failed {failed}).")
    print(f"Wrote summary: {SUMMARY_MD}")


if __name__ == "__main__":
    main()
