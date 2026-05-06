"""Evaluate ablation prediction variants with the original benchmark metrics."""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

from .common import CONSTRUCTION_ROOT, DATA_ROOT, EVALUATION_ROOT, OUTPUTS_ROOT, RESULTS_ROOT, YEAR_SLUG, import_module_from_path, read_json

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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ablation evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate ablation variant outputs with original metrics.")
    parser.add_argument("--settings", nargs="+", type=int, choices=[1, 2, 3, 4], default=[1, 2, 3, 4])
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--benchmark-json", type=Path, default=DATA_ROOT / "topic_benchmark_50.json")
    parser.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gpt-5.4"))
    parser.add_argument("--api-key", default=os.getenv("IKUNCODE_API_KEY"))
    parser.add_argument("--base-url", default=os.getenv("JUDGE_BASE_URL", "https://api.ikuncode.cc/v1"))
    parser.add_argument("--user-agent", default="Mozilla/5.0")
    parser.add_argument("--n-runs", type=int, default=3)
    return parser.parse_args()


def load_gt(domain_slug: str, topic_slug: str, direction: str) -> list[str]:
    """Load GT signals from construction outputs."""
    payload = read_json(CONSTRUCTION_ROOT / domain_slug / topic_slug / direction / "result_latest.json")
    return [item["signal"] for item in payload["result"]["weak_signals"]]


def load_pred(model: str, domain_slug: str, topic_slug: str, direction: str) -> list[str]:
    """Load ablation prediction signals from local outputs."""
    payload = read_json(OUTPUTS_ROOT / model / domain_slug / topic_slug / direction / YEAR_SLUG / "signals_latest.json")
    return payload.get("signals", [])


def available_models() -> list[str]:
    """List variant models currently present under `ablation/outputs`."""
    return sorted(path.name for path in OUTPUTS_ROOT.iterdir() if path.is_dir())


def evaluate_row(
    setting: int,
    gt_signals: list[str],
    pred_signals: list[str],
    llm_client: OpenAI | None,
    args: argparse.Namespace,
    eval_modules: dict[str, object],
) -> dict:
    """Run a single evaluation setting on one topic-direction pair."""
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
    raise ValueError(f"Unsupported setting: {setting}")


def summarize_csv(csv_path: Path) -> str:
    """Return a compact markdown summary of mean scores by model and setting."""
    grouped: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["model"], row["setting"])].append(
                {
                    "precision": float(row["precision"]),
                    "recall": float(row["recall"]),
                    "f1": float(row["f1"]),
                }
            )
    lines = ["# Budget Evaluation Summary", "", "| Model | Setting | N | P | R | F1 |", "|---|---|---:|---:|---:|---:|"]
    for (model, setting), rows in sorted(grouped.items()):
        n_rows = len(rows)
        precision = sum(item["precision"] for item in rows) / n_rows
        recall = sum(item["recall"] for item in rows) / n_rows
        f1 = sum(item["f1"] for item in rows) / n_rows
        lines.append(f"| `{model}` | `{setting}` | {n_rows} | {precision:.4f} | {recall:.4f} | {f1:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    benchmark = read_json(args.benchmark_json)
    models = args.models or available_models()

    if any(setting in {2, 4} for setting in args.settings) and not args.api_key:
        raise SystemExit("LLM evaluation requested but IKUNCODE_API_KEY / --api-key is missing.")

    eval_modules = {}
    if 1 in args.settings or 3 in args.settings:
        eval_modules["bertscore"] = import_module_from_path(
            "ablation_bertscore_eval",
            EVALUATION_ROOT / "bertscore_eval.py",
        )
    if 2 in args.settings:
        eval_modules["llm_set"] = import_module_from_path(
            "ablation_llm_set_eval",
            EVALUATION_ROOT / "llm_set_eval.py",
        )
    if 4 in args.settings:
        eval_modules["llm_signal"] = import_module_from_path(
            "ablation_llm_signal_eval",
            EVALUATION_ROOT / "llm_signal_eval.py",
        )
    llm_client = None
    if 2 in args.settings:
        from openai import OpenAI

        llm_client = OpenAI(
            api_key=args.api_key,
            base_url=args.base_url,
            default_headers={"User-Agent": args.user_agent},
        )

    setting_names = {1: "set_bertscore", 2: "set_llm", 3: "signal_bertscore", 4: "signal_llm"}
    out_path = RESULTS_ROOT / "budget_eval_all.csv"
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for item in benchmark["topics"]:
            domain_slug = item["domain"]
            topic_slug = item["topic"]
            for direction in ("problem", "solution"):
                gt_signals = load_gt(domain_slug, topic_slug, direction)
                for model in models:
                    pred_path = OUTPUTS_ROOT / model / domain_slug / topic_slug / direction / YEAR_SLUG / "signals_latest.json"
                    if not pred_path.exists():
                        continue
                    pred_signals = load_pred(model, domain_slug, topic_slug, direction)
                    for setting in args.settings:
                        result = evaluate_row(setting, gt_signals, pred_signals, llm_client, args, eval_modules)
                        row = {
                            "model": model,
                            "domain": domain_slug,
                            "topic": topic_slug,
                            "direction": direction,
                            "setting": setting_names[setting],
                            **{field: result.get(field) for field in RAW_FIELDS if field not in {"model", "domain", "topic", "direction", "setting"}},
                        }
                        writer.writerow(row)

    summary_path = RESULTS_ROOT / "budget_eval_summary.md"
    summary_path.write_text(summarize_csv(out_path), encoding="utf-8")
    print(f"Wrote evaluation CSV: {out_path}")
    print(f"Wrote evaluation summary: {summary_path}")


if __name__ == "__main__":
    main()
