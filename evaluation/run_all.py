#!/usr/bin/env python3
"""Main evaluation runner.

Iterates over all (model, domain, topic, direction) combinations and runs
the selected evaluation settings, saving results to evaluation/results/.

Usage:
    python evaluation/run_all.py --settings 1 3             # BERTScore only (no API needed)
    python evaluation/run_all.py --settings 2 4             # LLM only
    python evaluation/run_all.py --settings 1 2 3 4         # all settings
    python evaluation/run_all.py --models qwen3.5_397b gpt_5_3_chat --settings 1 3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Ensure evaluation/ is importable
_EVAL_DIR = Path(__file__).resolve().parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from loaders import CONSTRUCTION_ROOT, PREDICTION_ROOT, iter_eval_items, slugify

REPO_ROOT = _EVAL_DIR.parent
RESULTS_DIR = _EVAL_DIR / "results"
DIRECTIONS = ("problem", "solution")


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

RAW_FIELDS = [
    "model", "domain", "topic", "direction", "setting",
    "precision", "recall", "f1",
    "precision_std", "recall_std", "f1_std",
    "n_pred", "n_gt", "n_runs",
]


def _open_csv(path: Path) -> tuple:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=RAW_FIELDS, extrasaction="ignore")
    writer.writeheader()
    return fh, writer


def _open_master_csv(path: Path) -> tuple:
    """Open the master CSV in append mode; write header only if file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=RAW_FIELDS, extrasaction="ignore")
    if is_new:
        writer.writeheader()
    return fh, writer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_eval_env() -> None:
    env_candidates = [
        REPO_ROOT / ".env",
        REPO_ROOT / "prediction" / "python" / ".env",
    ]
    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(env_path, override=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all evaluation settings.")
    p.add_argument("--settings", nargs="+", type=int, choices=[1, 2, 3, 4], default=[1, 2, 3, 4],
                   help="Which settings to run (1=set-BERTScore, 2=set-LLM, 3=signal-BERTScore, 4=signal-LLM)")
    p.add_argument("--models", nargs="*", default=None,
                   help="Models to evaluate (default: all in prediction/python/outputs/)")
    p.add_argument("--domains", nargs="*", default=None,
                   help="Domains to filter by slug or display name, e.g. aerospace or 'Mobility and Transport'")
    p.add_argument("--directions", nargs="*", default=["problem", "solution"],
                   choices=["problem", "solution"])
    p.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL", "gpt-5.4"),
                   help="LLM model for settings 2 & 4 (default: gpt-5.4)")
    p.add_argument("--api-key", default=os.getenv("IKUNCODE_API_KEY"),
                   help="API key for LLM judge (default: $IKUNCODE_API_KEY)")
    p.add_argument("--base-url", default=os.getenv("JUDGE_BASE_URL", "https://api.ikuncode.cc/v1"),
                   help="Base URL for LLM judge API")
    p.add_argument("--user-agent", default="Mozilla/5.0")
    p.add_argument("--n-runs", type=int, default=3,
                   help="Number of LLM judge runs for settings 2 & 4 (default: 5)")
    p.add_argument("--skip-existing", dest="skip_existing", action="store_true", default=True,
                   help="Skip rows already present in evaluation/results/eval_all.csv (default: enabled)")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false",
                   help="Recompute rows even if they already exist in evaluation/results/eval_all.csv")
    p.add_argument("--output-tag", default="",
                   help="Optional tag appended to output filename")
    return p.parse_args()


def _result_key(row: dict) -> tuple:
    return (row["model"], row["domain"], row["topic"], row["direction"], row["setting"])


def _load_existing_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    keys = set()
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            keys.add(_result_key(row))
    return keys


def _domain_results_dir(domain_slug: str) -> Path:
    return RESULTS_DIR / domain_slug


def _domain_output_path(domain_slug: str, timestamp: str, tag: str) -> Path:
    suffix = f"_{tag}" if tag else ""
    return _domain_results_dir(domain_slug) / f"eval_{timestamp}{suffix}.csv"


def _domain_master_path(domain_slug: str) -> Path:
    return _domain_results_dir(domain_slug) / "eval_all.csv"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _load_eval_env()
    args = parse_args()
    requested_domains = {slugify(domain) for domain in args.domains} if args.domains else None

    need_llm = 2 in args.settings or 4 in args.settings
    if need_llm and not args.api_key:
        raise SystemExit("Error: --api-key or $IKUNCODE_API_KEY required for settings 2 and 4.")

    # Lazy imports — only load heavy libs when needed
    if 1 in args.settings or 3 in args.settings:
        from bertscore_eval import eval_set_bertscore, eval_signal_bertscore

    if 2 in args.settings:
        from llm_set_eval import eval_set_llm
        from openai import OpenAI
        llm_client = OpenAI(
            api_key=args.api_key,
            base_url=args.base_url,
            default_headers={"User-Agent": args.user_agent},
        )

    if 4 in args.settings:
        from llm_signal_eval import eval_signal_llm

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    setting_names = {1: "set_bertscore", 2: "set_llm", 3: "signal_bertscore", 4: "signal_llm"}

    total, skipped, errors = 0, 0, 0
    domain_existing_keys: dict[str, set[tuple]] = {}
    domain_outputs: dict[str, dict] = {}

    if args.skip_existing:
        print("Existing-result skipping enabled; each domain loads keys from results/<domain>/eval_all.csv.")
    else:
        print("Existing-result skipping disabled; recomputing all requested rows.")
    if requested_domains:
        print(f"Requested domains (normalized): {', '.join(sorted(requested_domains))}")

    try:
        for item in iter_eval_items(models=args.models):
            if item.direction not in args.directions:
                continue
            if requested_domains and item.domain_slug not in requested_domains:
                continue
            if not item.is_valid():
                continue

            base_row = {
                "model": item.model,
                "domain": item.domain_slug,
                "topic": item.topic_slug,
                "direction": item.direction,
                "n_pred": len(item.pred_signals),
                "n_gt": len(item.gt_signals),
            }

            print(f"\n{'─'*60}")
            print(f"  {item.key}  (GT={len(item.gt_signals)}, pred={len(item.pred_signals)})")

            if item.domain_slug not in domain_existing_keys:
                master_path = _domain_master_path(item.domain_slug)
                keys = _load_existing_keys(master_path) if args.skip_existing else set()
                domain_existing_keys[item.domain_slug] = keys
                if args.skip_existing:
                    print(f"  loaded {len(keys)} existing keys from {master_path}")

            for s in args.settings:
                sname = setting_names[s]
                row_key = (item.model, item.domain_slug, item.topic_slug, item.direction, sname)
                existing_keys = domain_existing_keys[item.domain_slug]
                if args.skip_existing and row_key in existing_keys:
                    skipped += 1
                    print(f"    [{sname}] skip")
                    continue

                try:
                    print(f"    [{sname}] running...")
                    if s == 1:
                        result = eval_set_bertscore(item.gt_signals, item.pred_signals)
                    elif s == 2:
                        result = eval_set_llm(
                            item.gt_signals, item.pred_signals,
                            client=llm_client,
                            judge_model=args.judge_model,
                            n_runs=args.n_runs,
                        )
                    elif s == 3:
                        result = eval_signal_bertscore(item.gt_signals, item.pred_signals)
                    elif s == 4:
                        result = eval_signal_llm(
                            item.gt_signals, item.pred_signals,
                            api_key=args.api_key,
                            base_url=args.base_url,
                            judge_model=args.judge_model,
                            user_agent=args.user_agent,
                            n_runs=args.n_runs,
                        )
                    else:
                        continue

                    row = {**base_row, **result}
                    if item.domain_slug not in domain_outputs:
                        out_path = _domain_output_path(item.domain_slug, ts, args.output_tag)
                        master_path = _domain_master_path(item.domain_slug)
                        out_fh, out_writer = _open_csv(out_path)
                        master_fh, master_writer = _open_master_csv(master_path)
                        domain_outputs[item.domain_slug] = {
                            "out_path": out_path,
                            "master_path": master_path,
                            "out_fh": out_fh,
                            "out_writer": out_writer,
                            "master_fh": master_fh,
                            "master_writer": master_writer,
                        }
                    handles = domain_outputs[item.domain_slug]
                    handles["out_writer"].writerow(row)
                    handles["out_fh"].flush()
                    handles["master_writer"].writerow(row)
                    handles["master_fh"].flush()
                    existing_keys.add(_result_key(row))
                    total += 1
                    print(f"    [{sname}] P={result['precision']} R={result['recall']} F1={result['f1']}")

                except Exception as exc:
                    errors += 1
                    print(f"    [{sname}] ERROR: {exc}")
                    traceback.print_exc()

    finally:
        for handles in domain_outputs.values():
            handles["out_fh"].close()
            handles["master_fh"].close()

    print(f"\n{'='*60}")
    print(f"Done. Written: {total}  Skipped: {skipped}  Errors: {errors}")
    if not domain_outputs:
        print("No new per-domain result files were created.")
        return

    print("Per-domain outputs:")
    for domain_slug, handles in sorted(domain_outputs.items()):
        print(f"  {domain_slug}:")
        print(f"    Results: {handles['out_path']}")
        print(f"    Master:  {handles['master_path']}")
        _print_summary(handles["out_path"])


def _print_summary(csv_path: Path) -> None:
    """Print mean P/R/F1 per (model, setting) across all topics."""
    if not csv_path.exists():
        return
    from collections import defaultdict
    data: dict[tuple, list] = defaultdict(list)
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["model"], row["setting"])
            try:
                data[key].append({
                    "precision": float(row["precision"]),
                    "recall":    float(row["recall"]),
                    "f1":        float(row["f1"]),
                })
            except (ValueError, KeyError):
                pass

    if not data:
        return

    print(f"\n{'='*60}")
    print(f"{'Model':<30} {'Setting':<20} {'N':>4}  {'P':>6}  {'R':>6}  {'F1':>6}")
    print(f"{'─'*30} {'─'*20} {'─'*4}  {'─'*6}  {'─'*6}  {'─'*6}")
    for (model, setting), rows in sorted(data.items()):
        n = len(rows)
        p = round(sum(r["precision"] for r in rows) / n, 4)
        r = round(sum(r["recall"]    for r in rows) / n, 4)
        f = round(sum(r["f1"]        for r in rows) / n, 4)
        print(f"{model:<30} {setting:<20} {n:>4}  {p:>6.4f}  {r:>6.4f}  {f:>6.4f}")


if __name__ == "__main__":
    main()
