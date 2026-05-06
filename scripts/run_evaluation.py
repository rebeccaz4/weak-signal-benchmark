#!/usr/bin/env python
"""Run LLM-as-a-Judge evaluation from a YAML config file.

Usage:
    python scripts/run_evaluation.py --config configs/dr_tulu.yaml
    python scripts/run_evaluation.py --config configs/dr_tulu.yaml --eval-index 0
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import yaml

from wsb.config import EVAL_DIR, PROJECT_ROOT
from wsb.evaluate import compute_summary, run_evaluation
from wsb.evaluate.batch import DEFAULT_CHUNK_SIZE, DEFAULT_POLL_INTERVAL, run_batch_evaluation
from wsb.evaluate.cost import UsageTracker, print_usage_summary
from wsb.evaluate.judge import DEFAULT_BATCH_SIZE, DEFAULT_N_WORKERS
from wsb.io import load_signals, save_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-Judge evaluation from a YAML config."
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--eval-index",
        type=int,
        default=None,
        help="Run only the Nth evaluation (0-indexed). Default: run all.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help="Override n_runs from config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Number of external signals per LLM call (default: 5).",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help=f"Max concurrent API calls (default: {DEFAULT_N_WORKERS}).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override judge_model from config (e.g. gemini-2.5-flash).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate API cost and exit without making any calls.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use the Batch API (50%% cost savings, results within 24h).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=None,
        help=f"Seconds between batch status polls (default: {DEFAULT_POLL_INTERVAL}).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=f"Requests per Batch API job (default: {DEFAULT_CHUNK_SIZE}). 0 = single job.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model_name"]
    judge_model = args.judge_model or cfg.get("judge_model", "gpt-5-mini")
    temperature = cfg.get("temperature", 1.0)
    n_runs = args.n_runs or cfg.get("n_runs", 10)
    batch_size = args.batch_size or cfg.get("batch_size", DEFAULT_BATCH_SIZE)
    n_workers = args.n_workers or cfg.get("n_workers", DEFAULT_N_WORKERS)
    seed = cfg.get("seed", 27)
    random.seed(seed)

    xlsx_path = Path(cfg.get("output_xlsx", f"Evaluations/llm_judge_summary_{model_name}.xlsx"))
    if not xlsx_path.is_absolute():
        xlsx_path = PROJECT_ROOT / xlsx_path

    evaluations = cfg["evaluations"]
    if args.eval_index is not None:
        evaluations = [evaluations[args.eval_index]]

    if args.dry_run:
        from wsb.evaluate.cost import estimate_cost, print_cost_report

        report = estimate_cost(
            evaluations,
            judge_model=judge_model,
            n_runs=n_runs,
            batch_size=batch_size,
            load_signals_fn=load_signals,
            project_root=PROJECT_ROOT,
        )
        print_cost_report(report)
        return

    if args.batch:
        poll_interval = args.poll_interval or DEFAULT_POLL_INTERVAL
        chunk_size = args.chunk_size if args.chunk_size is not None else 0
        eval_metrics, tracker = run_batch_evaluation(
            config=cfg,
            evaluations=evaluations,
            judge_model=judge_model,
            n_runs=n_runs,
            batch_size=batch_size,
            temperature=temperature,
            poll_interval=poll_interval,
            chunk_size=chunk_size,
        )

        for i, ev in enumerate(evaluations):
            signal_type = ev["signal_type"]
            year_bucket = ev["year_bucket"]
            topic = ev["topic"]

            metrics_df = eval_metrics[i]

            print(f"\n{'='*60}")
            print(f"Evaluation {i+1}: {signal_type} | {year_bucket} | {topic}")
            print(f"{'='*60}")
            print("\nPer-run metrics:")
            print(metrics_df.to_string(index=False))

            summary = compute_summary(metrics_df)
            print("\nSummary (mean / std):")
            print(summary)

            signal_label = signal_type.capitalize()
            sheet_name = f"{signal_label} {year_bucket} {topic}"
            save_summary(summary, xlsx_path, sheet_name)

        print_usage_summary(tracker, batch=True)
        return

    total_tracker = UsageTracker(judge_model)

    for i, ev in enumerate(evaluations):
        signal_type = ev["signal_type"]
        year_bucket = ev["year_bucket"]
        topic = ev["topic"]

        gt_path = Path(ev["ground_truth"])
        ext_path = Path(ev["external"])
        if not gt_path.is_absolute():
            gt_path = PROJECT_ROOT / gt_path
        if not ext_path.is_absolute():
            ext_path = PROJECT_ROOT / ext_path

        print(f"\n{'='*60}")
        print(f"Evaluation {i+1}: {signal_type} | {year_bucket} | {topic}")
        print(f"{'='*60}")

        ground_truth = load_signals(gt_path)
        external = load_signals(ext_path)

        metrics_df, tracker = run_evaluation(
            ground_truth=ground_truth,
            external=external,
            n_runs=n_runs,
            model_name=model_name,
            judge_model=judge_model,
            temperature=temperature,
            batch_size=batch_size,
            n_workers=n_workers,
        )

        total_tracker.prompt_tokens += tracker.prompt_tokens
        total_tracker.completion_tokens += tracker.completion_tokens
        total_tracker.api_calls += tracker.api_calls

        print("\nPer-run metrics:")
        print(metrics_df.to_string(index=False))

        summary = compute_summary(metrics_df)
        print("\nSummary (mean / std):")
        print(summary)

        signal_label = signal_type.capitalize()
        sheet_name = f"{signal_label} {year_bucket} {topic}"
        save_summary(summary, xlsx_path, sheet_name)

    print_usage_summary(total_tracker)


if __name__ == "__main__":
    main()
