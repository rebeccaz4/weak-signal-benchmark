#!/usr/bin/env python
"""Run LLM-as-a-Judge evaluation across ALL model configs.

Usage:
    # Real-time (sequential, one config at a time):
    python scripts/run_all.py

    # Batch mode — pool all configs, chunk into batch jobs:
    python scripts/run_all.py --batch

    # Batch mode with custom chunk size and poll interval:
    python scripts/run_all.py --batch --chunk-size 200 --poll-interval 60

    # Only specific configs:
    python scripts/run_all.py --batch --configs configs/dr_tulu.yaml configs/gpt_4_1.yaml
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from wsb.config import PROJECT_ROOT
from wsb.evaluate import compute_summary
from wsb.evaluate.batch import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_POLL_INTERVAL,
    run_batch_evaluation_multi,
)
from wsb.evaluate.cost import print_usage_summary
from wsb.io import save_summary

CONFIGS_DIR = PROJECT_ROOT / "configs"
COMBINED_XLSX = PROJECT_ROOT / "Evaluations" / "llm_judge_all_models.xlsx"
CSV_SUMMARY_FILE = PROJECT_ROOT / "Evaluations" / "pre-rec-f1-summary.csv"


def _load_configs(config_paths: list[Path] | None) -> list[dict]:
    """Load and return parsed YAML configs."""
    if config_paths:
        paths = config_paths
    else:
        paths = sorted(CONFIGS_DIR.glob("*.yaml"))
        if not paths:
            raise FileNotFoundError(f"No YAML configs found in {CONFIGS_DIR}")

    configs = []
    for p in paths:
        with open(p) as f:
            configs.append(yaml.safe_load(f))
    return configs


def _build_and_save_summary_from_xlsx(configs: list[dict]) -> None:
    """After sequential runs, read per-model Excel files and save cross-model summary."""
    rows: list[dict[str, Any]] = []

    for cfg in configs:
        model_name = cfg["model_name"]
        xlsx_path = Path(cfg.get("output_xlsx", f"Evaluations/llm_judge_summary_{model_name}.xlsx"))
        if not xlsx_path.is_absolute():
            xlsx_path = PROJECT_ROOT / xlsx_path

        if not xlsx_path.exists():
            print(f"  Warning: {xlsx_path} not found, skipping {model_name}")
            continue

        xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
        for sheet_name in xls.sheet_names:
            summary = pd.read_excel(xls, sheet_name=sheet_name, index_col=0)
            rows.append({
                "model": model_name,
                "evaluation": sheet_name,
                "precision_mean": summary.loc["precision", "mean"],
                "precision_std": summary.loc["precision", "std"],
                "recall_mean": summary.loc["recall", "mean"],
                "recall_std": summary.loc["recall", "std"],
                "f1_mean": summary.loc["f1", "mean"],
                "f1_std": summary.loc["f1", "std"],
            })

    if not rows:
        print("No results found to summarize.")
        return

    table = pd.DataFrame(rows)
    _print_cross_model_table(table)
    _save_combined_xlsx(table, COMBINED_XLSX)
    CSV_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CSV_SUMMARY_FILE, index=False)
    print(f"Combined summary CSV saved to {CSV_SUMMARY_FILE}")


def _build_cross_model_table(
    config_results: dict[int, tuple[dict[str, Any], dict[int, pd.DataFrame]]],
) -> pd.DataFrame:
    """Build a consolidated table with mean P/R/F1 per model per evaluation.

    Columns: model, evaluation, precision_mean, precision_std, recall_mean,
             recall_std, f1_mean, f1_std.
    """
    rows: list[dict[str, Any]] = []

    for _ci, (cfg, eval_metrics) in sorted(config_results.items()):
        model_name = cfg["model_name"]
        evaluations = cfg["evaluations"]

        for i, ev in enumerate(evaluations):
            if i not in eval_metrics:
                continue

            metrics_df = eval_metrics[i]
            summary = compute_summary(metrics_df)

            label = f"{ev['signal_type']} | {ev['year_bucket']} | {ev['topic']}"
            rows.append({
                "model": model_name,
                "evaluation": label,
                "precision_mean": summary.loc["precision", "mean"],
                "precision_std": summary.loc["precision", "std"],
                "recall_mean": summary.loc["recall", "mean"],
                "recall_std": summary.loc["recall", "std"],
                "f1_mean": summary.loc["f1", "mean"],
                "f1_std": summary.loc["f1", "std"],
            })

    return pd.DataFrame(rows)


def _print_cross_model_table(table: pd.DataFrame) -> None:
    """Print a formatted cross-model summary table grouped by evaluation."""
    print(f"\n{'='*90}")
    print("CROSS-MODEL SUMMARY")
    print(f"{'='*90}")

    for eval_label, group in table.groupby("evaluation", sort=False):
        print(f"\n  {eval_label}")
        print(f"  {'─'*80}")

        header = f"  {'Model':<25} {'Precision':>18} {'Recall':>18} {'F1':>18}"
        print(header)
        print(f"  {'─'*80}")

        for _, row in group.iterrows():
            p = f"{row['precision_mean']:.4f} +/- {row['precision_std']:.4f}"
            r = f"{row['recall_mean']:.4f} +/- {row['recall_std']:.4f}"
            f = f"{row['f1_mean']:.4f} +/- {row['f1_std']:.4f}"
            print(f"  {row['model']:<25} {p:>18} {r:>18} {f:>18}")

    print(f"\n{'='*90}")


def _save_combined_xlsx(table: pd.DataFrame, xlsx_path: Path) -> None:
    """Save the cross-model summary table to a single Excel file.

    Creates one sheet per evaluation with models as rows.
    """
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        # Full table on a "Summary" sheet
        table.to_excel(writer, sheet_name="Summary", index=False)

        # One sheet per evaluation with a pivot view
        for eval_label, group in table.groupby("evaluation", sort=False):
            sheet_name = eval_label[:31]  # Excel limit
            pivot = group.set_index("model")[
                ["precision_mean", "precision_std", "recall_mean", "recall_std", "f1_mean", "f1_std"]
            ]
            pivot.to_excel(writer, sheet_name=sheet_name)

    print(f"\nCombined results saved to {xlsx_path}")


def _process_results(
    config_results: dict[int, tuple[dict, dict[int, Any]]],
) -> pd.DataFrame:
    """Print per-model metrics, save per-model Excel, and return cross-model table."""
    for ci, (cfg, eval_metrics) in sorted(config_results.items()):
        model_name = cfg["model_name"]
        evaluations = cfg["evaluations"]
        xlsx_path = Path(cfg.get("output_xlsx", f"Evaluations/llm_judge_summary_{model_name}.xlsx"))
        if not xlsx_path.is_absolute():
            xlsx_path = PROJECT_ROOT / xlsx_path

        print(f"\n{'='*60}")
        print(f"MODEL: {model_name}")
        print(f"{'='*60}")

        for i, ev in enumerate(evaluations):
            signal_type = ev["signal_type"]
            year_bucket = ev["year_bucket"]
            topic = ev["topic"]

            if i not in eval_metrics:
                continue

            metrics_df = eval_metrics[i]

            print(f"\n  {signal_type} | {year_bucket} | {topic}")
            print(f"  {'─'*50}")
            print(metrics_df.to_string(index=False))

            summary = compute_summary(metrics_df)
            print(f"\n  Summary (mean / std):")
            print(summary)

            signal_label = signal_type.capitalize()
            sheet_name = f"{signal_label} {year_bucket} {topic}"
            save_summary(summary, xlsx_path, sheet_name)

    # Build and display cross-model table
    table = _build_cross_model_table(config_results)
    _print_cross_model_table(table)
    _save_combined_xlsx(table, COMBINED_XLSX)
    CSV_SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(CSV_SUMMARY_FILE, index=False)
    print(f"Combined summary CSV saved to {CSV_SUMMARY_FILE}")
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-Judge evaluation across all model configs."
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=Path,
        default=None,
        help="Specific config YAML files to run. Default: all in configs/.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Use the Batch API (50%% cost savings, results within 24h).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Requests per Batch API job (default: {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Seconds between batch status polls (default: {DEFAULT_POLL_INTERVAL}).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Override judge_model for all configs.",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help="Override n_runs for all configs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Signals per LLM call (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=27,
        help="Random seed (default: 27).",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    configs = _load_configs(args.configs)
    print(f"Loaded {len(configs)} config(s): {[c['model_name'] for c in configs]}")

    if not args.batch:
        # Sequential real-time mode: run each config one at a time
        import subprocess
        import sys

        script = Path(__file__).parent / "run_evaluation.py"
        paths = args.configs or sorted(CONFIGS_DIR.glob("*.yaml"))
        for cfg_path in paths:
            print(f"\n{'='*60}")
            print(f"Running: {cfg_path.name}")
            print(f"{'='*60}")
            cmd = [sys.executable, str(script), "--config", str(cfg_path)]
            if args.judge_model:
                cmd += ["--judge-model", args.judge_model]
            if args.n_runs:
                cmd += ["--n-runs", str(args.n_runs)]
            if args.batch_size != 5:
                cmd += ["--batch-size", str(args.batch_size)]
            subprocess.run(cmd, check=True)

        # Build cross-model summary from per-model Excel files
        _build_and_save_summary_from_xlsx(configs)
        print("\nAll evaluations complete.")
        return

    # Batch mode: pool all configs, chunk, submit together
    judge_model = args.judge_model or configs[0].get("judge_model", "gpt-5-mini")
    temperature = configs[0].get("temperature", 1.0)

    config_results, tracker = run_batch_evaluation_multi(
        configs,
        judge_model=judge_model,
        n_runs_override=args.n_runs,
        batch_size=args.batch_size,
        temperature=temperature,
        poll_interval=args.poll_interval,
        chunk_size=args.chunk_size,
    )

    _process_results(config_results)
    print_usage_summary(tracker, batch=True)
    print("\nAll evaluations complete.")


if __name__ == "__main__":
    main()
