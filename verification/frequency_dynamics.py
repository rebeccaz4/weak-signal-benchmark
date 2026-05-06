#!/usr/bin/env python
# coding: utf-8
"""
Step 3 – Frequency dynamics: compute f_early, f_later, Decline, and Impact.

Reads the per-(signal, year) matching counts produced by ``topic_extraction.py``
and computes:

    f_early  = (n_2023 + n_2024) / (N_2023 + N_2024)
    f_later  = n_2025 / N_2025
    Decline  = f_later - f_early
    Impact   = Decline * log(1 / f_early)

Constraints applied before output:
    1. f_early < F_EARLY_MAX   (the signal must be rare in the early window)
    2. Decline > 0             (the signal must grow)
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "README.md").exists()),
    Path(__file__).resolve().parent,
)

DATA_ROOT = PROJECT_ROOT / "data" / "verification"
MATCHING_DIR = DATA_ROOT / "matching"
METRICS_DIR = DATA_ROOT / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# ── Thresholds ───────────────────────────────────────────────────────────────
F_EARLY_MAX = float(os.getenv("VER_F_EARLY_MAX", "0.1"))
MATCHING_PATH = MATCHING_DIR / "matching_results.parquet"
METRICS_PATH = METRICS_DIR / "verification_metrics.parquet"
FILTERED_PATH = METRICS_DIR / "verified_signals.parquet"


# =====================================================================
# 1. Load matching results
# =====================================================================

def load_matching() -> pd.DataFrame:
    if not MATCHING_PATH.exists():
        raise FileNotFoundError(
            f"Matching results not found at {MATCHING_PATH}. "
            "Run topic_extraction.py first."
        )
    return pd.read_parquet(MATCHING_PATH)


# =====================================================================
# 2. Compute metrics
# =====================================================================

def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot matching results into one row per signal and compute
    f_early, f_later, Decline, Impact.
    """
    # Pivot to get n_year and N_year for each year as columns
    pivot = df.pivot_table(
        index=["signal_id", "domain", "mainframe_topic", "direction", "signal", "what_it_was"],
        columns="year",
        values=["n_year", "N_year"],
        aggfunc="first",
    )

    # Flatten multi-level columns
    pivot.columns = [f"{col[0]}_{col[1]}" for col in pivot.columns]
    pivot = pivot.reset_index()

    # Fill missing years with 0
    for year in [2023, 2024, 2025]:
        if f"n_year_{year}" not in pivot.columns:
            pivot[f"n_year_{year}"] = 0
        if f"N_year_{year}" not in pivot.columns:
            pivot[f"N_year_{year}"] = 0

    pivot = pivot.fillna(0)

    # ── f_early = (n_2023 + n_2024) / (N_2023 + N_2024) ─────────────
    n_early = pivot["n_year_2023"] + pivot["n_year_2024"]
    N_early = pivot["N_year_2023"] + pivot["N_year_2024"]
    pivot["n_early"] = n_early
    pivot["N_early"] = N_early
    pivot["f_early"] = n_early / N_early.replace(0, float("nan"))

    # ── f_later = n_2025 / N_2025 ───────────────────────────────────
    pivot["n_later"] = pivot["n_year_2025"]
    pivot["N_later"] = pivot["N_year_2025"]
    pivot["f_later"] = pivot["n_year_2025"] / pivot["N_year_2025"].replace(0, float("nan"))

    # ── Decline = f_later - f_early ──────────────────────────────────
    pivot["decline"] = pivot["f_later"] - pivot["f_early"]

    # ── Impact = Decline * log(1 / f_early) ──────────────────────────
    pivot["impact"] = pivot.apply(
        lambda row: (
            row["decline"] * math.log(1.0 / row["f_early"])
            if pd.notna(row["f_early"]) and row["f_early"] > 0 and pd.notna(row["decline"])
            else float("nan")
        ),
        axis=1,
    )

    return pivot


# =====================================================================
# 3. Filter verified signals
# =====================================================================

def filter_signals(metrics: pd.DataFrame) -> pd.DataFrame:
    """Apply constraints: f_early < F_EARLY_MAX and Decline > 0."""
    mask = (
        metrics["f_early"].notna()
        & metrics["f_later"].notna()
        & (metrics["f_early"] < F_EARLY_MAX)
        & (metrics["decline"] > 0)
    )
    filtered = metrics[mask].copy()
    filtered.sort_values("impact", ascending=False, inplace=True)
    return filtered.reset_index(drop=True)


# =====================================================================
# 4. Orchestrator
# =====================================================================

def run():
    """Load matching, compute metrics, filter, and save."""
    df = load_matching()
    print(f"Loaded {len(df):,} matching rows.")

    metrics = compute_metrics(df)
    metrics.to_parquet(METRICS_PATH, index=False)
    print(f"Saved all metrics ({len(metrics):,} signals) to {METRICS_PATH}")

    filtered = filter_signals(metrics)
    filtered.to_parquet(FILTERED_PATH, index=False)
    print(f"Verified signals (f_early < {F_EARLY_MAX}, Decline > 0): {len(filtered):,}")
    print(f"Saved to {FILTERED_PATH}")

    return metrics, filtered


# =====================================================================
# 5. Summary & visualization
# =====================================================================

def print_summary():
    if not METRICS_PATH.exists():
        print("No metrics found. Run frequency_dynamics.py first.")
        return

    metrics = pd.read_parquet(METRICS_PATH)
    filtered = pd.read_parquet(FILTERED_PATH) if FILTERED_PATH.exists() else pd.DataFrame()

    print(f"Total signals evaluated: {len(metrics):,}")
    print(f"Verified signals:        {len(filtered):,}")
    print(f"F_EARLY_MAX threshold:   {F_EARLY_MAX}")
    print()

    if not filtered.empty:
        print("Top 10 verified signals by Impact:")
        top = filtered.head(10)[["domain", "signal", "f_early", "f_later", "decline", "impact"]]
        print(top.to_string(index=False))

    print("\nPer-domain breakdown:")
    domain_stats = metrics.groupby("domain").agg(
        total_signals=("signal_id", "count"),
        verified=("decline", lambda s: int(((s > 0) & (metrics.loc[s.index, "f_early"] < F_EARLY_MAX)).sum())),
        mean_f_early=("f_early", "mean"),
        mean_decline=("decline", "mean"),
    ).round(4)
    print(domain_stats.to_string())



def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 3: Compute frequency dynamics metrics.",
    )
    parser.add_argument("--f-early-max", type=float, default=0.1,
                        help="f_early upper-bound threshold (default: 0.1).")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory (default: <project>/data/verification).")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix for matching_results / metrics / verified_signals filenames "
                             "(e.g. '_t0.5'). Must match the suffix used when running "
                             "topic_extraction.py. Default: '' (no suffix).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Override module-level config from args
    if args.data_dir is not None:
        DATA_ROOT = Path(args.data_dir)
        MATCHING_DIR = DATA_ROOT / "matching"
        METRICS_DIR = DATA_ROOT / "metrics"
        METRICS_DIR.mkdir(parents=True, exist_ok=True)

    suffix = args.output_suffix
    MATCHING_PATH = MATCHING_DIR / f"matching_results{suffix}.parquet"
    METRICS_PATH = METRICS_DIR / f"verification_metrics{suffix}.parquet"
    FILTERED_PATH = METRICS_DIR / f"verified_signals{suffix}.parquet"

    if args.f_early_max is not None:
        F_EARLY_MAX = args.f_early_max

    run()
    print("\n")
    print_summary()
