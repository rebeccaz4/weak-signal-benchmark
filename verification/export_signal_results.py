#!/usr/bin/env python
# coding: utf-8
"""
Export per-signal verification results from the metrics parquet to
human-friendly CSV + JSON, sorted by Impact descending.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "README.md").exists()),
    Path(__file__).resolve().parent,
)


COLUMNS = [
    "signal_id", "domain", "mainframe_topic", "direction",
    "signal", "what_it_was",
    "n_year_2023", "n_year_2024", "n_year_2025",
    "N_year_2023", "N_year_2024", "N_year_2025",
    "n_early", "N_early", "f_early",
    "n_later", "N_later", "f_later",
    "decline", "impact",
]


def export(metrics_path: Path, out_csv: Path, out_json: Path,
           only_verified: bool = False, top: int | None = None) -> None:
    df = pd.read_parquet(metrics_path)
    if only_verified:
        df = df[(df["f_early"] < 0.1) & (df["decline"] > 0)]
    # Sort by impact desc (NaN last)
    df = df.sort_values("impact", ascending=False, na_position="last").reset_index(drop=True)
    if top is not None:
        df = df.head(top)

    cols = [c for c in COLUMNS if c in df.columns]
    df_out = df[cols].copy()

    # Round floats for readability
    for c in ("f_early", "f_later", "decline", "impact"):
        if c in df_out.columns:
            df_out[c] = df_out[c].round(6)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"Saved CSV  → {out_csv}  ({len(df_out):,} rows)")

    records = df_out.to_dict(orient="records")
    out_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved JSON → {out_json}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export per-signal metrics to CSV/JSON.")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Data directory (e.g. paper/verification_domain_only).")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Matches frequency_dynamics.py --output-suffix.")
    parser.add_argument("--only-verified", action="store_true",
                        help="Only include signals passing f_early<0.1 and decline>0.")
    parser.add_argument("--top", type=int, default=None,
                        help="Limit to top-N signals by Impact.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_root = Path(args.data_dir)
    suffix = args.output_suffix
    metrics_path = data_root / "metrics" / f"verification_metrics{suffix}.parquet"
    out_dir = data_root / "metrics"
    tag = "_verified" if args.only_verified else ""
    top_tag = f"_top{args.top}" if args.top else ""
    out_csv = out_dir / f"signal_results{suffix}{tag}{top_tag}.csv"
    out_json = out_dir / f"signal_results{suffix}{tag}{top_tag}.json"
    export(metrics_path, out_csv, out_json,
           only_verified=args.only_verified, top=args.top)
