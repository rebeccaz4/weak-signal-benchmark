#!/usr/bin/env python
# coding: utf-8
"""
Extract verified weak signals from construction/outputs, keyed to
verification metrics. Also emits a duplicates log and a comparison report
contrasting construction vs verified signal distributions.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "README.md").exists()),
    Path(__file__).resolve().parent,
)

METRIC_COLS_FLOAT = ("f_early", "f_later", "decline", "impact")
METRIC_COLS_INT = ("n_early", "N_early", "n_later", "N_later",
                   "n_year_2023", "n_year_2024", "n_year_2025",
                   "N_year_2023", "N_year_2024", "N_year_2025")


# =====================================================================
# 1. Walk construction outputs once
# =====================================================================

def load_all_construction_signals(construction_dir: Path):
    """Return list of dicts: {meta, ws, src_rel}."""
    entries: List[Dict[str, Any]] = []
    for result_file in sorted(construction_dir.rglob("result_latest.json")):
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        meta = data.get("metadata", {})
        try:
            rel = result_file.relative_to(construction_dir)
        except ValueError:
            rel = result_file
        for ws in (data.get("result", {}).get("weak_signals", []) or []):
            signal = (ws.get("signal") or "").strip()
            if not signal:
                continue
            entries.append({"meta": meta, "ws": ws, "src_rel": str(rel)})
    return entries


# =====================================================================
# 2. Build verified key set from parquet
# =====================================================================

def build_verified_keys(verified_df: pd.DataFrame) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """(domain, signal, what_it_was) → verification metrics dict."""
    keys: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for _, row in verified_df.iterrows():
        k = (str(row["domain"]), str(row["signal"]), str(row["what_it_was"]))
        metrics: Dict[str, Any] = {}
        for c in METRIC_COLS_FLOAT:
            v = row.get(c)
            if pd.notna(v):
                metrics[c] = float(round(v, 6))
        for c in METRIC_COLS_INT:
            v = row.get(c)
            if pd.notna(v):
                metrics[c] = int(v)
        keys[k] = metrics
    return keys


# =====================================================================
# 3. Match verified keys against construction entries
# =====================================================================

def match_entries(construction_entries: List[Dict[str, Any]],
                  verified_keys: Dict[Tuple[str, str, str], Dict[str, Any]]):
    matches: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)

    for e in construction_entries:
        meta, ws, src_rel = e["meta"], e["ws"], e["src_rel"]
        domain = str(meta.get("domain", ""))
        signal = (ws.get("signal") or "").strip()
        what_it_was = (ws.get("what_it_was") or "").strip()
        key = (domain, signal, what_it_was)
        if key not in verified_keys:
            continue

        extra_ws_fields = {k: v for k, v in ws.items()
                           if k not in ("signal", "what_it_was")}
        entry = {
            "domain": domain,
            "mainframe_topic": meta.get("mainframe_topic", ""),
            "direction": meta.get("direction", ""),
            "signal": signal,
            "what_it_was": what_it_was,
            **extra_ws_fields,
            "source_file": src_rel,
            "verification": verified_keys[key],
        }
        matches.append(entry)
        by_key[key].append(entry)

    matches.sort(key=lambda e: e["verification"].get("impact", -math.inf), reverse=True)
    return matches, by_key


def duplicates_report(by_key) -> List[Dict[str, Any]]:
    dups: List[Dict[str, Any]] = []
    for (domain, signal, what_it_was), entries in by_key.items():
        if len(entries) <= 1:
            continue
        dups.append({
            "domain": domain,
            "signal": signal,
            "what_it_was": what_it_was,
            "n_occurrences": len(entries),
            "occurrences": [
                {
                    "mainframe_topic": e["mainframe_topic"],
                    "direction": e["direction"],
                    "source_file": e["source_file"],
                }
                for e in entries
            ],
            "verification": entries[0]["verification"],
        })
    dups.sort(key=lambda e: (e["n_occurrences"],
                             e["verification"].get("impact", -math.inf)),
              reverse=True)
    return dups


# =====================================================================
# 4. Comparison markdown
# =====================================================================

def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{nd}f}"


def _i(v):
    """Format as int or 0 if missing."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return 0
    return int(v)


def _paper_counts_line(r) -> str:
    """One-line paper count summary: n_year and N_year per year."""
    return (
        f"  - papers (n/N): "
        f"2023 `{_i(r.get('n_year_2023'))}/{_i(r.get('N_year_2023'))}`, "
        f"2024 `{_i(r.get('n_year_2024'))}/{_i(r.get('N_year_2024'))}`, "
        f"2025 `{_i(r.get('n_year_2025'))}/{_i(r.get('N_year_2025'))}` "
        f"| early `{_i(r.get('n_early'))}/{_i(r.get('N_early'))}`, "
        f"later `{_i(r.get('n_later'))}/{_i(r.get('N_later'))}`"
    )


def write_comparison(construction_entries, metrics_df, verified_df,
                     matches, duplicates, out_path: Path):
    # Construction totals
    construction_total = len(construction_entries)
    con_per_domain: Dict[str, int] = defaultdict(int)
    con_per_topic: Dict[str, int] = defaultdict(int)
    for e in construction_entries:
        con_per_domain[e["meta"].get("domain", "")] += 1
        con_per_topic[e["meta"].get("mainframe_topic", "")] += 1

    # Verified
    verified_count = len(verified_df)
    ver_per_domain = verified_df.groupby("domain").size().to_dict()
    ver_per_topic = (verified_df.groupby("mainframe_topic").size().to_dict()
                     if "mainframe_topic" in verified_df.columns else {})

    # Failure analysis
    failed_f_early = metrics_df[metrics_df["f_early"] >= 0.1] \
        if "f_early" in metrics_df.columns else pd.DataFrame()
    failed_decline = metrics_df[
        (metrics_df["f_early"] < 0.1) & ~(metrics_df["decline"] > 0)
    ] if "decline" in metrics_df.columns else pd.DataFrame()

    L: List[str] = []
    L.append("# Verification Pipeline Comparison Report")
    L.append("")
    L.append("## Summary")
    L.append("")
    L.append(f"- Construction total weak signals: **{construction_total:,}**")
    L.append(f"- Signals evaluated (metrics parquet): **{len(metrics_df):,}**")
    L.append(f"- **Verified** (`f_early < 0.1 AND decline > 0`): **{verified_count:,}** "
             f"({100 * verified_count / max(construction_total, 1):.1f}% of construction)")
    L.append(f"- Construction occurrences matching verified (flat list incl. duplicates): **{len(matches):,}**")
    L.append(f"- Verified signals with >1 construction context: **{len(duplicates):,}**")
    L.append("")

    # Per-domain paper counts: N_year from metrics (per-domain constant),
    # mean n_year per verified signal.
    N_per_domain = metrics_df.groupby("domain").agg(
        N_2023=("N_year_2023", "first"),
        N_2024=("N_year_2024", "first"),
        N_2025=("N_year_2025", "first"),
    ) if "N_year_2023" in metrics_df.columns else pd.DataFrame()

    ver_n_per_domain = verified_df.groupby("domain").agg(
        mean_n_2023=("n_year_2023", "mean"),
        mean_n_2024=("n_year_2024", "mean"),
        mean_n_2025=("n_year_2025", "mean"),
    ) if not verified_df.empty and "n_year_2023" in verified_df.columns else pd.DataFrame()

    # Per-domain
    L.append("## Per-Domain Distribution")
    L.append("")
    L.append("`mean n` = average matched papers per **verified** signal in that domain. "
             "`N` = total papers retrieved for the domain that year (paper pool size).")
    L.append("")
    L.append("| Domain | Construction | Verified | Pass rate "
             "| mean n 2023 | N 2023 | mean n 2024 | N 2024 | mean n 2025 | N 2025 |")
    L.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for d in sorted(set(con_per_domain) | set(ver_per_domain)):
        c = con_per_domain.get(d, 0)
        v = ver_per_domain.get(d, 0)
        rate = f"{100 * v / c:.1f}%" if c else "—"

        # Paper counts
        def _n(col):
            if d in ver_n_per_domain.index and col in ver_n_per_domain.columns:
                val = ver_n_per_domain.loc[d, col]
                return f"{val:.1f}" if pd.notna(val) else "—"
            return "—"

        def _N(col):
            if d in N_per_domain.index and col in N_per_domain.columns:
                val = N_per_domain.loc[d, col]
                return f"{int(val):,}" if pd.notna(val) else "—"
            return "—"

        L.append(
            f"| {d} | {c} | {v} | {rate} "
            f"| {_n('mean_n_2023')} | {_N('N_2023')} "
            f"| {_n('mean_n_2024')} | {_N('N_2024')} "
            f"| {_n('mean_n_2025')} | {_N('N_2025')} |"
        )
    L.append("")

    # Per-mainframe_topic (top 20 by construction count)
    if con_per_topic:
        L.append("## Per–Mainframe Topic Distribution (top 20 by construction count)")
        L.append("")
        L.append("| Mainframe Topic | Construction | Verified | Pass rate |")
        L.append("| --- | ---: | ---: | ---: |")
        topics_sorted = sorted(con_per_topic.items(), key=lambda kv: kv[1], reverse=True)[:20]
        for t, c in topics_sorted:
            v = ver_per_topic.get(t, 0)
            rate = f"{100 * v / c:.1f}%" if c else "—"
            L.append(f"| {t} | {c} | {v} | {rate} |")
        L.append("")

    # Why removed
    L.append("## Why Signals Were Removed")
    L.append("")
    if not failed_f_early.empty:
        L.append(f"- Failed `f_early < 0.1` (too common early): **{len(failed_f_early):,}** signals")
    if not failed_decline.empty:
        L.append(f"- Failed `decline > 0` (not growing): **{len(failed_decline):,}** signals")
    L.append("")

    # Examples: removed
    L.append("## Examples — Removed Signals")
    L.append("")
    if not failed_f_early.empty:
        L.append("### Failed `f_early < 0.1` (too common early)")
        L.append("")
        for _, r in failed_f_early.nlargest(3, "f_early").iterrows():
            L.append(f"- **{r.get('signal','')}**  _({r.get('domain','')})_")
            L.append(f"  - f_early = `{_fmt(r.get('f_early'))}`, "
                     f"f_later = `{_fmt(r.get('f_later'))}`, "
                     f"decline = `{_fmt(r.get('decline'))}`")
            L.append(_paper_counts_line(r))
            wiw = str(r.get("what_it_was", ""))
            L.append(f"  - what_it_was: {wiw[:300]}{'...' if len(wiw) > 300 else ''}")
        L.append("")
    if not failed_decline.empty:
        L.append("### Failed `decline > 0` (didn't grow)")
        L.append("")
        for _, r in failed_decline.nsmallest(3, "decline").iterrows():
            L.append(f"- **{r.get('signal','')}**  _({r.get('domain','')})_")
            L.append(f"  - f_early = `{_fmt(r.get('f_early'))}`, "
                     f"f_later = `{_fmt(r.get('f_later'))}`, "
                     f"decline = `{_fmt(r.get('decline'))}`")
            L.append(_paper_counts_line(r))
            wiw = str(r.get("what_it_was", ""))
            L.append(f"  - what_it_was: {wiw[:300]}{'...' if len(wiw) > 300 else ''}")
        L.append("")

    # Examples: kept
    L.append("## Examples — Verified (Kept) Signals")
    L.append("")
    if not verified_df.empty:
        L.append("### Top 3 by Impact")
        L.append("")
        for _, r in verified_df.nlargest(3, "impact").iterrows():
            L.append(f"- **{r.get('signal','')}**  _({r.get('domain','')})_")
            L.append(f"  - f_early = `{_fmt(r.get('f_early'))}`, "
                     f"f_later = `{_fmt(r.get('f_later'))}`, "
                     f"decline = `{_fmt(r.get('decline'))}`, "
                     f"impact = `{_fmt(r.get('impact'))}`")
            L.append(_paper_counts_line(r))
            wiw = str(r.get("what_it_was", ""))
            L.append(f"  - what_it_was: {wiw[:300]}{'...' if len(wiw) > 300 else ''}")
        L.append("")

        # Mid-impact examples
        sorted_by_impact = verified_df.sort_values("impact", ascending=False).reset_index(drop=True)
        mid_start = max(0, len(sorted_by_impact) // 2 - 1)
        mid = sorted_by_impact.iloc[mid_start: mid_start + 3]
        if not mid.empty:
            L.append("### 3 mid-impact signals")
            L.append("")
            for _, r in mid.iterrows():
                L.append(f"- **{r.get('signal','')}**  _({r.get('domain','')})_")
                L.append(f"  - f_early = `{_fmt(r.get('f_early'))}`, "
                         f"decline = `{_fmt(r.get('decline'))}`, "
                         f"impact = `{_fmt(r.get('impact'))}`")
                L.append(_paper_counts_line(r))
                wiw = str(r.get("what_it_was", ""))
                L.append(f"  - what_it_was: {wiw[:300]}{'...' if len(wiw) > 300 else ''}")
            L.append("")

        # Lowest-positive-decline (marginal verified signals)
        L.append("### 3 signals with smallest positive `decline` (marginal)")
        L.append("")
        for _, r in verified_df.nsmallest(3, "decline").iterrows():
            L.append(f"- **{r.get('signal','')}**  _({r.get('domain','')})_")
            L.append(f"  - f_early = `{_fmt(r.get('f_early'))}`, "
                     f"decline = `{_fmt(r.get('decline'))}`, "
                     f"impact = `{_fmt(r.get('impact'))}`")
            L.append(_paper_counts_line(r))
            wiw = str(r.get("what_it_was", ""))
            L.append(f"  - what_it_was: {wiw[:300]}{'...' if len(wiw) > 300 else ''}")
        L.append("")

    # Impact distribution
    if "impact" in verified_df.columns and not verified_df.empty:
        L.append("## Verified Signals — Impact Distribution")
        L.append("")
        imp = verified_df["impact"].dropna()
        if not imp.empty:
            L.append(f"- mean: `{imp.mean():.4f}`, median: `{imp.median():.4f}`, "
                     f"min: `{imp.min():.4f}`, max: `{imp.max():.4f}`")
            L.append("")

    out_path.write_text("\n".join(L), encoding="utf-8")


# =====================================================================
# 5. CLI
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract verified weak signals and generate a comparison report.",
    )
    p.add_argument("--data-dir", type=str, required=True,
                   help="Data directory (e.g. paper/verification_domain_only).")
    p.add_argument("--output-suffix", type=str, required=True,
                   help="Matches frequency_dynamics.py --output-suffix (e.g. '_norerank', '_rerank0.8').")
    p.add_argument("--construction-dir", type=str, default=None,
                   help="Construction outputs directory "
                        "(default: <project-root>/construction/outputs, i.e. sibling of verification/).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_root = Path(args.data_dir)
    suffix = args.output_suffix
    construction_dir = (
        Path(args.construction_dir) if args.construction_dir
        else PROJECT_ROOT.parent / "construction" / "outputs"
    )

    metrics_path = data_root / "metrics" / f"verification_metrics{suffix}.parquet"
    verified_path = data_root / "metrics" / f"verified_signals{suffix}.parquet"

    if not metrics_path.exists() or not verified_path.exists():
        raise SystemExit(
            f"Missing input:\n  {metrics_path}\n  {verified_path}\n"
            "Run frequency_dynamics.py with the matching --output-suffix first."
        )

    print(f"Loading metrics and verified parquets...")
    metrics_df = pd.read_parquet(metrics_path)
    verified_df = pd.read_parquet(verified_path)
    print(f"  {len(verified_df):,} verified, {len(metrics_df):,} total.")

    verified_keys = build_verified_keys(verified_df)
    print(f"  {len(verified_keys):,} unique (domain, signal, what_it_was) keys verified.")

    print(f"Walking {construction_dir}...")
    construction_entries = load_all_construction_signals(construction_dir)
    print(f"  {len(construction_entries):,} weak signals in construction.")

    matches, by_key = match_entries(construction_entries, verified_keys)
    print(f"  {len(matches):,} construction occurrences matched (flat list).")

    duplicates = duplicates_report(by_key)
    print(f"  {len(duplicates):,} verified signals appear in >1 construction context.")

    out_dir = data_root / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_verified = out_dir / f"verified_weak_signals{suffix}.json"
    out_duplicates = out_dir / f"verified_duplicates{suffix}.json"
    out_comparison = out_dir / f"comparison{suffix}.md"

    out_verified.write_text(json.dumps(matches, ensure_ascii=False, indent=2), encoding="utf-8")
    out_duplicates.write_text(json.dumps(duplicates, ensure_ascii=False, indent=2), encoding="utf-8")
    write_comparison(construction_entries, metrics_df, verified_df,
                     matches, duplicates, out_comparison)

    print()
    print(f"✓ Verified signals → {out_verified}")
    print(f"✓ Duplicates       → {out_duplicates}")
    print(f"✓ Comparison md    → {out_comparison}")
