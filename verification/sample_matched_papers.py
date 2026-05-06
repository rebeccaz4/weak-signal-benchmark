#!/usr/bin/env python
# coding: utf-8
"""
Sample matched papers for manual inspection.

For each weak signal, recompute cosine similarity against paper abstracts
using cached embeddings, find papers above the threshold, randomly sample
a few, and save the results as JSON for human review.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from domain_descriptions import DOMAIN_DESCRIPTIONS, domain_slug

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "README.md").exists()),
    Path(__file__).resolve().parent,
)

DATA_ROOT = PROJECT_ROOT / "data" / "verification"
MANIFEST_DIR = DATA_ROOT / "manifests"
EMBED_CACHE_DIR = DATA_ROOT / "embedding_cache"
RERANK_CACHE_DIR = DATA_ROOT / "rerank_cache"
CONSTRUCTION_OUTPUTS = PROJECT_ROOT.parent / "construction" / "outputs"

YEARS = [2023, 2024, 2025]
SIMILARITY_THRESHOLD = float(os.getenv("VER_SIMILARITY_THRESHOLD", "0.5"))


def load_all_weak_signals(signal_only: bool = False) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for result_file in sorted(CONSTRUCTION_OUTPUTS.rglob("result_latest.json")):
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        meta = data.get("metadata", {})
        domain = meta.get("domain", "")
        for ws in (data.get("result", {}).get("weak_signals", []) or []):
            signal = (ws.get("signal") or "").strip()
            what_it_was = (ws.get("what_it_was") or "").strip()
            if not signal:
                continue
            rows.append({
                "domain": domain,
                "signal": signal,
                "what_it_was": what_it_was,
            })
    df = pd.DataFrame(rows)
    df["signal_id"] = df.index.astype(str)
    if signal_only:
        df["match_text"] = df["signal"]
    else:
        df["match_text"] = df["signal"] + ". " + df["what_it_was"]
    return df


def load_cached_embeddings(cache_path: Path) -> np.ndarray:
    cache_df = pd.read_parquet(cache_path)
    cache_df["embedding"] = cache_df["embedding"].apply(
        lambda x: list(x) if not isinstance(x, list) else x
    )
    return np.array(cache_df["embedding"].tolist(), dtype=np.float32)


def normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def load_rerank_cache(cache_path: Path) -> Dict[tuple, float]:
    if not cache_path.exists():
        return {}
    df = pd.read_parquet(cache_path)
    return {
        (str(row["signal_id"]), str(row["paper_id"])): float(row["rerank_score"])
        for _, row in df.iterrows()
    }


def sample_for_domain(domain: str, signals_df: pd.DataFrame,
                      n_per_signal: int, seed: int,
                      signal_only: bool = False,
                      use_reranker: bool = False,
                      rerank_threshold: float = 0.8,
                      rerank_cache_suffix: str = "") -> List[Dict[str, Any]]:
    slug = domain_slug(domain)
    domain_signals = signals_df[signals_df["domain"] == domain].reset_index(drop=True)
    if domain_signals.empty:
        return []

    sig_cache_suffix = "_signal_only" if signal_only else ""
    signal_cache = EMBED_CACHE_DIR / f"signals_{slug}{sig_cache_suffix}.parquet"
    if not signal_cache.exists():
        print(f"  [{slug}] No signal embedding cache; skipping.")
        return []

    signal_embeds = normalize(load_cached_embeddings(signal_cache))
    rng = random.Random(seed)
    samples: List[Dict[str, Any]] = []

    for year in YEARS:
        mp = MANIFEST_DIR / f"{slug}_{year}.parquet"
        abstract_cache = EMBED_CACHE_DIR / f"abstracts_{slug}_{year}.parquet"
        if not mp.exists() or not abstract_cache.exists():
            continue

        papers_df = pd.read_parquet(mp)
        papers_df = papers_df[papers_df["abstract"].fillna("").str.strip() != ""].reset_index(drop=True)
        if papers_df.empty:
            continue

        abstract_embeds = normalize(load_cached_embeddings(abstract_cache))

        rerank_cache: Dict[tuple, float] = {}
        if use_reranker:
            rerank_cache_path = RERANK_CACHE_DIR / f"rerank_{slug}_{year}{rerank_cache_suffix}.parquet"
            if not rerank_cache_path.exists():
                print(f"  [{slug}/{year}] No rerank cache at {rerank_cache_path.name}; skipping this year.")
                continue
            rerank_cache = load_rerank_cache(rerank_cache_path)

        for sig_idx, sig_row in domain_signals.iterrows():
            sig_vec = signal_embeds[sig_idx:sig_idx + 1]
            sims = (abstract_embeds @ sig_vec.T).flatten()
            matched_indices = np.where(sims >= SIMILARITY_THRESHOLD)[0]

            if len(matched_indices) == 0:
                continue

            # Rerank filter: keep only papers whose (signal_id, paper_id) has cached rerank ≥ threshold
            filtered_pairs: List[tuple] = []  # (idx, rerank_score or None)
            if use_reranker:
                signal_id = str(sig_row["signal_id"])
                for idx in matched_indices:
                    paper_id = str(papers_df.iloc[idx].get("paperId", ""))
                    score = rerank_cache.get((signal_id, paper_id))
                    if score is not None and score >= rerank_threshold:
                        filtered_pairs.append((int(idx), score))
            else:
                filtered_pairs = [(int(idx), None) for idx in matched_indices]

            if not filtered_pairs:
                continue

            k = min(n_per_signal, len(filtered_pairs))
            sampled = rng.sample(filtered_pairs, k)

            for idx, rerank_score in sampled:
                paper = papers_df.iloc[idx]
                entry = {
                    "domain": domain,
                    "year": year,
                    "signal": sig_row["signal"],
                    "what_it_was": sig_row["what_it_was"],
                    "similarity": round(float(sims[idx]), 4),
                    "paper_title": paper.get("title", ""),
                    "paper_abstract": paper.get("abstract", ""),
                    "paper_id": paper.get("paperId", ""),
                }
                if rerank_score is not None:
                    entry["rerank_score"] = round(float(rerank_score), 4)
                samples.append(entry)

    print(f"  [{slug}] Sampled {len(samples)} signal-paper pairs.")
    return samples


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample matched papers for manual inspection.",
    )
    parser.add_argument("--n-per-signal", type=int, default=3,
                        help="Number of matched papers to sample per signal per year (default: 3).")
    parser.add_argument("--similarity-threshold", type=float, default=0.7,
                        help="Cosine similarity threshold (default: from env or 0.5).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42).")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: <data-dir>/matching/sampled_matches.json).")
    parser.add_argument("--domains", nargs="+", default=None,
                        help="Domain names to process (default: all).")
    parser.add_argument("--signal-only", action="store_true",
                        help="Use signal-only embedding cache (signals_<slug>_signal_only.parquet). "
                             "Must match the mode used when running topic_extraction.py.")
    parser.add_argument("--use-reranker", action="store_true",
                        help="Only sample papers that pass both cosine and rerank thresholds. "
                             "Requires rerank cache from topic_extraction.py --use-reranker.")
    parser.add_argument("--rerank-threshold", type=float, default=0.8,
                        help="Minimum rerank score (default: 0.8). Only used with --use-reranker.")
    parser.add_argument("--rerank-cache-suffix", type=str, default="",
                        help="Rerank cache suffix (must match topic_extraction.py --rerank-cache-suffix).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.data_dir is not None:
        DATA_ROOT = Path(args.data_dir)
        MANIFEST_DIR = DATA_ROOT / "manifests"
        EMBED_CACHE_DIR = DATA_ROOT / "embedding_cache"
        RERANK_CACHE_DIR = DATA_ROOT / "rerank_cache"

    if args.similarity_threshold is not None:
        SIMILARITY_THRESHOLD = args.similarity_threshold

    signals_df = load_all_weak_signals(signal_only=args.signal_only)
    print(f"Loaded {len(signals_df):,} weak signals.")

    domains = args.domains or sorted(signals_df["domain"].unique())
    all_samples: List[Dict[str, Any]] = []

    for domain in domains:
        print(f"Domain: {domain}")
        all_samples.extend(
            sample_for_domain(
                domain, signals_df, args.n_per_signal, args.seed,
                signal_only=args.signal_only,
                use_reranker=args.use_reranker,
                rerank_threshold=args.rerank_threshold,
                rerank_cache_suffix=args.rerank_cache_suffix,
            )
        )

    # Sort by rerank_score (if present) then cosine, both descending
    if args.use_reranker:
        all_samples.sort(key=lambda x: (x.get("rerank_score", 0.0), x["similarity"]), reverse=True)
    else:
        all_samples.sort(key=lambda x: x["similarity"], reverse=True)

    output_path = Path(args.output) if args.output else DATA_ROOT / "matching" / "sampled_matches.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_samples, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaved {len(all_samples)} sampled pairs to {output_path}")
