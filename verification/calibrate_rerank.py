#!/usr/bin/env python
# coding: utf-8
"""
Calibrate reranker threshold.

Stratified-sample N signals per domain, retrieve all papers with cosine
similarity >= --embedding-threshold from cached embeddings, rerank each
(signal, paper) pair with a cross-encoder, and dump the results as JSON
sorted by rerank score for manual threshold picking.

Prerequisite: run topic_extraction.py first so that embedding caches exist.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from tqdm.auto import tqdm

from domain_descriptions import domain_slug

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "README.md").exists()),
    Path(__file__).resolve().parent,
)

DATA_ROOT = PROJECT_ROOT / "data" / "verification"
MANIFEST_DIR = DATA_ROOT / "manifests"
EMBED_CACHE_DIR = DATA_ROOT / "embedding_cache"
CONSTRUCTION_OUTPUTS = PROJECT_ROOT.parent / "construction" / "outputs"

YEARS = [2023, 2024, 2025]


# =====================================================================
# 1. Load weak signals (mirrors topic_extraction.py)
# =====================================================================

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
    if df.empty:
        raise RuntimeError("No weak signals loaded from construction/outputs.")
    df["signal_id"] = df.index.astype(str)
    if signal_only:
        df["match_text"] = df["signal"]
    else:
        df["match_text"] = df["signal"] + ". " + df["what_it_was"]
    return df


# =====================================================================
# 2. Stratified sampling
# =====================================================================

def sample_signals_stratified(
    signals_df: pd.DataFrame, n_per_domain: int, seed: int
) -> pd.DataFrame:
    rng = random.Random(seed)
    sampled_indices: List[int] = []
    for domain in sorted(signals_df["domain"].unique()):
        group = signals_df[signals_df["domain"] == domain]
        k = min(n_per_domain, len(group))
        sampled_indices.extend(rng.sample(list(group.index), k))
    return signals_df.loc[sampled_indices].reset_index(drop=True)


# =====================================================================
# 3. Embedding cache (content-keyed; safer than position-based)
# =====================================================================

def load_cache_as_map(cache_path: Path) -> Dict[str, np.ndarray]:
    cache_df = pd.read_parquet(cache_path)
    m: Dict[str, np.ndarray] = {}
    for _, row in cache_df.iterrows():
        emb = row["embedding"]
        if not isinstance(emb, list):
            emb = list(emb)
        m[row["text"]] = np.array(emb, dtype=np.float32)
    return m


def _normalize(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


# =====================================================================
# 4. Collect candidates (all papers with cosine >= threshold)
# =====================================================================

def collect_candidates(
    sampled_signals: pd.DataFrame,
    embedding_threshold: float,
    signal_only: bool,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    for domain in sorted(sampled_signals["domain"].unique()):
        slug = domain_slug(domain)
        sig_cache_suffix = "_signal_only" if signal_only else ""
        signal_cache = EMBED_CACHE_DIR / f"signals_{slug}{sig_cache_suffix}.parquet"

        if not signal_cache.exists():
            print(f"  [{slug}] No signal cache ({signal_cache.name}); skipping domain.")
            continue

        signal_map = load_cache_as_map(signal_cache)
        domain_sampled = sampled_signals[sampled_signals["domain"] == domain]

        for year in YEARS:
            mp = MANIFEST_DIR / f"{slug}_{year}.parquet"
            abstract_cache = EMBED_CACHE_DIR / f"abstracts_{slug}_{year}.parquet"
            if not mp.exists():
                continue
            if not abstract_cache.exists():
                print(f"  [{slug}/{year}] No abstract cache; skipping.")
                continue

            papers_df = pd.read_parquet(mp)
            papers_df = papers_df[papers_df["abstract"].fillna("").str.strip() != ""].reset_index(drop=True)
            if papers_df.empty:
                continue

            abstract_map = load_cache_as_map(abstract_cache)
            abstract_texts = (papers_df["title"].fillna("") + ". " + papers_df["abstract"]).tolist()

            valid_mask = np.array([t in abstract_map for t in abstract_texts])
            if not valid_mask.any():
                continue
            valid_papers = papers_df[valid_mask].reset_index(drop=True)
            valid_texts = [t for t, ok in zip(abstract_texts, valid_mask) if ok]

            abstract_mat = np.array([abstract_map[t] for t in valid_texts], dtype=np.float32)
            a_norms = np.linalg.norm(abstract_mat, axis=1, keepdims=True)
            a_norms[a_norms == 0] = 1.0
            abstract_normed = abstract_mat / a_norms

            for _, sig_row in domain_sampled.iterrows():
                match_text = sig_row["match_text"]
                if match_text not in signal_map:
                    print(f"  [{slug}] signal_id={sig_row['signal_id']} missing from signal cache; skipping.")
                    continue
                sig_vec = _normalize(signal_map[match_text]).reshape(1, -1)
                sims = (abstract_normed @ sig_vec.T).flatten()
                matched = np.where(sims >= embedding_threshold)[0]

                for idx in matched:
                    paper = valid_papers.iloc[idx]
                    candidates.append({
                        "domain": domain,
                        "year": year,
                        "signal_id": sig_row["signal_id"],
                        "signal": sig_row["signal"],
                        "what_it_was": sig_row["what_it_was"],
                        "cosine": float(sims[idx]),
                        "paper_title": paper.get("title", ""),
                        "paper_abstract": paper.get("abstract", ""),
                        "paper_id": paper.get("paperId", ""),
                    })

        n_this_domain = sum(1 for c in candidates if c["domain"] == domain)
        print(f"  [{slug}] {n_this_domain} candidates above cosine {embedding_threshold}.")

    return candidates


# =====================================================================
# 5. Rerank candidates
# =====================================================================

def rerank_candidates(
    candidates: List[Dict[str, Any]],
    reranker_model: str,
    batch_size: int,
    rerank_signal_only: bool,
) -> List[Dict[str, Any]]:
    if not candidates:
        return candidates

    print(f"Loading reranker: {reranker_model}")
    reranker = CrossEncoder(reranker_model, max_length=512)

    pairs: List[List[str]] = []
    for c in candidates:
        query = c["signal"] if rerank_signal_only else f"{c['signal']}. {c['what_it_was']}"
        doc = f"{c['paper_title']}. {c['paper_abstract']}"
        pairs.append([query, doc])

    print(f"Reranking {len(pairs):,} pairs...")
    raw_scores = reranker.predict(pairs, batch_size=batch_size, show_progress_bar=True)
    raw_scores = np.asarray(raw_scores, dtype=np.float32)

    # BGE-reranker-v2-m3 returns logits; sigmoid → 0..1 for readability.
    # If the model already sigmoid-activated (some CrossEncoder configs do),
    # values are already in 0..1; sigmoid would compress but preserve order.
    # We only apply sigmoid when scores fall outside [0, 1].
    if raw_scores.min() < 0.0 or raw_scores.max() > 1.0:
        rerank_scores = 1.0 / (1.0 + np.exp(-raw_scores))
    else:
        rerank_scores = raw_scores

    for c, s in zip(candidates, rerank_scores):
        c["rerank_score"] = round(float(s), 4)
        c["cosine"] = round(float(c["cosine"]), 4)

    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates


# =====================================================================
# 6. Distribution summary
# =====================================================================

def print_distribution(results: List[Dict[str, Any]]) -> None:
    if not results:
        return
    scores = np.array([r["rerank_score"] for r in results])
    bins = np.arange(0.0, 1.01, 0.1)
    hist, _ = np.histogram(scores, bins=bins)
    max_h = int(max(hist)) if len(hist) else 1
    print("\nRerank score distribution:")
    for i in range(len(bins) - 1):
        bar_len = int(hist[i] / max_h * 40) if max_h > 0 else 0
        print(f"  [{bins[i]:.1f}, {bins[i + 1]:.1f})  {hist[i]:5d}  {'█' * bar_len}")
    print(f"  mean={scores.mean():.4f}  median={np.median(scores):.4f}  "
          f"min={scores.min():.4f}  max={scores.max():.4f}")


# =====================================================================
# 7. CLI
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate reranker threshold on sampled (signal, paper) pairs.",
    )
    parser.add_argument("--n-per-domain", type=int, default=2,
                        help="Number of signals to sample per domain (default: 2).")
    parser.add_argument("--embedding-threshold", type=float, default=0.7,
                        help="Cosine threshold for candidate recall (default: 0.7).")
    parser.add_argument("--reranker-model", type=str, default="BAAI/bge-reranker-v2-m3",
                        help="Cross-encoder reranker model (default: BAAI/bge-reranker-v2-m3).")
    parser.add_argument("--reranker-batch-size", type=int, default=32,
                        help="Reranker batch size (default: 32).")
    parser.add_argument("--signal-only", action="store_true",
                        help="Use signal-only embedding cache for recall (must match "
                             "topic_extraction.py mode). Does NOT affect the reranker text.")
    parser.add_argument("--rerank-signal-only", action="store_true",
                        help="Also feed only `signal` to the reranker. Default: reranker receives "
                             "`signal + what_it_was` regardless of embedding mode, since "
                             "cross-encoders benefit from more context.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for signal sampling (default: 42).")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory.")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: <data-dir>/matching/rerank_calibration.json).")
    parser.add_argument("--domains", nargs="+", default=None,
                        help="Only sample from specified domains (default: all).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.data_dir is not None:
        DATA_ROOT = Path(args.data_dir)
        MANIFEST_DIR = DATA_ROOT / "manifests"
        EMBED_CACHE_DIR = DATA_ROOT / "embedding_cache"

    signals_df = load_all_weak_signals(signal_only=args.signal_only)
    if args.domains:
        signals_df = signals_df[signals_df["domain"].isin(args.domains)].reset_index(drop=True)
        if signals_df.empty:
            raise SystemExit(f"No signals for domains: {args.domains}")
    print(f"Loaded {len(signals_df):,} weak signals across {signals_df['domain'].nunique()} domains.")

    sampled = sample_signals_stratified(signals_df, args.n_per_domain, args.seed)
    print(f"Sampled {len(sampled):,} signals ({args.n_per_domain} per domain).")

    candidates = collect_candidates(sampled, args.embedding_threshold, args.signal_only)
    print(f"\nTotal candidates above cosine {args.embedding_threshold}: {len(candidates):,}")

    if not candidates:
        print("No candidates to rerank. Try lowering --embedding-threshold or check caches.")
        raise SystemExit(0)

    results = rerank_candidates(
        candidates, args.reranker_model, args.reranker_batch_size, args.rerank_signal_only
    )

    output_path = (
        Path(args.output) if args.output
        else DATA_ROOT / "matching" / "rerank_calibration.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {len(results)} rerank results to {output_path}")

    print_distribution(results)
