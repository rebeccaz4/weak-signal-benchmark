#!/usr/bin/env python
# coding: utf-8
"""
Step 2 – Embedding-based similarity matching.

For every weak signal produced by the construction step, compute cosine
similarity between the signal text (``signal`` + ``what_it_was``) and every
paper abstract retrieved for that signal's domain in each year (2023-2025).

If the similarity exceeds a threshold the paper counts toward *n_year*.
The total number of papers for that (domain, year) is *N_year*.

Outputs are written to ``data/verification/matching/``.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm.auto import tqdm

from domain_descriptions import DOMAIN_DESCRIPTIONS, domain_slug

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "README.md").exists()),
    Path(__file__).resolve().parent,
)

DATA_ROOT = PROJECT_ROOT / "data" / "verification"
MANIFEST_DIR = DATA_ROOT / "manifests"
MATCHING_DIR = DATA_ROOT / "matching"
EMBED_CACHE_DIR = DATA_ROOT / "embedding_cache"
RERANK_CACHE_DIR = DATA_ROOT / "rerank_cache"

for d in (MATCHING_DIR, EMBED_CACHE_DIR, RERANK_CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)

CONSTRUCTION_OUTPUTS = PROJECT_ROOT.parent / "construction" / "outputs"

YEARS = [2023, 2024, 2025]

# ── Embedding config ─────────────────────────────────────────────────────────
EMBED_MODEL = os.getenv("VER_EMBED_MODEL", "BAAI/bge-large-en-v1.5")
EMBED_BATCH_SIZE = int(os.getenv("VER_EMBED_BATCH_SIZE", "128"))
SIMILARITY_THRESHOLD = float(os.getenv("VER_SIMILARITY_THRESHOLD", "0.5"))

st_model = SentenceTransformer(EMBED_MODEL)

# =====================================================================
# 1. Load weak signals from construction/outputs
# =====================================================================

def load_all_weak_signals(signal_only: bool = False) -> pd.DataFrame:
    """
    Walk ``construction/outputs/`` and load every ``result_latest.json``.
    Return a DataFrame with columns:
        domain, mainframe_topic, direction, signal, what_it_was, signal_id

    If signal_only=True, match_text is just `signal` (no what_it_was appended).
    """
    rows: List[Dict[str, Any]] = []
    if not CONSTRUCTION_OUTPUTS.exists():
        raise FileNotFoundError(f"Construction outputs not found: {CONSTRUCTION_OUTPUTS}")

    for result_file in sorted(CONSTRUCTION_OUTPUTS.rglob("result_latest.json")):
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        meta = data.get("metadata", {})
        domain = meta.get("domain", "")
        mainframe_topic = meta.get("mainframe_topic", "")
        direction = meta.get("direction", "")

        for ws in (data.get("result", {}).get("weak_signals", []) or []):
            signal = (ws.get("signal") or "").strip()
            what_it_was = (ws.get("what_it_was") or "").strip()
            if not signal:
                continue
            rows.append({
                "domain": domain,
                "mainframe_topic": mainframe_topic,
                "direction": direction,
                "signal": signal,
                "what_it_was": what_it_was,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No weak signals loaded from construction/outputs.")

    # Create a unique signal_id
    df["signal_id"] = df.index.astype(str)
    # Build the matching text: signal only, or signal + what_it_was
    if signal_only:
        df["match_text"] = df["signal"]
    else:
        df["match_text"] = df["signal"] + ". " + df["what_it_was"]

    print(f"Loaded {len(df):,} weak signals across {df['domain'].nunique()} domains.")
    return df


# =====================================================================
# 2. Embedding utilities (with disk cache)
# =====================================================================

def _embed_batch(texts: List[str]) -> List[List[float]]:
    embeddings = st_model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()


def embed_texts_cached(texts: List[str], cache_path: Path, label: str = "") -> np.ndarray:
    """
    Embed a list of texts using OpenAI, caching results to *cache_path*.
    Returns an (N, dim) float32 numpy array.
    """
    if cache_path.exists():
        cache_df = pd.read_parquet(cache_path)
    else:
        cache_df = pd.DataFrame(columns=["text", "embedding"])

    # text -> row-index map (later wins on duplicate, matches original dict(zip) semantics)
    text_to_idx: Dict[str, int] = {}
    for i, t in enumerate(cache_df["text"].values):
        text_to_idx[t] = i

    needed = [t for t in texts if t not in text_to_idx]
    if needed:
        new_texts: List[str] = []
        new_embs: List[List[float]] = []
        desc = f"Embedding {label}" if label else "Embedding"
        for start in tqdm(range(0, len(needed), EMBED_BATCH_SIZE), desc=desc, unit="batch"):
            batch = needed[start:start + EMBED_BATCH_SIZE]
            embeddings = _embed_batch(batch)
            for txt, emb in zip(batch, embeddings):
                new_texts.append(txt)
                new_embs.append(emb)

        new_df = pd.DataFrame({"text": new_texts, "embedding": new_embs})
        cache_df = pd.concat([cache_df, new_df], ignore_index=True)
        cache_df.drop_duplicates(subset="text", keep="last", inplace=True)
        cache_df.reset_index(drop=True, inplace=True)
        cache_df.to_parquet(cache_path, index=False)
        print(f"  Cached {len(new_texts):,} new embeddings to {cache_path.name}")

        text_to_idx = {}
        for i, t in enumerate(cache_df["text"].values):
            text_to_idx[t] = i
    else:
        if label:
            print(f"  {label}: all embeddings already cached.")

    # Index directly into the underlying object array — avoids building a dict copy of all embeddings
    cached_embeds = cache_df["embedding"].values
    result = np.stack(
        [np.asarray(cached_embeds[text_to_idx[t]], dtype=np.float32) for t in texts]
    )
    return result


# =====================================================================
# 2b. Reranker utilities (optional two-stage pipeline)
# =====================================================================

def _load_rerank_cache(cache_path: Path) -> Dict[tuple, float]:
    """Load rerank score cache keyed by (signal_id, paper_id)."""
    if not cache_path.exists():
        return {}
    df = pd.read_parquet(cache_path)
    return {
        (str(sid), str(pid)): float(score)
        for sid, pid, score in zip(
            df["signal_id"].values,
            df["paper_id"].values,
            df["rerank_score"].values,
        )
    }


def _save_rerank_cache(cache_path: Path, cache: Dict[tuple, float]) -> None:
    rows = [
        {"signal_id": k[0], "paper_id": k[1], "rerank_score": v}
        for k, v in cache.items()
    ]
    df = pd.DataFrame(rows)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)


def _rerank_score_pairs(reranker: CrossEncoder,
                        pairs: List[List[str]],
                        batch_size: int) -> np.ndarray:
    """Run the cross-encoder, apply sigmoid only if output is outside [0, 1]."""
    raw = np.asarray(
        reranker.predict(pairs, batch_size=batch_size, show_progress_bar=True),
        dtype=np.float32,
    )
    if raw.size and (raw.min() < 0.0 or raw.max() > 1.0):
        return 1.0 / (1.0 + np.exp(-raw))
    return raw


# =====================================================================
# 3. Per-domain similarity matching
# =====================================================================

def match_domain(domain: str, signals_df: pd.DataFrame,
                 signal_only: bool = False,
                 reranker: CrossEncoder | None = None,
                 rerank_threshold: float = 0.8,
                 reranker_batch_size: int = 32,
                 rerank_signal_only: bool = False,
                 rerank_cache_suffix: str = "") -> pd.DataFrame:
    """
    For one domain, embed all paper abstracts (across years) and all weak
    signals, compute cosine similarities, optionally rerank with a cross-encoder,
    and return per-(signal, year) counts.

    Returns DataFrame with columns:
        signal_id, domain, year, n_year, N_year
        n_year_cosine (only when reranker is used; cosine-only baseline count)
    """
    slug = domain_slug(domain)
    domain_signals = signals_df[signals_df["domain"] == domain].reset_index(drop=True)
    if domain_signals.empty:
        print(f"  [{slug}] No weak signals for this domain; skipping.")
        return pd.DataFrame()

    use_reranker = reranker is not None

    # ── Embed weak signals ───────────────────────────────────────────
    sig_cache_suffix = "_signal_only" if signal_only else ""
    signal_cache = EMBED_CACHE_DIR / f"signals_{slug}{sig_cache_suffix}.parquet"
    signal_texts = domain_signals["match_text"].tolist()
    signal_embeds = embed_texts_cached(signal_texts, signal_cache, label=f"{slug} signals")

    # Normalize signal embeddings
    norms = np.linalg.norm(signal_embeds, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    signal_embeds_normed = signal_embeds / norms

    results: List[Dict[str, Any]] = []

    for year in YEARS:
        mp = MANIFEST_DIR / f"{slug}_{year}.parquet"
        if not mp.exists():
            print(f"  [{slug}/{year}] No manifest found; skipping.")
            for sid in domain_signals["signal_id"]:
                row = {"signal_id": sid, "domain": domain, "year": year, "n_year": 0, "N_year": 0}
                if use_reranker:
                    row["n_year_cosine"] = 0
                results.append(row)
            continue

        papers_df = pd.read_parquet(mp)
        # Keep only papers with non-empty abstracts
        papers_df = papers_df[papers_df["abstract"].fillna("").str.strip() != ""].reset_index(drop=True)
        N_year = len(papers_df)
        print(N_year)

        if N_year == 0:
            for sid in domain_signals["signal_id"]:
                row = {"signal_id": sid, "domain": domain, "year": year, "n_year": 0, "N_year": 0}
                if use_reranker:
                    row["n_year_cosine"] = 0
                results.append(row)
            continue

        # ── Embed abstracts ──────────────────────────────────────────
        abstract_cache = EMBED_CACHE_DIR / f"abstracts_{slug}_{year}.parquet"
        abstract_texts = (papers_df["title"].fillna("") + ". " + papers_df["abstract"]).tolist()
        abstract_embeds = embed_texts_cached(abstract_texts, abstract_cache, label=f"{slug}/{year} abstracts")

        # Normalize
        a_norms = np.linalg.norm(abstract_embeds, axis=1, keepdims=True)
        a_norms[a_norms == 0] = 1.0
        abstract_embeds_normed = abstract_embeds / a_norms

        CHUNK_SIZE = int(os.getenv("VER_SIM_CHUNK_SIZE", "5000"))

        # ── Stage 1: cosine filter, collect all candidates ──────────
        # per-signal list of paper indices that passed cosine
        candidates_per_signal: Dict[str, List[int]] = {}
        for sig_idx, sig_row in domain_signals.iterrows():
            sig_vec = signal_embeds_normed[sig_idx:sig_idx + 1]
            matched_idx_list: List[int] = []
            for chunk_start in range(0, N_year, CHUNK_SIZE):
                chunk = abstract_embeds_normed[chunk_start:chunk_start + CHUNK_SIZE]
                sims = (chunk @ sig_vec.T).flatten()
                hits = np.where(sims >= SIMILARITY_THRESHOLD)[0]
                matched_idx_list.extend((chunk_start + hits).tolist())
            candidates_per_signal[sig_row["signal_id"]] = matched_idx_list

        cosine_total = sum(len(v) for v in candidates_per_signal.values())

        # ── Stage 2: rerank filter (optional) ────────────────────────
        if use_reranker:
            rerank_cache_path = (
                RERANK_CACHE_DIR / f"rerank_{slug}_{year}{rerank_cache_suffix}.parquet"
            )
            rerank_cache = _load_rerank_cache(rerank_cache_path)

            # Gather pairs missing from cache
            pairs_to_score: List[List[str]] = []
            keys_to_score: List[tuple] = []
            for sig_idx, sig_row in domain_signals.iterrows():
                signal_id = sig_row["signal_id"]
                query = sig_row["signal"] if rerank_signal_only else f"{sig_row['signal']}. {sig_row['what_it_was']}"
                for paper_idx in candidates_per_signal[signal_id]:
                    paper = papers_df.iloc[paper_idx]
                    paper_id = str(paper.get("paperId", ""))
                    if not paper_id:
                        continue
                    if (signal_id, paper_id) in rerank_cache:
                        continue
                    doc = f"{paper.get('title', '')}. {paper.get('abstract', '')}"
                    pairs_to_score.append([query, doc])
                    keys_to_score.append((signal_id, paper_id))

            if pairs_to_score:
                print(f"  [{slug}/{year}] Reranking {len(pairs_to_score):,} new pairs "
                      f"({cosine_total:,} total candidates, {cosine_total - len(pairs_to_score):,} cached)...")
                scores = _rerank_score_pairs(reranker, pairs_to_score, reranker_batch_size)
                for key, s in zip(keys_to_score, scores):
                    rerank_cache[key] = float(s)
                _save_rerank_cache(rerank_cache_path, rerank_cache)
            else:
                print(f"  [{slug}/{year}] All {cosine_total:,} candidates already reranked (cache hit).")

            # Count per signal: both cosine & rerank thresholds must pass
            for sig_idx, sig_row in domain_signals.iterrows():
                signal_id = sig_row["signal_id"]
                n_cosine = len(candidates_per_signal[signal_id])
                n_matched = 0
                for paper_idx in candidates_per_signal[signal_id]:
                    paper_id = str(papers_df.iloc[paper_idx].get("paperId", ""))
                    score = rerank_cache.get((signal_id, paper_id), 0.0)
                    if score >= rerank_threshold:
                        n_matched += 1
                results.append({
                    "signal_id": signal_id,
                    "domain": domain,
                    "year": year,
                    "n_year": n_matched,
                    "n_year_cosine": n_cosine,
                    "N_year": N_year,
                })
        else:
            # Cosine-only pipeline (original behavior)
            for sig_idx, sig_row in domain_signals.iterrows():
                signal_id = sig_row["signal_id"]
                results.append({
                    "signal_id": signal_id,
                    "domain": domain,
                    "year": year,
                    "n_year": len(candidates_per_signal[signal_id]),
                    "N_year": N_year,
                })

        suffix = f" (rerank ≥ {rerank_threshold})" if use_reranker else ""
        print(f"  [{slug}/{year}] N={N_year:,}, {len(domain_signals)} signals{suffix}.")

    return pd.DataFrame(results)


# =====================================================================
# 4. Orchestrator
# =====================================================================

def match_all(*, domains: list[str] | None = None, output_suffix: str = "",
              signal_only: bool = False,
              use_reranker: bool = False,
              reranker_model: str = "BAAI/bge-reranker-v2-m3",
              rerank_threshold: float = 0.8,
              reranker_batch_size: int = 32,
              rerank_signal_only: bool = False,
              rerank_cache_suffix: str = "") -> pd.DataFrame:
    """Run similarity matching for every domain. Returns combined results."""
    signals_df = load_all_weak_signals(signal_only=signal_only)

    # Save loaded signals for reference
    signals_path = DATA_ROOT / "weak_signals_loaded.parquet"
    signals_df.to_parquet(signals_path, index=False)
    print(f"Saved loaded weak signals to {signals_path}")

    reranker = None
    if use_reranker:
        print(f"Loading reranker: {reranker_model}")
        reranker = CrossEncoder(reranker_model, max_length=512)

    domains_with_signals = sorted(signals_df["domain"].unique())
    if domains:
        domains_with_signals = [d for d in domains_with_signals if d in domains]
    all_results: List[pd.DataFrame] = []

    for domain in domains_with_signals:
        print("=" * 90)
        print(f"Domain: {domain}")
        result = match_domain(
            domain, signals_df,
            signal_only=signal_only,
            reranker=reranker,
            rerank_threshold=rerank_threshold,
            reranker_batch_size=reranker_batch_size,
            rerank_signal_only=rerank_signal_only,
            rerank_cache_suffix=rerank_cache_suffix,
        )
        if not result.empty:
            all_results.append(result)

    if not all_results:
        print("No matching results produced.")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    # Merge signal metadata back
    combined = combined.merge(
        signals_df[["signal_id", "mainframe_topic", "direction", "signal", "what_it_was"]],
        on="signal_id",
        how="left",
    )

    output_path = MATCHING_DIR / f"matching_results{output_suffix}.parquet"
    combined.to_parquet(output_path, index=False)
    print(f"\nSaved {len(combined):,} matching rows to {output_path}")
    return combined


# =====================================================================
# 5. Summary
# =====================================================================

def print_summary(output_suffix: str = ""):
    output_path = MATCHING_DIR / f"matching_results{output_suffix}.parquet"
    if not output_path.exists():
        print("No matching results found. Run match_all() first.")
        return
    df = pd.read_parquet(output_path)
    print(f"Total matching rows: {len(df):,}")
    print(f"Domains: {df['domain'].nunique()}")
    print(f"Unique signals: {df['signal_id'].nunique()}")
    print(f"\nPer-year summary:")
    summary = df.groupby("year").agg(
        total_N=("N_year", "mean"),
        mean_n=("n_year", "mean"),
        max_n=("n_year", "max"),
    ).round(1)
    print(summary.to_string())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 2: Embedding-based similarity matching.",
    )
    parser.add_argument("--domains", nargs="+", default=None,
                        help="Domain names to process (default: all with signals).")
    parser.add_argument("--similarity-threshold", type=float, default=0.7,
                        help="Cosine similarity threshold (default: 0.5).")
    parser.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5",
                        help="SentenceTransformer embedding model (default: BAAI/bge-large-en-v1.5).")
    parser.add_argument("--embed-batch-size", type=int, default=None,
                        help="Embedding batch size (default: 128).")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override data directory (default: <project>/data/verification).")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix appended to matching_results filename, "
                             "e.g. '_t0.5' → matching_results_t0.5.parquet. Default: '' (no suffix).")
    parser.add_argument("--signal-only", action="store_true",
                        help="Embed only `signal` text, without appending `what_it_was`. "
                             "Uses a separate cache file signals_<slug>_signal_only.parquet.")
    parser.add_argument("--use-reranker", action="store_true",
                        help="Enable two-stage pipeline: cosine recall + cross-encoder rerank. "
                             "Papers must pass BOTH thresholds to count toward n_year.")
    parser.add_argument("--reranker-model", type=str, default="BAAI/bge-reranker-v2-m3",
                        help="Cross-encoder reranker model (default: BAAI/bge-reranker-v2-m3).")
    parser.add_argument("--rerank-threshold", type=float, default=0.8,
                        help="Minimum rerank score to count a paper (default: 0.8).")
    parser.add_argument("--reranker-batch-size", type=int, default=32,
                        help="Reranker batch size (default: 32).")
    parser.add_argument("--rerank-signal-only", action="store_true",
                        help="Feed only `signal` (no `what_it_was`) to the reranker. "
                             "Default: reranker uses `signal + what_it_was` regardless of embedding mode.")
    parser.add_argument("--rerank-cache-suffix", type=str, default="",
                        help="Suffix for rerank cache filenames (e.g. '_v2_full'). "
                             "Use different suffixes when switching reranker model or rerank text mode.")
    parser.add_argument("--list-domains", action="store_true",
                        help="Print available domain names and exit.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_domains:
        for i, d in enumerate(DOMAIN_DESCRIPTIONS, 1):
            print(f"  {i:2d}. {d}")
        raise SystemExit(0)

    # Override module-level config from args
    if args.data_dir is not None:
        DATA_ROOT = Path(args.data_dir)
        MANIFEST_DIR = DATA_ROOT / "manifests"
        MATCHING_DIR = DATA_ROOT / "matching"
        EMBED_CACHE_DIR = DATA_ROOT / "embedding_cache"
        RERANK_CACHE_DIR = DATA_ROOT / "rerank_cache"
        for d in (MATCHING_DIR, EMBED_CACHE_DIR, RERANK_CACHE_DIR):
            d.mkdir(parents=True, exist_ok=True)

    if args.similarity_threshold is not None:
        SIMILARITY_THRESHOLD = args.similarity_threshold
    if args.embed_model is not None:
        EMBED_MODEL = args.embed_model
    if args.embed_batch_size is not None:
        EMBED_BATCH_SIZE = args.embed_batch_size

    match_all(
        domains=args.domains,
        output_suffix=args.output_suffix,
        signal_only=args.signal_only,
        use_reranker=args.use_reranker,
        reranker_model=args.reranker_model,
        rerank_threshold=args.rerank_threshold,
        reranker_batch_size=args.reranker_batch_size,
        rerank_signal_only=args.rerank_signal_only,
        rerank_cache_suffix=args.rerank_cache_suffix,
    )
    print("\n")
    print_summary(output_suffix=args.output_suffix)
