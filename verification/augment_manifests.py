#!/usr/bin/env python
# coding: utf-8
"""
Augment existing paper manifests with additional Semantic Scholar searches
for domains whose paper pool is too small.

For each target domain, each extra query text is fetched per year; newly
returned papers are merged into the domain's manifest, dedup by paperId.
Existing manifests are backed up before write.
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

import data_acquisition as da
from domain_descriptions import domain_slug

load_dotenv()


# ── Augmentation map: target_domain → list of custom S2 queries ──────────────
DOMAIN_AUGMENTATIONS: Dict[str, List[str]] = {
    "Advanced materials and advanced manufacturing": ["Advanced materials", "advanced manufacturing"],
    "Mobility and Transport":                         ["Mobility", "Transport"],
    "Quantum and Cryptography":                       ["Quantum", "Cryptography"],
    "Therapeutics and Biotechnologies":               ["Therapeutics", "Biotechnologies"],
}

YEARS = [2023, 2024, 2025]


# ── S2 param builder (bypasses domain→description mapping) ───────────────────

def build_custom_params(query: str, year: int, limit: int,
                        token: str | None = None) -> Dict:
    params = {
        "query": query,
        "year": year,
        "limit": limit,
        "fields": ",".join(da.SEARCH_FIELDS),
    }
    if da.LANGUAGE_FILTER:
        params["language"] = da.LANGUAGE_FILTER
    if token:
        params["token"] = token
    return params


# ── Manifest I/O ─────────────────────────────────────────────────────────────

def backup_manifest(mpath: Path) -> Path | None:
    if not mpath.exists():
        return None
    ts = int(time.time())
    backup = mpath.with_suffix(f".parquet.bak.{ts}")
    shutil.copy2(mpath, backup)
    return backup


def load_existing(mpath: Path) -> Tuple[pd.DataFrame, set]:
    if not mpath.exists():
        return pd.DataFrame(), set()
    df = pd.read_parquet(mpath)
    if df.empty:
        return df, set()
    df = df.copy()
    if "paperId" in df.columns:
        df["paperId"] = df["paperId"].astype(str)
        seen = set(df["paperId"].dropna())
    else:
        seen = set()
    return df, seen


def save_manifest(df: pd.DataFrame, mpath: Path) -> None:
    try:
        df.to_parquet(mpath, index=False)
    except UnicodeEncodeError:
        df = da.drop_bad_unicode_rows(df)
        df.to_parquet(mpath, index=False)


# ── Fetch one (target_domain, query, year) ───────────────────────────────────

def fetch_with_query(target_domain: str, query: str, year: int,
                     max_papers: int) -> int:
    """Fetch S2 results for a custom query, append to target_domain's manifest.
    Returns count of newly added papers."""
    slug = domain_slug(target_domain)
    mpath = da.MANIFEST_DIR / f"{slug}_{year}.parquet"

    existing_df, seen_ids = load_existing(mpath)
    n_before = len(existing_df)
    added = 0
    fetched = 0
    current_token: str | None = None

    progress = tqdm(total=max_papers, desc=f'{slug}/{year} "{query}"',
                    unit="papers", dynamic_ncols=True, leave=False)

    while True:
        if max_papers and fetched >= max_papers:
            break
        limit = (min(da.PER_PAGE, max_papers - fetched)
                 if max_papers else da.PER_PAGE)
        params = build_custom_params(query, year, limit, current_token)

        try:
            payload = da.request_with_retry(params)
        except RuntimeError as e:
            print(f"  [{slug}/{year}] S2 error: {e}")
            break

        data = payload.get("data", []) or []
        if not data:
            break

        fetched += len(data)
        progress.update(len(data))

        df_chunk = da.normalize_papers(data)
        if df_chunk.empty:
            break
        df_chunk["domain"] = target_domain
        df_chunk["query_year"] = year

        if "paperId" in df_chunk.columns:
            df_chunk["paperId"] = df_chunk["paperId"].astype(str)
            new_chunk = df_chunk[~df_chunk["paperId"].isin(seen_ids)].copy()
            seen_ids.update(new_chunk["paperId"].dropna())
        else:
            new_chunk = df_chunk

        if not new_chunk.empty:
            added += len(new_chunk)
            existing_df = (pd.concat([existing_df, new_chunk], ignore_index=True)
                           if not existing_df.empty else new_chunk)
            existing_df.drop_duplicates(subset="paperId", keep="first", inplace=True)
            save_manifest(existing_df, mpath)

        next_token = payload.get("token")
        if not next_token:
            break
        current_token = next_token

    progress.close()
    n_after = len(existing_df)
    print(f"  [{slug}/{year}] query={query!r:35s}  "
          f"before={n_before:>6,}  added={added:>6,}  after={n_after:>6,}")
    return added


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Augment existing paper manifests with additional S2 searches.",
    )
    p.add_argument("--data-dir", type=str, required=True,
                   help="Data directory (e.g. paper/verification_domain_only).")
    p.add_argument("--max-papers", type=int, default=30000,
                   help="Max papers per (extra-query, year) pair (default: 30000).")
    p.add_argument("--domains", nargs="+", default=None,
                   help="Subset of target domains to augment (default: all 4 in map).")
    p.add_argument("--no-backup", action="store_true",
                   help="Skip backing up existing manifests before writing.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Point data_acquisition at the right data dir
    da.DATA_ROOT = Path(args.data_dir)
    da.RAW_DIR = da.DATA_ROOT / "raw"
    da.MANIFEST_DIR = da.DATA_ROOT / "manifests"
    for d in (da.RAW_DIR, da.MANIFEST_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # Resolve which targets to run
    if args.domains:
        for d in args.domains:
            if d not in DOMAIN_AUGMENTATIONS:
                raise SystemExit(f"Unknown augmentation target: {d!r}. "
                                 f"Valid: {list(DOMAIN_AUGMENTATIONS)}")
        targets = {d: DOMAIN_AUGMENTATIONS[d] for d in args.domains}
    else:
        targets = dict(DOMAIN_AUGMENTATIONS)

    print(f"Augmenting {len(targets)} domain(s), max_papers={args.max_papers:,}")
    for d, qs in targets.items():
        print(f"  {d}  →  {qs}")
    print()

    # Backup phase
    if not args.no_backup:
        print("Backing up existing manifests...")
        for domain in targets:
            slug = domain_slug(domain)
            for year in YEARS:
                mpath = da.MANIFEST_DIR / f"{slug}_{year}.parquet"
                b = backup_manifest(mpath)
                if b:
                    print(f"  {mpath.name}  →  {b.name}")
        print()

    # Fetch phase
    total_added = 0
    for domain, queries in targets.items():
        print("=" * 90)
        print(f"Domain: {domain}")
        for query in queries:
            for year in YEARS:
                added = fetch_with_query(domain, query, year,
                                         max_papers=args.max_papers)
                total_added += added

    # Final summary
    print()
    print("=" * 90)
    print(f"Augmentation complete. Total new papers added: {total_added:,}")
    print()
    print("Final manifest sizes:")
    for domain in targets:
        slug = domain_slug(domain)
        for year in YEARS:
            mpath = da.MANIFEST_DIR / f"{slug}_{year}.parquet"
            n = len(pd.read_parquet(mpath, columns=["paperId"])) if mpath.exists() else 0
            print(f"  {slug}/{year}: {n:,}")
