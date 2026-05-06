"""
Step 1 – Domain-level paper acquisition from Semantic Scholar.

For each domain defined in ``domain_descriptions.py``, fetch papers published
in 2023, 2024, and 2025 using the S2 bulk-search endpoint.  Results are stored
as per-(domain, year) Parquet manifests under ``data/verification/manifests/``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

from domain_descriptions import DOMAIN_DESCRIPTIONS, domain_slug, search_query_for_domain

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = next(
    (p for p in Path(__file__).resolve().parents if (p / "README.md").exists()),
    Path(__file__).resolve().parent,
)

DATA_ROOT = PROJECT_ROOT / "data" / "verification"
RAW_DIR = DATA_ROOT / "raw"
MANIFEST_DIR = DATA_ROOT / "manifests"

for d in (RAW_DIR, MANIFEST_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Year range ───────────────────────────────────────────────────────────────
YEARS = [2023, 2024, 2025]

# ── Semantic Scholar API ─────────────────────────────────────────────────────
API_BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEARCH_ENDPOINT = f"{API_BASE_URL}/paper/search/bulk"

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY")
if not SEMANTIC_SCHOLAR_API_KEY:
    raise RuntimeError("Set SEMANTIC_SCHOLAR_API_KEY (or S2_API_KEY) in your environment or .env file.")

SEARCH_FIELDS = [
    "paperId", "title", "abstract", "year", "publicationDate",
    "venue", "journal", "externalIds", "fieldsOfStudy", "isOpenAccess",
    "openAccessPdf", "authors", "citationCount", "referenceCount", "url",
]

LANGUAGE_FILTER = os.getenv("VER_LANGUAGE_FILTER", "English").strip() or None
PER_PAGE = int(os.getenv("VER_SEARCH_PAGE_SIZE", "100"))

_max_env = (os.getenv("VER_MAX_PAPERS_PER_DOMAIN_YEAR") or "").strip()
if _max_env and _max_env.lower() != "none":
    _parsed = int(_max_env)
    MAX_PAPERS_PER_DOMAIN_YEAR = _parsed if _parsed > 0 else None
else:
    MAX_PAPERS_PER_DOMAIN_YEAR = None

REQUEST_TIMEOUT = float(os.getenv("VER_REQUEST_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("VER_MAX_RETRIES", "5"))
RETRY_BACKOFF = float(os.getenv("VER_RETRY_BACKOFF", "1.5"))
RETRY_BACKOFF_MAX = float(os.getenv("VER_RETRY_BACKOFF_MAX", "60"))
THROTTLE_SECONDS = float(os.getenv("VER_THROTTLE_SECONDS", "0.2"))

session = requests.Session()
session.headers.update({"x-api-key": SEMANTIC_SCHOLAR_API_KEY})

# ── Request helpers ──────────────────────────────────────────────────────────
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _sleep_with_jitter(base_delay: float) -> None:
    if base_delay <= 0:
        return
    jitter = base_delay * 0.1
    time.sleep(base_delay + (jitter * (2 * (os.urandom(1)[0] / 255) - 1)))


def _maybe_throttle() -> None:
    if THROTTLE_SECONDS > 0:
        time.sleep(THROTTLE_SECONDS)


def request_with_retry(params: Dict[str, Any], *, endpoint: str = SEARCH_ENDPOINT) -> Dict[str, Any]:
    attempt = 0
    while True:
        attempt += 1
        _maybe_throttle()
        try:
            response = session.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            if attempt > MAX_RETRIES:
                raise RuntimeError(f"Request failed after retries: {exc}") from exc
            delay = min(RETRY_BACKOFF * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX)
            _sleep_with_jitter(delay)
            continue

        if response.status_code == 200:
            return response.json()

        if response.status_code in RETRYABLE_STATUS and attempt <= MAX_RETRIES:
            retry_after = response.headers.get("Retry-After")
            delay = min(float(retry_after), RETRY_BACKOFF_MAX) if retry_after else min(RETRY_BACKOFF * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX)
            _sleep_with_jitter(delay)
            continue

        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        raise RuntimeError(f"S2 request failed: status={response.status_code}, payload={payload}")


# ── Path helpers & state management ──────────────────────────────────────────
def manifest_path(domain: str, year: int) -> Path:
    return MANIFEST_DIR / f"{domain_slug(domain)}_{year}.parquet"


def raw_log_path(domain: str, year: int) -> Path:
    return RAW_DIR / f"{domain_slug(domain)}_{year}.jsonl"


def state_path(domain: str, year: int) -> Path:
    return RAW_DIR / f"{domain_slug(domain)}_{year}_state.json"


def load_state(domain: str, year: int) -> Dict[str, Any]:
    path = state_path(domain, year)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_state(domain: str, year: int, *, next_token: str | None, total_estimate: int | None, fetched_raw: int) -> None:
    payload = {"next_token": next_token, "total_estimate": total_estimate, "fetched_raw": fetched_raw, "updated_at": time.time()}
    state_path(domain, year).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def clear_state(domain: str, year: int) -> None:
    path = state_path(domain, year)
    if path.exists():
        path.unlink()


# ── Normalize & log ──────────────────────────────────────────────────────────
def normalize_papers(papers: List[Dict[str, Any]]) -> pd.DataFrame:
    if not papers:
        return pd.DataFrame()
    df = pd.json_normalize(papers, max_level=1)
    df.rename(columns={c: c.replace(".", "_") for c in df.columns}, inplace=True)
    return df


def append_raw_payload(domain: str, year: int, payload: Dict[str, Any]) -> None:
    log = raw_log_path(domain, year)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=True)
        fp.write("\n")


def _row_has_surrogate(row: pd.Series) -> bool:
    for val in row:
        if isinstance(val, str):
            for ch in val:
                if "\ud800" <= ch <= "\udfff":
                    return True
    return False


def drop_bad_unicode_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bad_mask = df.apply(_row_has_surrogate, axis=1)
    if bad_mask.any():
        print(f"  [drop_bad_unicode_rows] Dropping {bad_mask.sum()} rows with invalid Unicode.")
        df = df.loc[~bad_mask].copy()
    return df


# ── Search param builder ─────────────────────────────────────────────────────
def build_search_params(domain: str, year: int, limit: int, token: str | None = None, *, domain_only: bool = False) -> Dict[str, Any]:
    params = {
        "query": search_query_for_domain(domain, use_description=not domain_only),
        "year": year,
        "limit": limit,
        "fields": ",".join(SEARCH_FIELDS),
    }
    if LANGUAGE_FILTER:
        params["language"] = LANGUAGE_FILTER
    if token:
        params["token"] = token
    return params


# ── Core fetch routine ───────────────────────────────────────────────────────
def fetch_domain_year(domain: str, year: int, *, force: bool = False, domain_only: bool = False) -> pd.DataFrame:
    """Fetch papers for one (domain, year) pair. Resumable & idempotent."""
    mpath = manifest_path(domain, year)
    slug = domain_slug(domain)

    # Force refresh
    if force:
        if mpath.exists():
            backup = mpath.with_suffix(f".parquet.bak.{int(time.time())}")
            mpath.rename(backup)
            print(f"  [{slug}/{year}] Existing manifest backed up to {backup.name}.")
        log = raw_log_path(domain, year)
        if log.exists():
            backup = log.with_suffix(f".jsonl.bak.{int(time.time())}")
            log.rename(backup)
        clear_state(domain, year)

    # Load existing
    existing_df = pd.read_parquet(mpath) if mpath.exists() else None
    if existing_df is not None and not existing_df.empty:
        existing_df = existing_df.copy()
        existing_df["paperId"] = existing_df["paperId"].astype(str)
        seen_ids = set(existing_df["paperId"].dropna())
    else:
        existing_df = None
        seen_ids = set()

    # Skip if complete
    if existing_df is not None and not existing_df.empty and not force:
        state = load_state(domain, year)
        if not state:
            print(f"  [{slug}/{year}] Already complete ({len(existing_df):,} records). Skipping.")
            return existing_df

    target_desc = f"<= {MAX_PAPERS_PER_DOMAIN_YEAR:,}" if MAX_PAPERS_PER_DOMAIN_YEAR else "all"
    print(f"  [{slug}/{year}] Fetching (have {len(seen_ids):,}, target {target_desc})...")

    state = load_state(domain, year)
    current_token = state.get("next_token")
    total_estimate = state.get("total_estimate")
    fetched_raw = state.get("fetched_raw", 0)

    progress = tqdm(total=total_estimate, initial=fetched_raw, desc=f"{slug}/{year}", unit="papers", dynamic_ncols=True)

    while True:
        if MAX_PAPERS_PER_DOMAIN_YEAR and len(seen_ids) >= MAX_PAPERS_PER_DOMAIN_YEAR:
            progress.write(f"  [{slug}/{year}] Reached max {MAX_PAPERS_PER_DOMAIN_YEAR:,}; stopping.")
            break

        limit = min(PER_PAGE, MAX_PAPERS_PER_DOMAIN_YEAR - len(seen_ids)) if MAX_PAPERS_PER_DOMAIN_YEAR else PER_PAGE
        params = build_search_params(domain, year, limit, current_token, domain_only=domain_only)
        payload = request_with_retry(params)
        append_raw_payload(domain, year, payload)

        data = payload.get("data", []) or []
        if not data:
            clear_state(domain, year)
            break

        fetched_raw += len(data)
        progress.update(len(data))

        df_chunk = normalize_papers(data)
        if df_chunk.empty:
            clear_state(domain, year)
            break

        df_chunk["domain"] = domain
        df_chunk["query_year"] = year

        if "paperId" in df_chunk.columns:
            df_chunk["paperId"] = df_chunk["paperId"].astype(str)
            new_chunk = df_chunk[~df_chunk["paperId"].isin(seen_ids)].copy()
            seen_ids.update(new_chunk["paperId"].dropna())
        else:
            new_chunk = df_chunk

        if not new_chunk.empty:
            existing_df = pd.concat([existing_df, new_chunk], ignore_index=True) if existing_df is not None else new_chunk
            existing_df.drop_duplicates(subset="paperId", keep="first", inplace=True)
            try:
                existing_df.to_parquet(mpath, index=False)
            except UnicodeEncodeError:
                existing_df = drop_bad_unicode_rows(existing_df)
                existing_df.to_parquet(mpath, index=False)

        reported_total = payload.get("total")
        if isinstance(reported_total, int) and reported_total > 0:
            total_estimate = reported_total
            progress.total = max(progress.initial, total_estimate)
            progress.refresh()

        next_token = payload.get("token")
        save_state(domain, year, next_token=next_token, total_estimate=total_estimate, fetched_raw=fetched_raw)
        if not next_token:
            clear_state(domain, year)
            break
        current_token = next_token

    progress.close()

    if existing_df is not None:
        existing_df.drop_duplicates(subset="paperId", keep="first", inplace=True)
        try:
            existing_df.to_parquet(mpath, index=False)
        except UnicodeEncodeError:
            existing_df = drop_bad_unicode_rows(existing_df)
            existing_df.to_parquet(mpath, index=False)
        return existing_df

    return pd.DataFrame()


# ── Orchestrator ─────────────────────────────────────────────────────────────
def fetch_all(*, domains: list[str] | None = None, force_domains: set[str] | None = None, domain_only: bool = False):
    """Fetch papers for every (domain, year) pair."""
    _domains = domains if domains is not None else list(DOMAIN_DESCRIPTIONS.keys())
    for domain in _domains:
        print("=" * 90)
        print(f"Domain: {domain}")
        force = domain in (force_domains or set())
        for year in YEARS:
            fetch_domain_year(domain, year, force=force, domain_only=domain_only)


# ── Summary ──────────────────────────────────────────────────────────────────
def print_summary():
    rows = []
    for domain in DOMAIN_DESCRIPTIONS:
        for year in YEARS:
            mp = manifest_path(domain, year)
            if mp.exists():
                n = len(pd.read_parquet(mp, columns=["paperId"]))
                mb = round(mp.stat().st_size / (1024 ** 2), 2)
            else:
                n, mb = 0, 0.0
            rows.append({"domain": domain, "year": year, "records": n, "size_mb": mb})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print(f"\nTotal records: {df['records'].sum():,}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Step 1: Domain-level paper acquisition from Semantic Scholar.",
    )
    parser.add_argument("--domains", nargs="+", default=None,
                        help="Domain names to fetch (default: all). Use --list-domains to see options.")
    parser.add_argument("--domain-only", action="store_true",
                        help="Search using domain name only, without appending the description.")
    parser.add_argument("--force", action="store_true",
                        help="Force re-fetch, backing up existing manifests.")
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="Years to fetch (default: 2023 2024 2025).")
    parser.add_argument("--max-papers", type=int, default=None,
                        help="Max papers per (domain, year). Default: unlimited.")
    parser.add_argument("--page-size", type=int, default=None,
                        help="Results per API page (default: 100).")
    parser.add_argument("--language", default=None,
                        help="Language filter (default: English).")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Override output data directory (default: <project>/data/verification).")
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
        RAW_DIR = DATA_ROOT / "raw"
        MANIFEST_DIR = DATA_ROOT / "manifests"
        for d in (RAW_DIR, MANIFEST_DIR):
            d.mkdir(parents=True, exist_ok=True)

    if args.years is not None:
        YEARS = args.years
        print(f"Fetching years: {YEARS}")
    if args.page_size is not None:
        PER_PAGE = args.page_size
        print(f"Page size: {PER_PAGE}")
    if args.language is not None:
        LANGUAGE_FILTER = args.language
        print(f"Language filter: {LANGUAGE_FILTER}")
    if args.max_papers is not None:
        MAX_PAPERS_PER_DOMAIN_YEAR = args.max_papers
        print(f"Max papers per domain year: {MAX_PAPERS_PER_DOMAIN_YEAR}")

    # Validate domains
    selected_domains = args.domains
    if selected_domains:
        for d in selected_domains:
            if d not in DOMAIN_DESCRIPTIONS:
                print(f"Error: Unknown domain '{d}'")
                print("Use --list-domains to see available options.")
                raise SystemExit(1)

    force_domains = set(selected_domains or list(DOMAIN_DESCRIPTIONS.keys())) if args.force else None

    fetch_all(domains=selected_domains, force_domains=force_domains, domain_only=args.domain_only)
    print("\n")
    print_summary()
