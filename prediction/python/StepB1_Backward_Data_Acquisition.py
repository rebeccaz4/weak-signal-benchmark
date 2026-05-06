#!/usr/bin/env python
# coding: utf-8

# # 1. Subfield-based Paper Obtaining Pipeline

# In[ ]:


# Cell 1 – Imports & Path Setup
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

PROJECT_ROOT = next(
  (p for p in Path.cwd().resolve().parents if (p / "README.md").exists()),
  Path.cwd().resolve(),
)

DATA_ROOT = PROJECT_ROOT / "data" / "backward_construction"
RAW_DIR = DATA_ROOT / "raw"
MANIFEST_DIR = DATA_ROOT / "manifests"

for directory in (RAW_DIR, MANIFEST_DIR):
  directory.mkdir(parents=True, exist_ok=True)

BASELINE_YEAR = 2024
MIN_YEAR = 2020
YEARS = list(range(BASELINE_YEAR, MIN_YEAR - 1, -1))

SEMANTIC_SCHOLAR_API_KEY = (
  os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY")
)
if not SEMANTIC_SCHOLAR_API_KEY:
  raise RuntimeError(
      "Set SEMANTIC_SCHOLAR_API_KEY (or S2_API_KEY) in your environment or .env file."
  )


# In[ ]:


# Cell 2 – Inspect key directories
print(f"Project root: {PROJECT_ROOT}")
print(f"Data root:    {DATA_ROOT}")
print(f"Raw dir:      {RAW_DIR}  | exists={RAW_DIR.exists()}")
print(f"Manifest dir: {MANIFEST_DIR}  | exists={MANIFEST_DIR.exists()}")
print(f"Years (baseline -> oldest): {YEARS}")


# In[ ]:


# Cell 3 – Semantic Scholar API configuration (bulk search, dual-query, English-only default)
API_BASE_URL = "https://api.semanticscholar.org/graph/v1"
SEARCH_ENDPOINT = f"{API_BASE_URL}/paper/search/bulk"

_default_queries = ["artificial intelligence", "machine learning", "robotics", "computer vision", "natural language processing"]
_env_queries = os.getenv("BWD_SEARCH_QUERIES")
if _env_queries:
  SEARCH_QUERIES = [q.strip() for q in _env_queries.split(";") if q.strip()]
else:
  SEARCH_QUERIES = _default_queries

SEARCH_FIELDS = [
  "paperId",
  "title",
  "abstract",
  "year",
  "publicationDate",
  "venue",
  "journal",
  "externalIds",
  "fieldsOfStudy",
  "isOpenAccess",
  "openAccessPdf",
  "authors",
  "citationCount",
  "referenceCount",
  "url",
]

LANGUAGE_FILTER = os.getenv("BWD_LANGUAGE_FILTER", "English")
LANGUAGE_FILTER = LANGUAGE_FILTER.strip() or None

PER_PAGE = int(os.getenv("BWD_SEARCH_PAGE_SIZE", "100"))

_max_env = (os.getenv("BWD_MAX_PAPERS_PER_YEAR") or "").strip()
if _max_env and _max_env.lower() != "none":
  parsed = int(_max_env)
  MAX_PAPERS_PER_YEAR = parsed if parsed > 0 else None
else:
  MAX_PAPERS_PER_YEAR = None  # default: fetch the entire year

REQUEST_TIMEOUT = float(os.getenv("BWD_REQUEST_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("BWD_MAX_RETRIES", "5"))
RETRY_BACKOFF = float(os.getenv("BWD_RETRY_BACKOFF", "1.5"))
RETRY_BACKOFF_MAX = float(os.getenv("BWD_RETRY_BACKOFF_MAX", "60"))
THROTTLE_SECONDS = float(os.getenv("BWD_THROTTLE_SECONDS", "0.2"))

session = requests.Session()
session.headers.update({"x-api-key": SEMANTIC_SCHOLAR_API_KEY})


# In[ ]:


# Cell 3b – total count for "papers" in a given year
# Run this cell if you want to quickly test how many papers will be returned for a year of your choice.

import os, requests, time

API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
HEADERS = {"x-api-key": API_KEY} if API_KEY else {}
SEARCH_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

preview_query = "robotics"
preview_year = 2024
per_page = 100
throttle_seconds = 0.5      # slow down to reduce 429s
max_pages = None            # set to 1 to avoid paging, trust reported_total
max_retries = 5
backoff = 1.5

def fetch_page(token=None):
  params = {
      "query": preview_query,
      "year": preview_year,
      "limit": per_page,
      "fields": "paperId",
  }
  if token:
      params["token"] = token

  delay = throttle_seconds
  for attempt in range(1, max_retries + 1):
      resp = requests.get(SEARCH_ENDPOINT, headers=HEADERS, params=params, timeout=30)
      if resp.status_code == 429:
          time.sleep(delay)
          delay *= backoff
          continue
      resp.raise_for_status()
      return resp.json()
  raise RuntimeError(f"Exhausted retries; last status: {resp.status_code}")

token = None
total_seen = 0
iteration = 0
reported_total = None

while True:
  payload = fetch_page(token)
  if reported_total is None:
      reported_total = payload.get("total") or payload.get("totalResults")
      print(f"API-reported total: {reported_total}")

  data = payload.get("data", []) or []
  iteration += 1
  total_seen += len(data)
  print(f"Iter {iteration}: fetched {len(data)} rows; cumulative={total_seen}")

  token = payload.get("token")
  if not data or not token:
      break
  if max_pages and iteration >= max_pages:
      print(f"Stopped at max_pages={max_pages}; token available for more.")
      break

  time.sleep(throttle_seconds)

print(f"Final seen rows for query='{preview_query}', year={preview_year}: {total_seen}")


# In[ ]:


# Cell 4 – Request helpers with retry & throttling (bulk endpoint aware)
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


def _sleep_with_jitter(base_delay: float) -> None:
  if base_delay <= 0:
      return
  jitter = base_delay * 0.1
  time.sleep(base_delay + (jitter * (2 * (os.urandom(1)[0] / 255) - 1)))


def _maybe_throttle() -> None:
  if THROTTLE_SECONDS > 0:
      time.sleep(THROTTLE_SECONDS)


def request_with_retry(
  params: Dict[str, Any],
  *,
  endpoint: str = SEARCH_ENDPOINT,
) -> Dict[str, Any]:
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
          if retry_after:
              delay = min(float(retry_after), RETRY_BACKOFF_MAX)
          else:
              delay = min(RETRY_BACKOFF * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX)
          _sleep_with_jitter(delay)
          continue

      try:
          payload = response.json()
      except ValueError:
          payload = response.text
      raise RuntimeError(
          f"Semantic Scholar request failed: status={response.status_code}, payload={payload}"
      )


# In[ ]:


# Cell 5 – Path helpers, manifest utilities, resumable state (per query)
import re

def query_slug(query: str) -> str:
  slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
  return slug or "query"

def manifest_path(year: int) -> Path:
  return MANIFEST_DIR / f"semantic_scholar_{year}.parquet"

def raw_log_path(year: int, query: str) -> Path:
  return RAW_DIR / f"semantic_scholar_{year}_{query_slug(query)}.jsonl"

def state_path(year: int, query: str) -> Path:
  return RAW_DIR / f"semantic_scholar_{year}_{query_slug(query)}_state.json"

def load_existing_manifest(year: int) -> pd.DataFrame | None:
  path = manifest_path(year)
  if not path.exists():
      return None
  return pd.read_parquet(path)

def manifest_summary(df: pd.DataFrame | None) -> str:
  if df is None or df.empty:
      return "0 records"
  return f"{len(df):,} records | columns={list(df.columns)}"

def load_state(year: int, query: str) -> Dict[str, Any]:
  path = state_path(year, query)
  if not path.exists():
      return {}
  with path.open("r", encoding="utf-8") as fp:
      try:
          return json.load(fp)
      except json.JSONDecodeError:
          return {}

def save_state(
  year: int,
  query: str,
  *,
  next_token: str | None,
  total_estimate: int | None,
  fetched_raw: int,
) -> None:
  payload = {
      "next_token": next_token,
      "total_estimate": total_estimate,
      "fetched_raw": fetched_raw,
      "updated_at": time.time(),
  }
  with state_path(year, query).open("w", encoding="utf-8") as fp:
      json.dump(payload, fp, ensure_ascii=False, indent=2)

def clear_state(year: int, query: str | None = None) -> None:
  if query is not None:
      path = state_path(year, query)
      if path.exists():
          path.unlink()
      return

  pattern = f"semantic_scholar_{year}_*_state.json"
  for path in RAW_DIR.glob(pattern):
      path.unlink()


# In[ ]:


# Cell 6 – Normalization & raw logging utilities
import json
from typing import List, Dict, Any

import pandas as pd


def normalize_papers(papers: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize a list of Semantic Scholar paper dicts into a flat DataFrame.
    No heavy sanitization here for performance; we only clean on error later.
    """
    if not papers:
        return pd.DataFrame()
    df = pd.json_normalize(papers, max_level=1)
    # Replace dots in column names so they are easier to work with in pandas
    df.rename(columns={c: c.replace(".", "_") for c in df.columns}, inplace=True)
    return df


def append_raw_payload(year: int, query: str, payload: Dict[str, Any]) -> None:
    """
    Append raw JSONL payload for logging/debugging.

    We use ensure_ascii=True so that the written file only contains ASCII
    characters and never triggers UnicodeEncodeError, even if the payload
    itself has weird characters.
    """
    log_path = raw_log_path(year, query)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=True)
        fp.write("\n")


def _row_has_surrogate(row: pd.Series) -> bool:
    """
    Return True if any string cell in this row contains a UTF-16 surrogate
    codepoint (which will cause UnicodeEncodeError when encoding as UTF-8).
    """
    for val in row:
        if isinstance(val, str):
            for ch in val:
                if "\ud800" <= ch <= "\udfff":
                    return True
    return False


def drop_bad_unicode_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop any rows that contain surrogate codepoints in any string cell.

    This is called only inside fetch_year_manifest when a UnicodeEncodeError
    actually occurs on to_parquet(), so the normal path stays fast.
    """
    df = df.copy()
    bad_mask = df.apply(_row_has_surrogate, axis=1)

    if bad_mask.any():
        print(f"[drop_bad_unicode_rows] Dropping {bad_mask.sum()} rows with invalid Unicode.")
        df = df.loc[~bad_mask].copy()
    else:
        print("[drop_bad_unicode_rows] No rows with invalid Unicode found.")

    return df


# In[ ]:


# Cell 7 – Existing manifest & raw/state summary
status_rows = []
for year in YEARS:
  df = load_existing_manifest(year)
  row = {
      "year": year,
      "manifest_exists": df is not None,
      "records": 0 if df is None else len(df),
      "summary": manifest_summary(df),
      "manifest_path": manifest_path(year),
  }
  for query in SEARCH_QUERIES:
      slug = query_slug(query)
      row[f"raw_exists_{slug}"] = raw_log_path(year, query).exists()
      row[f"state_exists_{slug}"] = state_path(year, query).exists()
  status_rows.append(row)

status_df = pd.DataFrame(status_rows)
status_df


# In[ ]:


# Cell 8 – Query parameter builder (token-based, per query)
def build_search_params(
  year: int,
  query: str,
  limit: int,
  token: str | None = None,
) -> Dict[str, Any]:
  params = {
      "query": query,
      "year": year,
      "limit": limit,
      "fields": ",".join(SEARCH_FIELDS),
  }
  if LANGUAGE_FILTER:
      params["language"] = LANGUAGE_FILTER
  if token:
      params["token"] = token
  return params


# In[ ]:


# Preview – Quick sample fetch per query (no persistence)
preview_year = BASELINE_YEAR

for query in SEARCH_QUERIES:
  params = build_search_params(year=preview_year, query=query, limit=3)
  payload = request_with_retry(params)
  data = payload.get("data", [])
  print(f"\nPreview year {preview_year} | query='{query}'")
  print(f"Records returned: {len(data)}")
  df = normalize_papers(data)
  print(f"Preview columns ({len(df.columns)}): {list(df.columns)}")
  if not df.empty:
      display(df[["paperId", "title", "year"]].head(3))


# In[ ]:


# # Cell 9 – Year-level fetch routine (bulk API, dual-query, resumable)
# def fetch_year_manifest(year: int, *, force: bool = False) -> pd.DataFrame:
#   manifest_file = manifest_path(year)

#   if force:
#       if manifest_file.exists():
#           backup = manifest_file.with_suffix(f".parquet.bak.{int(time.time())}")
#           manifest_file.rename(backup)
#           print(f"[{year}] Existing manifest moved to {backup.name}.")
#       for query in SEARCH_QUERIES:
#           log_path = raw_log_path(year, query)
#           if log_path.exists():
#               backup = log_path.with_suffix(f".jsonl.bak.{int(time.time())}")
#               log_path.rename(backup)
#               print(f"[{year}] Raw log for '{query}' moved to {backup.name}.")
#           clear_state(year, query)

#   existing_df = load_existing_manifest(year)
#   if existing_df is not None and not existing_df.empty:
#       existing_df = existing_df.copy()
#       if "paperId" in existing_df.columns:
#           existing_df["paperId"] = existing_df["paperId"].astype(str)
#           seen_ids = set(existing_df["paperId"].dropna())
#       else:
#           seen_ids = set()
#       if "source_query" not in existing_df.columns:
#           existing_df["source_query"] = pd.NA
#       existing_count = len(seen_ids) if seen_ids else len(existing_df)
#   else:
#       existing_df = None
#       seen_ids = set()
#       existing_count = 0

#   target_desc = (
#       f"target ≤ {MAX_PAPERS_PER_YEAR:,}"
#       if MAX_PAPERS_PER_YEAR is not None
#       else "target = all available"
#   )
#   print(
#       f"[{year}] Starting fetch | already have {existing_count:,} unique records ({target_desc})."
#   )

#   for query in SEARCH_QUERIES:
#       state = load_state(year, query)
#       current_token = state.get("next_token")
#       total_estimate = state.get("total_estimate")
#       fetched_raw = state.get("fetched_raw", 0)

#       progress = tqdm(
#           total=total_estimate,
#           initial=fetched_raw,
#           desc=f"{year} | {query}",
#           unit="papers",
#           dynamic_ncols=True,
#       )

#       while True:
#           if MAX_PAPERS_PER_YEAR is not None and len(seen_ids) >= MAX_PAPERS_PER_YEAR:
#               progress.write(
#                   f"[{year}] Reached MAX_PAPERS_PER_YEAR={MAX_PAPERS_PER_YEAR:,}; stopping."
#               )
#               break

#           limit = (
#               min(PER_PAGE, MAX_PAPERS_PER_YEAR - len(seen_ids))
#               if MAX_PAPERS_PER_YEAR is not None
#               else PER_PAGE
#           )

#           params = build_search_params(year=year, query=query, limit=limit, token=current_token)
#           payload = request_with_retry(params)
#           append_raw_payload(year, query, payload)

#           data = payload.get("data", []) or []
#           if not data:
#               progress.write(f"[{year}] '{query}': no more data.")
#               clear_state(year, query)
#               break

#           raw_count = len(data)
#           fetched_raw += raw_count
#           progress.update(raw_count)

#           df_chunk = normalize_papers(data)
#           if df_chunk.empty:
#               progress.write(f"[{year}] '{query}': empty chunk, stopping.")
#               clear_state(year, query)
#               break

#           df_chunk["query_year"] = year
#           df_chunk["source_query"] = query

#           if "paperId" in df_chunk.columns:
#               df_chunk["paperId"] = df_chunk["paperId"].astype(str)
#               new_chunk = df_chunk[~df_chunk["paperId"].isin(seen_ids)].copy()
#               seen_ids.update(new_chunk["paperId"].dropna())
#           else:
#               new_chunk = df_chunk

#           if new_chunk.empty:
#               progress.write(f"[{year}] '{query}': chunk contained only duplicates.")
#           else:
#               if existing_df is not None:
#                   existing_df = pd.concat([existing_df, new_chunk], ignore_index=True)
#               else:
#                   existing_df = new_chunk
#               existing_df.drop_duplicates(subset="paperId", keep="first", inplace=True)
#               existing_df.to_parquet(manifest_file, index=False)

#           reported_total = payload.get("total")
#           if isinstance(reported_total, int) and reported_total > 0:
#               total_estimate = reported_total
#               progress.total = max(progress.initial, total_estimate)
#               progress.refresh()

#           next_token = payload.get("token")
#           save_state(
#               year,
#               query,
#               next_token=next_token,
#               total_estimate=total_estimate,
#               fetched_raw=fetched_raw,
#           )
#           if not next_token:
#               clear_state(year, query)
#               break
#           current_token = next_token

#       progress.close()

#       if MAX_PAPERS_PER_YEAR is not None and len(seen_ids) >= MAX_PAPERS_PER_YEAR:
#           break

#   if existing_df is not None:
#       existing_df.drop_duplicates(subset="paperId", keep="first", inplace=True)
#       existing_df.to_parquet(manifest_file, index=False)
#       summary = manifest_summary(existing_df)
#       print(f"[{year}] Completed fetch → {summary}")
#       return existing_df

#   print(f"[{year}] Nothing fetched; manifest untouched.")
#   return pd.DataFrame()


# In[ ]:


# Cell 9 – Year-level fetch routine (bulk API, dual-query, resumable)
def fetch_year_manifest(year: int, *, force: bool = False) -> pd.DataFrame:
    manifest_file = manifest_path(year)

    # =============================
    # Optional force-refresh logic
    # =============================
    if force:
        if manifest_file.exists():
            backup = manifest_file.with_suffix(f".parquet.bak.{int(time.time())}")
            manifest_file.rename(backup)
            print(f"[{year}] Existing manifest moved to {backup.name}.")
        for query in SEARCH_QUERIES:
            log_path = raw_log_path(year, query)
            if log_path.exists():
                backup = log_path.with_suffix(f".jsonl.bak.{int(time.time())}")
                log_path.rename(backup)
                print(f"[{year}] Raw log for '{query}' moved to {backup.name}.")
            clear_state(year, query)

    # =============================
    # Load existing manifest
    # =============================
    existing_df = load_existing_manifest(year)
    if existing_df is not None and not existing_df.empty:
        existing_df = existing_df.copy()
        if "paperId" in existing_df.columns:
            existing_df["paperId"] = existing_df["paperId"].astype(str)
            seen_ids = set(existing_df["paperId"].dropna())
        else:
            seen_ids = set()
        if "source_query" not in existing_df.columns:
            existing_df["source_query"] = pd.NA
        existing_count = len(seen_ids) if seen_ids else len(existing_df)
    else:
        existing_df = None
        seen_ids = set()
        existing_count = 0

    # =============================
    # NEW: if manifest exists and there is NO pending state → skip year
    # =============================
    if existing_df is not None and not existing_df.empty and not force:
        any_pending_state = False
        for query in SEARCH_QUERIES:
            state = load_state(year, query)
            # After a fully completed run we call clear_state, so all states are {}.
            # If ANY state dict is non-empty, we assume the year is mid-run and should resume.
            if state:  # non-empty dict
                any_pending_state = True
                break

        if not any_pending_state:
            summary = manifest_summary(existing_df)
            print(f"[{year}] Manifest already complete; skipping fetch → {summary}")
            return existing_df

    # =============================
    # Otherwise, either no manifest OR pending state → fetch/resume
    # =============================
    target_desc = (
        f"target ≤ {MAX_PAPERS_PER_YEAR:,}"
        if MAX_PAPERS_PER_YEAR is not None
        else "target = all available"
    )
    print(
        f"[{year}] Starting fetch | already have {existing_count:,} unique records ({target_desc})."
    )

    for query in SEARCH_QUERIES:
        state = load_state(year, query)
        current_token = state.get("next_token")
        total_estimate = state.get("total_estimate")
        fetched_raw = state.get("fetched_raw", 0)

        progress = tqdm(
            total=total_estimate,
            initial=fetched_raw,
            desc=f"{year} | {query}",
            unit="papers",
            dynamic_ncols=True,
        )

        while True:
            if MAX_PAPERS_PER_YEAR is not None and len(seen_ids) >= MAX_PAPERS_PER_YEAR:
                progress.write(
                    f"[{year}] Reached MAX_PAPERS_PER_YEAR={MAX_PAPERS_PER_YEAR:,}; stopping."
                )
                break

            limit = (
                min(PER_PAGE, MAX_PAPERS_PER_YEAR - len(seen_ids))
                if MAX_PAPERS_PER_YEAR is not None
                else PER_PAGE
            )

            params = build_search_params(
                year=year,
                query=query,
                limit=limit,
                token=current_token,
            )
            payload = request_with_retry(params)
            append_raw_payload(year, query, payload)

            data = payload.get("data", []) or []
            if not data:
                progress.write(f"[{year}] '{query}': no more data.")
                clear_state(year, query)
                break

            raw_count = len(data)
            fetched_raw += raw_count
            progress.update(raw_count)

            df_chunk = normalize_papers(data)
            if df_chunk.empty:
                progress.write(f"[{year}] '{query}': empty chunk, stopping.")
                clear_state(year, query)
                break

            df_chunk["query_year"] = year
            df_chunk["source_query"] = query

            if "paperId" in df_chunk.columns:
                df_chunk["paperId"] = df_chunk["paperId"].astype(str)
                new_chunk = df_chunk[~df_chunk["paperId"].isin(seen_ids)].copy()
                seen_ids.update(new_chunk["paperId"].dropna())
            else:
                new_chunk = df_chunk

            if new_chunk.empty:
                progress.write(f"[{year}] '{query}': chunk contained only duplicates.")
            else:
                if existing_df is not None:
                    existing_df = pd.concat([existing_df, new_chunk], ignore_index=True)
                else:
                    existing_df = new_chunk
                existing_df.drop_duplicates(subset="paperId", keep="first", inplace=True)

                # Write manifest, dropping rows with bad Unicode only if needed
                try:
                    existing_df.to_parquet(manifest_file, index=False)
                except UnicodeEncodeError:
                    print(
                        f"[{year}] UnicodeEncodeError when writing manifest; "
                        f"dropping bad rows and retrying."
                    )
                    existing_df = drop_bad_unicode_rows(existing_df)
                    existing_df.to_parquet(manifest_file, index=False)

            reported_total = payload.get("total")
            if isinstance(reported_total, int) and reported_total > 0:
                total_estimate = reported_total
                progress.total = max(progress.initial, total_estimate)
                progress.refresh()

            next_token = payload.get("token")
            save_state(
                year,
                query,
                next_token=next_token,
                total_estimate=total_estimate,
                fetched_raw=fetched_raw,
            )
            if not next_token:
                clear_state(year, query)
                break
            current_token = next_token

        progress.close()

        if MAX_PAPERS_PER_YEAR is not None and len(seen_ids) >= MAX_PAPERS_PER_YEAR:
            break

    if existing_df is not None:
        existing_df.drop_duplicates(subset="paperId", keep="first", inplace=True)

        # Final write with the same protective wrapper
        try:
            existing_df.to_parquet(manifest_file, index=False)
        except UnicodeEncodeError:
            print(
                f"[{year}] UnicodeEncodeError on final write; "
                f"dropping bad rows and retrying."
            )
            existing_df = drop_bad_unicode_rows(existing_df)
            existing_df.to_parquet(manifest_file, index=False)

        summary = manifest_summary(existing_df)
        print(f"[{year}] Completed fetch → {summary}")
        return existing_df

    print(f"[{year}] Nothing fetched; manifest untouched.")
    return pd.DataFrame()


# In[ ]:


# Cell 10 – Orchestrate year-by-year downloads
years_to_fetch = YEARS  # e.g., set to [2024] for a single-year test
force_refresh_years = set()  # e.g., {2020} to re-pull one year from scratch

for year in years_to_fetch:
  print("=" * 90)
  force = year in force_refresh_years
  fetch_year_manifest(year, force=force)


# In[ ]:


# Cell 10b – Re-run a single year manually
# You can run this cell for a single year if you are not satisfied with the volume of papers returned in prior downloads.

# Choose the year you want to re-fetch
single_year = 2023  # <-- change this by hand as needed

# Choose whether to force a full refresh for that year.
# force = True: back up existing manifest + logs, clear state, start fresh.
# force = False: resume from any saved state (if mid-run); otherwise behave like Cell 10.
force = True

print("=" * 90)
print(f"[{single_year}] Single-year fetch (force={force})")
fetch_year_manifest(single_year, force=force)


# In[ ]:


# Cell 11 – Manifest & raw-log summary
def bytes_to_mb(num_bytes: int) -> float:
  return round(num_bytes / (1024 ** 2), 2)

summary_rows: List[Dict[str, Any]] = []
for year in YEARS:
  row: Dict[str, Any] = {"year": year}
  manifest_file = manifest_path(year)
  if manifest_file.exists():
      row["records"] = len(pd.read_parquet(manifest_file))
      row["manifest_mb"] = bytes_to_mb(manifest_file.stat().st_size)
  else:
      row["records"] = 0
      row["manifest_mb"] = 0.0

  for query in SEARCH_QUERIES:
      slug = query_slug(query)
      log_file = raw_log_path(year, query)
      row[f"{slug}_raw_mb"] = (
          bytes_to_mb(log_file.stat().st_size) if log_file.exists() else 0.0
      )

  summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).set_index("year").sort_index(ascending=False)
display(summary_df)

total_records = summary_df["records"].sum()
total_manifest_mb = summary_df["manifest_mb"].sum()
print(f"Total records: {total_records:,}")
print(f"Total manifest size: {total_manifest_mb:.2f} MB")


# In[ ]:


# Cell 12 – Abstract availability & sample previews (with totals)
sample_years = [2024, 2023, 2022, 2021, 2020]  # adjust as needed

abstract_stats: List[Dict[str, Any]] = []
for year in YEARS:
  manifest_file = manifest_path(year)
  if not manifest_file.exists():
      continue
  df = pd.read_parquet(manifest_file, columns=["paperId", "abstract"])
  total = len(df)
  not_null = df["abstract"].notna().sum()
  non_empty = df["abstract"].fillna("").str.strip().ne("").sum()
  abstract_stats.append(
      {
          "year": year,
          "total_records": total,
          "abstract_not_null": not_null,
          "abstract_non_empty": non_empty,
          "pct_non_empty": round((non_empty / total) * 100, 2) if total else 0.0,
      }
  )

abstract_df = pd.DataFrame(abstract_stats).set_index("year").sort_index(ascending=False)
display(abstract_df)

overall_total = abstract_df["total_records"].sum()
overall_non_empty = abstract_df["abstract_non_empty"].sum()
overall_pct = round((overall_non_empty / overall_total) * 100, 2) if overall_total else 0.0

print(f"Overall abstract coverage: {overall_non_empty:,} / {overall_total:,} "
    f"({overall_pct:.2f}%) have non-empty abstracts.")

for year in sample_years:
  print(f"\nPreview for {year}")
  manifest_file = manifest_path(year)
  if not manifest_file.exists():
      print(f"  Manifest missing at {manifest_file}")
      continue
  df_sample = pd.read_parquet(
      manifest_file,
      columns=[
          "paperId",
          "title",
          "year",
          "source_query",
          "abstract",
          "authors",
          "fieldsOfStudy",
      ],
  )
  display(df_sample.head(3))


# In[ ]:


# Cell 13 – Subfield breakdown for a single year
target_year = 2024  # ← change to any year in YEARS

manifest_file = manifest_path(target_year)
if not manifest_file.exists():
  print(f"No manifest found for {target_year}: {manifest_file}")
else:
  cols = ["paperId", "source_query"]
  df = pd.read_parquet(manifest_file, columns=cols)
  if df.empty:
      print(f"Manifest for {target_year} is empty.")
  else:
      df["source_query"] = df["source_query"].fillna("<missing>")
      by_query = (
          df.groupby("source_query")["paperId"]
            .nunique()
            .reset_index(name="unique_papers")
            .sort_values("unique_papers", ascending=False)
      )

      missing_queries = [
          q for q in SEARCH_QUERIES if q not in by_query["source_query"].values
      ]
      if missing_queries:
          pad = pd.DataFrame(
              [{"source_query": q, "unique_papers": 0} for q in missing_queries]
          )
          by_query = pd.concat([by_query, pad], ignore_index=True)

      by_query = by_query.set_index("source_query").sort_index()

      total_unique = df["paperId"].nunique()
      print(f"{target_year} manifest: {total_unique:,} unique papers across queries")
      display(by_query)

