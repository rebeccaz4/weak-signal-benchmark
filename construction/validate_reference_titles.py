#!/usr/bin/env python3
"""Validate whether reference titles match the papers pointed to by their URLs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - optional dependency fallback
    fuzz = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "construction" / "outputs"
ENV_CANDIDATES = [
    REPO_ROOT / ".env",
    REPO_ROOT / "prediction" / "python" / ".env",
]

S2_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
S2_FIELDS = "title,year,url"
ARXIV_API_URL = "https://export.arxiv.org/api/query"


@dataclass
class FetchResult:
    source: str
    resolved_url: str | None
    remote_title: str | None
    remote_year: int | None
    status: str
    error: str | None = None


def load_env() -> None:
    load_dotenv()
    for env_path in ENV_CANDIDATES:
        if env_path.exists():
            load_dotenv(env_path, override=False)


def normalize_title(title: str) -> str:
    title = title.casefold()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    return " ".join(title.split())


def similarity_score(local_title: str, remote_title: str) -> float:
    normalized_local = normalize_title(local_title)
    normalized_remote = normalize_title(remote_title)
    if fuzz is None:
        from difflib import SequenceMatcher
        return 100.0 * SequenceMatcher(None, normalized_local, normalized_remote).ratio()
    return max(
        fuzz.ratio(normalized_local, normalized_remote),
        fuzz.token_sort_ratio(normalized_local, normalized_remote),
        fuzz.token_set_ratio(normalized_local, normalized_remote),
    )


def titles_match(local_title: str, remote_title: str, threshold: float) -> bool:
    normalized_local = normalize_title(local_title)
    normalized_remote = normalize_title(remote_title)
    if normalized_local == normalized_remote:
        return True
    return similarity_score(local_title, remote_title) >= threshold


def extract_semantic_scholar_paper_id(url: str) -> str | None:
    parsed = urlparse(url)
    if "semanticscholar.org" not in parsed.netloc:
        return None

    path = parsed.path.strip("/")
    if not path.startswith("paper/"):
        return None

    tail = path[len("paper/"):].strip("/")
    if not tail:
        return None

    segments = [segment for segment in tail.split("/") if segment]
    candidate = segments[-1]

    if re.fullmatch(r"[0-9a-f]{40}", candidate, flags=re.IGNORECASE):
        return candidate
    if re.fullmatch(r"CorpusID:\d+", candidate, flags=re.IGNORECASE):
        return candidate
    if re.fullmatch(r"\d+", candidate):
        return candidate

    match = re.search(r"([0-9a-f]{40}|CorpusID:\d+|\d+)$", candidate, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def extract_arxiv_id(url: str) -> str | None:
    parsed = urlparse(url)
    if "arxiv.org" not in parsed.netloc:
        return None

    match = re.search(r"/(?:abs|pdf)/([^/?#]+)", parsed.path)
    if not match:
        return None

    arxiv_id = match.group(1)
    arxiv_id = arxiv_id.removesuffix(".pdf")
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
    return arxiv_id or None


def fetch_semantic_scholar(
    session: requests.Session,
    url: str,
    timeout: float,
) -> FetchResult:
    paper_id = extract_semantic_scholar_paper_id(url)
    if not paper_id:
        return FetchResult(
            source="semantic_scholar",
            resolved_url=url,
            remote_title=None,
            remote_year=None,
            status="unsupported_url",
            error="Could not extract Semantic Scholar paper id from URL.",
        )

    headers: dict[str, str] = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    try:
        response = session.get(
            S2_PAPER_URL.format(paper_id=paper_id),
            params={"fields": S2_FIELDS},
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return FetchResult(
            source="semantic_scholar",
            resolved_url=url,
            remote_title=None,
            remote_year=None,
            status="fetch_error",
            error=str(exc),
        )

    return FetchResult(
        source="semantic_scholar",
        resolved_url=payload.get("url") or url,
        remote_title=payload.get("title"),
        remote_year=payload.get("year"),
        status="ok" if payload.get("title") else "missing_remote_title",
        error=None if payload.get("title") else "Semantic Scholar response did not include a title.",
    )


def fetch_arxiv(
    session: requests.Session,
    url: str,
    timeout: float,
) -> FetchResult:
    arxiv_id = extract_arxiv_id(url)
    if not arxiv_id:
        return FetchResult(
            source="arxiv",
            resolved_url=url,
            remote_title=None,
            remote_year=None,
            status="unsupported_url",
            error="Could not extract arXiv id from URL.",
        )

    # Prefer Semantic Scholar for arXiv ids to avoid arXiv API rate limits.
    try:
        response = session.get(
            S2_PAPER_URL.format(paper_id=f"ARXIV:{arxiv_id}"),
            params={"fields": S2_FIELDS},
            headers=_semantic_scholar_headers(),
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("title"):
            return FetchResult(
                source="semantic_scholar_arxiv",
                resolved_url=payload.get("url") or f"https://arxiv.org/abs/{arxiv_id}",
                remote_title=payload.get("title"),
                remote_year=payload.get("year"),
                status="ok",
                error=None,
            )
    except Exception:
        pass

    try:
        response = session.get(
            ARXIV_API_URL,
            params={"id_list": arxiv_id},
            timeout=timeout,
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
    except Exception as exc:
        return FetchResult(
            source="arxiv",
            resolved_url=url,
            remote_title=None,
            remote_year=None,
            status="fetch_error",
            error=str(exc),
        )

    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", namespace)
    if entry is None:
        return FetchResult(
            source="arxiv",
            resolved_url=url,
            remote_title=None,
            remote_year=None,
            status="missing_remote_title",
            error=f"No arXiv entry returned for id {arxiv_id}.",
        )

    title = entry.findtext("atom:title", default="", namespaces=namespace).strip()
    published = entry.findtext("atom:published", default="", namespaces=namespace).strip()
    year = int(published[:4]) if len(published) >= 4 and published[:4].isdigit() else None

    return FetchResult(
        source="arxiv",
        resolved_url=f"https://arxiv.org/abs/{arxiv_id}",
        remote_title=title or None,
        remote_year=year,
        status="ok" if title else "missing_remote_title",
        error=None if title else "arXiv entry did not include a title.",
    )


def _semantic_scholar_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def fetch_remote_metadata(
    session: requests.Session,
    url: str,
    timeout: float,
    cache: dict[str, FetchResult],
) -> FetchResult:
    if url in cache:
        return cache[url]

    parsed = urlparse(url)
    netloc = parsed.netloc.casefold()
    if "semanticscholar.org" in netloc:
        result = fetch_semantic_scholar(session, url, timeout)
    elif "arxiv.org" in netloc:
        result = fetch_arxiv(session, url, timeout)
    else:
        result = FetchResult(
            source=parsed.netloc or "unknown",
            resolved_url=url,
            remote_title=None,
            remote_year=None,
            status="unsupported_url",
            error=f"Unsupported reference host: {parsed.netloc or 'unknown'}",
        )

    cache[url] = result
    return result


def iter_result_files(input_root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in input_root.rglob(pattern) if path.is_file())


def build_record(
    json_path: Path,
    weak_signal_index: int,
    weak_signal: dict[str, Any],
    reference_index: int,
    reference: dict[str, Any],
    fetch_result: FetchResult,
    threshold: float,
) -> dict[str, Any]:
    local_title = str(reference.get("title") or "")
    local_year = reference.get("year")
    remote_title = fetch_result.remote_title
    remote_year = fetch_result.remote_year

    record: dict[str, Any] = {
        "file": str(json_path),
        "weak_signal_index": weak_signal_index,
        "weak_signal": weak_signal.get("signal"),
        "reference_index": reference_index,
        "local_title": local_title,
        "local_year": local_year,
        "reference_url": reference.get("url"),
        "source": fetch_result.source,
        "resolved_url": fetch_result.resolved_url,
        "remote_title": remote_title,
        "remote_year": remote_year,
        "fetch_status": fetch_result.status,
        "error": fetch_result.error,
    }

    if remote_title:
        score = similarity_score(local_title, remote_title)
        exact_normalized_match = normalize_title(local_title) == normalize_title(remote_title)
        record["title_similarity"] = round(score, 2)
        record["exact_normalized_match"] = exact_normalized_match
        record["year_matches"] = None if local_year is None or remote_year is None else (local_year == remote_year)
        record["status"] = "match" if titles_match(local_title, remote_title, threshold) else "mismatch"
    else:
        record["title_similarity"] = None
        record["exact_normalized_match"] = False
        record["year_matches"] = None
        record["status"] = fetch_result.status

    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate whether reference titles correspond to the papers in their URLs."
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pattern", default="result_latest.json",
                        help="Glob used under --input-root, e.g. result_latest.json or result_*.json")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional path to write the JSON report.")
    parser.add_argument("--threshold", type=float, default=95.0,
                        help="Similarity threshold for considering a title a match.")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Optional delay between uncached network fetches.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after validating this many references (0 = no limit).")
    parser.add_argument("--only-mismatches", action="store_true",
                        help="Print only non-match records to stdout.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env()

    if not args.input_root.exists():
        raise SystemExit(f"Input root does not exist: {args.input_root}")

    files = iter_result_files(args.input_root, args.pattern)
    if not files:
        raise SystemExit(f"No files found under {args.input_root} matching pattern {args.pattern!r}")

    session = requests.Session()
    cache: dict[str, FetchResult] = {}
    records: list[dict[str, Any]] = []
    fetch_count = 0

    for json_path in files:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            records.append({
                "file": str(json_path),
                "status": "invalid_json",
                "error": str(exc),
            })
            continue

        weak_signals = ((payload.get("result") or {}).get("weak_signals") or [])
        if not isinstance(weak_signals, list):
            records.append({
                "file": str(json_path),
                "status": "invalid_shape",
                "error": "result.weak_signals is missing or not a list.",
            })
            continue

        for weak_signal_index, weak_signal in enumerate(weak_signals):
            references = weak_signal.get("references") or []
            if not isinstance(references, list):
                records.append({
                    "file": str(json_path),
                    "weak_signal_index": weak_signal_index,
                    "weak_signal": weak_signal.get("signal"),
                    "status": "invalid_references",
                    "error": "references is missing or not a list.",
                })
                continue

            for reference_index, reference in enumerate(references):
                if args.limit and len(records) >= args.limit:
                    break

                url = str(reference.get("url") or "").strip()
                if not url:
                    records.append({
                        "file": str(json_path),
                        "weak_signal_index": weak_signal_index,
                        "weak_signal": weak_signal.get("signal"),
                        "reference_index": reference_index,
                        "local_title": reference.get("title"),
                        "status": "missing_url",
                        "error": "Reference URL is empty.",
                    })
                    continue

                should_sleep = url not in cache and args.sleep > 0 and fetch_count > 0
                if should_sleep:
                    time.sleep(args.sleep)

                fetch_result = fetch_remote_metadata(session, url, args.timeout, cache)
                if url in cache:
                    fetch_count += 1
                record = build_record(
                    json_path=json_path,
                    weak_signal_index=weak_signal_index,
                    weak_signal=weak_signal,
                    reference_index=reference_index,
                    reference=reference,
                    fetch_result=fetch_result,
                    threshold=args.threshold,
                )
                records.append(record)

            if args.limit and len(records) >= args.limit:
                break

        if args.limit and len(records) >= args.limit:
            break

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "summary": summarize(records),
                    "records": records,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    printed = 0
    for record in records:
        if args.only_mismatches and record.get("status") == "match":
            continue
        print(json.dumps(record, ensure_ascii=False))
        printed += 1

    summary = summarize(records)
    print("\nSummary:", file=sys.stderr)
    for key, value in summary.items():
        print(f"  {key}: {value}", file=sys.stderr)
    if args.output:
        print(f"  report_path: {args.output}", file=sys.stderr)
    print(f"  printed_records: {printed}", file=sys.stderr)
    return 0


def summarize(records: list[dict[str, Any]]) -> dict[str, int]:
    summary: dict[str, int] = {
        "total_records": len(records),
        "match": 0,
        "mismatch": 0,
        "fetch_error": 0,
        "unsupported_url": 0,
        "missing_remote_title": 0,
        "missing_url": 0,
        "invalid_json": 0,
        "invalid_shape": 0,
        "invalid_references": 0,
    }
    for record in records:
        status = record.get("status")
        if status in summary:
            summary[status] += 1
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
