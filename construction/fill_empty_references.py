#!/usr/bin/env python3
"""Fill empty weak-signal references using Semantic Scholar search."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "construction" / "outputs"
DEFAULT_EMPTY_SUMMARY = REPO_ROOT / "construction" / "reference_cleanup_summary.json"
ENV_CANDIDATES = [
    REPO_ROOT / ".env",
    REPO_ROOT / "prediction" / "python" / ".env",
]

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "paperId,title,year,url,abstract"
STOPWORDS = {
    "about", "across", "after", "again", "against", "already", "also", "among", "and", "are",
    "because", "became", "before", "being", "between", "both", "builds", "could", "current",
    "design", "directly", "emerged", "emerging", "fact", "flagship", "focused", "form", "forms",
    "for", "from", "functioned", "generic", "highly", "idea", "initially", "inside", "into",
    "its", "itself", "just", "key", "known", "later", "main", "more", "most", "moving", "new",
    "niche", "not", "now", "offered", "only", "other", "outcome", "overcome", "planar", "problem",
    "problems", "process", "pursuing", "rather", "represented", "research", "route", "signal",
    "signals", "since", "solution", "solutions", "space", "specific", "strategy", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "this", "through", "toward",
    "treated", "unusual", "using", "very", "was", "were", "what", "when", "which", "while", "why",
    "with", "within", "yield", "yields",
}


def load_env() -> None:
    load_dotenv()
    for env_path in ENV_CANDIDATES:
        if env_path.exists():
            load_dotenv(env_path, override=False)


def normalize_text(text: str) -> str:
    text = text.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def tokenize(text: str) -> set[str]:
    tokens = normalize_text(text).split()
    return {t for t in tokens if len(t) > 2}


def keyword_list(text: str, max_terms: int) -> list[str]:
    seen: list[str] = []
    for token in normalize_text(text).split():
        if len(token) <= 2 or token in STOPWORDS or token.isdigit():
            continue
        if token not in seen:
            seen.append(token)
        if len(seen) >= max_terms:
            break
    return seen


def sentence_excerpt(text: str, max_words: int) -> str:
    words = text.split()
    return " ".join(words[:max_words]).strip()


def build_queries(topic: str, weak_signal: dict[str, Any]) -> list[str]:
    signal = str(weak_signal.get("signal") or "").strip()
    what = str(weak_signal.get("what_it_was") or "").strip()
    why = str(weak_signal.get("why_weak_signal") or "").strip()

    topic_keywords = keyword_list(topic, 4)
    signal_keywords = keyword_list(signal, 6)
    what_keywords = keyword_list(what, 6)
    why_keywords = keyword_list(why, 6)

    queries = [
        topic.strip(),
        f"{topic} {' '.join(signal_keywords[:4])}".strip(),
        f"{topic} {' '.join((what_keywords[:2] + why_keywords[:2])[:4])}".strip(),
    ]

    deduped: list[str] = []
    seen: set[str] = set()
    for query in queries:
        query = " ".join(query.split())
        if query and query not in seen:
            seen.add(query)
            deduped.append(query)
    return deduped


def semantic_scholar_headers() -> dict[str, str]:
    headers: dict[str, str] = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def search_semantic_scholar(
    session: requests.Session,
    query: str,
    *,
    limit: int,
    timeout: float,
) -> list[dict[str, Any]]:
    response = session.get(
        S2_SEARCH_URL,
        params={"query": query, "limit": limit, "fields": S2_FIELDS},
        headers=semantic_scholar_headers(),
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("data") or []


def candidate_score(topic: str, weak_signal: dict[str, Any], candidate: dict[str, Any]) -> float:
    signal = str(weak_signal.get("signal") or "")
    what = str(weak_signal.get("what_it_was") or "")
    why = str(weak_signal.get("why_weak_signal") or "")
    title = str(candidate.get("title") or "")
    abstract = str(candidate.get("abstract") or "")

    text_all = " ".join([topic, signal, what, why])
    signal_tokens = tokenize(signal)
    topic_tokens = tokenize(topic)
    context_tokens = tokenize(text_all)
    candidate_title_tokens = tokenize(title)
    candidate_all_tokens = tokenize(f"{title} {abstract}")

    score = 0.0
    score += 6.0 * len(signal_tokens & candidate_title_tokens)
    score += 3.0 * len(topic_tokens & candidate_title_tokens)
    score += 1.2 * len(context_tokens & candidate_all_tokens)

    norm_signal = normalize_text(signal)
    norm_title = normalize_text(title)
    if norm_signal and norm_signal in norm_title:
        score += 30.0
    if normalize_text(topic) and normalize_text(topic) in norm_title:
        score += 10.0

    year = candidate.get("year")
    if isinstance(year, int):
        if year >= 2020:
            score += 2.0
        if year >= 2023:
            score += 1.0
    return score


def select_references(
    session: requests.Session,
    topic: str,
    weak_signal: dict[str, Any],
    *,
    refs_per_signal: int,
    search_limit: int,
    timeout: float,
    sleep: float,
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, dict[str, Any]]] = []
    seen_paper_ids: set[str] = set()

    for query in build_queries(topic, weak_signal):
        candidates = search_semantic_scholar(session, query, limit=search_limit, timeout=timeout)
        for candidate in candidates:
            paper_id = str(candidate.get("paperId") or "").strip()
            if not paper_id or paper_id in seen_paper_ids:
                continue
            seen_paper_ids.add(paper_id)
            score = candidate_score(topic, weak_signal, candidate)
            ranked.append((score, candidate))
        if sleep > 0:
            time.sleep(sleep)

    ranked.sort(key=lambda item: (item[0], item[1].get("year") or 0), reverse=True)

    references: list[dict[str, Any]] = []
    for score, candidate in ranked:
        title = str(candidate.get("title") or "").strip()
        year = candidate.get("year")
        paper_id = str(candidate.get("paperId") or "").strip()
        if not title or not paper_id:
            continue
        references.append({
            "title": title,
            "year": year,
            "url": candidate.get("url") or f"https://www.semanticscholar.org/paper/{paper_id}",
        })
        if len(references) >= refs_per_signal:
            break
    return references


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill empty references using Semantic Scholar search.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--empty-summary", type=Path, default=DEFAULT_EMPTY_SUMMARY)
    parser.add_argument("--refs-per-signal", type=int, default=3)
    parser.add_argument("--search-limit", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0,
                        help="Only fill this many empty weak signals (0 = all).")
    parser.add_argument("--output-summary", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env()

    summary_payload = json.loads(args.empty_summary.read_text(encoding="utf-8"))
    empties = summary_payload.get("empty_weak_signals") or []

    if args.limit > 0:
        empties = empties[:args.limit]

    session = requests.Session()
    modified_files: set[str] = set()
    filled = 0
    still_empty = 0
    records: list[dict[str, Any]] = []
    file_cache: dict[str, dict[str, Any]] = {}

    for item in empties:
        path_str = item["file"]
        weak_signal_index = int(item["weak_signal_index"])
        path = Path(path_str)
        if path_str not in file_cache:
            file_cache[path_str] = json.loads(path.read_text(encoding="utf-8"))
        data = file_cache[path_str]
        metadata = data.get("metadata") or {}
        topic = str(metadata.get("mainframe_topic") or "")
        weak_signals = (((data.get("result") or {}).get("weak_signals")) or [])
        if weak_signal_index >= len(weak_signals):
            records.append({
                "file": path_str,
                "weak_signal_index": weak_signal_index,
                "status": "missing_weak_signal",
            })
            still_empty += 1
            continue

        weak_signal = weak_signals[weak_signal_index]
        current_refs = weak_signal.get("references") or []
        if isinstance(current_refs, list) and current_refs:
            records.append({
                "file": path_str,
                "weak_signal_index": weak_signal_index,
                "signal": weak_signal.get("signal"),
                "status": "already_nonempty",
                "filled_count": len(current_refs),
            })
            continue

        try:
            new_refs = select_references(
                session,
                topic,
                weak_signal,
                refs_per_signal=args.refs_per_signal,
                search_limit=args.search_limit,
                timeout=args.timeout,
                sleep=args.sleep,
            )
        except Exception as exc:
            new_refs = []
            records.append({
                "file": path_str,
                "weak_signal_index": weak_signal_index,
                "signal": weak_signal.get("signal"),
                "status": "search_error",
                "error": str(exc),
            })

        if new_refs:
            weak_signal["references"] = new_refs
            modified_files.add(path_str)
            filled += 1
            records.append({
                "file": path_str,
                "weak_signal_index": weak_signal_index,
                "signal": weak_signal.get("signal"),
                "status": "filled",
                "filled_count": len(new_refs),
                "references": new_refs,
            })
        else:
            weak_signal["references"] = []
            still_empty += 1
            if not records or records[-1].get("file") != path_str or records[-1].get("weak_signal_index") != weak_signal_index:
                records.append({
                    "file": path_str,
                    "weak_signal_index": weak_signal_index,
                    "signal": weak_signal.get("signal"),
                    "status": "no_candidates",
                })

        processed = filled + still_empty
        if processed % 50 == 0 or processed == len(empties):
            print(
                f"[progress] processed={processed}/{len(empties)} filled={filled} still_empty={still_empty}",
                flush=True,
            )

    for path_str in modified_files:
        path = Path(path_str)
        path.write_text(json.dumps(file_cache[path_str], indent=2, ensure_ascii=False), encoding="utf-8")

    output_summary = args.output_summary or (REPO_ROOT / "construction" / "reference_refill_summary.json")
    payload = {
        "summary": {
            "target_empty_weak_signals": len(empties),
            "filled_weak_signals": filled,
            "still_empty_weak_signals": still_empty,
            "modified_files": len(modified_files),
        },
        "records": records,
    }
    output_summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(payload["summary"], ensure_ascii=False))
    print(str(output_summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
