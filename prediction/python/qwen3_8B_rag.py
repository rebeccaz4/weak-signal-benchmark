#!/usr/bin/env python
# coding: utf-8
"""
Qwen3-8B + LlamaIndex RAG -- weak-signal prediction (prediction only, no evaluation).

Pipeline:
  1. Fetch papers from Semantic Scholar (with year cutoff)
  2. Rank papers via LlamaIndex vector similarity
  3. Build evidence block from top-K papers
  4. Append evidence to prompt and call Qwen3-8B via local vLLM
  5. Extract signals and save results

Usage example:
    python qwen3_8B_rag.py \
        --spaces problem solution \
        --domain "Natural Language Processing" \
        --output-dir ./outputs \
        --retrieval-queries "reward type NLP" "process or outcome NLP"
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests as http_requests
from dotenv import load_dotenv

load_dotenv()

from prediction_prompts import (
    YEAR_RANGE,
    YEAR_SLUG,
    build_prompt,
    extract_candidate_signals,
    make_topic_slug,
)


def year_range_to_cutoff(year_range: str) -> int:
    return int(year_range.split("-")[0]) - 1


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


# ---------------------------------------------------------------------------
# Semantic Scholar bulk retrieval
# ---------------------------------------------------------------------------

S2_BULK_ENDPOINT = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"


def s2_bulk_fetch_with_cutoff(
    query: str,
    cutoff_year: int,
    max_total: int,
    page_size: int,
    api_key: str,
    max_retries: int = 5,
    retry_backoff: float = 1.5,
) -> List[Dict[str, Any]]:
    """Fetch papers from Semantic Scholar bulk API with year cutoff filtering."""
    if not api_key:
        raise RuntimeError(
            "Missing Semantic Scholar API key. "
            "Set SEMANTIC_SCHOLAR_API_KEY or S2_API_KEY."
        )

    headers = {"x-api-key": api_key}
    fields = "paperId,title,year,abstract,url,citationCount"
    token = None
    scanned = 0
    kept: List[Dict[str, Any]] = []

    while scanned < max_total:
        limit = min(page_size, max_total - scanned)
        params: Dict[str, Any] = {
            "query": query,
            "limit": limit,
            "fields": fields,
        }
        if token:
            params["token"] = token

        payload = None
        last_exc = None

        for attempt in range(1, max_retries + 1):
            try:
                resp = http_requests.get(
                    S2_BULK_ENDPOINT, headers=headers, params=params, timeout=60,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    raise http_requests.HTTPError(
                        f"Transient HTTP {resp.status_code}", response=resp,
                    )
                resp.raise_for_status()
                payload = resp.json()
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= max_retries:
                    break
                sleep_s = retry_backoff * attempt
                print(
                    f"  [warn] S2 bulk error for query='{query}' "
                    f"(attempt {attempt}/{max_retries}): {exc}. "
                    f"Retrying in {sleep_s:.1f}s..."
                )
                time.sleep(sleep_s)

        if payload is None:
            raise RuntimeError(
                f"S2 bulk request failed after {max_retries} attempts "
                f"for query='{query}': {last_exc}"
            )

        batch = payload.get("data", []) or []
        if not batch:
            break

        for p in batch:
            scanned += 1
            y = p.get("year")
            if isinstance(y, int) and y <= cutoff_year:
                kept.append({
                    "paperId": p.get("paperId") or "",
                    "title": p.get("title") or "Untitled",
                    "year": y,
                    "abstract": (p.get("abstract") or "").strip(),
                    "url": p.get("url") or "N/A",
                    "citationCount": int(p.get("citationCount") or 0),
                    "query": query,
                })

        token = payload.get("token")
        if not token:
            break

    return kept


# ---------------------------------------------------------------------------
# LlamaIndex RAG helpers
# ---------------------------------------------------------------------------

def papers_to_documents(papers: List[Dict[str, Any]]):
    """Convert retrieved papers into LlamaIndex Document objects."""
    from llama_index.core import Document

    docs = []
    for i, p in enumerate(papers, start=1):
        text = (
            f"[Paper {i}]\n"
            f"Title: {p.get('title', 'Untitled')}\n"
            f"Year: {p.get('year', 'N/A')}\n"
            f"Citations: {p.get('citationCount', 0)}\n"
            f"URL: {p.get('url', 'N/A')}\n"
            f"Abstract: {p.get('abstract', '')}"
        )
        docs.append(Document(
            text=text,
            metadata={
                "paper_id": p.get("paperId", ""),
                "title": p.get("title", "Untitled"),
                "year": p.get("year"),
                "url": p.get("url", "N/A"),
                "citationCount": p.get("citationCount", 0),
                "query": p.get("query", ""),
            },
        ))
    return docs


def llamaindex_rank_topk(documents, query: str, top_k: int):
    """Build in-memory VectorStoreIndex and return top-k ranked nodes."""
    from llama_index.core import VectorStoreIndex, Settings

    if not documents:
        return []
    try:
        # Try OpenAI embedding first (requires valid OPENAI_API_KEY)
        index = VectorStoreIndex.from_documents(documents)
    except Exception:
        # Fallback to local HuggingFace embedding
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        print("  OpenAI embedding unavailable, using local BAAI/bge-base-en-v1.5")
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        index = VectorStoreIndex.from_documents(documents)
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(query)


def build_evidence_block(ranked_nodes) -> tuple[str, List[Dict]]:
    """Build evidence text block and metadata list from ranked nodes."""
    evidence_lines: List[str] = []
    metadata_list: List[Dict] = []

    for rank, nws in enumerate(ranked_nodes, start=1):
        md = nws.node.metadata or {}
        title = md.get("title", "Untitled")
        year = md.get("year")
        url = md.get("url", "N/A")
        score = float(nws.score) if nws.score is not None else None

        evidence_lines.append(
            f"[{rank}] Title: {title}\nYear: {year}\nURL: {url}\nScore: {score}"
        )
        metadata_list.append({
            "rank": rank,
            "title": title,
            "year": year,
            "url": url,
            "score": score,
            "query": md.get("query", ""),
        })

    return "\n\n".join(evidence_lines), metadata_list


# ---------------------------------------------------------------------------
# RAG retrieval pipeline
# ---------------------------------------------------------------------------

def run_rag_retrieval(
    retrieval_queries: List[str],
    cutoff_year: int,
    s2_max_total: int,
    s2_page_size: int,
    s2_api_key: str,
    s2_max_retries: int,
    s2_retry_backoff: float,
    rag_top_k: int,
    space: str,
    mainframe_topic: str,
    domain: str,
) -> tuple[str, List[Dict]]:
    """Run full RAG retrieval: S2 fetch -> deduplicate -> LlamaIndex rank -> evidence block."""
    # Fetch from Semantic Scholar
    all_candidates: List[Dict[str, Any]] = []
    for q in retrieval_queries:
        papers = s2_bulk_fetch_with_cutoff(
            query=q,
            cutoff_year=cutoff_year,
            max_total=s2_max_total,
            page_size=s2_page_size,
            api_key=s2_api_key,
            max_retries=s2_max_retries,
            retry_backoff=s2_retry_backoff,
        )
        all_candidates.extend(papers)
        print(f"  Query '{q}': {len(papers)} papers (cutoff={cutoff_year})")

    # Deduplicate
    dedup: Dict[str, Dict] = {}
    for p in all_candidates:
        key = (
            p.get("paperId")
            or p.get("url")
            or (p.get("title") or "").lower().strip()
        )
        if key and key not in dedup:
            dedup[key] = p
    unique_papers = list(dedup.values())
    print(f"  Candidates: {len(all_candidates)} raw -> {len(unique_papers)} unique")

    # LlamaIndex ranking
    documents = papers_to_documents(unique_papers)
    ranking_query = (
        f"Identify papers most useful for finding early weak {space}-space "
        f"signals in {domain} that later relate to the mainframe topic: {mainframe_topic}."
    )
    ranked_nodes = llamaindex_rank_topk(documents, ranking_query, rag_top_k)
    print(f"  Ranked top-k: {len(ranked_nodes)}")

    return build_evidence_block(ranked_nodes)


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------

def start_vllm_server(
    model: str,
    host: str,
    port: int,
    tensor_parallel: int = 1,
    max_model_len: int = 32768,
    gpu_mem: float = 0.9,
) -> subprocess.Popen | None:
    """Start vLLM OpenAI-compatible server if not already running."""
    if port_open(port, host):
        print(f"vLLM already running on {host}:{port}. Reusing.")
        return None

    log_file = f"vllm_server_{port}.log"
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", host,
        "--port", str(port),
        "--dtype", "bfloat16",
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem),
    ]
    if tensor_parallel > 1:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel)])

    log_fh = open(log_file, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Tee subprocess output to both terminal and log file
    import threading

    def _tee_output(pipe, log_fh):
        for line in pipe:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_fh.write(line)
            log_fh.flush()
        log_fh.close()

    tee_thread = threading.Thread(target=_tee_output, args=(proc.stdout, log_fh), daemon=True)
    tee_thread.start()

    # Wait for server readiness
    base_url = f"http://{host}:{port}"
    max_wait = 900  # 15 minutes
    interval = 5
    waited = 0
    while waited < max_wait:
        time.sleep(interval)
        waited += interval
        # Check if the process has crashed
        if proc.poll() is not None:
            tee_thread.join(timeout=5)
            raise RuntimeError(
                f"vLLM server died with exit code {proc.returncode} after {waited}s. "
                f"This is likely a GPU OOM error. Try reducing --max-model-len or "
                f"--gpu-memory-utilization. Check {log_file} for details."
            )
        try:
            # Use /health endpoint which checks engine core, not just API server
            r = http_requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                print(f"vLLM ready (pid={proc.pid}) on {host}:{port} after {waited}s")
                return proc
        except Exception:
            pass

    print(f"[warn] vLLM may not be ready after {max_wait}s. Check {log_file}")
    return proc


# ---------------------------------------------------------------------------
# Qwen generation
# ---------------------------------------------------------------------------

def generate_with_qwen(
    base_url: str,
    model: str,
    user_prompt: str,
    temperature: float = 0.7,
    max_tokens: int | None = None,
) -> str:
    """Send prompt to local vLLM Qwen server and return response text."""
    from openai import OpenAI

    client = OpenAI(api_key="EMPTY", base_url=f"{base_url}/v1", timeout=3600)
    kwargs: dict = dict(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens

    # First try with requests to get full error details on failure
    import requests as _req
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    raw = _req.post(f"{base_url}/v1/chat/completions", json=payload, timeout=3600)
    if raw.status_code != 200:
        print(f"[vLLM error] status={raw.status_code}")
        print(f"[vLLM error] body={raw.text}")
        raw.raise_for_status()

    import json as _json
    data = raw.json()
    return (data["choices"][0]["message"]["content"] or "").strip()


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    output_dir: Path,
    space: str,
    topic: str,
    domain: str,
    response_text: str,
    signals: List[str],
    retrieved_metadata: List[Dict],
) -> Path:
    from mainframe_topics import make_domain_slug

    topic_slug = make_topic_slug(topic)
    domain_slug = make_domain_slug(domain)
    result_dir = output_dir / "qwen3_8b_rag" / domain_slug / topic_slug / space / YEAR_SLUG
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Raw response
    (result_dir / f"response_{timestamp}.txt").write_text(
        response_text, encoding="utf-8"
    )
    (result_dir / "response_latest.txt").write_text(
        response_text, encoding="utf-8"
    )

    # Signals + retrieval metadata
    payload = {
        "domain": domain,
        "space": space,
        "mainframe_topic": topic,
        "year_range": YEAR_RANGE,
        "timestamp": timestamp,
        "signals": signals,
        "retrieved_papers": retrieved_metadata,
    }
    (result_dir / f"signals_{timestamp}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (result_dir / "signals_latest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result_dir


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen3-8B + LlamaIndex RAG weak-signal prediction (no evaluation)."
    )
    # Core
    p.add_argument("--spaces", nargs="+", default=["problem", "solution"],
                   choices=["problem", "solution"])
    p.add_argument("--domain", nargs="+", default=None,
                   help="Domains to predict. If omitted, uses all domains.")
    p.add_argument("--output-dir", required=True, type=Path)

    # RAG retrieval queries (topic-specific, user must provide)
    p.add_argument(
        "--retrieval-queries", nargs="+", default=None,
        help='Semantic Scholar queries for paper retrieval. If omitted, uses the topic name as query.',
    )

    # Qwen / vLLM
    p.add_argument("--qwen-model", default=os.getenv("QWEN_MODEL_PATH", "Qwen/Qwen3-8B"))
    p.add_argument("--vllm-host", default=os.getenv("QWEN_VLLM_HOST", "127.0.0.1"))
    p.add_argument("--vllm-port", type=int, default=int(os.getenv("QWEN_VLLM_PORT", "6003")))
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.7)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--skip-vllm-start", action="store_true", help="Skip vLLM server startup")

    # RAG parameters
    p.add_argument("--rag-top-k", type=int, default=int(os.getenv("RAG_TOP_K", "30")))
    p.add_argument("--s2-max-total", type=int, default=10000)
    p.add_argument("--s2-page-size", type=int, default=100)
    p.add_argument("--s2-api-key", default=os.getenv("SEMANTIC_SCHOLAR_API_KEY") or os.getenv("S2_API_KEY", ""))
    p.add_argument("--s2-max-retries", type=int, default=5)
    p.add_argument("--s2-retry-backoff", type=float, default=1.5)

    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from mainframe_topics import TOPICS_BY_DOMAIN, ALL_DOMAINS, make_domain_slug

    args = parse_args()
    domains = args.domain if args.domain else ALL_DOMAINS
    random.seed(args.seed)

    base_url = f"http://{args.vllm_host}:{args.vllm_port}"

    print("=" * 60)
    print(f"Spaces:            {args.spaces}")
    print(f"Domains:           {domains}")
    print(f"Year range:        {YEAR_RANGE}")
    print(f"Model:             {args.qwen_model}")
    print(f"vLLM:              {base_url}")
    print(f"RAG top-k:         {args.rag_top_k}")
    print(f"Retrieval queries: {args.retrieval_queries}")
    print(f"Output dir:        {args.output_dir}")
    print("=" * 60)

    # Start vLLM if needed
    if not args.skip_vllm_start:
        start_vllm_server(
            model=args.qwen_model,
            host=args.vllm_host,
            port=args.vllm_port,
            tensor_parallel=args.tensor_parallel,
            max_model_len=args.max_model_len,
            gpu_mem=args.gpu_memory_utilization,
        )

    cutoff_year = year_range_to_cutoff(YEAR_RANGE)

    for domain in domains:
        topics = TOPICS_BY_DOMAIN.get(domain, [])
        domain_slug = make_domain_slug(domain)

        for topic in topics:
            for space in args.spaces:
                print(f"\n{'─' * 60}")
                print(f"Domain: {domain}  |  Topic: {topic}  |  Space: {space}")
                print(f"{'─' * 60}")

                topic_slug = make_topic_slug(topic)
                result_dir = args.output_dir / "qwen3_8b_rag" / domain_slug / topic_slug / space / YEAR_SLUG
                if result_dir.exists():
                    print(f"[skip] Already exists: {result_dir}")
                    continue

                print(f"S2 cutoff year: {cutoff_year}")

                # Step 1: RAG retrieval
                print("Running RAG retrieval ...")
                evidence_block, retrieved_metadata = run_rag_retrieval(
                    retrieval_queries=args.retrieval_queries or [topic],
                    cutoff_year=cutoff_year,
                    s2_max_total=args.s2_max_total,
                    s2_page_size=args.s2_page_size,
                    s2_api_key=args.s2_api_key,
                    s2_max_retries=args.s2_max_retries,
                    s2_retry_backoff=args.s2_retry_backoff,
                    rag_top_k=args.rag_top_k,
                    space=space,
                    mainframe_topic=topic,
                    domain=domain,
                )

                # Step 2: Build augmented prompt (with dynamic truncation)
                base_prompt = build_prompt(space, domain, topic)
                suffix = "\n\nUse this evidence when generating the weak signals."
                evidence_papers = evidence_block.split("\n\n")
                max_input_tokens = args.max_model_len  # will truncate if needed

                while True:
                    cur_evidence = "\n\n".join(evidence_papers)
                    augmented_prompt = (
                        f"{base_prompt}\n\n"
                        "Retrieved paper evidence (already time-cutoff filtered):\n"
                        f"{cur_evidence}"
                        f"{suffix}"
                    )
                    estimated_tokens = len(augmented_prompt) // 3
                    if estimated_tokens <= max_input_tokens or len(evidence_papers) <= 1:
                        break
                    evidence_papers.pop()

                print(f"Augmented prompt length: {len(augmented_prompt)} chars "
                      f"(~{estimated_tokens} tokens, {len(evidence_papers)} papers kept)")

                # Step 3: Generate (let vLLM auto-allocate remaining context for output)
                print("Calling Qwen via vLLM ...")
                response_text = generate_with_qwen(
                    base_url=base_url,
                    model=args.qwen_model,
                    user_prompt=augmented_prompt,
                    temperature=args.temperature,
                )
                print(f"Response length: {len(response_text)} chars")

                # Step 4: Extract signals
                signals = extract_candidate_signals(response_text)
                print(f"Extracted {len(signals)} candidate signals:")
                for i, sig in enumerate(signals, 1):
                    print(f"  {i}. {sig}")

                # Step 5: Save
                result_dir = save_results(
                    args.output_dir, space, topic, domain,
                    response_text, signals, retrieved_metadata,
                )
                print(f"Results saved to: {result_dir}")

    print(f"\n{'=' * 60}")
    print("All predictions complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
