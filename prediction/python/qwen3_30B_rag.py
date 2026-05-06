#!/usr/bin/env python
# coding: utf-8
"""
Qwen3-30B + LlamaIndex RAG -- weak-signal prediction (prediction only, no evaluation).

Same RAG pipeline as qwen3_8B_rag.py but defaults to the larger Qwen3-30B model
with 4-way tensor parallelism.

Usage example:
    python qwen3_30B_rag.py \
        --spaces problem solution \
        --domain "Natural Language Processing" \
        --output-dir ./outputs \
        --retrieval-queries "model-based reinforcement learning language models"
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import socket
import subprocess
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
# Semantic Scholar bulk retrieval (identical to qwen3_8B_rag.py)
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
        params: Dict[str, Any] = {"query": query, "limit": limit, "fields": fields}
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
                print(f"  [warn] S2 error '{query}' ({attempt}/{max_retries}): {exc}. Retrying...")
                time.sleep(sleep_s)

        if payload is None:
            raise RuntimeError(f"S2 failed after {max_retries} attempts for '{query}': {last_exc}")

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
# LlamaIndex RAG helpers (identical to qwen3_8B_rag.py)
# ---------------------------------------------------------------------------

def papers_to_documents(papers: List[Dict[str, Any]]):
    from llama_index.core import Document
    docs = []
    for i, p in enumerate(papers, start=1):
        text = (
            f"[Paper {i}]\nTitle: {p.get('title', 'Untitled')}\n"
            f"Year: {p.get('year', 'N/A')}\nCitations: {p.get('citationCount', 0)}\n"
            f"URL: {p.get('url', 'N/A')}\nAbstract: {p.get('abstract', '')}"
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
    return index.as_retriever(similarity_top_k=top_k).retrieve(query)


def build_evidence_block(ranked_nodes) -> tuple[str, List[Dict]]:
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
            "rank": rank, "title": title, "year": year,
            "url": url, "score": score, "query": md.get("query", ""),
        })
    return "\n\n".join(evidence_lines), metadata_list


def run_rag_retrieval(
    retrieval_queries, cutoff_year, s2_max_total, s2_page_size,
    s2_api_key, s2_max_retries, s2_retry_backoff, rag_top_k,
    space, mainframe_topic, domain,
) -> tuple[str, List[Dict]]:
    all_candidates: List[Dict[str, Any]] = []
    for q in retrieval_queries:
        papers = s2_bulk_fetch_with_cutoff(
            q, cutoff_year, s2_max_total, s2_page_size,
            s2_api_key, s2_max_retries, s2_retry_backoff,
        )
        all_candidates.extend(papers)
        print(f"  Query '{q}': {len(papers)} papers (cutoff={cutoff_year})")

    dedup: Dict[str, Dict] = {}
    for p in all_candidates:
        key = p.get("paperId") or p.get("url") or (p.get("title") or "").lower().strip()
        if key and key not in dedup:
            dedup[key] = p
    unique_papers = list(dedup.values())
    print(f"  Candidates: {len(all_candidates)} raw -> {len(unique_papers)} unique")

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
    model, host, port, tensor_parallel=4, max_model_len=32768, gpu_mem=0.9,
) -> subprocess.Popen | None:
    if port_open(port, host):
        print(f"vLLM already running on {host}:{port}. Reusing.")
        return None
    log_file = f"vllm_server_{port}.log"
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model, "--host", host, "--port", str(port),
        "--dtype", "auto", "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem),
        "--enforce-eager",
    ]
    if tensor_parallel > 1:
        cmd.extend(["--tensor-parallel-size", str(tensor_parallel)])
    print(f"[vLLM] Starting server: {' '.join(cmd)}")
    print(f"[vLLM] Log file: {log_file}")
    log_fh = open(log_file, "w")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    import threading
    def _stream_output():
        for line in proc.stdout:
            line_stripped = line.rstrip("\n")
            print(f"[vLLM] {line_stripped}", flush=True)
            log_fh.write(line)
            log_fh.flush()
        log_fh.close()
    reader_thread = threading.Thread(target=_stream_output, daemon=True)
    reader_thread.start()

    base_url = f"http://{host}:{port}"
    max_wait, interval, waited = 900, 5, 0
    while waited < max_wait:
        time.sleep(interval)
        waited += interval
        # Check if process has crashed
        if proc.poll() is not None:
            reader_thread.join(timeout=5)
            print(f"[vLLM] ERROR: Server process exited with code {proc.returncode} after {waited}s")
            raise RuntimeError(f"vLLM server died with exit code {proc.returncode}. Check {log_file}")
        try:
            r = http_requests.get(f"{base_url}/v1/models", timeout=5)
            if r.status_code == 200:
                print(f"[vLLM] Server ready (pid={proc.pid}) on {host}:{port} after {waited}s")
                return proc
        except Exception:
            print(f"[vLLM] Waiting for server... ({waited}s / {max_wait}s)", flush=True)
    print(f"[vLLM] WARNING: Server may not be ready after {max_wait}s. Check {log_file}")
    return proc


def generate_with_qwen(base_url, model, user_prompt, temperature=0.6) -> str:
    from openai import OpenAI
    client = OpenAI(api_key="EMPTY", base_url=f"{base_url}/v1")
    # Let vLLM use all remaining context for generation (no explicit max_tokens)
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(output_dir, space, domain, topic, response_text, signals, retrieved_metadata) -> Path:
    from mainframe_topics import make_domain_slug
    topic_slug = make_topic_slug(topic)
    domain_slug = make_domain_slug(domain)
    result_dir = output_dir / "qwen3_30b_awq_rag" / domain_slug / topic_slug / space / YEAR_SLUG
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    (result_dir / f"response_{timestamp}.txt").write_text(response_text, encoding="utf-8")
    (result_dir / "response_latest.txt").write_text(response_text, encoding="utf-8")

    payload = {
        "domain": domain, "space": space, "mainframe_topic": topic,
        "year_range": YEAR_RANGE, "timestamp": timestamp,
        "signals": signals, "retrieved_papers": retrieved_metadata,
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
        description="Qwen3-30B + LlamaIndex RAG weak-signal prediction (no evaluation)."
    )
    p.add_argument("--spaces", nargs="+", default=["problem", "solution"],
                   choices=["problem", "solution"],
                   help="Signal spaces to predict.")
    p.add_argument("--domain", nargs="+", default=None,
                   help="Domains to predict. If omitted, uses all domains.")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--retrieval-queries", nargs="+", default=None,
                   help="Semantic Scholar queries. If omitted, uses the topic name as query.")

    p.add_argument("--qwen-model", default=os.getenv("QWEN_MODEL_PATH", "stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ"))
    p.add_argument("--vllm-host", default=os.getenv("QWEN_VLLM_HOST", "127.0.0.1"))
    p.add_argument("--vllm-port", type=int, default=int(os.getenv("QWEN_VLLM_PORT", "6004")))
    p.add_argument("--tensor-parallel", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--skip-vllm-start", action="store_true")

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
    args = parse_args()
    from mainframe_topics import TOPICS_BY_DOMAIN, ALL_DOMAINS, make_domain_slug

    domains = args.domain if args.domain else ALL_DOMAINS
    random.seed(args.seed)
    base_url = f"http://{args.vllm_host}:{args.vllm_port}"

    print("=" * 60)
    print(f"Spaces:            {args.spaces}")
    print(f"Domains:           {domains}")
    print(f"Year range:        {YEAR_RANGE}")
    print(f"Model:             {args.qwen_model}")
    print(f"vLLM:              {base_url}")
    print(f"Tensor parallel:   {args.tensor_parallel}")
    print(f"RAG top-k:         {args.rag_top_k}")
    print(f"Retrieval queries: {args.retrieval_queries}")
    print(f"Output dir:        {args.output_dir}")
    print("=" * 60)

    if not args.skip_vllm_start:
        start_vllm_server(
            args.qwen_model, args.vllm_host, args.vllm_port,
            args.tensor_parallel, args.max_model_len, args.gpu_memory_utilization,
        )

    cutoff_year = year_range_to_cutoff(YEAR_RANGE)

    for domain in domains:
        topics = TOPICS_BY_DOMAIN.get(domain, [])
        domain_slug = make_domain_slug(domain)
        for topic in topics:
            for space in args.spaces:
                print(f"\n{'---' * 20}")
                print(f"Domain: {domain}  |  Topic: {topic}  |  Space: {space}")
                print(f"{'---' * 20}")

                topic_slug = make_topic_slug(topic)
                result_dir = args.output_dir / "qwen3_30b_awq_rag" / domain_slug / topic_slug / space / YEAR_SLUG
                if result_dir.exists():
                    print(f"[skip] Already exists: {result_dir}")
                    continue

                print(f"S2 cutoff year: {cutoff_year}")

                print("Running RAG retrieval ...")
                evidence_block, retrieved_metadata = run_rag_retrieval(
                    args.retrieval_queries or [topic], cutoff_year,
                    args.s2_max_total, args.s2_page_size,
                    args.s2_api_key, args.s2_max_retries, args.s2_retry_backoff,
                    args.rag_top_k, space, topic, domain,
                )

                base_prompt = build_prompt(space, domain, topic)
                suffix = "\n\nUse this evidence when generating the weak signals."
                evidence_papers = evidence_block.split("\n\n")
                max_input_tokens = args.max_model_len

                while True:
                    cur_evidence = "\n\n".join(evidence_papers)
                    augmented_prompt = (
                        f"{base_prompt}\n\nRetrieved paper evidence "
                        f"(already time-cutoff filtered):\n{cur_evidence}"
                        f"{suffix}"
                    )
                    estimated_tokens = len(augmented_prompt) // 3
                    if estimated_tokens <= max_input_tokens or len(evidence_papers) <= 1:
                        break
                    evidence_papers.pop()

                print(f"Augmented prompt length: {len(augmented_prompt)} chars "
                      f"(~{estimated_tokens} tokens, {len(evidence_papers)} papers kept)")

                print("Calling Qwen via vLLM ...")
                response_text = generate_with_qwen(
                    base_url, args.qwen_model, augmented_prompt,
                    args.temperature,
                )
                print(f"Response length: {len(response_text)} chars")

                signals = extract_candidate_signals(response_text)
                print(f"Extracted {len(signals)} candidate signals:")
                for i, sig in enumerate(signals, 1):
                    print(f"  {i}. {sig}")

                result_dir = save_results(
                    args.output_dir, space, domain, topic,
                    response_text, signals, retrieved_metadata,
                )
                print(f"Results saved to: {result_dir}")

    print(f"\n{'=' * 60}")
    print("All predictions complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
