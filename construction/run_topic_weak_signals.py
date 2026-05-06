#!/usr/bin/env python3
"""Generate problem-space and solution-space weak signals for one topic."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from openai import OpenAI

from direction_prompts import format_problem_prompt, format_solution_prompt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "construction" / "outputs"
DEFAULT_LOG_ROOT = REPO_ROOT / "construction" / "logs"
ENV_CANDIDATES = [
    REPO_ROOT / ".env",
    REPO_ROOT / "prediction" / "python" / ".env",
]

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,year,url,abstract,externalIds"
S2_LIMIT = 5  # papers per search call

WEB_SEARCH_TOOL_DEF: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for academic papers and research information. "
            "Use this to find and verify real publications."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for finding academic papers.",
                }
            },
            "required": ["query"],
        },
    },
}

MAX_TOOL_ROUNDS = 15  # safety cap on agentic loop iterations


def load_env() -> None:
    load_dotenv()
    for env_path in ENV_CANDIDATES:
        if env_path.exists():
            load_dotenv(env_path, override=False)


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def ensure_payload_shape(payload: dict[str, Any], *, topic: str, direction: str) -> None:
    if not isinstance(payload, dict) or "weak_signals" not in payload:
        raise ValueError(f"{direction} response for topic '{topic}' is missing 'weak_signals'.")
    if not isinstance(payload["weak_signals"], list):
        raise ValueError(f"{direction} response for topic '{topic}' has non-list 'weak_signals'.")


def build_prompt(direction: str, *, field: str, topic: str) -> str:
    if direction == "problem":
        return format_problem_prompt(mainframe_topic=topic, field=field)
    if direction == "solution":
        return format_solution_prompt(mainframe_topic=topic, field=field)
    raise ValueError(f"Unsupported direction: {direction}")


# ---------------------------------------------------------------------------
# Semantic Scholar search (used as web_search tool backend)
# ---------------------------------------------------------------------------

def semantic_scholar_search(query: str) -> str:
    """Execute a search against Semantic Scholar and return formatted results."""
    s2_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    headers: dict[str, str] = {}
    if s2_key:
        headers["x-api-key"] = s2_key

    try:
        resp = requests.get(
            S2_SEARCH_URL,
            params={"query": query, "limit": S2_LIMIT, "fields": S2_FIELDS},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"[search error] {exc}"

    papers = data.get("data") or []
    if not papers:
        return "[no results found]"

    lines: list[str] = []
    for p in papers:
        arxiv_id = (p.get("externalIds") or {}).get("ArXiv")
        url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else (p.get("url") or "")
        abstract = (p.get("abstract") or "")[:300]
        lines.append(
            f"- title: {p.get('title', '?')}\n"
            f"  year: {p.get('year', '?')}\n"
            f"  url: {url}\n"
            f"  abstract: {abstract}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model calling
# ---------------------------------------------------------------------------

def extract_text(response: Any) -> str:
    """Extract text from a ChatCompletion response."""
    choice = response.choices[0]
    text = choice.message.content
    if not text:
        raise RuntimeError("Model returned no text output.")
    return text.strip()


def call_model(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    max_retries: int,
    retry_backoff: float,
    web_search: bool = False,
) -> tuple[str, dict[str, Any], int]:
    """Call the model, optionally with an agentic web-search tool-use loop.

    Returns (raw_text, parsed_json, web_search_count).
    """
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            if web_search:
                return _call_model_with_search(client, model=model, prompt=prompt)
            else:
                return _call_model_plain(client, model=model, prompt=prompt)
        except Exception as exc:
            last_error = exc
            if attempt == max_retries:
                break
            sleep_s = retry_backoff * attempt
            print(f"[warn] attempt {attempt} failed: {exc}; retrying in {sleep_s:.1f}s", flush=True)
            time.sleep(sleep_s)

    raise RuntimeError(f"OpenAI request failed after {max_retries} attempts: {last_error}") from last_error


def _call_model_plain(
    client: OpenAI, *, model: str, prompt: str
) -> tuple[str, dict[str, Any], int]:
    """Single-shot call without tools."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    raw_text = extract_text(response)
    parsed = json.loads(strip_code_fences(raw_text))
    if not isinstance(parsed, dict):
        raise ValueError("Response JSON root must be an object.")
    return raw_text, parsed, 0


def _call_model_with_search(
    client: OpenAI, *, model: str, prompt: str
) -> tuple[str, dict[str, Any], int]:
    """Agentic tool-use loop: model calls web_search, we execute via Semantic Scholar."""
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    web_search_count = 0

    for round_idx in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[WEB_SEARCH_TOOL_DEF],
            tool_choice="auto",
        )

        choice = response.choices[0]
        assistant_msg = choice.message

        # Append the assistant message to conversation history
        messages.append(_message_to_dict(assistant_msg))

        # If model is done (no more tool calls), extract final text
        if choice.finish_reason != "tool_calls" or not assistant_msg.tool_calls:
            raw_text = (assistant_msg.content or "").strip()
            if not raw_text:
                raise RuntimeError("Model finished without producing text output.")
            parsed = json.loads(strip_code_fences(raw_text))
            if not isinstance(parsed, dict):
                raise ValueError("Response JSON root must be an object.")
            return raw_text, parsed, web_search_count

        # Execute each tool call
        for tc in assistant_msg.tool_calls:
            if tc.function.name == "web_search":
                args = json.loads(tc.function.arguments)
                query = args.get("query", "")
                web_search_count += 1
                print(f"    [search #{web_search_count}] {query}", flush=True)
                result = semantic_scholar_search(query)
            else:
                result = f"[error] Unknown tool: {tc.function.name}"

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # Exhausted rounds — try to extract whatever we have
    raise RuntimeError(
        f"Agentic loop did not finish after {MAX_TOOL_ROUNDS} rounds "
        f"({web_search_count} searches executed)"
    )


def _message_to_dict(msg: Any) -> dict[str, Any]:
    """Convert an OpenAI message object to a plain dict for the messages list."""
    d: dict[str, Any] = {"role": msg.role}
    if msg.content:
        d["content"] = msg.content
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return d


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_direction_result(
    *,
    output_root: Path,
    domain: str,
    topic: str,
    direction: str,
    field: str,
    model: str,
    response_text: str,
    payload: dict[str, Any],
    web_search_count: int = 0,
) -> Path:
    result_dir = output_root / slugify(domain) / slugify(topic) / direction
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tool_type = "web_search_preview" if web_search_count > 0 else "none"
    metadata = {
        "domain": domain,
        "field": field,
        "mainframe_topic": topic,
        "direction": direction,
        "model": model,
        "timestamp": timestamp,
        "tool": tool_type,
        "web_search_count": web_search_count,
    }

    # Inject web_search count into the result payload alongside weak_signals
    payload["web_search"] = web_search_count

    (result_dir / f"response_{timestamp}.txt").write_text(response_text, encoding="utf-8")
    (result_dir / "response_latest.txt").write_text(response_text, encoding="utf-8")
    (result_dir / f"result_{timestamp}.json").write_text(
        json.dumps({"metadata": metadata, "result": payload}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (result_dir / "result_latest.json").write_text(
        json.dumps({"metadata": metadata, "result": payload}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate weak signals for one topic.")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--field", default=None, help="Override the research field inserted into the prompt.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-5.4"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument("--user-agent", default=os.getenv("OPENAI_USER_AGENT"))
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--web-search", action="store_true", help="Enable agentic web search via Semantic Scholar.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env()

    api_key = os.getenv("IKUNCODE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing IKUNCODE_API_KEY.")

    field = args.field or args.domain
    args.log_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    if args.user_agent:
        client_kwargs["default_headers"] = {"User-Agent": args.user_agent}
    client = OpenAI(**client_kwargs)

    print("=" * 72, flush=True)
    print(f"domain: {args.domain}", flush=True)
    print(f"topic: {args.topic}", flush=True)
    print(f"field: {field}", flush=True)
    print(f"model: {args.model}", flush=True)
    print(f"web_search: {args.web_search}", flush=True)
    if args.base_url:
        print(f"base_url: {args.base_url}", flush=True)
    print("=" * 72, flush=True)

    for direction in ("problem", "solution"):
        result_dir = args.output_root / slugify(args.domain) / slugify(args.topic) / direction
        latest_json = result_dir / "result_latest.json"
        if args.skip_existing and latest_json.exists():
            print(f"[skip] {direction}: {latest_json}", flush=True)
            continue

        prompt = build_prompt(direction, field=field, topic=args.topic)
        print(f"[start] {direction}", flush=True)
        response_text, payload, web_search_count = call_model(
            client,
            model=args.model,
            prompt=prompt,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
            web_search=args.web_search,
        )
        ensure_payload_shape(payload, topic=args.topic, direction=direction)
        saved_dir = save_direction_result(
            output_root=args.output_root,
            domain=args.domain,
            topic=args.topic,
            direction=direction,
            field=field,
            model=args.model,
            response_text=response_text,
            payload=payload,
            web_search_count=web_search_count,
        )
        print(f"[done] {direction}: {saved_dir} (web_searches={web_search_count})", flush=True)

    print("[complete] topic finished", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        raise
