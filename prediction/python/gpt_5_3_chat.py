#!/usr/bin/env python
# coding: utf-8
"""
GPT-5.3-chat – weak-signal prediction (prediction only, no evaluation).

Usage example:
    python gpt_5_3_chat.py \
        --domain "Natural Language Processing" \
        --spaces problem solution \
        --output-dir ./outputs
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

from prediction_prompts import (
    YEAR_RANGE,
    YEAR_SLUG,
    build_prompt,
    extract_candidate_signals,
    make_topic_slug,
)


# ---------------------------------------------------------------------------
# OpenAI API call
# ---------------------------------------------------------------------------

def run_openrouter_once(
    client,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 8,
    retry_backoff: float = 3.0,
) -> str:
    """Call OpenAI chat completion with retry logic. Returns response text."""
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if not resp.choices:
                raise RuntimeError("OpenRouter returned no choices.")
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("OpenRouter returned empty text.")
            return text
        except Exception as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"OpenRouter request failed after {attempt} attempts: {exc}"
                ) from exc
            sleep_s = retry_backoff ** attempt
            print(f"  [warn] attempt {attempt}: {exc}. Retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    output_dir: Path,
    domain: str,
    space: str,
    topic: str,
    response_text: str,
    signals: List[str],
) -> Path:
    from mainframe_topics import make_domain_slug
    domain_slug = make_domain_slug(domain)
    topic_slug = make_topic_slug(topic)
    result_dir = output_dir / "gpt_5_3_chat" / domain_slug / topic_slug / space / YEAR_SLUG
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    (result_dir / f"response_{timestamp}.txt").write_text(
        response_text, encoding="utf-8"
    )
    (result_dir / "response_latest.txt").write_text(
        response_text, encoding="utf-8"
    )

    signals_payload = {
        "domain": domain,
        "space": space,
        "mainframe_topic": topic,
        "year_range": YEAR_RANGE,
        "timestamp": timestamp,
        "signals": signals,
    }
    (result_dir / f"signals_{timestamp}.json").write_text(
        json.dumps(signals_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (result_dir / "signals_latest.json").write_text(
        json.dumps(signals_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result_dir


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GPT-5.3 Chat via OpenRouter weak-signal prediction (no evaluation)."
    )
    p.add_argument(
        "--domain", nargs="+", default=None,
        help="Domain(s) to predict. If omitted, runs all domains.",
    )
    p.add_argument(
        "--spaces", nargs="+", default=["problem", "solution"],
        choices=["problem", "solution"],
        help="Signal spaces to predict (default: problem solution).",
    )
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument(
        "--model", "--openai-model", dest="model",
        default=os.getenv("OPENROUTER_MODEL", "openai/gpt-5.3-chat"),
        help="OpenRouter model id (default: $OPENROUTER_MODEL or openai/gpt-5.3-chat).",
    )
    p.add_argument(
        "--api-key", "--openai-api-key", dest="api_key",
        default=os.getenv("OPENROUTER_API_KEY"),
        help="OpenRouter API key (default: $OPENROUTER_API_KEY).",
    )
    p.add_argument(
        "--base-url", "--api-base-url", dest="base_url",
        default=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        help="OpenRouter base URL (default: https://openrouter.ai/api/v1).",
    )
    p.add_argument(
        "--http-referer",
        default=os.getenv("OPENROUTER_HTTP_REFERER", ""),
        help="Optional HTTP-Referer header for OpenRouter rankings/analytics.",
    )
    p.add_argument(
        "--x-title",
        default=os.getenv("OPENROUTER_X_TITLE", "weak-signal-benchmark"),
        help="Optional X-Title header for OpenRouter rankings/analytics.",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=32768,
                   help="Maximum output tokens. OpenRouter GPT-5.3 Chat requires >= 16.")
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--retry-backoff", type=float, default=2.0)
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

    if not args.api_key:
        raise RuntimeError(
            "Missing OpenRouter API key. Set OPENROUTER_API_KEY or use --api-key."
        )
    if args.max_tokens < 16:
        raise RuntimeError("Invalid --max-tokens: OpenRouter GPT-5.3 Chat requires a value >= 16.")

    from openai import OpenAI

    default_headers = {}
    if args.http_referer:
        default_headers["HTTP-Referer"] = args.http_referer
    if args.x_title:
        default_headers["X-Title"] = args.x_title

    client = OpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        default_headers=default_headers or None,
    )

    print("=" * 60)
    print(f"Domains:     {domains}")
    print(f"Spaces:      {args.spaces}")
    print(f"Year range:  {YEAR_RANGE}")
    print(f"Model:       {args.model}")
    print(f"Base URL:    {args.base_url}")
    print(f"Output dir:  {args.output_dir}")
    print("=" * 60)

    for domain in domains:
        topics = TOPICS_BY_DOMAIN.get(domain)
        if topics is None:
            print(f"[warn] Unknown domain: {domain}. Skipping.")
            continue
        domain_slug = make_domain_slug(domain)

        for topic in topics:
            for space in args.spaces:
                print(f"\n{'─' * 60}")
                print(f"Domain: {domain}  |  Topic: {topic}  |  Space: {space}")
                print(f"{'─' * 60}")

                topic_slug = make_topic_slug(topic)
                result_dir = args.output_dir / "gpt_5_3_chat" / domain_slug / topic_slug / space / YEAR_SLUG
                if result_dir.exists():
                    print(f"[skip] Already exists: {result_dir}")
                    continue

                prompt = build_prompt(space, domain, topic)
                print(f"Prompt length: {len(prompt)} chars")
                print("Calling OpenRouter ...")

                response_text = run_openrouter_once(
                    client=client,
                    model=args.model,
                    user_prompt=prompt,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    max_retries=args.max_retries,
                    retry_backoff=args.retry_backoff,
                )
                print(f"Response length: {len(response_text)} chars")

                signals = extract_candidate_signals(response_text)
                print(f"Extracted {len(signals)} candidate signals:")
                for i, sig in enumerate(signals, 1):
                    print(f"  {i}. {sig}")

                result_dir = save_results(
                    args.output_dir, domain, space, topic,
                    response_text, signals,
                )
                print(f"Results saved to: {result_dir}")
                time.sleep(10)

    print(f"\n{'=' * 60}")
    print("All predictions complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
