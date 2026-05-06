"""LLM judge execution — async pairwise matching with retry and multi-run orchestration."""

from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Optional

import pandas as pd
from openai import AsyncOpenAI

from wsb.config import GEMINI_API_KEY, GEMINI_BASE_URL
from wsb.evaluate.cost import UsageTracker
from wsb.evaluate.metrics import compute_metrics, flatten_metric_runs
from wsb.evaluate.prompts import PAIRWISE_SYSTEM_PROMPT, build_pairwise_prompt

DEFAULT_BATCH_SIZE = 5
DEFAULT_N_WORKERS = 8


def _make_client(judge_model: str) -> AsyncOpenAI:
    """Create an AsyncOpenAI client, using Gemini's endpoint for Gemini models."""
    if judge_model.startswith("gemini"):
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required for Gemini models."
            )
        return AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=GEMINI_API_KEY)
    return AsyncOpenAI()


def _safe_json_loads(text: str) -> Optional[dict]:
    """Robust JSON parsing with regex fallback."""
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None


async def _async_judge_match(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    expected_count: int,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    max_retries: int = 4,
    retry_backoff: float = 2.0,
    semaphore: asyncio.Semaphore | None = None,
    tracker: UsageTracker | None = None,
) -> list[int]:
    """Batched binary matching call with retry. Returns list of 0/1 values."""
    if semaphore is not None:
        async with semaphore:
            return await _async_judge_match(
                client, model, system_prompt, user_prompt, expected_count,
                temperature, max_tokens, max_retries, retry_backoff,
                semaphore=None, tracker=tracker,
            )
    attempt = 0
    while True:
        attempt += 1
        try:
            kwargs: dict = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if model.startswith("gemini"):
                # Gemini's OpenAI-compat endpoint does not support
                # temperature, max_tokens, or response_format.
                pass
            else:
                kwargs["temperature"] = temperature
                kwargs["max_completion_tokens"] = max_tokens
                kwargs["response_format"] = {"type": "json_object"}
            resp = await client.chat.completions.create(**kwargs)

            if tracker is not None:
                tracker.record(resp.usage)

            if not resp.choices:
                raise ValueError("No choices returned from model.")

            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("Empty response from model.")

            payload = _safe_json_loads(text)
            if payload is None:
                raise ValueError("Could not parse JSON from model response.")

            matches = payload.get("matches")
            if not isinstance(matches, list):
                raise ValueError(f"Expected 'matches' list, got: {type(matches)}")

            if len(matches) != expected_count:
                raise ValueError(
                    f"Expected {expected_count} matches, got {len(matches)}"
                )

            result = [int(m) for m in matches]
            if not all(v in (0, 1) for v in result):
                raise ValueError(f"Expected all 0/1 values, got: {result}")

            return result

        except Exception as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"LLM judge failed after {attempt} attempts: {exc}"
                ) from exc
            sleep_s = retry_backoff * attempt
            print(
                f"[warn] LLM error (attempt {attempt}): {exc}. "
                f"Retrying in {sleep_s:.1f}s..."
            )
            await asyncio.sleep(sleep_s)


async def _async_batch_match(
    client: AsyncOpenAI,
    model: str,
    reference: list[str],
    candidates: list[str],
    temperature: float,
    batch_size: int,
    max_retries: int = 4,
    retry_backoff: float = 2.0,
    semaphore: asyncio.Semaphore | None = None,
    tracker: UsageTracker | None = None,
) -> list[int]:
    """Run batched matching: for each candidate, does it match any reference? Returns 0/1 list."""
    n_batches = math.ceil(len(candidates) / batch_size)
    batches = [
        candidates[i * batch_size : (i + 1) * batch_size]
        for i in range(n_batches)
    ]

    tasks = [
        _async_judge_match(
            client=client,
            model=model,
            system_prompt=PAIRWISE_SYSTEM_PROMPT,
            user_prompt=build_pairwise_prompt(reference, batch),
            expected_count=len(batch),
            temperature=temperature,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            semaphore=semaphore,
            tracker=tracker,
        )
        for batch in batches
    ]

    batch_results = await asyncio.gather(*tasks)
    matches: list[int] = []
    for result in batch_results:
        matches.extend(result)
    return matches


async def _async_run_single_eval(
    client: AsyncOpenAI,
    ground_truth: list[str],
    external: list[str],
    model: str,
    temperature: float = 1.0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_retries: int = 4,
    retry_backoff: float = 2.0,
    semaphore: asyncio.Semaphore | None = None,
    tracker: UsageTracker | None = None,
) -> dict:
    """One evaluation run: precision pass + recall pass in parallel."""
    # Precision: for each external, does it match any GT?
    precision_task = _async_batch_match(
        client, model, ground_truth, external,
        temperature, batch_size, max_retries, retry_backoff, semaphore, tracker,
    )
    # Recall: for each GT, does it match any external?
    recall_task = _async_batch_match(
        client, model, external, ground_truth,
        temperature, batch_size, max_retries, retry_backoff, semaphore, tracker,
    )

    precision_matches, recall_matches = await asyncio.gather(precision_task, recall_task)

    return compute_metrics(precision_matches, recall_matches, len(ground_truth), len(external))


async def _async_run_evaluation(
    ground_truth: list[str],
    external: list[str],
    n_runs: int = 10,
    model_name: str = "external",
    judge_model: str = "gpt-5-mini",
    temperature: float = 1.0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_workers: int = DEFAULT_N_WORKERS,
) -> tuple[pd.DataFrame, UsageTracker]:
    """Run N judge iterations concurrently and return metrics + usage tracker."""
    async with _make_client(judge_model) as client:
        semaphore = asyncio.Semaphore(n_workers)
        tracker = UsageTracker(judge_model)

        n_p_batches = math.ceil(len(external) / batch_size)
        n_r_batches = math.ceil(len(ground_truth) / batch_size)
        print(f"Judge model: {judge_model} | temperature={temperature} | batch_size={batch_size} | workers={n_workers}")
        print(f"Ground-truth count: {len(ground_truth)}")
        print(f"External count:     {len(external)}")
        print(f"Running {n_runs} iterations (async, {n_p_batches}+{n_r_batches} batches/run)...\n")

        tasks = [
            _async_run_single_eval(
                client=client,
                ground_truth=ground_truth,
                external=external,
                model=judge_model,
                temperature=temperature,
                batch_size=batch_size,
                semaphore=semaphore,
                tracker=tracker,
            )
            for _ in range(n_runs)
        ]

        raw_runs = await asyncio.gather(*tasks)
        raw_runs = list(raw_runs)

        for i, result in enumerate(raw_runs, 1):
            p = result.get("precision", "?")
            r = result.get("recall", "?")
            f = result.get("f1", "?")
            print(f"  Run {i}/{n_runs}: P={p}  R={r}  F1={f}")

        return flatten_metric_runs(raw_runs, model_name), tracker


def run_evaluation(
    ground_truth: list[str],
    external: list[str],
    n_runs: int = 10,
    model_name: str = "external",
    judge_model: str = "gpt-5-mini",
    temperature: float = 1.0,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_workers: int = DEFAULT_N_WORKERS,
) -> tuple[pd.DataFrame, UsageTracker]:
    """Sync wrapper around async evaluation. Returns (metrics_df, usage_tracker)."""
    return asyncio.run(
        _async_run_evaluation(
            ground_truth=ground_truth,
            external=external,
            n_runs=n_runs,
            model_name=model_name,
            judge_model=judge_model,
            temperature=temperature,
            batch_size=batch_size,
            n_workers=n_workers,
        )
    )
