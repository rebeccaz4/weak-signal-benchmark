"""Batch API evaluation — submit all judge calls as a single batch job (50% cost savings).

Supports both OpenAI and Gemini batch endpoints via the OpenAI SDK.
Supports chunked submission (multiple batch jobs) for large request sets.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from wsb.config import GEMINI_API_KEY, GEMINI_BASE_URL, PROJECT_ROOT
from wsb.evaluate.cost import UsageTracker, print_usage_summary
from wsb.evaluate.judge import DEFAULT_BATCH_SIZE, _safe_json_loads
from wsb.evaluate.metrics import compute_metrics, flatten_metric_runs
from wsb.evaluate.prompts import PAIRWISE_SYSTEM_PROMPT, build_pairwise_prompt
from wsb.io import load_signals

DEFAULT_POLL_INTERVAL = 30  # seconds
DEFAULT_CHUNK_SIZE = 100  # requests per Batch API job


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def _make_sync_client(judge_model: str) -> OpenAI:
    """Create a sync OpenAI client, using Gemini's endpoint for Gemini models."""
    if judge_model.startswith("gemini"):
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY environment variable is required for Gemini models."
            )
        return OpenAI(base_url=GEMINI_BASE_URL, api_key=GEMINI_API_KEY)
    return OpenAI()


# ---------------------------------------------------------------------------
# Custom ID encoding / decoding
# ---------------------------------------------------------------------------

def encode_custom_id(
    eval_idx: int, run_idx: int, pass_type: str, batch_idx: int, cfg_idx: int = 0
) -> str:
    """Build custom_id: cfg_{c}_eval_{e}_run_{r}_{pass_type}_batch_{b}."""
    return f"cfg_{cfg_idx}_eval_{eval_idx}_run_{run_idx}_{pass_type}_batch_{batch_idx}"


def decode_custom_id(custom_id: str) -> dict[str, Any]:
    """Parse custom_id → dict with cfg_idx, eval_idx, run_idx, pass_type, batch_idx."""
    parts = custom_id.split("_")
    # cfg_{c}_eval_{e}_run_{r}_{pass_type}_batch_{b}
    return {
        "cfg_idx": int(parts[1]),
        "eval_idx": int(parts[3]),
        "run_idx": int(parts[5]),
        "pass_type": parts[6],
        "batch_idx": int(parts[8]),
    }


# ---------------------------------------------------------------------------
# Request collection
# ---------------------------------------------------------------------------

def collect_batch_requests(
    evaluations_data: list[dict[str, Any]],
    judge_model: str,
    n_runs: int,
    batch_size: int,
    temperature: float = 1.0,
    cfg_idx: int = 0,
) -> tuple[list[dict], dict[str, int]]:
    """Build all JSONL request dicts and a metadata mapping custom_id → expected_count.

    Args:
        evaluations_data: list of dicts with keys ground_truth, external (as loaded signal lists),
                          and eval_idx.
        judge_model: model name for the batch body.
        n_runs: number of judge iterations per evaluation.
        batch_size: signals per LLM call.
        temperature: sampling temperature.
        cfg_idx: config index (for multi-config batch mode).

    Returns:
        (requests, request_meta) where requests is a list of JSONL-line dicts and
        request_meta maps custom_id → expected match count.
    """
    requests: list[dict] = []
    request_meta: dict[str, int] = {}

    for ed in evaluations_data:
        eval_idx = ed["eval_idx"]
        ground_truth: list[str] = ed["ground_truth"]
        external: list[str] = ed["external"]

        for run_idx in range(n_runs):
            # Precision pass: for each external batch, does it match any GT?
            n_p_batches = math.ceil(len(external) / batch_size)
            for b in range(n_p_batches):
                batch = external[b * batch_size : (b + 1) * batch_size]
                cid = encode_custom_id(eval_idx, run_idx, "precision", b, cfg_idx)
                prompt = build_pairwise_prompt(ground_truth, batch)
                requests.append(_build_request_line(cid, judge_model, prompt, temperature))
                request_meta[cid] = len(batch)

            # Recall pass: for each GT batch, does it match any external?
            n_r_batches = math.ceil(len(ground_truth) / batch_size)
            for b in range(n_r_batches):
                batch = ground_truth[b * batch_size : (b + 1) * batch_size]
                cid = encode_custom_id(eval_idx, run_idx, "recall", b, cfg_idx)
                prompt = build_pairwise_prompt(external, batch)
                requests.append(_build_request_line(cid, judge_model, prompt, temperature))
                request_meta[cid] = len(batch)

    return requests, request_meta


def _build_request_line(
    custom_id: str, model: str, user_prompt: str, temperature: float
) -> dict:
    """Build a single JSONL request line for the Batch API."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": PAIRWISE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    if not model.startswith("gemini"):
        body["temperature"] = temperature
        body["max_completion_tokens"] = 4096
        body["response_format"] = {"type": "json_object"}

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def write_batch_jsonl(requests: list[dict], output_path: Path) -> Path:
    """Write request dicts as a JSONL file. Returns the path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")
    return output_path


# ---------------------------------------------------------------------------
# Batch submission, polling, download
# ---------------------------------------------------------------------------

def submit_batch(client: OpenAI, jsonl_path: Path) -> Any:
    """Upload JSONL file and create a batch job. Returns the batch object."""
    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Batch submitted: id={batch.id}  status={batch.status}")
    return batch


def poll_batch(client: OpenAI, batch_id: str, poll_interval: int = DEFAULT_POLL_INTERVAL) -> Any:
    """Poll until batch completes or fails. Returns the final batch object."""
    while True:
        batch = client.batches.retrieve(batch_id)
        completed = batch.request_counts.completed if batch.request_counts else 0
        total = batch.request_counts.total if batch.request_counts else 0
        print(f"  [{batch.status}] {completed}/{total} requests completed")

        if batch.status == "completed":
            return batch
        if batch.status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch {batch_id} ended with status: {batch.status}")

        time.sleep(poll_interval)


def submit_and_poll_chunked(
    client: OpenAI,
    requests: list[dict],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    batch_dir: Path | None = None,
) -> list[Any]:
    """Split requests into chunks, submit each as a separate batch job, poll all to completion.

    Returns list of completed batch objects.
    """
    if batch_dir is None:
        batch_dir = PROJECT_ROOT / ".batch"

    n_chunks = math.ceil(len(requests) / chunk_size)
    chunks = [
        requests[i * chunk_size : (i + 1) * chunk_size]
        for i in range(n_chunks)
    ]

    print(f"\nSubmitting {len(requests)} requests in {n_chunks} chunk(s) of up to {chunk_size}...")

    # Submit all chunks
    batch_objects: list[Any] = []
    for ci, chunk in enumerate(chunks):
        jsonl_path = batch_dir / f"chunk_{ci}.jsonl"
        write_batch_jsonl(chunk, jsonl_path)
        print(f"Chunk {ci+1}/{n_chunks} ({len(chunk)} requests):")
        batch = submit_batch(client, jsonl_path)
        batch_objects.append(batch)

    # Poll all until complete
    print(f"\nPolling {n_chunks} batch job(s) every {poll_interval}s...")
    completed_batches: list[Any] = []
    pending = {i: b.id for i, b in enumerate(batch_objects)}

    while pending:
        still_pending: dict[int, str] = {}
        for ci, bid in pending.items():
            batch = client.batches.retrieve(bid)
            done = batch.request_counts.completed if batch.request_counts else 0
            total = batch.request_counts.total if batch.request_counts else 0

            if batch.status == "completed":
                print(f"  Chunk {ci+1}/{n_chunks}: completed ({done}/{total})")
                completed_batches.append(batch)
            elif batch.status in ("failed", "expired", "cancelled"):
                raise RuntimeError(
                    f"Chunk {ci+1} (batch {bid}) ended with status: {batch.status}"
                )
            else:
                print(f"  Chunk {ci+1}/{n_chunks}: [{batch.status}] {done}/{total}")
                still_pending[ci] = bid

        pending = still_pending
        if pending:
            time.sleep(poll_interval)

    return completed_batches


def download_and_parse_results(
    client: OpenAI,
    batch: Any,
    request_meta: dict[str, int],
    judge_model: str,
) -> tuple[dict[str, list[int]], UsageTracker]:
    """Download batch output, parse responses, validate matches.

    Returns:
        (results, tracker) where results maps custom_id → list of 0/1 match values.
    """
    if not batch.output_file_id:
        raise RuntimeError("Batch completed but has no output_file_id.")

    content = client.files.content(batch.output_file_id)
    raw_text = content.read().decode("utf-8")

    tracker = UsageTracker(judge_model)
    results: dict[str, list[int]] = {}
    errors: list[str] = []

    for line in raw_text.strip().split("\n"):
        if not line.strip():
            continue
        entry = json.loads(line)
        cid = entry["custom_id"]
        response = entry.get("response", {})

        if response.get("status_code") != 200:
            errors.append(f"{cid}: HTTP {response.get('status_code')}")
            continue

        body = response.get("body", {})

        # Track usage
        usage = body.get("usage")
        if usage:
            tracker.prompt_tokens += usage.get("prompt_tokens", 0)
            tracker.completion_tokens += usage.get("completion_tokens", 0)
            tracker.api_calls += 1

        # Parse response content
        choices = body.get("choices", [])
        if not choices:
            errors.append(f"{cid}: no choices in response")
            continue

        text = (choices[0].get("message", {}).get("content", "") or "").strip()
        payload = _safe_json_loads(text)
        if payload is None:
            errors.append(f"{cid}: could not parse JSON from response")
            continue

        matches = payload.get("matches")
        expected = request_meta.get(cid)
        if not isinstance(matches, list):
            errors.append(f"{cid}: 'matches' is not a list")
            continue
        if expected is not None and len(matches) != expected:
            errors.append(f"{cid}: expected {expected} matches, got {len(matches)}")
            continue

        result = [int(m) for m in matches]
        if not all(v in (0, 1) for v in result):
            errors.append(f"{cid}: non-binary values in matches")
            continue

        results[cid] = result

    if errors:
        summary = "\n  ".join(errors[:20])
        raise RuntimeError(
            f"{len(errors)} request(s) failed in batch output:\n  {summary}"
        )

    return results, tracker


def download_and_parse_multi(
    client: OpenAI,
    batches: list[Any],
    request_meta: dict[str, int],
    judge_model: str,
) -> tuple[dict[str, list[int]], UsageTracker]:
    """Download and parse results from multiple completed batch objects.

    Merges results and usage across all batches.
    """
    all_results: dict[str, list[int]] = {}
    total_tracker = UsageTracker(judge_model)

    for batch in batches:
        results, tracker = download_and_parse_results(client, batch, request_meta, judge_model)
        all_results.update(results)
        total_tracker.prompt_tokens += tracker.prompt_tokens
        total_tracker.completion_tokens += tracker.completion_tokens
        total_tracker.api_calls += tracker.api_calls

    return all_results, total_tracker


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

def assemble_evaluation_results(
    results: dict[str, list[int]],
    evaluations_data: list[dict[str, Any]],
    n_runs: int,
    batch_size: int,
    cfg_idx: int = 0,
) -> dict[int, list[dict]]:
    """Reconstruct per-eval, per-run metric dicts from flat batch results.

    Returns:
        {eval_idx: [run_0_metrics, run_1_metrics, ...]}
    """
    eval_results: dict[int, list[dict]] = {}

    for ed in evaluations_data:
        eval_idx = ed["eval_idx"]
        ground_truth = ed["ground_truth"]
        external = ed["external"]
        n_gt = len(ground_truth)
        n_ext = len(external)

        runs: list[dict] = []
        for run_idx in range(n_runs):
            # Collect precision matches
            n_p_batches = math.ceil(n_ext / batch_size)
            precision_matches: list[int] = []
            for b in range(n_p_batches):
                cid = encode_custom_id(eval_idx, run_idx, "precision", b, cfg_idx)
                precision_matches.extend(results[cid])

            # Collect recall matches
            n_r_batches = math.ceil(n_gt / batch_size)
            recall_matches: list[int] = []
            for b in range(n_r_batches):
                cid = encode_custom_id(eval_idx, run_idx, "recall", b, cfg_idx)
                recall_matches.extend(results[cid])

            runs.append(compute_metrics(precision_matches, recall_matches, n_gt, n_ext))

        eval_results[eval_idx] = runs

    return eval_results


# ---------------------------------------------------------------------------
# Top-level orchestrators
# ---------------------------------------------------------------------------

def _load_evaluations_data(evaluations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load signals for a list of evaluation configs. Returns enriched dicts."""
    evaluations_data: list[dict[str, Any]] = []
    for i, ev in enumerate(evaluations):
        gt_path = Path(ev["ground_truth"])
        ext_path = Path(ev["external"])
        if not gt_path.is_absolute():
            gt_path = PROJECT_ROOT / gt_path
        if not ext_path.is_absolute():
            ext_path = PROJECT_ROOT / ext_path

        ground_truth = load_signals(gt_path)
        external = load_signals(ext_path)

        evaluations_data.append({
            "eval_idx": i,
            "ground_truth": ground_truth,
            "external": external,
            "config": ev,
        })
    return evaluations_data


def run_batch_evaluation(
    config: dict[str, Any],
    evaluations: list[dict[str, Any]],
    *,
    judge_model: str = "gpt-5-mini",
    n_runs: int = 10,
    batch_size: int = DEFAULT_BATCH_SIZE,
    temperature: float = 1.0,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    chunk_size: int = 0,
) -> tuple[dict[int, pd.DataFrame], UsageTracker]:
    """Run the full batch evaluation pipeline for a single config.

    Args:
        chunk_size: If > 0, split requests into chunks of this size.
                    If 0, submit all as a single batch job.

    Returns:
        (eval_metrics, tracker) where eval_metrics maps eval_idx → metrics DataFrame.
    """
    model_name = config["model_name"]

    # 1. Load signals
    evaluations_data = _load_evaluations_data(evaluations)
    for ed in evaluations_data:
        i = ed["eval_idx"]
        print(f"Evaluation {i}: GT={len(ed['ground_truth'])} signals, External={len(ed['external'])} signals")

    # 2. Build all requests
    print(f"\nBuilding batch requests (model={judge_model}, n_runs={n_runs}, batch_size={batch_size})...")
    requests, request_meta = collect_batch_requests(
        evaluations_data, judge_model, n_runs, batch_size, temperature,
    )
    print(f"Total requests: {len(requests)}")

    # 3–6. Submit, poll, download
    client = _make_sync_client(judge_model)

    if chunk_size > 0:
        completed_batches = submit_and_poll_chunked(
            client, requests,
            chunk_size=chunk_size,
            poll_interval=poll_interval,
        )
        print("All chunks completed! Downloading results...")
        results, tracker = download_and_parse_multi(
            client, completed_batches, request_meta, judge_model,
        )
    else:
        jsonl_path = PROJECT_ROOT / ".batch" / "batch_input.jsonl"
        write_batch_jsonl(requests, jsonl_path)
        print(f"JSONL written to: {jsonl_path}")
        batch = submit_batch(client, jsonl_path)
        print(f"\nPolling every {poll_interval}s...")
        batch = poll_batch(client, batch.id, poll_interval)
        print("Batch completed! Downloading results...")
        results, tracker = download_and_parse_results(client, batch, request_meta, judge_model)

    # 7. Assemble per-eval metrics
    eval_run_metrics = assemble_evaluation_results(results, evaluations_data, n_runs, batch_size)

    eval_metrics: dict[int, pd.DataFrame] = {}
    for eval_idx, runs in eval_run_metrics.items():
        for run_idx, result in enumerate(runs, 1):
            p = result.get("precision", "?")
            r = result.get("recall", "?")
            f = result.get("f1", "?")
            print(f"  Eval {eval_idx} Run {run_idx}/{n_runs}: P={p}  R={r}  F1={f}")
        eval_metrics[eval_idx] = flatten_metric_runs(runs, model_name)

    return eval_metrics, tracker


def run_batch_evaluation_multi(
    configs: list[dict[str, Any]],
    *,
    judge_model: str = "gpt-5-mini",
    n_runs_override: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    temperature: float = 1.0,
    poll_interval: int = DEFAULT_POLL_INTERVAL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[dict[int, tuple[dict[str, Any], dict[int, pd.DataFrame]]], UsageTracker]:
    """Run batch evaluation across multiple configs in a single submission.

    All requests from all configs are pooled, chunked, and submitted together.

    Args:
        configs: list of parsed YAML config dicts.
        judge_model: judge model override (if None-ish, uses each config's setting).
        n_runs_override: override n_runs for all configs (uses config value if None).
        batch_size: signals per LLM call.
        temperature: sampling temperature.
        poll_interval: seconds between batch status polls.
        chunk_size: requests per Batch API job.

    Returns:
        (config_results, tracker) where config_results maps cfg_idx →
        (config_dict, {eval_idx: metrics_df}).
    """
    # 1. Load signals and build requests for all configs
    all_requests: list[dict] = []
    all_meta: dict[str, int] = {}
    config_info: dict[int, dict[str, Any]] = {}

    for ci, cfg in enumerate(configs):
        model_name = cfg["model_name"]
        n_runs = n_runs_override or cfg.get("n_runs", 10)
        evaluations = cfg["evaluations"]

        print(f"\n{'─'*60}")
        print(f"Config {ci}: {model_name} ({len(evaluations)} evaluations, {n_runs} runs)")
        print(f"{'─'*60}")

        evaluations_data = _load_evaluations_data(evaluations)
        for ed in evaluations_data:
            print(f"  Eval {ed['eval_idx']}: GT={len(ed['ground_truth'])}, Ext={len(ed['external'])}")

        requests, meta = collect_batch_requests(
            evaluations_data, judge_model, n_runs, batch_size, temperature, cfg_idx=ci,
        )
        all_requests.extend(requests)
        all_meta.update(meta)

        config_info[ci] = {
            "config": cfg,
            "evaluations_data": evaluations_data,
            "n_runs": n_runs,
        }
        print(f"  Requests: {len(requests)}")

    print(f"\nTotal requests across all configs: {len(all_requests)}")

    # 2. Submit chunked
    client = _make_sync_client(judge_model)
    completed_batches = submit_and_poll_chunked(
        client, all_requests,
        chunk_size=chunk_size,
        poll_interval=poll_interval,
    )

    # 3. Download and parse all results
    print("\nDownloading and parsing results from all chunks...")
    results, tracker = download_and_parse_multi(
        client, completed_batches, all_meta, judge_model,
    )

    # 4. Assemble per-config, per-eval metrics
    config_results: dict[int, tuple[dict[str, Any], dict[int, pd.DataFrame]]] = {}

    for ci, info in config_info.items():
        cfg = info["config"]
        model_name = cfg["model_name"]
        n_runs = info["n_runs"]
        evaluations_data = info["evaluations_data"]

        eval_run_metrics = assemble_evaluation_results(
            results, evaluations_data, n_runs, batch_size, cfg_idx=ci,
        )

        eval_metrics: dict[int, pd.DataFrame] = {}
        for eval_idx, runs in eval_run_metrics.items():
            eval_metrics[eval_idx] = flatten_metric_runs(runs, model_name)

        config_results[ci] = (cfg, eval_metrics)

    return config_results, tracker
