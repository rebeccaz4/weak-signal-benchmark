"""Cost estimation and usage tracking for LLM judge evaluations."""

from __future__ import annotations

import math
import threading
from typing import Any

import tiktoken

from wsb.evaluate.prompts import PAIRWISE_SYSTEM_PROMPT, build_pairwise_prompt

# Pricing per 1M tokens: (input_dollars, output_dollars)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5-mini": (0.25, 2),
    "gpt-5.2": (1.75, 14.00),
    "gemini-3-flash-preview": (0.15, 0.60),
    "gemini-3.1-flash-lite-preview": (0.25, 1.25),
    "gemini-3.1-pro-preview": (2.00, 12.00),
}

# Conservative estimate: JSON output with binary matches + reasoning tokens
ESTIMATED_OUTPUT_TOKENS = 200

# Batch API discount (both OpenAI and Gemini offer 50% off for batch)
BATCH_DISCOUNT = 0.5


class UsageTracker:
    """Thread-safe accumulator for API token usage across async calls."""

    def __init__(self, model: str) -> None:
        self.model = model
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.api_calls = 0

    def record(self, usage: Any) -> None:
        """Record usage from an API response. Accepts None gracefully."""
        if usage is None:
            return
        with self._lock:
            self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            self.api_calls += 1

    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def cost_usd(self, batch: bool = False) -> float | None:
        """Compute dollar cost from MODEL_PRICING. Returns None if model unknown.

        Args:
            batch: If True, apply BATCH_DISCOUNT to the total cost.
        """
        pricing = MODEL_PRICING.get(self.model)
        if pricing is None:
            return None
        input_cost = self.prompt_tokens * pricing[0] / 1_000_000
        output_cost = self.completion_tokens * pricing[1] / 1_000_000
        total = input_cost + output_cost
        if batch:
            total *= BATCH_DISCOUNT
        return total


def print_usage_summary(tracker: UsageTracker, batch: bool = False) -> None:
    """Print actual token usage and cost after a run."""
    print(f"\n{'─'*60}")
    print("API USAGE SUMMARY" + (" (Batch)" if batch else ""))
    print(f"{'─'*60}")
    print(f"  Model:             {tracker.model}")
    print(f"  API calls:         {tracker.api_calls}")
    print(f"  Prompt tokens:     {tracker.prompt_tokens:,}")
    print(f"  Completion tokens: {tracker.completion_tokens:,}")
    print(f"  Total tokens:      {tracker.total_tokens():,}")
    cost = tracker.cost_usd(batch=batch)
    if cost is not None:
        label = "Total cost (50% batch discount):" if batch else "Total cost:"
        print(f"  {label}  ${cost:.4f}")
    else:
        print(f"  (Pricing not available for model '{tracker.model}')")
    print(f"{'─'*60}")


def _get_encoding(model: str) -> tiktoken.Encoding:
    """Get tiktoken encoding for a model, falling back to cl100k_base."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _count_prompt_tokens(
    encoding: tiktoken.Encoding,
    reference: list[str],
    candidates: list[str],
) -> int:
    """Count tokens for a single judge call (system + user prompt)."""
    user_prompt = build_pairwise_prompt(reference, candidates)
    system_tokens = len(encoding.encode(PAIRWISE_SYSTEM_PROMPT))
    user_tokens = len(encoding.encode(user_prompt))
    # Chat overhead: ~4 tokens per message + 2 for reply priming
    return system_tokens + user_tokens + 10


def estimate_cost(
    evaluations: list[dict[str, Any]],
    *,
    judge_model: str,
    n_runs: int,
    batch_size: int,
    load_signals_fn: Any,
    project_root: Any,
) -> dict[str, Any]:
    """Estimate API cost without making any calls.

    Returns a dict with per-evaluation breakdowns and totals.
    """
    from pathlib import Path

    encoding = _get_encoding(judge_model)
    pricing = MODEL_PRICING.get(judge_model)

    results: list[dict[str, Any]] = []
    total_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for ev in evaluations:
        gt_path = Path(ev["ground_truth"])
        ext_path = Path(ev["external"])
        if not gt_path.is_absolute():
            gt_path = project_root / gt_path
        if not ext_path.is_absolute():
            ext_path = project_root / ext_path

        ground_truth = load_signals_fn(gt_path)
        external = load_signals_fn(ext_path)

        n_gt = len(ground_truth)
        n_ext = len(external)
        precision_batches = math.ceil(n_ext / batch_size)
        recall_batches = math.ceil(n_gt / batch_size)
        calls_per_run = precision_batches + recall_batches
        eval_calls = n_runs * calls_per_run

        # Sample token counts from representative batches
        precision_tokens = 0
        for i in range(precision_batches):
            batch = external[i * batch_size : (i + 1) * batch_size]
            precision_tokens += _count_prompt_tokens(encoding, ground_truth, batch)

        recall_tokens = 0
        for i in range(recall_batches):
            batch = ground_truth[i * batch_size : (i + 1) * batch_size]
            recall_tokens += _count_prompt_tokens(encoding, external, batch)

        input_tokens_per_run = precision_tokens + recall_tokens
        eval_input_tokens = n_runs * input_tokens_per_run
        eval_output_tokens = eval_calls * ESTIMATED_OUTPUT_TOKENS

        total_calls += eval_calls
        total_input_tokens += eval_input_tokens
        total_output_tokens += eval_output_tokens

        label = f"{ev.get('signal_type', '?')} | {ev.get('year_bucket', '?')} | {ev.get('topic', '?')}"
        results.append({
            "label": label,
            "gt_count": n_gt,
            "ext_count": n_ext,
            "precision_batches": precision_batches,
            "recall_batches": recall_batches,
            "calls_per_run": calls_per_run,
            "total_calls": eval_calls,
            "input_tokens": eval_input_tokens,
            "output_tokens": eval_output_tokens,
        })

    # Compute dollar cost
    if pricing:
        input_cost = total_input_tokens * pricing[0] / 1_000_000
        output_cost = total_output_tokens * pricing[1] / 1_000_000
    else:
        input_cost = None
        output_cost = None

    return {
        "judge_model": judge_model,
        "n_runs": n_runs,
        "batch_size": batch_size,
        "evaluations": results,
        "total_calls": total_calls,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": (input_cost + output_cost) if input_cost is not None else None,
        "pricing_known": pricing is not None,
    }


def print_cost_report(report: dict[str, Any]) -> None:
    """Print a formatted cost estimation report."""
    print(f"\n{'='*60}")
    print("DRY-RUN COST ESTIMATE")
    print(f"{'='*60}")
    print(f"Judge model:  {report['judge_model']}")
    print(f"N runs:       {report['n_runs']}")
    print(f"Batch size:   {report['batch_size']}")

    print(f"\n{'─'*60}")
    for ev in report["evaluations"]:
        print(f"\n  {ev['label']}")
        print(f"    GT signals: {ev['gt_count']}  |  External signals: {ev['ext_count']}")
        print(f"    Batches/run: {ev['precision_batches']} (precision) + {ev['recall_batches']} (recall) = {ev['calls_per_run']}")
        print(f"    Total API calls: {ev['total_calls']}")
        print(f"    Input tokens:  {ev['input_tokens']:,}")
        print(f"    Output tokens: {ev['output_tokens']:,} (estimated)")

    print(f"\n{'─'*60}")
    print(f"TOTALS")
    print(f"  API calls:     {report['total_calls']}")
    print(f"  Input tokens:  {report['total_input_tokens']:,}")
    print(f"  Output tokens: {report['total_output_tokens']:,} (estimated)")

    if report["pricing_known"]:
        print(f"\n  Input cost:    ${report['input_cost_usd']:.4f}")
        print(f"  Output cost:   ${report['output_cost_usd']:.4f}")
        print(f"  TOTAL COST:    ${report['total_cost_usd']:.4f}")
    else:
        print(f"\n  (Pricing not available for model '{report['judge_model']}')")
    print(f"{'='*60}\n")
