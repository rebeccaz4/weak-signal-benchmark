#!/usr/bin/env python
"""Test the baseline prompt with different batch sizes and measure variance.

Batch size controls how many candidate topics are sent per LLM call.
Smaller batches = more calls but potentially more focused judgments.
Larger batches = fewer calls but the LLM sees more context at once.
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import time
from typing import Optional

from openai import AsyncOpenAI

from wsb.config import PROJECT_ROOT
from wsb.io import load_signals


# ---------------------------------------------------------------------------
# Baseline prompt (fixed)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a careful, fair, and specialty-agnostic evaluator of research topics.
You must judge topics without favoring any domain; treat all fields as equally important.
Focus on semantic similarity rather than exact wording.
Only consider two topics a match if they are highly close in meaning."""

USER_TEMPLATE = """\
I have a list of reference research topics and a set of candidate topics to evaluate.

Your task: for each candidate topic, determine whether it is a close semantic match to \
ANY of the reference topics. Two topics match ONLY if they describe essentially the \
same research trend, phenomenon, or insight — mere thematic overlap is NOT enough.

Reference topics:
{ref_block}

Candidate topics to evaluate:
{cand_block}

Return ONLY valid JSON with this schema:
{{"matches": [<int>, ...]}}

Where matches is an array with exactly {n_cand} elements (one per candidate topic, in order).
Each element is:
- 1 if the candidate topic closely matches any reference topic
- 0 if it does not"""


def _fmt(signals: list[str]) -> str:
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(signals))


# ---------------------------------------------------------------------------
# LLM call infrastructure
# ---------------------------------------------------------------------------

def _safe_json_loads(text: str) -> Optional[dict]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


async def _call_llm(
    client: AsyncOpenAI,
    model: str,
    user: str,
    expected_count: int,
    max_retries: int = 6,
    retry_backoff: float = 2.0,
) -> list[int]:
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=1.0,
                max_completion_tokens=4096,
                response_format={"type": "json_object"},
            )
            if not resp.choices:
                raise ValueError("No choices")
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("Empty response")
            payload = _safe_json_loads(text)
            if payload is None:
                raise ValueError("Unparseable JSON")
            matches = payload.get("matches")
            if not isinstance(matches, list):
                raise ValueError(f"No matches list: {payload}")
            if len(matches) != expected_count:
                raise ValueError(f"Expected {expected_count}, got {len(matches)}")
            result = [int(m) for m in matches]
            if not all(v in (0, 1) for v in result):
                raise ValueError(f"Non-binary: {result}")
            return result
        except Exception as exc:
            if attempt >= max_retries:
                raise RuntimeError(f"Failed after {attempt} attempts: {exc}") from exc
            await asyncio.sleep(retry_backoff * attempt)


async def _batch_match(
    client: AsyncOpenAI,
    model: str,
    reference: list[str],
    candidates: list[str],
    batch_size: int,
) -> list[int]:
    """Run matching with specified batch size. Returns 0/1 list for all candidates."""
    if batch_size >= len(candidates):
        # Single batch
        prompt = USER_TEMPLATE.format(
            ref_block=_fmt(reference),
            cand_block=_fmt(candidates),
            n_cand=len(candidates),
        )
        return await _call_llm(client, model, prompt, len(candidates))

    # Multiple batches
    n_batches = math.ceil(len(candidates) / batch_size)
    batches = [
        candidates[i * batch_size : (i + 1) * batch_size]
        for i in range(n_batches)
    ]
    tasks = [
        _call_llm(
            client, model,
            USER_TEMPLATE.format(
                ref_block=_fmt(reference),
                cand_block=_fmt(batch),
                n_cand=len(batch),
            ),
            len(batch),
        )
        for batch in batches
    ]
    batch_results = await asyncio.gather(*tasks)
    matches: list[int] = []
    for result in batch_results:
        matches.extend(result)
    return matches


async def _run_single_eval(
    client: AsyncOpenAI,
    model: str,
    gt: list[str],
    ext: list[str],
    batch_size: int,
) -> dict:
    """One eval run: precision pass + recall pass."""
    p_task = _batch_match(client, model, gt, ext, batch_size)
    r_task = _batch_match(client, model, ext, gt, batch_size)
    p_matches, r_matches = await asyncio.gather(p_task, r_task)

    p_count = sum(p_matches)
    r_count = sum(r_matches)
    precision = p_count / len(ext) if ext else 0.0
    recall = r_count / len(gt) if gt else 0.0
    denom = precision + recall
    f1 = (2 * precision * recall / denom) if denom > 0 else 0.0

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "p_matches": p_matches, "r_matches": r_matches,
    }


async def test_batch_size(
    client: AsyncOpenAI,
    model: str,
    gt: list[str],
    ext: list[str],
    batch_size: int,
    n_runs: int,
) -> dict:
    """Run baseline prompt with given batch_size n_runs times and return stats."""
    tasks = [_run_single_eval(client, model, gt, ext, batch_size) for _ in range(n_runs)]
    results = await asyncio.gather(*tasks)

    ps = [r["precision"] for r in results]
    rs = [r["recall"] for r in results]
    f1s = [r["f1"] for r in results]

    def mean(xs): return sum(xs) / len(xs) if xs else 0.0
    def std(xs):
        m = mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0

    n_p_batches = math.ceil(len(ext) / batch_size)
    n_r_batches = math.ceil(len(gt) / batch_size)

    return {
        "batch_size": batch_size,
        "n_runs": n_runs,
        "n_llm_calls_per_run": n_p_batches + n_r_batches,
        "p_mean": round(mean(ps), 4), "p_std": round(std(ps), 4),
        "r_mean": round(mean(rs), 4), "r_std": round(std(rs), 4),
        "f1_mean": round(mean(f1s), 4), "f1_std": round(std(f1s), 4),
        "total_std": round(std(ps) + std(rs) + std(f1s), 4),
        "runs": [{"p": round(r["precision"], 4), "r": round(r["recall"], 4),
                  "f1": round(r["f1"], 4),
                  "p_matches": r["p_matches"], "r_matches": r["r_matches"]}
                 for r in results],
    }


async def main():
    model = "gpt-5-mini"
    n_runs = 5
    batch_sizes = [1, 2, 3, 5, 10]

    evals = [
        {
            "label": "Problem 2020-2022",
            "gt": load_signals(PROJECT_ROOT / "data/ground_truth/problem/2020-2022/reward_type_process_or_outcome.json"),
            "ext": load_signals(PROJECT_ROOT / "data/external/dr_tulu/problem/2020-2022/reward_type_process_or_outcome.json"),
        },
        {
            "label": "Solution 2020-2022",
            "gt": load_signals(PROJECT_ROOT / "data/ground_truth/solution/2020-2022/model_based_rl_for_llms.json"),
            "ext": load_signals(PROJECT_ROOT / "data/external/dr_tulu/solution/2020-2022/model_based_rl_for_llms.json"),
        },
    ]

    client = AsyncOpenAI()
    all_results = {}

    for ev in evals:
        label = ev["label"]
        gt, ext = ev["gt"], ev["ext"]
        print(f"\n{'='*70}")
        print(f"Dataset: {label} | GT={len(gt)}, Ext={len(ext)}")
        print(f"{'='*70}")

        ev_results = []
        for bs in batch_sizes:
            n_p = math.ceil(len(ext) / bs)
            n_r = math.ceil(len(gt) / bs)
            print(f"\n  batch_size={bs} ({n_p}+{n_r} calls/run)...", end=" ", flush=True)
            t0 = time.time()
            stats = await test_batch_size(client, model, gt, ext, bs, n_runs)
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s) — "
                  f"P={stats['p_mean']}±{stats['p_std']}  "
                  f"R={stats['r_mean']}±{stats['r_std']}  "
                  f"F1={stats['f1_mean']}±{stats['f1_std']}")
            ev_results.append(stats)

        all_results[label] = ev_results

    # Write results
    out_path = PROJECT_ROOT / "batch_size_experiment_results.md"
    with open(out_path, "w") as f:
        f.write("# Batch Size Experiment Results\n\n")
        f.write(f"Model: `{model}` | Temperature: 1.0 | Runs per batch size: {n_runs}\n")
        f.write(f"Prompt: **baseline** (fixed)\n\n")
        f.write("Batch size controls how many candidate topics are sent per LLM call.\n")
        f.write("- batch_size=1: each candidate evaluated individually (most calls)\n")
        f.write("- batch_size=N: N candidates per call (fewer calls, more context per call)\n\n")

        for label, ev_results in all_results.items():
            gt_size = [e for e in evals if e["label"] == label][0]
            f.write(f"## {label}\n\n")
            f.write("| Batch Size | Calls/Run | P mean | P std | R mean | R std | F1 mean | F1 std | Total std |\n")
            f.write("|------------|-----------|--------|-------|--------|-------|---------|--------|----------|\n")
            for r in ev_results:
                f.write(f"| {r['batch_size']} | {r['n_llm_calls_per_run']} | "
                        f"{r['p_mean']} | {r['p_std']} | "
                        f"{r['r_mean']} | {r['r_std']} | "
                        f"{r['f1_mean']} | {r['f1_std']} | "
                        f"{r['total_std']} |\n")
            f.write("\n")

            # Per-run details
            f.write("<details><summary>Per-run details</summary>\n\n")
            for r in ev_results:
                f.write(f"### batch_size={r['batch_size']}\n")
                for j, run in enumerate(r["runs"]):
                    f.write(f"- Run {j+1}: P={run['p']}, R={run['r']}, F1={run['f1']} | "
                            f"p_matches={run['p_matches']} r_matches={run['r_matches']}\n")
                f.write("\n")
            f.write("</details>\n\n")

        # Summary: rank by average total_std
        f.write("## Overall Ranking (by average total_std across datasets)\n\n")
        avg_std = {}
        for bs in batch_sizes:
            stds = []
            for label in all_results:
                r = next(r for r in all_results[label] if r["batch_size"] == bs)
                stds.append(r["total_std"])
            avg_std[bs] = sum(stds) / len(stds)
        ranked = sorted(avg_std.items(), key=lambda x: x[1])
        f.write("| Rank | Batch Size | Avg Total Std |\n")
        f.write("|------|------------|---------------|\n")
        for rank, (bs, std_val) in enumerate(ranked, 1):
            marker = " **← LOWEST VARIANCE**" if rank == 1 else ""
            f.write(f"| {rank} | {bs} | {std_val:.4f}{marker} |\n")
        f.write("\n")

    print(f"\n\nResults written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
