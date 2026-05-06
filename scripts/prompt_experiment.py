#!/usr/bin/env python
"""Test multiple prompt variations and measure variance.

Runs each prompt N times on two eval topics, records P/R/F1 stats and variance,
and writes results to a markdown file for comparison.

An LLM judges whether each candidate matches a ground-truth answer and returns
a 0/1 label. Precision, recall and F1 are then computed from the aggregated
counts:

    precision = true_positives / (true_positives + false_positives)
    recall    = true_positives / (true_positives + false_negatives)
    f1        = 2 * precision * recall / (precision + recall)
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

from wsb.config import PROJECT_ROOT
from wsb.io import load_signals


# ---------------------------------------------------------------------------
# Prompt variants
# ---------------------------------------------------------------------------

@dataclass
class PromptVariant:
    name: str
    system: str
    user_template: str  # must contain {ref_block}, {cand_block}, {n_cand}


def _fmt(signals: list[str]) -> str:
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(signals))


VARIANTS: list[PromptVariant] = [
    # 0 — Current baseline
    PromptVariant(
        name="baseline",
        system="""\
You are a careful, fair, and specialty-agnostic evaluator of research topics.
You must judge topics without favoring any domain; treat all fields as equally important.
Focus on semantic similarity rather than exact wording.
Only consider two topics a match if they are highly close in meaning.""",
        user_template="""\
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
- 0 if it does not""",
    ),

    # 1 — Explicit definition of "match"
    PromptVariant(
        name="explicit_definition",
        system="""\
You are a research topic evaluator. Be precise and consistent.""",
        user_template="""\
Below are reference topics and candidate topics.

DEFINITION OF MATCH: A candidate matches a reference topic if and only if \
they describe the SAME specific research idea, method, or finding. \
Sharing a broad theme (e.g., both about "reinforcement learning") is NOT a match. \
They must target the same narrow sub-problem or propose the same core technique.

Reference topics:
{ref_block}

Candidate topics:
{cand_block}

For each of the {n_cand} candidate(s), output 1 (match) or 0 (no match).
Return ONLY JSON: {{"matches": [<int>, ...]}}""",
    ),

    # 2 — With concrete examples
    PromptVariant(
        name="with_examples",
        system="""\
You are a research topic evaluator. Be precise and consistent.""",
        user_template="""\
For each candidate topic, decide if it closely matches any reference topic.

MATCH means the two topics describe the same specific research idea or method.
NOT A MATCH if they only share a broad theme.

Examples:
- "Chain-of-thought prompting for math" vs "Step-by-step reasoning in LLMs" → MATCH (same idea)
- "Chain-of-thought prompting for math" vs "Reinforcement learning from feedback" → NO MATCH (different ideas)
- "Process reward models for math" vs "Outcome vs process supervision" → MATCH (same core concept)
- "Process reward models for math" vs "Reward shaping in robotics" → NO MATCH (different domains/ideas)

Reference topics:
{ref_block}

Candidate topics:
{cand_block}

Return JSON with exactly {n_cand} elements: {{"matches": [0 or 1, ...]}}""",
    ),

    # 3 — Analyze-then-answer (reasoning first)
    PromptVariant(
        name="reason_first",
        system="""\
You are a research topic evaluator. Think carefully before answering.""",
        user_template="""\
For each candidate topic below, determine if it semantically matches any reference topic.
Two topics match only if they describe the same narrow research idea — not just a shared theme.

Reference topics:
{ref_block}

Candidate topics:
{cand_block}

For each candidate, first briefly state which reference topic (if any) it might match and why, \
then give your binary verdict (1=match, 0=no match).

Format your response as JSON:
{{"reasoning": ["<brief explanation for candidate 1>", ...], "matches": [<int>, ...]}}

The matches array must have exactly {n_cand} elements.""",
    ),

    # 4 — Strict/conservative (bias toward 0)
    PromptVariant(
        name="strict_conservative",
        system="""\
You are a strict research topic evaluator. When in doubt, say NO MATCH. \
Only mark a match when you are highly confident the two topics describe the exact same idea.""",
        user_template="""\
For each candidate topic, decide if it is a near-exact semantic match to any reference topic.

The bar for matching is HIGH. Mark 1 only if the topics are essentially paraphrases of the \
same research idea. If there is any ambiguity, mark 0.

Reference topics:
{ref_block}

Candidate topics:
{cand_block}

Return ONLY JSON: {{"matches": [<int>, ...]}} with exactly {n_cand} elements.""",
    ),

    # 5 — Lenient/inclusive (bias toward 1)
    PromptVariant(
        name="lenient_inclusive",
        system="""\
You are a research topic evaluator. Be inclusive — if two topics address \
substantially overlapping research questions or methods, consider them a match.""",
        user_template="""\
For each candidate topic, decide if it substantially overlaps with any reference topic.

Two topics match if they address the same research question, use the same core method, \
or describe the same phenomenon — even if from different angles or with different terminology.

Reference topics:
{ref_block}

Candidate topics:
{cand_block}

Return ONLY JSON: {{"matches": [<int>, ...]}} with exactly {n_cand} elements.
1 = substantial overlap with any reference topic, 0 = no overlap.""",
    ),

    # 6 — Pairwise comparison table
    PromptVariant(
        name="pairwise_table",
        system="""\
You are a research topic evaluator. Be systematic and thorough.""",
        user_template="""\
I need to check if each candidate topic matches any reference topic.

Reference topics:
{ref_block}

Candidate topics:
{cand_block}

Instructions:
1. For each candidate, compare it against EVERY reference topic.
2. A match means the topics describe the same specific research idea or method.
3. Sharing a broad area (e.g., "RL" or "NLP") is NOT enough.
4. If a candidate matches at least one reference, mark 1. Otherwise mark 0.

Return ONLY JSON: {{"matches": [<int>, ...]}} with exactly {n_cand} elements.""",
    ),

    # 7 — Keyword-focused
    PromptVariant(
        name="keyword_focused",
        system="""\
You are a research topic evaluator.""",
        user_template="""\
For each candidate topic, determine if it matches any reference topic.

To decide, focus on the CORE CONCEPTS, not surface keywords:
- What specific problem or question does each topic address?
- What specific technique or approach does each topic describe?
- Two topics match only if they share the same core concept on BOTH dimensions.

Reference topics:
{ref_block}

Candidate topics:
{cand_block}

Return JSON: {{"matches": [<int>, ...]}} with exactly {n_cand} elements (1=match, 0=no match).""",
    ),

    # 8 — Structured rubric
    PromptVariant(
        name="structured_rubric",
        system="""\
You are a research topic evaluator. Apply the rubric below consistently.""",
        user_template="""\
For each candidate topic, check if it matches any reference topic using this rubric:

MATCH (1) if ALL of:
  - Same core research problem or question
  - Same general methodological approach
  - A domain expert would consider them "about the same thing"

NO MATCH (0) if ANY of:
  - Different core problems, even if same method
  - Different methods, even if same broad area
  - Only surface-level keyword overlap

Reference topics:
{ref_block}

Candidate topics:
{cand_block}

Return ONLY JSON: {{"matches": [<int>, ...]}} with exactly {n_cand} elements.""",
    ),

    # 9 — Minimal/terse prompt
    PromptVariant(
        name="minimal_terse",
        system="You match research topics. Return JSON only.",
        user_template="""\
Do any candidate topics match (=same specific idea) any reference topics?

Reference:
{ref_block}

Candidates:
{cand_block}

{{"matches": [<1 or 0>, ...]}} — {n_cand} elements, 1=same idea, 0=different.""",
    ),
]


# ---------------------------------------------------------------------------
# LLM call infrastructure (copied/adapted from judge.py)
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
    system: str,
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
                    {"role": "system", "content": system},
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


async def _run_one_pass(
    client: AsyncOpenAI,
    model: str,
    variant: PromptVariant,
    reference: list[str],
    candidates: list[str],
    batch_size: int = 10,
) -> list[int]:
    """Run one direction of matching (all candidates in one batch)."""
    prompt = variant.user_template.format(
        ref_block=_fmt(reference),
        cand_block=_fmt(candidates),
        n_cand=len(candidates),
    )
    return await _call_llm(client, model, variant.system, prompt, len(candidates))


async def _run_single_eval(
    client: AsyncOpenAI,
    model: str,
    variant: PromptVariant,
    gt: list[str],
    ext: list[str],
) -> dict:
    """One eval run: precision pass + recall pass."""
    p_task = _run_one_pass(client, model, variant, gt, ext)
    r_task = _run_one_pass(client, model, variant, ext, gt)
    p_matches, r_matches = await asyncio.gather(p_task, r_task)

    p_count = sum(p_matches)
    r_count = sum(r_matches)
    precision = p_count / len(ext) if ext else 0.0
    recall = r_count / len(gt) if gt else 0.0
    denom = precision + recall
    f1 = (2 * precision * recall / denom) if denom > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1,
            "p_matches": p_matches, "r_matches": r_matches}


async def test_variant(
    client: AsyncOpenAI,
    model: str,
    variant: PromptVariant,
    gt: list[str],
    ext: list[str],
    n_runs: int,
) -> dict:
    """Run a variant n_runs times and return stats."""
    tasks = [_run_single_eval(client, model, variant, gt, ext) for _ in range(n_runs)]
    results = await asyncio.gather(*tasks)

    ps = [r["precision"] for r in results]
    rs = [r["recall"] for r in results]
    f1s = [r["f1"] for r in results]

    def mean(xs): return sum(xs) / len(xs) if xs else 0.0
    def std(xs):
        m = mean(xs)
        return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0

    return {
        "name": variant.name,
        "n_runs": n_runs,
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

    # Load two eval datasets for robustness
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
        for i, variant in enumerate(VARIANTS):
            print(f"\n  [{i+1}/{len(VARIANTS)}] Testing '{variant.name}'...", end=" ", flush=True)
            t0 = time.time()
            stats = await test_variant(client, model, variant, gt, ext, n_runs)
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s) — "
                  f"P={stats['p_mean']}±{stats['p_std']}  "
                  f"R={stats['r_mean']}±{stats['r_std']}  "
                  f"F1={stats['f1_mean']}±{stats['f1_std']}")
            ev_results.append(stats)

        all_results[label] = ev_results

    # Write results
    out_path = PROJECT_ROOT / "prompt_experiment_results.md"
    with open(out_path, "w") as f:
        f.write("# Prompt Experiment Results\n\n")
        f.write(f"Model: `{model}` | Temperature: 1.0 | Runs per variant: {n_runs}\n\n")

        for label, ev_results in all_results.items():
            f.write(f"## {label}\n\n")
            f.write("| # | Prompt | P mean | P std | R mean | R std | F1 mean | F1 std | Total std |\n")
            f.write("|---|--------|--------|-------|--------|-------|---------|--------|----------|\n")
            for i, r in enumerate(ev_results):
                f.write(f"| {i+1} | {r['name']} | {r['p_mean']} | {r['p_std']} | "
                        f"{r['r_mean']} | {r['r_std']} | {r['f1_mean']} | {r['f1_std']} | "
                        f"{r['total_std']} |\n")
            f.write("\n")

            # Per-run details
            f.write("<details><summary>Per-run details</summary>\n\n")
            for r in ev_results:
                f.write(f"### {r['name']}\n")
                for j, run in enumerate(r["runs"]):
                    f.write(f"- Run {j+1}: P={run['p']}, R={run['r']}, F1={run['f1']} | "
                            f"p_matches={run['p_matches']} r_matches={run['r_matches']}\n")
                f.write("\n")
            f.write("</details>\n\n")

        # Summary: rank by total_std across both datasets
        f.write("## Overall Ranking (by average total_std across datasets)\n\n")
        avg_std = {}
        for i, variant in enumerate(VARIANTS):
            stds = [all_results[label][i]["total_std"] for label in all_results]
            avg_std[variant.name] = sum(stds) / len(stds)
        ranked = sorted(avg_std.items(), key=lambda x: x[1])
        f.write("| Rank | Prompt | Avg Total Std |\n")
        f.write("|------|--------|---------------|\n")
        for rank, (name, std_val) in enumerate(ranked, 1):
            marker = " **← BEST**" if rank == 1 else ""
            f.write(f"| {rank} | {name} | {std_val:.4f}{marker} |\n")
        f.write("\n")

        # Write the winning prompt text
        best_name = ranked[0][0]
        best_variant = next(v for v in VARIANTS if v.name == best_name)
        f.write(f"## Best Prompt: `{best_name}`\n\n")
        f.write("### System prompt\n```\n" + best_variant.system + "\n```\n\n")
        f.write("### User template\n```\n" + best_variant.user_template + "\n```\n")

    print(f"\n\nResults written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
