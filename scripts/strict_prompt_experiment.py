#!/usr/bin/env python
"""Test strict prompt variants for lower variance. Batch size fixed at 1."""
from __future__ import annotations

import asyncio
import json
import math
import re
import time
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

from wsb.config import PROJECT_ROOT
from wsb.io import load_signals


@dataclass
class PromptVariant:
    name: str
    system: str
    user_template: str  # must contain {ref_block}, {cand_block}, {n_cand}


def _fmt(signals: list[str]) -> str:
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(signals))


VARIANTS: list[PromptVariant] = [
    # 0 — Current baseline (for comparison)
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

    # 1 — Strict: paraphrase-level only
    PromptVariant(
        name="strict_paraphrase",
        system="""\
You are an extremely strict research topic matcher. \
Two topics match ONLY if one is essentially a paraphrase or restatement of the other. \
When in doubt, say NO.""",
        user_template="""\
For each candidate topic, decide: is it a paraphrase or restatement of any reference topic?

A match means the two topics describe the EXACT SAME specific research idea, just worded differently. \
It is NOT enough to share a theme, method, or domain.

Reference topics:
{ref_block}

Candidate topics:
{cand_block}

Return ONLY JSON: {{"matches": [<int>, ...]}} with exactly {n_cand} elements.
1 = paraphrase of a reference topic, 0 = not a paraphrase.""",
    ),

    # 2 — Strict: would a reviewer consider them the same contribution?
    PromptVariant(
        name="strict_reviewer",
        system="""\
You are an academic peer reviewer checking for novelty. \
You must decide if two topics describe the same research contribution.""",
        user_template="""\
Imagine you are a peer reviewer. For each candidate topic, would you reject a paper on it \
as a duplicate of any reference topic? That is, do they describe the same specific contribution?

Only say YES (1) if a paper on the candidate would be rejected as redundant with a reference topic. \
Different aspects of the same broad area do NOT count.

Reference topics:
{ref_block}

Candidate topics:
{cand_block}

Return ONLY JSON: {{"matches": [<int>, ...]}} with exactly {n_cand} elements.
1 = same contribution (duplicate), 0 = different contribution.""",
    ),

    # 3 — Strict: identical research question test
    PromptVariant(
        name="strict_research_question",
        system="""\
You match research topics. Be very strict. Only mark a match when the underlying \
research question is identical.""",
        user_template="""\
For each candidate topic, determine if it addresses the IDENTICAL research question \
as any reference topic.

Two topics match only if they ask the same question about the same phenomenon. \
Merely studying the same broad area or using the same method is NOT a match.

Reference topics:
{ref_block}

Candidate topics:
{cand_block}

Return ONLY JSON: {{"matches": [<int>, ...]}} with exactly {n_cand} elements.
1 = identical research question, 0 = different question.""",
    ),

    # 4 — Strict with negative examples baked in
    PromptVariant(
        name="strict_with_negatives",
        system="""\
You are a strict research topic evaluator. Default to 0 (no match). \
Only output 1 when you are certain the topics are about the same specific idea.""",
        user_template="""\
For each candidate, is it about the SAME specific research idea as any reference topic?

MATCH (1): Topics that a researcher would file under the exact same project.
  e.g. "Chain-of-thought prompting for math" ≈ "Step-by-step reasoning in LLM math solving" → 1

NOT A MATCH (0): Topics that share a theme but differ in specifics.
  e.g. "Chain-of-thought prompting" vs "In-context learning" → 0
  e.g. "RLHF for language models" vs "Reward modeling" → 0
  e.g. "Diffusion models for images" vs "Diffusion models for audio" → 0

Reference topics:
{ref_block}

Candidate topics:
{cand_block}

Return ONLY JSON: {{"matches": [<int>, ...]}} with exactly {n_cand} elements.""",
    ),

    # 5 — Ultra-terse strict
    PromptVariant(
        name="strict_terse",
        system="Match research topics. Be strict. Default to 0.",
        user_template="""\
Same specific idea = 1. Different idea = 0. Shared theme alone = 0.

Reference:
{ref_block}

Candidates:
{cand_block}

Return JSON: {{"matches": [<0 or 1>, ...]}} — {n_cand} elements.""",
    ),

    # 6 — Strict with step-by-step
    PromptVariant(
        name="strict_stepbystep",
        system="""\
You are a strict research topic matcher. Think step by step, then give your final answer.""",
        user_template="""\
For each candidate topic:
1. Identify its core specific idea (not the broad area).
2. Check each reference topic for the same core idea.
3. Mark 1 ONLY if the core ideas are the same. Mark 0 otherwise.

Two topics sharing a broad area (e.g., both about "reinforcement learning") is NOT a match. \
They must target the same narrow problem or finding.

Reference topics:
{ref_block}

Candidate topics:
{cand_block}

Return ONLY JSON: {{"matches": [<int>, ...]}} with exactly {n_cand} elements.""",
    ),

    # 7 — Strict: "could be merged into one survey section"
    PromptVariant(
        name="strict_survey_section",
        system="""\
You are a research topic evaluator. Be conservative — only mark matches for \
topics that are clearly about the same thing.""",
        user_template="""\
For each candidate topic, would it belong in the EXACT SAME narrow subsection \
of a survey paper as any reference topic? Not the same section — the same sub-subsection, \
covering the same specific technique or finding.

Reference topics:
{ref_block}

Candidate topics:
{cand_block}

Return ONLY JSON: {{"matches": [<int>, ...]}} with exactly {n_cand} elements.
1 = same narrow sub-subsection, 0 = different.""",
    ),

    # 8 — Two-sentence core idea extraction
    PromptVariant(
        name="strict_core_extraction",
        system="""\
You are a precise research topic matcher. Extract core ideas before comparing.""",
        user_template="""\
For each candidate topic, compare its core idea to each reference topic's core idea.

A "core idea" is the specific phenomenon, method, or finding — not the broad field.
Two topics match (1) only if their core ideas are the same.
Two topics do NOT match (0) if they merely share a field, method class, or application domain.

Reference topics:
{ref_block}

Candidate topics:
{cand_block}

Return ONLY JSON: {{"matches": [<int>, ...]}} with exactly {n_cand} elements.""",
    ),

    # 9 — Strict: "would you cite both for the same claim?"
    PromptVariant(
        name="strict_citation",
        system="""\
You are a strict academic topic matcher.""",
        user_template="""\
For each candidate topic: would a researcher cite a paper on this topic AND a paper on \
a reference topic to support the EXACT SAME specific claim in a literature review?

Not "related work" — the SAME claim. If the topics address different aspects of a \
broader area, that is NOT a match.

Reference topics:
{ref_block}

Candidate topics:
{cand_block}

Return ONLY JSON: {{"matches": [<int>, ...]}} with exactly {n_cand} elements.
1 = same specific claim, 0 = different claims.""",
    ),
]


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
) -> list[int]:
    """Evaluate all candidates in a single call."""
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
    out_path = PROJECT_ROOT / "strict_prompt_experiment_results.md"
    with open(out_path, "w") as f:
        f.write("# Strict Prompt Experiment Results\n\n")
        f.write(f"Model: `{model}` | Temperature: 1.0 | Runs per variant: {n_runs}\n")
        f.write(f"All candidates sent in a single batch per pass.\n\n")

        for label, ev_results in all_results.items():
            f.write(f"## {label}\n\n")
            f.write("| # | Prompt | P mean | P std | R mean | R std | F1 mean | F1 std | Total std |\n")
            f.write("|---|--------|--------|-------|--------|-------|---------|--------|----------|\n")
            for i, r in enumerate(ev_results):
                f.write(f"| {i+1} | {r['name']} | {r['p_mean']} | {r['p_std']} | "
                        f"{r['r_mean']} | {r['r_std']} | {r['f1_mean']} | {r['f1_std']} | "
                        f"{r['total_std']} |\n")
            f.write("\n")

            f.write("<details><summary>Per-run details</summary>\n\n")
            for r in ev_results:
                f.write(f"### {r['name']}\n")
                for j, run in enumerate(r["runs"]):
                    f.write(f"- Run {j+1}: P={run['p']}, R={run['r']}, F1={run['f1']} | "
                            f"p_matches={run['p_matches']} r_matches={run['r_matches']}\n")
                f.write("\n")
            f.write("</details>\n\n")

        # Overall ranking
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

        best_name = ranked[0][0]
        best_variant = next(v for v in VARIANTS if v.name == best_name)
        f.write(f"## Best Prompt: `{best_name}`\n\n")
        f.write("### System prompt\n```\n" + best_variant.system + "\n```\n\n")
        f.write("### User template\n```\n" + best_variant.user_template + "\n```\n")

    print(f"\n\nResults written to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
