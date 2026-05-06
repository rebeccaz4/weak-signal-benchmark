"""Prompt templates for the LLM-as-a-Judge evaluation."""

from __future__ import annotations

PAIRWISE_SYSTEM_PROMPT = """\
You are a research direction evaluator. Be inclusive — if two research directions address substantially overlapping research questions or methods, consider them a match.
"""

PAIRWISE_USER_TEMPLATE = """\
For each candidate research direction, decide if it substantially overlaps with any reference research direction.

Two research directions match if they address the same research question, use the same core method, or describe the same phenomenon — even if from different angles or with different terminology.

Reference research directions:
{ref_block}

Candidate research directions:
{cand_block}

Return ONLY JSON: {{"matches": [<int>, ...]}} with exactly {n_cand} elements.
1 = substantial overlap with any reference research direction, 0 = no overlap.
"""


def format_signal_block(signals: list[str]) -> str:
    """Format signals as a numbered list."""
    return "\n".join(f"{i+1}. {s}" for i, s in enumerate(signals))


def build_pairwise_prompt(
    reference: list[str],
    candidates: list[str],
) -> str:
    """Build the user prompt for a batched pairwise matching call."""
    return PAIRWISE_USER_TEMPLATE.format(
        ref_block=format_signal_block(reference),
        cand_block=format_signal_block(candidates),
        n_cand=len(candidates),
    )
