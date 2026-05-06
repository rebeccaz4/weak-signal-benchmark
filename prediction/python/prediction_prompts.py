#!/usr/bin/env python
# coding: utf-8
"""
Shared prompt templates and utilities for all weak-signal prediction scripts.

All prediction scripts import from here to guarantee identical prompts.
"""
from __future__ import annotations

import json
import re
from typing import List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

YEAR_RANGE = "2023-2024"
YEAR_SLUG = "2023_2024"

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROBLEM_PROMPT_TEMPLATE = """\
You are an expert analyst of frontier {domain} research with the highest IQ in the world identifying EARLY WEAK SIGNALS. There are two categories of weak signals. The early weak signals that you are identifying belong to a specific category of weak signals that I call "problem-space weak signals". Problem-space weak signals represent problems that have previously been largely UNKNOWN to researchers. In other words, before those weak signals emerge, scientists are not familiar with the "problems" represented by those signals.

The other category of weak signals that is opposed to problem-space weak signals are what I call "solution-space weak signals". Solution-space weak signals are different from the problem-space ones, as solution-space weak signals represent research techniques that are previously not widely available to scientists but are later shown to solve already-known problems. In other words, unlike problem-space weak signals, the "problems" for solution-space weak signals were already present and known to the research community, but the solution to the problems has not been recognized.

Given the information above, I will ask you a direct question to have you identify the "problem-space weak signals" that are the predecessors of some mainframe topics nowadays. The question will have the following format:
What are the early weak signals in {domain} that emerged between [2023-2024] that led to the mainframe topic [{mainframe_topic}]?

Regarding "weak signals" in general (not specific to problem-space weak signals or solution-space weak signals):
Definition-wise: Weak signals are those early research ideas that were not paid much attention to but grew significantly in prominence throughout subsequent years and ultimately led to the current framing of the mainframe topic [{mainframe_topic}].
Logistics-wise: A weak signal here is a concrete research direction OR idea

Avoid emitting topics that are:
- generic techniques that could apply anywhere (e.g., "manifold optimization", "benchmark datasets"),
- purely application labels ("for medical imaging", "for finance", "for ..." in general)

You must return ONLY valid JSON in exactly this format (no other text):

{{"weak_signals": [
  {{
    "signal": "<name of the weak signal>",
    "what_it_was": "<sentences describing what it was, including the year>",
    "why_weak_signal": "<sentences on why it was a problem-space weak signal for [{mainframe_topic}]>",
  }}
]}}

Rules:
- Return ONLY the JSON object above, no markdown fences, no explanation.
- Each weak signal must have all three fields: signal, what_it_was, why_weak_signal.
- Do not include generic techniques that could apply anywhere."""

SOLUTION_PROMPT_TEMPLATE = """\
You are an expert analyst of frontier {domain} research with the highest IQ in the world identifying EARLY WEAK SIGNALS. There are two categories of weak signals. One category of weak signals is what I call "problem-space weak signals". What this means is that those problem-space weak signals represent problems that were previously UNKNOWN to researchers. In other words, before those weak signals emerged, the "problems" represented by them were not even known.

The other category of weak signals that is opposed to problem-space weak signals are what I call "solution-space weak signals", which are also the weak signals that you will help me identify. Solution-space weak signals are different from the problem-space ones, as solution-space weak signals represent RESEARCH METHODS that are previously unknown to researchers but are later shown to solve ALREADY KNOWN problems. In other words, unlike problem-space weak signals, the "problems" for solution-space weak signals were already present and known to the research community, but the solution to the problems were NOT KNOWN.

Given the information above, I will ask you a direct question to have you identify the "solution-space weak signals" that are the predecessors of some mainframe topics nowadays. The question will have the following format:
What are the solution-space early weak signals in {domain} that emerged between [2023-2024] that led to the mainframe topic [{mainframe_topic}]?

Regarding "weak signals" in general (not specific to problem-space weak signals or solution-space weak signals):
Definition-wise: Weak signals are those early research ideas that were not paid much attention to but grew significantly in prominence throughout subsequent years and ultimately led to the current framing of the mainframe topic [{mainframe_topic}].
Logistics-wise: A weak signal here is a concrete RESEARCH DIRECTION OR IDEA

Avoid emitting topics that are:
- generic techniques that could apply anywhere (e.g., "manifold optimization", "benchmark datasets"),
- purely application labels ("for medical imaging", "for finance", "for ..." in general)

You must return ONLY valid JSON in exactly this format (no other text):

{{"weak_signals": [
  {{
    "signal": "<name of the weak signal>",
    "what_it_was": "<sentences describing what it was, including the year>",
    "why_weak_signal": "<sentences on why it was a solution-space weak signal for [{mainframe_topic}]>",
  }}
]}}

Rules:
- Return ONLY the JSON object above, no markdown fences, no explanation.
- Each weak signal must have all three fields: signal, what_it_was, why_weak_signal.
- Do not include generic techniques that could apply anywhere."""

PROMPT_TEMPLATES = {
    "problem": PROBLEM_PROMPT_TEMPLATE,
    "solution": SOLUTION_PROMPT_TEMPLATE,
}


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def make_topic_slug(topic: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", topic.lower()).strip("_")


def build_prompt(space: str, domain: str, mainframe_topic: str) -> str:
    return PROMPT_TEMPLATES[space].format(
        domain=domain,
        mainframe_topic=mainframe_topic,
    )


def extract_candidate_signals(text: str) -> List[str]:
    """Extract weak-signal names from model response (JSON preferred, regex fallback)."""
    signals: List[str] = []

    # 1) Direct JSON parse
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for ws in data.get("weak_signals", []):
                s = (ws.get("signal") or "").strip()
                if s:
                    signals.append(s)
        if signals:
            return signals
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    # 2) JSON inside markdown code block
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        try:
            data = json.loads(m.group(1))
            if isinstance(data, dict):
                for ws in data.get("weak_signals", []):
                    s = (ws.get("signal") or "").strip()
                    if s:
                        signals.append(s)
            if signals:
                return signals
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    # 3) Regex for "signal": "..." fields
    signals.extend(s.strip() for s in re.findall(r'"signal"\s*:\s*"([^"]+)"', text) if s.strip())

    # 4) Numbered list lines
    for line in text.splitlines():
        m2 = re.match(r"^\s*\d+[.)]+\s+(.+?)\s*$", line)
        if m2:
            s = m2.group(1).strip()
            if s:
                signals.append(s)

    seen: set[str] = set()
    deduped: List[str] = []
    for s in signals:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped
