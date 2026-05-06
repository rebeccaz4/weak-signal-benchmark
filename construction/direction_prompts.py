"""
Prompt templates for generating direction-based (problem-space & solution-space)
weak signals benchmark data.

Usage:
    from direction_prompts import (
        PROBLEM_SPACE_PROMPT,
        SOLUTION_SPACE_PROMPT,
        format_problem_prompt,
        format_solution_prompt,
    )

    prompt = format_problem_prompt(
        field="natural language processing (NLP)",
        mainframe_topic="verifiable rewards",
    )
"""

# ---------------------------------------------------------------------------
# Problem-space weak signal prompt
# ---------------------------------------------------------------------------
PROBLEM_SPACE_PROMPT = """\
You are an expert analyst of {field} research with the \
highest IQ in the world identifying EARLY WEAK SIGNALS. There are two categories \
of weak signals. The early weak signals that you are identifying belong to a \
specific category of weak signals that I call "problem-space weak signals". \
Problem-space weak signals represent problems that have previously been largely \
UNKNOWN to researchers. In other words, before those weak signals emerge, \
scientists are not familiar with the "problems" represented by those signals.

The other category of weak signals that is opposed to problem-space weak signals \
are what I call "solution-space weak signals". Solution-space weak signals are \
different from the problem-space ones, as solution-space weak signals represent \
research techniques that are previously not widely available to scientists but \
are later shown to solve already-known problems. In other words, unlike \
problem-space weak signals, the "problems" for solution-space weak signals were \
already present and known to the research community, but the solution to the \
problems has not been recognized.

Given the information above, I will ask you a direct question to have you \
identify the "problem-space weak signals" that are the predecessors of some \
mainframe topics nowadays. The question will have the following format:
What are the early weak signals in {field} that emerged between [2023-2024] \
that led to the mainframe topic [{mainframe_topic}]?

Regarding "weak signals" in general (not specific to problem-space weak signals \
or solution-space weak signals):
Definition-wise: Weak signals are those early research ideas that were not paid \
much attention to but grew significantly in prominence throughout subsequent \
years and ultimately led to the current framing of the mainframe topic \
[{mainframe_topic}].
Logistics-wise: A weak signal here is a concrete research direction OR idea

Avoid emitting topics that are:
- generic techniques that could apply anywhere (e.g., "manifold optimization", \
"benchmark datasets"),
- purely application labels ("for medical imaging", "for finance", "for ..." \
in general)

IMPORTANT: You MUST use web search several times to find and verify the real academic papers \
that support each weak signal you identify. Do NOT rely solely on memory — \
actively search the web for relevant publications to ensure all references are \
real, accurate, and verifiable.

Return only JSON in the following format:
{{
  "weak_signals": [
    {{
      "signal": "<name of the weak signal>",
      "what_it_was": "<describing what it was specifically>",
      "why_weak_signal": "<explaining why it was a problem-space weak signal for [{mainframe_topic}] specifically>",
      "references": [
        {{
          "title": "<exact paper title>",
          "year": <4-digit year>,
          "url": "<paper URL>"
        }}
      ]
    }}
  ]
}}

Rules for references:
- Each weak signal must include references.
- Use real papers only; do NOT invent titles, years, or URLs.
- The URL should be a direct paper link (e.g., arxiv, doi) when possible.
- If a URL is unavailable, still provide exact title and year.\
"""

# ---------------------------------------------------------------------------
# Solution-space weak signal prompt
# ---------------------------------------------------------------------------
SOLUTION_SPACE_PROMPT = """\
You are an expert analyst of {field} research with the \
highest IQ in the world identifying EARLY WEAK SIGNALS. There are two categories \
of weak signals. One category of weak signals is what I call "problem-space \
weak signals". What this means is that those problem-space weak signals \
represent problems that were previously UNKNOWN to researchers. In other words, \
before those weak signals emerged, the "problems" represented by them were not \
even known.

The other category of weak signals that is opposed to problem-space weak signals \
are what I call "solution-space weak signals", which are also the weak signals \
that you will help me identify. Solution-space weak signals are different from \
the problem-space ones, as solution-space weak signals represent RESEARCH \
METHODS that are previously unknown to researchers but are later shown to solve \
ALREADY KNOWN problems. In other words, unlike problem-space weak signals, the \
"problems" for solution-space weak signals were already present and known to \
the research community, but the solution to the problems were NOT KNOWN.

Given the information above, I will ask you a direct question to have you \
identify the "solution-space weak signals" that are the predecessors of some \
mainframe topics nowadays. The question will have the following format:
What are the solution-space early weak signals in {field} that emerged between \
[2023-2024] that led to the mainframe topic [{mainframe_topic}]?

Regarding "weak signals" in general (not specific to problem-space weak signals \
or solution-space weak signals):
Definition-wise: Weak signals are those early research ideas that were not paid \
much attention to but grew significantly in prominence throughout subsequent \
years and ultimately led to the current framing of the mainframe topic \
[{mainframe_topic}].
Logistics-wise: A weak signal here is a concrete RESEARCH DIRECTION OR IDEA

Avoid emitting topics that are:
- generic techniques that could apply anywhere (e.g., "manifold optimization", \
"benchmark datasets"),
- purely application labels ("for medical imaging", "for finance", "for ..." \
in general)

IMPORTANT: You MUST use web search several times to find and verify the real academic papers \
that support each weak signal you identify. Do NOT rely solely on memory — \
actively search the web for relevant publications to ensure all references are \
real, accurate, and verifiable.

Return only JSON in the following format:
{{
  "weak_signals": [
    {{
      "signal": "<name of the weak signal>",
      "what_it_was": "<describing what it was specifically>",
      "why_weak_signal": "<explaining why it was a solution-space weak signal for [{mainframe_topic}] specifically>",
      "references": [
        {{
          "title": "<exact paper title>",
          "year": <4-digit year>,
          "url": "<paper URL>"
        }}
      ]
    }}
  ]
}}

Rules for references:
- Each weak signal must include references.
- Use real papers only; do NOT invent titles, years, or URLs.
- The URL should be a direct paper link (e.g., arxiv, doi) when possible.
- If a URL is unavailable, still provide exact title and year.

Going forward, I will only modify one-by-one the mainframe topic that is \
substituted into the [{mainframe_topic}] handle in the question format \
mentioned above.\
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def format_problem_prompt(
    mainframe_topic: str,
    field: str = "natural language processing (NLP)",
) -> str:
    """Return a ready-to-send problem-space weak signal prompt."""
    return PROBLEM_SPACE_PROMPT.format(
        field=field,
        mainframe_topic=mainframe_topic,
    )


def format_solution_prompt(
    mainframe_topic: str,
    field: str = "natural language processing (NLP)",
) -> str:
    """Return a ready-to-send solution-space weak signal prompt."""
    return SOLUTION_SPACE_PROMPT.format(
        field=field,
        mainframe_topic=mainframe_topic,
    )
