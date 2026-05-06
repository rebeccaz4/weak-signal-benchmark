"""Setting 2 — Set-level LLM Judgment.

Give both GT and predicted signal sets to an LLM in one prompt.
The LLM returns precision and recall directly; we validate them and compute F1.

Runs N times and returns mean ± std.
"""
from __future__ import annotations

import json
import re
import statistics
import time

from openai import OpenAI


SYSTEM_PROMPT = """\
You are a research direction evaluator. Be inclusive — if two research directions address substantially overlapping research questions or methods, consider them a match.
"""


USER_TEMPLATE = """\
I have two sets of research weak signals. Compare the PREDICTED set against the GROUND TRUTH set.

Rules:
- A match means two research directions substantially overlap in research question, core method, or described phenomenon.
- Different angles or terminology are acceptable if the overlap is substantial.
- Return set-level scores, not matched pairs.
- Precision = how well the predicted set is covered by the ground-truth set.
- Recall = how well the ground-truth set is covered by the predicted set.
- Both scores must be numbers between 0 and 1 inclusive.

Ground truth signals:
{gt_block}

Predicted signals:
{pred_block}

Return ONLY valid JSON:
{{
  "precision": <float>,
  "recall": <float>
}}
"""


def _format_block(signals: list[str]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(signals))


def _safe_json(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


def _run_once(
    client: OpenAI,
    model: str,
    gt: list[str],
    pred: list[str],
    temperature: float,
    max_retries: int = 4,
    retry_backoff: float = 2.0,
) -> dict:
    """One LLM call. Returns validated P/R/F1."""
    prompt = USER_TEMPLATE.format(
        gt_block=_format_block(gt),
        pred_block=_format_block(pred),
    )
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise ValueError("Empty response")
            payload = _safe_json(text)
            if payload is None:
                raise ValueError("Could not parse JSON")
            precision_raw = payload.get("precision")
            recall_raw = payload.get("recall")
            if not isinstance(precision_raw, (int, float)):
                raise ValueError(f"precision is not numeric: {precision_raw!r}")
            if not isinstance(recall_raw, (int, float)):
                raise ValueError(f"recall is not numeric: {recall_raw!r}")

            precision = float(precision_raw)
            recall = float(recall_raw)
            if not 0.0 <= precision <= 1.0:
                raise ValueError(f"precision out of range [0,1]: {precision}")
            if not 0.0 <= recall <= 1.0:
                raise ValueError(f"recall out of range [0,1]: {recall}")

            denom = precision + recall
            f1 = 2 * precision * recall / denom if denom > 0 else 0.0

            return {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }

        except Exception as exc:
            if attempt >= max_retries:
                raise RuntimeError(f"LLM set judge failed after {attempt} attempts: {exc}") from exc
            sleep_s = retry_backoff * attempt
            print(f"  [warn] attempt {attempt}: {exc}. Retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)


def eval_set_llm(
    gt: list[str],
    pred: list[str],
    client: OpenAI,
    judge_model: str,
    n_runs: int = 5,
    temperature: float = 1.0,
) -> dict:
    """Setting 2: set-level LLM judgment. Returns mean/std over n_runs."""
    runs = []
    for i in range(n_runs):
        result = _run_once(client, judge_model, gt, pred, temperature)
        runs.append(result)
        print(f"    run {i+1}/{n_runs}: P={result['precision']} R={result['recall']} F1={result['f1']}")

    def _agg(key: str) -> tuple[float, float]:
        vals = [r[key] for r in runs]
        mean = round(statistics.mean(vals), 4)
        std  = round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4)
        return mean, std

    p_mean, p_std = _agg("precision")
    r_mean, r_std = _agg("recall")
    f_mean, f_std = _agg("f1")

    return {
        "setting": "set_llm",
        "precision": p_mean, "precision_std": p_std,
        "recall":    r_mean, "recall_std":    r_std,
        "f1":        f_mean, "f1_std":        f_std,
        "n_runs": n_runs,
        "n_pred": len(pred),
        "n_gt":   len(gt),
    }
