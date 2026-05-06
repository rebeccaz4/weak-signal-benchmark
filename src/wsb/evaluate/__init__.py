"""Evaluation subpackage — prompts, judge calls, and metric aggregation."""

from wsb.evaluate.batch import run_batch_evaluation, run_batch_evaluation_multi
from wsb.evaluate.judge import run_evaluation
from wsb.evaluate.metrics import compute_metrics, compute_summary, flatten_metric_runs
from wsb.evaluate.prompts import (
    PAIRWISE_SYSTEM_PROMPT,
    PAIRWISE_USER_TEMPLATE,
    build_pairwise_prompt,
    format_signal_block,
)

__all__ = [
    "PAIRWISE_SYSTEM_PROMPT",
    "PAIRWISE_USER_TEMPLATE",
    "build_pairwise_prompt",
    "format_signal_block",
    "run_evaluation",
    "run_batch_evaluation",
    "run_batch_evaluation_multi",
    "compute_metrics",
    "flatten_metric_runs",
    "compute_summary",
]
