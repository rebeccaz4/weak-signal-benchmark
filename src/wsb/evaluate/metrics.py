"""Metric aggregation helpers."""

from __future__ import annotations

import pandas as pd


def compute_metrics(
    precision_matches: list[int],
    recall_matches: list[int],
    n_gt: int,
    n_ext: int,
) -> dict:
    """Compute P/R/F1 from two binary match vectors.

    Args:
        precision_matches: 0/1 per external topic (does it match any GT?).
        recall_matches: 0/1 per GT topic (does it match any external?).
        n_gt: number of ground-truth topics.
        n_ext: number of external topics.

    Returns:
        dict with precision, recall, f1, precision_matched, recall_matched.
    """
    precision_matched = sum(precision_matches)
    recall_matched = sum(recall_matches)

    precision = precision_matched / n_ext if n_ext > 0 else 0.0
    recall = recall_matched / n_gt if n_gt > 0 else 0.0
    denom = precision + recall
    f1 = (2 * precision * recall / denom) if denom > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "precision_matched": precision_matched,
        "recall_matched": recall_matched,
        "n_ext": n_ext,
        "n_gt": n_gt,
    }


def flatten_metric_runs(raw_runs: list[dict], model_name: str = "external") -> pd.DataFrame:
    """Convert a list of run result dicts into a tidy per-run DataFrame."""
    rows = []
    for run_idx, payload in enumerate(raw_runs, start=1):
        rows.append(
            {
                "run": run_idx,
                "model": payload.get("model", model_name),
                "precision": payload.get("precision"),
                "recall": payload.get("recall"),
                "f1": payload.get("f1"),
                "precision_matched": payload.get("precision_matched"),
                "recall_matched": payload.get("recall_matched"),
                "n_ext": payload.get("n_ext"),
                "n_gt": payload.get("n_gt"),
            }
        )
    return pd.DataFrame(rows)


def compute_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean/std summary from a per-run metrics DataFrame."""
    return (
        metrics_df[["precision", "recall", "f1"]]
        .agg(["mean", "std"])
        .T.round(4)
    )
