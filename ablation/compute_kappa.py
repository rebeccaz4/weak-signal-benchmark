"""Compute Cohen's kappa between unified human labels and LLM judge labels."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from .common import EVALUATION_RESULTS_ROOT, RESULTS_ROOT, read_json, write_json

DEFAULT_UNIFIED_HUMAN_JUDGMENTS_JSON = (
    EVALUATION_RESULTS_ROOT / "sampled_20_topics_gt_vs_all_models_zh_judged_unified.json"
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for kappa computation."""
    parser = argparse.ArgumentParser(description="Compute Cohen's kappa for human vs judge labels.")
    parser.add_argument("--human-judgments-json", type=Path, default=DEFAULT_UNIFIED_HUMAN_JUDGMENTS_JSON)
    parser.add_argument("--judge-jsonl", type=Path, default=RESULTS_ROOT / "judge_raw_decisions.jsonl")
    return parser.parse_args()


def parse_unified_human_judgments_json(human_judgments_json: Path) -> dict[tuple[str, str, str, str, int], int]:
    """Parse binary human labels from the unified sample20 JSON."""
    payload = read_json(human_judgments_json)
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise RuntimeError(f"Invalid unified human judgments JSON: {human_judgments_json}")
    records: dict[tuple[str, str, str, str, int], int] = {}
    for row in payload["records"]:
        human_match = row.get("human_match")
        if not isinstance(human_match, bool):
            raise RuntimeError(
                "Unified human judgments JSON contains a non-boolean human_match for "
                f"{row.get('domain')}/{row.get('topic')}/{row.get('direction')}/{row.get('model')}/"
                f"{row.get('prediction_index')}"
            )
        key = (
            row["domain"],
            row["topic"],
            row["direction"],
            row["model"],
            int(row["prediction_index"]),
        )
        if key in records:
            raise RuntimeError(f"Duplicate human judgment key in {human_judgments_json}: {key}")
        records[key] = int(human_match)
    return records


def parse_judge_jsonl(judge_jsonl: Path) -> dict[tuple[str, str, str, str, int], int]:
    """Load judge replay labels from JSONL."""
    records = {}
    for line in judge_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        key = (
            payload["domain"],
            payload["topic"],
            payload["direction"],
            payload["model"],
            int(payload["prediction_index"]),
        )
        records[key] = int(payload["judge_match"])
    return records


def compute_cohen_kappa(human_labels: list[int], judge_labels: list[int]) -> float:
    """Compute Cohen's kappa for two aligned binary label lists."""
    if len(human_labels) != len(judge_labels) or not human_labels:
        raise ValueError("Human and judge labels must be non-empty and aligned.")
    n_items = len(human_labels)
    agreement = sum(1 for h, j in zip(human_labels, judge_labels) if h == j)
    p0 = agreement / n_items

    human_pos = sum(human_labels) / n_items
    human_neg = 1.0 - human_pos
    judge_pos = sum(judge_labels) / n_items
    judge_neg = 1.0 - judge_pos
    pe = human_pos * judge_pos + human_neg * judge_neg
    if pe == 1.0:
        return 1.0
    return (p0 - pe) / (1.0 - pe)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    human_records = parse_unified_human_judgments_json(args.human_judgments_json)
    judge_records = parse_judge_jsonl(args.judge_jsonl)

    shared_keys = sorted(key for key in human_records if key in judge_records)
    expected_human_keys = set(human_records)
    if shared_keys != sorted(expected_human_keys):
        missing = sorted(expected_human_keys - set(shared_keys))
        raise RuntimeError(f"Missing judge labels for {len(missing)} human-labeled predictions.")

    human_labels = [human_records[key] for key in shared_keys]
    judge_labels = [judge_records[key] for key in shared_keys]
    overall_kappa = compute_cohen_kappa(human_labels, judge_labels)

    per_model: dict[str, dict[str, float | int]] = {}
    model_groups: dict[str, list[tuple[tuple[str, str, str, str, int], int, int]]] = defaultdict(list)
    for key in shared_keys:
        model_groups[key[3]].append((key, human_records[key], judge_records[key]))
    for model, entries in sorted(model_groups.items()):
        per_model[model] = {
            "n_predictions": len(entries),
            "kappa": round(
                compute_cohen_kappa([entry[1] for entry in entries], [entry[2] for entry in entries]),
                6,
            ),
        }

    output = {
        "metadata": {
            "human_judgments_json": str(args.human_judgments_json),
            "judge_jsonl": str(args.judge_jsonl),
            "n_predictions": len(shared_keys),
        },
        "overall": {"cohen_kappa": round(overall_kappa, 6)},
        "per_model": per_model,
    }
    json_path = RESULTS_ROOT / "cohen_kappa.json"
    md_path = RESULTS_ROOT / "cohen_kappa.md"
    write_json(json_path, output)

    lines = [
        "# Cohen's Kappa",
        "",
        f"- overall_kappa: `{output['overall']['cohen_kappa']}`",
        f"- n_predictions: `{output['metadata']['n_predictions']}`",
        "",
        "## Per Model",
        "",
        "| Model | N | Kappa |",
        "|---|---:|---:|",
    ]
    for model, payload in sorted(per_model.items()):
        lines.append(f"| `{model}` | {payload['n_predictions']} | {payload['kappa']:.6f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote kappa JSON: {json_path}")
    print(f"Wrote kappa Markdown: {md_path}")


if __name__ == "__main__":
    main()
