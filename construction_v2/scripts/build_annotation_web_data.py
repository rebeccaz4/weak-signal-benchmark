#!/usr/bin/env python
"""Build annotation_web/data.js from final weak-signal outputs."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CONSTRUCTION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = DEFAULT_CONSTRUCTION_DIR / "final_results_core"
DEFAULT_WEB_DIR = DEFAULT_CONSTRUCTION_DIR / "annotation_web"
DEFAULT_OUTPUT = DEFAULT_WEB_DIR / "data.js"
EARLY_YEARS = [2019, 2020, 2021, 2022, 2023]
DEFAULT_STORAGE_KEYS = {
    "core": "weak-signal-core-manual-annotation-v2",
    "strict": "weak-signal-strict-annotation-v1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build annotation web data.js.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--dataset-mode",
        choices=["core", "strict"],
        default="core",
        help="Annotation dataset mode. Used for selection_source labels and browser storage isolation.",
    )
    parser.add_argument(
        "--storage-key",
        help="Override the browser localStorage key. Defaults to a mode-specific key.",
    )
    parser.add_argument(
        "--page-title",
        help="Override the annotation page title written to data.js.",
    )
    parser.add_argument(
        "--include-manual-rescue",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include final_results_core/manual_rescue/manual_rescue.csv.",
    )
    return parser.parse_args()


def safe_name(text: str, max_len: int = 90) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text.lower())
    out = re.sub(r"_+", "_", out).strip("_")
    return (out or "item")[:max_len]


def json_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return ""
    if pd.isna(value):
        return ""
    return value


def rel_to_web(path: Path, web_dir: Path) -> str:
    return Path(os.path.relpath(path, web_dir)).as_posix()


def build_core_items(results_dir: Path, web_dir: Path, selection_source: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for csv_path in sorted(results_dir.glob("*/[ps]*/passed_all_gates/weak_signals.csv")):
        parts = csv_path.relative_to(results_dir).parts
        topic_slug = parts[0]
        space = parts[1]
        df = pd.read_csv(csv_path)
        for rank, (_, row) in enumerate(df.iterrows(), start=1):
            image_path = (
                results_dir
                / topic_slug
                / space
                / "passed_all_gates"
                / "frequency_individual"
                / f"{rank:02d}_{safe_name(str(row['candidate_topic']))}.png"
            )
            item = {
                "mature_topic": topic_slug,
                "space": space,
                "rank": f"{rank:02d}",
                "candidate_topic": row["candidate_topic"],
                "selection_source": selection_source,
                "original_image_path": image_path.relative_to(DEFAULT_CONSTRUCTION_DIR.parents[0]).as_posix(),
                "local_image_path": rel_to_web(image_path, web_dir),
                "github_image_url": "",
                "score": json_value(row.get("score", "")),
                "passed_all_gates": json_value(row.get("passed_all_gates", "")),
            }
            for year in EARLY_YEARS:
                item[f"topic_f_{year}"] = json_value(row.get(f"topic_f_{year}", ""))
            item["ref_f_2024"] = json_value(row.get("ref_f_2024", ""))
            items.append(item)
    return items


def build_manual_items(results_dir: Path, web_dir: Path) -> list[dict[str, Any]]:
    csv_path = results_dir / "manual_rescue" / "manual_rescue.csv"
    if not csv_path.exists():
        return []

    items: list[dict[str, Any]] = []
    df = pd.read_csv(csv_path)
    for (topic_slug, space), group in df.groupby(["manual_topic", "manual_space"], sort=True):
        for rank, (_, row) in enumerate(group.iterrows(), start=1):
            image_path = Path(str(row.get("manual_image_path", "")))
            if not image_path.is_absolute():
                image_path = DEFAULT_CONSTRUCTION_DIR.parents[0] / image_path
            item = {
                "mature_topic": topic_slug,
                "space": space,
                "rank": f"M{rank:02d}",
                "candidate_topic": row["candidate_topic"],
                "selection_source": "manual_rescue",
                "manual_reason": json_value(row.get("manual_reason", "")),
                "manual_lift_threshold": json_value(row.get("manual_lift_threshold", "")),
                "manual_lift_2024_vs_peak": json_value(row.get("manual_lift_2024_vs_peak", "")),
                "original_image_path": image_path.relative_to(DEFAULT_CONSTRUCTION_DIR.parents[0]).as_posix(),
                "local_image_path": rel_to_web(image_path, web_dir),
                "github_image_url": "",
                "score": json_value(row.get("score", "")),
                "passed_all_gates": json_value(row.get("passed_all_gates", "")),
            }
            for year in EARLY_YEARS:
                item[f"topic_f_{year}"] = json_value(row.get(f"topic_f_{year}", ""))
            item["ref_f_2024"] = json_value(row.get("ref_f_2024", ""))
            items.append(item)
    return items


def assign_annotation_ids(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for index, item in enumerate(items, start=1):
        item["annotation_id"] = f"ANN{index:04d}"
    return items


def main() -> None:
    args = parse_args()
    web_dir = args.output.parent.resolve()
    results_dir = args.results_dir.resolve()

    items = build_core_items(results_dir, web_dir, args.dataset_mode)
    if args.include_manual_rescue:
        items.extend(build_manual_items(results_dir, web_dir))
    items = assign_annotation_ids(items)

    missing_images = [item["local_image_path"] for item in items if not (web_dir / item["local_image_path"]).exists()]
    if missing_images:
        preview = "\n".join(missing_images[:10])
        raise FileNotFoundError(f"{len(missing_images)} image paths do not exist. First examples:\n{preview}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "datasetMode": args.dataset_mode,
        "storageKey": args.storage_key or DEFAULT_STORAGE_KEYS[args.dataset_mode],
        "pageTitle": args.page_title or f"Weak Signal Annotation ({args.dataset_mode.title()} Mode)",
    }
    payload = json.dumps(items, ensure_ascii=False, indent=2)
    config_payload = json.dumps(config, ensure_ascii=False, indent=2)
    args.output.write_text(
        f"window.WEAK_SIGNAL_ANNOTATION_CONFIG = {config_payload};\n"
        f"window.WEAK_SIGNAL_DATA = {payload};\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(items)} annotation items to {args.output}")
    print(f"{args.dataset_mode}={sum(item['selection_source'] == args.dataset_mode for item in items)}")
    print(f"manual_rescue={sum(item['selection_source'] == 'manual_rescue' for item in items)}")


if __name__ == "__main__":
    main()
