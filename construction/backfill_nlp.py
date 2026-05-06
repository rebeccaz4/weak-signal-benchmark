#!/usr/bin/env python3
"""Backfill missing problem/solution directions for the NLP domain.

Step 1: Fix metadata — rename domain/field from "nlp" to full name.
Step 2: Generate the missing direction for each topic via run_topic_weak_signals.py.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONSTRUCTION_DIR = REPO_ROOT / "construction"
DEFAULT_OUTPUT_ROOT = CONSTRUCTION_DIR / "outputs"
DEFAULT_LOG_ROOT = CONSTRUCTION_DIR / "logs"

DOMAIN_FULL_NAME = "Natural Language Processing"
DOMAIN_SLUG = "natural_language_processing"


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def fix_metadata(output_root: Path) -> None:
    nlp_dir = output_root / DOMAIN_SLUG
    if not nlp_dir.exists():
        print("[fix_metadata] NLP directory not found, skipping.", flush=True)
        return
    count = 0
    for json_path in nlp_dir.rglob("result_*.json"):
        text = json_path.read_text(encoding="utf-8")
        data = json.loads(text)
        meta = data.get("metadata", {})
        changed = False
        if meta.get("domain") == "nlp":
            meta["domain"] = DOMAIN_FULL_NAME
            changed = True
        if meta.get("field") == "nlp":
            meta["field"] = DOMAIN_FULL_NAME
            changed = True
        if changed:
            json_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            count += 1
    print(f"[fix_metadata] Updated {count} JSON files.", flush=True)


def discover_missing(output_root: Path) -> list[tuple[str, str, str]]:
    nlp_dir = output_root / DOMAIN_SLUG
    missing: list[tuple[str, str, str]] = []
    for topic_dir in sorted(nlp_dir.iterdir()):
        if not topic_dir.is_dir():
            continue
        topic_slug = topic_dir.name
        has_problem = (topic_dir / "problem" / "result_latest.json").exists()
        has_solution = (topic_dir / "solution" / "result_latest.json").exists()
        existing_dir = "problem" if has_problem else "solution" if has_solution else None
        if existing_dir is None:
            continue
        result_file = topic_dir / existing_dir / "result_latest.json"
        data = json.loads(result_file.read_text(encoding="utf-8"))
        original_topic = data.get("metadata", {}).get("mainframe_topic", topic_slug)
        if not has_problem:
            missing.append((topic_slug, original_topic, "problem"))
        if not has_solution:
            missing.append((topic_slug, original_topic, "solution"))
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill missing directions for NLP domain.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--base-url", default="https://api.ikuncode.cc/v1")
    parser.add_argument("--user-agent", default="Mozilla/5.0")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--web-search", action="store_true")
    parser.add_argument("--max-workers", type=int, default=6,
                        help="Max concurrent topic processes (0 = unlimited).")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 72, flush=True)
    print("Step 1: Fix metadata (nlp -> Natural Language Processing)", flush=True)
    print("=" * 72, flush=True)
    fix_metadata(args.output_root)

    print(f"\n{'=' * 72}", flush=True)
    print("Step 2: Discover missing directions", flush=True)
    print("=" * 72, flush=True)
    missing = discover_missing(args.output_root)

    if not missing:
        print("Nothing to backfill -- all topics have both directions.", flush=True)
        return 0

    problem_missing = sum(1 for _, _, d in missing if d == "problem")
    solution_missing = sum(1 for _, _, d in missing if d == "solution")
    print(f"Missing problem: {problem_missing}", flush=True)
    print(f"Missing solution: {solution_missing}", flush=True)
    print(f"Total to generate: {len(missing)}", flush=True)

    if args.dry_run:
        print("\n[dry-run] Would generate:", flush=True)
        for slug, topic, direction in missing:
            print(f"  {slug}/{direction}  (topic: {topic})", flush=True)
        return 0

    print(f"\n{'=' * 72}", flush=True)
    print(f"Step 3: Generate missing directions (max_workers={args.max_workers})", flush=True)
    print("=" * 72, flush=True)

    max_workers = args.max_workers if args.max_workers > 0 else len(missing)
    worker_script = CONSTRUCTION_DIR / "run_topic_weak_signals.py"
    log_dir = args.log_root / DOMAIN_SLUG
    log_dir.mkdir(parents=True, exist_ok=True)

    pending = list(missing)
    active: list[tuple[str, str, str, Path, subprocess.Popen[str]]] = []
    failures: list[tuple[str, str, int, Path]] = []
    done_count = 0

    while pending or active:
        while pending and len(active) < max_workers:
            slug, topic, direction = pending.pop(0)
            log_path = log_dir / f"{slug}_backfill_{direction}.log"
            cmd = [
                sys.executable, str(worker_script),
                "--domain", DOMAIN_FULL_NAME,
                "--topic", topic,
                "--field", DOMAIN_FULL_NAME,
                "--output-root", str(args.output_root),
                "--log-root", str(args.log_root),
                "--model", args.model,
                "--max-retries", str(args.max_retries),
                "--retry-backoff", str(args.retry_backoff),
                "--skip-existing",
            ]
            if args.base_url:
                cmd.extend(["--base-url", args.base_url])
            if args.user_agent:
                cmd.extend(["--user-agent", args.user_agent])
            if args.web_search:
                cmd.append("--web-search")
            log_file = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd, cwd=str(REPO_ROOT),
                stdout=log_file, stderr=subprocess.STDOUT, text=True,
            )
            active.append((slug, topic, direction, log_path, proc))
            print(f"[spawned] {slug}/{direction} pid={proc.pid}", flush=True)

        still_active = []
        for slug, topic, direction, log_path, proc in active:
            ret = proc.poll()
            if ret is None:
                still_active.append((slug, topic, direction, log_path, proc))
            else:
                done_count += 1
                if ret == 0:
                    print(f"[ok {done_count}/{len(missing)}] {slug}/{direction}", flush=True)
                else:
                    failures.append((slug, direction, ret, log_path))
                    print(f"[fail {done_count}/{len(missing)}] {slug}/{direction} exit={ret} log={log_path}", flush=True)

        if still_active and len(still_active) == len(active):
            time.sleep(0.5)
        active = still_active

    print(f"\n{'=' * 72}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 72}", flush=True)
    print(f"Generated: {len(missing) - len(failures)}/{len(missing)}", flush=True)
    if failures:
        print(f"Failed: {len(failures)}", flush=True)
        for slug, direction, code, log_path in failures:
            print(f"  - {slug}/{direction} exit={code} log={log_path}", flush=True)
        return 1

    print("All backfill tasks completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
