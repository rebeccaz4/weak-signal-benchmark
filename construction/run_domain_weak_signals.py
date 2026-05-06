#!/usr/bin/env python3
"""Launch worker processes for topics in a selected domain.

By default all topics run in parallel.  Use --max-workers to cap concurrency.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONSTRUCTION_DIR = REPO_ROOT / "construction"
DOMAIN_FILE = CONSTRUCTION_DIR / "weak_signals_by_domain.json"
DEFAULT_OUTPUT_ROOT = CONSTRUCTION_DIR / "outputs"
DEFAULT_LOG_ROOT = CONSTRUCTION_DIR / "logs"


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_domains() -> dict[str, list[str]]:
    return json.loads(DOMAIN_FILE.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all topics for one domain in parallel.")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--field", default=None, help="Override the prompt field for every topic in the domain.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--user-agent", default=None)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--web-search", action="store_true", help="Enable agentic web search via Semantic Scholar.")
    parser.add_argument("--max-workers", type=int, default=0,
                        help="Max concurrent topic processes (0 = unlimited, all at once).")
    return parser.parse_args()


def build_cmd(args: argparse.Namespace, topic: str) -> list[str]:
    """Build the subprocess command for one topic."""
    worker_script = CONSTRUCTION_DIR / "run_topic_weak_signals.py"
    cmd = [
        sys.executable,
        str(worker_script),
        "--domain", args.domain,
        "--topic", topic,
        "--output-root", str(args.output_root),
        "--log-root", str(args.log_root),
        "--model", args.model,
        "--max-retries", str(args.max_retries),
        "--retry-backoff", str(args.retry_backoff),
    ]
    if args.field:
        cmd.extend(["--field", args.field])
    if args.base_url:
        cmd.extend(["--base-url", args.base_url])
    if args.user_agent:
        cmd.extend(["--user-agent", args.user_agent])
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.web_search:
        cmd.append("--web-search")
    return cmd


def main() -> int:
    args = parse_args()
    domains = load_domains()
    if args.domain not in domains:
        available = ", ".join(sorted(domains))
        raise SystemExit(f"Unknown domain: {args.domain}\nAvailable domains: {available}")

    topics = domains[args.domain]
    domain_log_dir = args.log_root / slugify(args.domain)
    domain_log_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    max_workers = args.max_workers if args.max_workers > 0 else len(topics)

    print("=" * 72, flush=True)
    print(f"domain:      {args.domain}", flush=True)
    print(f"topics:      {len(topics)}", flush=True)
    print(f"max_workers: {max_workers}", flush=True)
    print(f"model:       {args.model}", flush=True)
    print(f"web_search:  {args.web_search}", flush=True)
    if args.base_url:
        print(f"base_url:    {args.base_url}", flush=True)
    print(f"output root: {args.output_root}", flush=True)
    print(f"log root:    {args.log_root}", flush=True)
    print("=" * 72, flush=True)

    # Track active and finished processes
    pending_topics = list(topics)
    active: list[tuple[str, Path, subprocess.Popen[str]]] = []
    failures: list[tuple[str, int, Path]] = []
    done_count = 0

    while pending_topics or active:
        # Fill up to max_workers
        while pending_topics and len(active) < max_workers:
            topic = pending_topics.pop(0)
            log_path = domain_log_dir / f"{slugify(topic)}.log"
            cmd = build_cmd(args, topic)
            log_file = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append((topic, log_path, proc))
            print(f"[spawned] pid={proc.pid} topic={topic} log={log_path}", flush=True)

        # Wait for any one process to finish
        still_active: list[tuple[str, Path, subprocess.Popen[str]]] = []
        for topic, log_path, proc in active:
            ret = proc.poll()
            if ret is None:
                still_active.append((topic, log_path, proc))
            else:
                done_count += 1
                if ret == 0:
                    print(f"[ok {done_count}/{len(topics)}] topic={topic}", flush=True)
                else:
                    failures.append((topic, ret, log_path))
                    print(f"[fail {done_count}/{len(topics)}] topic={topic} exit={ret} log={log_path}", flush=True)

        if still_active and len(still_active) == len(active):
            # Nothing finished this iteration — brief sleep to avoid busy-wait
            import time
            time.sleep(0.5)

        active = still_active

    if failures:
        print(f"\nFailed topics ({len(failures)}):", flush=True)
        for topic, code, log_path in failures:
            print(f"- {topic} | exit={code} | log={log_path}", flush=True)
        return 1

    print("\nAll topic processes finished successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
