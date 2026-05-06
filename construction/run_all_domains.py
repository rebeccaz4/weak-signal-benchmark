#!/usr/bin/env python3
"""Run all domains sequentially; within each domain, topics run in parallel.

Usage:
    python construction/run_all_domains.py \
        --base-url "https://api.ikuncode.cc/v1" \
        --user-agent "Mozilla/5.0" \
        --web-search \
        --skip-existing
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
    parser = argparse.ArgumentParser(
        description="Run ALL domains sequentially (topics within each domain run in parallel)."
    )
    parser.add_argument(
        "--domains",
        nargs="*",
        default=None,
        help="Subset of domains to run (default: all). Quote names with spaces.",
    )
    parser.add_argument("--field", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--base-url", default="https://api.ikuncode.cc/v1")
    parser.add_argument("--user-agent", default="Mozilla/5.0")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff", type=float, default=2.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--web-search", action="store_true")
    parser.add_argument("--max-workers", type=int, default=6,
                        help="Max concurrent topic processes per domain (0 = unlimited).")
    return parser.parse_args()


def run_domain(domain: str, args: argparse.Namespace) -> tuple[str, int]:
    """Spawn run_domain_weak_signals.py for one domain and wait for it to finish."""
    cmd = [
        sys.executable,
        str(CONSTRUCTION_DIR / "run_domain_weak_signals.py"),
        "--domain", domain,
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
    if args.max_workers:
        cmd.extend(["--max-workers", str(args.max_workers)])

    result = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
    )
    return domain, result.returncode


def main() -> int:
    args = parse_args()
    all_domains = load_domains()

    if args.domains:
        # Validate requested domains
        for d in args.domains:
            if d not in all_domains:
                available = ", ".join(sorted(all_domains))
                raise SystemExit(f"Unknown domain: {d}\nAvailable: {available}")
        domains_to_run = args.domains
    else:
        domains_to_run = list(all_domains.keys())

    total_topics = sum(len(all_domains[d]) for d in domains_to_run)
    print("=" * 72, flush=True)
    print(f"Domains to run: {len(domains_to_run)}", flush=True)
    print(f"Total topics:   {total_topics}", flush=True)
    print(f"Model:          {args.model}", flush=True)
    print(f"Web search:     {args.web_search}", flush=True)
    print(f"Skip existing:  {args.skip_existing}", flush=True)
    print("=" * 72, flush=True)

    failures: list[tuple[str, int]] = []

    for i, domain in enumerate(domains_to_run, 1):
        n_topics = len(all_domains[domain])
        print(f"\n{'#' * 72}", flush=True)
        print(f"[{i}/{len(domains_to_run)}] {domain} ({n_topics} topics)", flush=True)
        print(f"{'#' * 72}", flush=True)

        domain_name, exit_code = run_domain(domain, args)

        if exit_code == 0:
            print(f"\n[domain ok] {domain_name}", flush=True)
        else:
            print(f"\n[domain FAIL] {domain_name} exit={exit_code}", flush=True)
            failures.append((domain_name, exit_code))

    # Summary
    print(f"\n{'=' * 72}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 72}", flush=True)
    print(f"Total domains: {len(domains_to_run)}", flush=True)
    print(f"Succeeded:     {len(domains_to_run) - len(failures)}", flush=True)
    print(f"Failed:        {len(failures)}", flush=True)
    if failures:
        for d, code in failures:
            print(f"  - {d} (exit={code})", flush=True)
        return 1

    print("\nAll domains completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
