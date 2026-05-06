#!/usr/bin/env python3
"""Check construction/outputs for failed or incomplete weak-signal results.

Detects three failure modes per direction (problem/solution):
  - missing  : directory exists but result_latest.json is absent
  - empty    : result_latest.json exists but weak_signals == []
  - invalid  : result_latest.json exists but cannot be parsed or has wrong shape

Usage:
    python tools/check_construction_outputs.py [--output-root construction/outputs]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "construction" / "outputs"
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
DIRECTIONS = ("problem", "solution")


def check_direction(direction_dir: Path) -> tuple[str, str] | None:
    """Return (status, detail) if a problem is found, else None."""
    latest = direction_dir / "result_latest.json"
    if not latest.exists():
        return "missing", "result_latest.json not found"
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
        signals = data["result"]["weak_signals"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return "invalid", str(exc)
    if not isinstance(signals, list):
        return "invalid", f"weak_signals is not a list: {type(signals)}"
    if len(signals) == 0:
        return "empty", "weak_signals == []"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Check construction outputs for failures.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    output_root = args.output_root
    if not output_root.exists():
        raise SystemExit(f"Output root not found: {output_root}")

    failures: list[dict] = []
    total_checked = 0

    for domain_dir in sorted(output_root.iterdir()):
        if not domain_dir.is_dir():
            continue
        for topic_dir in sorted(domain_dir.iterdir()):
            if not topic_dir.is_dir():
                continue
            for direction in DIRECTIONS:
                direction_dir = topic_dir / direction
                if not direction_dir.exists():
                    # Direction dir missing entirely — topic may not have started
                    failures.append({
                        "status": "missing",
                        "domain": domain_dir.name,
                        "topic": topic_dir.name,
                        "direction": direction,
                        "detail": f"{direction}/ directory does not exist",
                        "path": str(direction_dir),
                    })
                    continue
                total_checked += 1
                result = check_direction(direction_dir)
                if result is not None:
                    status, detail = result
                    failures.append({
                        "status": status,
                        "domain": domain_dir.name,
                        "topic": topic_dir.name,
                        "direction": direction,
                        "detail": detail,
                        "path": str(direction_dir),
                    })

    # ── Summary ──────────────────────────────────────────────────────────────
    status_counts: dict[str, int] = {}
    for f in failures:
        status_counts[f["status"]] = status_counts.get(f["status"], 0) + 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines: list[str] = []
    lines.append(f"Construction outputs check — {timestamp}")
    lines.append(f"Output root : {output_root}")
    lines.append(f"Checked     : {total_checked} direction folders")
    lines.append(f"Failures    : {len(failures)}")
    for status, count in sorted(status_counts.items()):
        lines.append(f"  {status}: {count}")

    if failures:
        lines.append("\nDetails:")
        for f in failures:
            lines.append(
                f"  [{f['status']:7s}] {f['domain']} / {f['topic']} / {f['direction']}"
                f"\n           {f['detail']}"
            )
    else:
        lines.append("\nAll checked directories look healthy.")

    report = "\n".join(lines)
    print(report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"construction_check_{timestamp}.txt"
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
