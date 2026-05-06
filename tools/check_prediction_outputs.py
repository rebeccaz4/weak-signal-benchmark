#!/usr/bin/env python3
"""Check prediction/python/outputs for missing or empty weak-signal results.

Two modes:

  --mode existing  (default)
      Walk every signals_latest.json that already exists on disk and report
      empty or invalid ones. Does NOT compare against GT coverage.
      Failure modes detected:
        - empty   : signals_latest.json exists but signals == []
        - invalid : signals_latest.json exists but cannot be parsed or has wrong shape

  --mode coverage
      Compare each model's outputs against the GT topics in construction/outputs.
      Additionally reports topics that have not been started yet.
      Failure modes detected:
        - not_started : signals_latest.json does not exist at all
        - empty       : signals_latest.json exists but signals == []
        - invalid     : signals_latest.json exists but cannot be parsed or has wrong shape

Usage:
    python tools/check_prediction_outputs.py                          # existing mode, all models
    python tools/check_prediction_outputs.py --models qwen3.5_397b   # existing mode, one model
    python tools/check_prediction_outputs.py --mode coverage          # coverage mode
    python tools/check_prediction_outputs.py --mode coverage --show-missing
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT       = Path(__file__).resolve().parents[1]
GT_ROOT         = REPO_ROOT / "construction" / "outputs"
PREDICTION_ROOT = REPO_ROOT / "prediction" / "python" / "outputs"
REPORTS_DIR     = Path(__file__).resolve().parent / "reports"
YEAR_SLUG       = "2023_2024"
DIRECTIONS      = ("problem", "solution")


# ---------------------------------------------------------------------------
# Per-file check
# ---------------------------------------------------------------------------

def check_prediction(path: Path) -> tuple[str, str] | None:
    """Return (status, detail) if a problem is found, else None."""
    if not path.exists():
        return "not_started", "signals_latest.json not found"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        signals = data["signals"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return "invalid", str(exc)
    if not isinstance(signals, list):
        return "invalid", f"signals is not a list: {type(signals)}"
    if len(signals) == 0:
        return "empty", "signals == []"
    return None


# ---------------------------------------------------------------------------
# Existing-mode check (walk what's on disk, no GT comparison)
# ---------------------------------------------------------------------------

def check_existing(models: list[str], prediction_root: Path) -> list[str]:
    """Walk all existing signals_latest.json files and report empty/invalid ones."""
    lines: list[str] = []
    total_checked = 0
    all_failures: list[dict] = []

    for model in models:
        model_root = prediction_root / model
        if not model_root.exists():
            continue
        failures: list[dict] = []
        n_ok = 0

        for sig_file in sorted(model_root.rglob("signals_latest.json")):
            total_checked += 1
            # Path pattern: {model}/{domain}/{topic}/{direction}/{year_slug}/signals_latest.json
            parts = sig_file.relative_to(model_root).parts  # (domain, topic, direction, year, filename)
            domain    = parts[0] if len(parts) > 0 else "?"
            topic     = parts[1] if len(parts) > 1 else "?"
            direction = parts[2] if len(parts) > 2 else "?"

            try:
                data = json.loads(sig_file.read_text(encoding="utf-8"))
                signals = data["signals"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                failures.append({"status": "invalid", "domain": domain, "topic": topic,
                                  "direction": direction, "detail": str(exc)})
                continue
            if not isinstance(signals, list):
                failures.append({"status": "invalid", "domain": domain, "topic": topic,
                                  "direction": direction,
                                  "detail": f"signals is not a list: {type(signals)}"})
            elif len(signals) == 0:
                failures.append({"status": "empty", "domain": domain, "topic": topic,
                                  "direction": direction, "detail": "signals == []"})
            else:
                n_ok += 1

        counts: dict[str, int] = {}
        for f in failures:
            counts[f["status"]] = counts.get(f["status"], 0) + 1

        lines.append(f"{'─'*60}")
        lines.append(f"Model : {model}")
        lines.append(f"  Checked : {n_ok + len(failures)}")
        lines.append(f"  OK      : {n_ok}")
        lines.append(f"  empty   : {counts.get('empty', 0)}")
        lines.append(f"  invalid : {counts.get('invalid', 0)}")
        if failures:
            lines.append("  Details:")
            for f in failures:
                lines.append(
                    f"    [{f['status']:7s}] {f['domain']} / {f['topic']} / {f['direction']}"
                    + (f"  — {f['detail']}" if f["status"] == "invalid" else "")
                )
        all_failures.extend(failures)

    lines.append(f"{'─'*60}")
    lines.append(f"Total checked : {total_checked}")
    lines.append(f"Total failures: {len(all_failures)}")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Check prediction outputs for each model.")
    parser.add_argument(
        "--mode", default="existing", choices=["existing", "coverage"],
        help="existing: check only files on disk (default). coverage: compare against GT.",
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help="Model names to check (default: all found in prediction/python/outputs/)",
    )
    parser.add_argument(
        "--show-missing", action="store_true",
        help="(coverage mode only) Also list not_started items",
    )
    args = parser.parse_args()

    if not PREDICTION_ROOT.exists():
        raise SystemExit(f"Prediction root not found: {PREDICTION_ROOT}")

    available_models = sorted(d.name for d in PREDICTION_ROOT.iterdir() if d.is_dir())
    models = args.models if args.models else available_models
    unknown = [m for m in models if m not in available_models]
    if unknown:
        raise SystemExit(f"Unknown model(s): {unknown}\nAvailable: {available_models}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines: list[str] = []

    if args.mode == "existing":
        lines.append(f"Prediction outputs check (existing files only) — {timestamp}")
        lines.append(f"Models : {models}")
        lines.append("")
        lines.extend(check_existing(models, PREDICTION_ROOT))
        report_name = f"prediction_check_existing_{timestamp}.txt"

    else:  # coverage mode
        if not GT_ROOT.exists():
            raise SystemExit(f"GT root not found: {GT_ROOT}")

        gt_pairs: list[tuple[str, str]] = []
        for domain_dir in sorted(GT_ROOT.iterdir()):
            if not domain_dir.is_dir():
                continue
            for topic_dir in sorted(domain_dir.iterdir()):
                if topic_dir.is_dir():
                    gt_pairs.append((domain_dir.name, topic_dir.name))

        total_expected = len(gt_pairs) * len(DIRECTIONS)
        lines.append(f"Prediction outputs check (coverage vs GT) — {timestamp}")
        lines.append(f"GT topics : {len(gt_pairs)} topics × {len(DIRECTIONS)} directions = {total_expected} expected per model")
        lines.append(f"Models    : {models}")
        lines.append("")

        for model in models:
            model_root = PREDICTION_ROOT / model
            failures: list[dict] = []
            n_ok = 0

            for domain_slug, topic_slug in gt_pairs:
                for direction in DIRECTIONS:
                    path = (
                        model_root / domain_slug / topic_slug / direction
                        / YEAR_SLUG / "signals_latest.json"
                    )
                    result = check_prediction(path)
                    if result is None:
                        n_ok += 1
                    else:
                        status, detail = result
                        failures.append({
                            "status": status,
                            "domain": domain_slug,
                            "topic": topic_slug,
                            "direction": direction,
                            "detail": detail,
                        })

            counts: dict[str, int] = {}
            for f in failures:
                counts[f["status"]] = counts.get(f["status"], 0) + 1

            coverage_pct = round(100 * n_ok / total_expected, 1) if total_expected else 0.0
            lines.append(f"{'─'*60}")
            lines.append(f"Model : {model}")
            lines.append(f"  OK          : {n_ok} / {total_expected}  ({coverage_pct}%)")
            lines.append(f"  not_started : {counts.get('not_started', 0)}")
            lines.append(f"  empty       : {counts.get('empty', 0)}")
            lines.append(f"  invalid     : {counts.get('invalid', 0)}")

            detail_failures = [
                f for f in failures
                if f["status"] in ("empty", "invalid")
                or (f["status"] == "not_started" and args.show_missing)
            ]
            if detail_failures:
                lines.append("  Details:")
                for f in detail_failures:
                    lines.append(
                        f"    [{f['status']:11s}] {f['domain']} / {f['topic']} / {f['direction']}"
                        + (f"  — {f['detail']}" if f["status"] != "not_started" else "")
                    )

        lines.append(f"{'─'*60}")
        report_name = f"prediction_check_coverage_{timestamp}.txt"

    report = "\n".join(lines)
    print(report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / report_name
    report_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
