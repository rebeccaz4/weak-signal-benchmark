#!/usr/bin/env python
"""Compute early/later frequency metrics for matched candidate topics.

Run examples:
  conda run -n osworld python construction_v2/scripts/compute_candidate_frequency.py --input-suffix _t0.72 --output-suffix _t0.72
  conda run -n osworld python construction_v2/scripts/compute_candidate_frequency.py --topic "trustworthy AI" --input-suffix _t0.72 --output-suffix _trustworthy_t0.72
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CONSTRUCTION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TOPICS_JSON = DEFAULT_CONSTRUCTION_DIR / "topics.json"
DEFAULT_DEDUP_DIR = DEFAULT_CONSTRUCTION_DIR / "candidate_dedup"
DEFAULT_PAPERS_DIR = DEFAULT_CONSTRUCTION_DIR / "papers"
DEFAULT_MATCHING_DIR = DEFAULT_CONSTRUCTION_DIR / "candidate_matching"
DEFAULT_OUTPUT_DIR = DEFAULT_CONSTRUCTION_DIR / "candidate_frequency"
DEFAULT_EARLY_YEARS = [2019, 2020, 2021, 2022, 2023]
DEFAULT_LATER_YEAR = 2024

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dedupe_candidates import (  # noqa: E402
    load_topics,
    normalize_candidate_topic,
    read_candidate_json,
    topic_candidate_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute candidate topic frequency metrics.")
    parser.add_argument("--topics-json", type=Path, default=DEFAULT_TOPICS_JSON)
    parser.add_argument("--dedup-dir", type=Path, default=DEFAULT_DEDUP_DIR)
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    parser.add_argument("--papers-suffix", default="")
    parser.add_argument("--matching-dir", type=Path, default=DEFAULT_MATCHING_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--topic", help="Process only one topic from topics.json.")
    parser.add_argument(
        "--candidate-topic-type",
        choices=["all", "problem-space", "solution-space"],
        default="all",
    )
    parser.add_argument("--early-years", type=int, nargs="+", default=DEFAULT_EARLY_YEARS)
    parser.add_argument("--later-year", type=int, default=DEFAULT_LATER_YEAR)
    parser.add_argument("--f-early-max", type=float, default=0.1)
    parser.add_argument(
        "--exclude-target-named-candidates",
        action="store_true",
        help="Exclude candidates that directly use the target/mainframe topic name from weak-signal output.",
    )
    parser.add_argument(
        "--exclude-late-target-era-candidates",
        action="store_true",
        help="Exclude candidates with late target-era terms such as jailbreak/hallucination for LLM.",
    )
    parser.add_argument(
        "--exclude-over-specific-candidates",
        action="store_true",
        help="Exclude long/prepositional candidates that look like application or subfield phrases.",
    )
    parser.add_argument(
        "--min-early-papers",
        type=int,
        default=1,
        help="Minimum early-stage matched papers required to call a candidate a weak signal.",
    )
    parser.add_argument("--impact-epsilon", type=float, default=1e-6)
    parser.add_argument("--input-suffix", default="")
    parser.add_argument("--dedup-suffix", default="")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument(
        "--candidate-source",
        choices=["index", "cluster"],
        default="index",
        help="Use exact-deduped candidate_index files or clustered candidate_clusters files.",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "item"


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", str(text)))


def is_target_named_candidate(candidate_topic: str, target_topic: str, target_topic_slug: str) -> bool:
    text = str(candidate_topic).lower()
    target = str(target_topic).lower()
    target_words = [word for word in re.findall(r"[a-z0-9]+", target) if len(word) > 2]
    if target and target in text:
        return True
    if target_words and all(re.search(rf"\b{re.escape(word)}s?\b", text) for word in target_words):
        return True
    if target_topic_slug == "large-language-models":
        return bool(
            re.search(
                r"\b(large language models?|llms?|chatgpt|gpt-?\d*|foundation models?)\b",
                text,
                flags=re.IGNORECASE,
            )
        )
    return False


def is_late_target_era_candidate(candidate_topic: str, target_topic_slug: str) -> bool:
    text = str(candidate_topic).lower()
    if target_topic_slug == "large-language-models":
        return bool(re.search(r"\b(jailbreak\w*|prompt injection|red teaming)\b", text))
    return False


def is_over_specific_candidate(candidate_topic: str) -> bool:
    text = str(candidate_topic).lower()
    wc = word_count(text)
    if wc > 8:
        return True
    if wc > 6 and re.search(r"\b(for|in|with|from|via|using|on|under|during)\b", text):
        return True
    if re.search(r"\b(medical|clinical|finance|financial|education|software engineering|cybersecurity)\b", text) and wc > 5:
        return True
    return False


def papers_path(papers_dir: Path, topic_slug: str, year: int, suffix: str = "") -> Path:
    return papers_dir / topic_slug / f"papers_{topic_slug}_{year}{suffix}.parquet"


def count_total_papers(papers_dir: Path, topic_slug: str, year: int, suffix: str = "") -> int:
    path = papers_path(papers_dir, topic_slug, year, suffix)
    if not path.exists():
        return 0
    df = pd.read_parquet(path, columns=["paperId"])
    return int(df["paperId"].dropna().astype(str).nunique())


def load_matches(matching_dir: Path, suffix: str) -> pd.DataFrame:
    matches_path = matching_dir / f"matched_papers{suffix}.parquet"
    if not matches_path.exists():
        raise FileNotFoundError(f"Matched papers not found: {matches_path}")
    return pd.read_parquet(matches_path)


def topic_matches_path(matching_dir: Path, topic_slug: str, suffix: str) -> Path:
    return matching_dir / topic_slug / f"matched_papers_{topic_slug}{suffix}.parquet"


def load_matches_for_topics(args: argparse.Namespace) -> pd.DataFrame:
    root_path = args.matching_dir / f"matched_papers{args.input_suffix}.parquet"
    if args.topic:
        topic_slug = slugify(args.topic)
        topic_path = topic_matches_path(args.matching_dir, topic_slug, args.input_suffix)
        if topic_path.exists():
            return pd.read_parquet(topic_path)
        if root_path.exists():
            matches = pd.read_parquet(root_path)
            return matches[matches["target_topic_slug"] == topic_slug].reset_index(drop=True)
        raise FileNotFoundError(f"Matched papers not found: {topic_path}")

    topic_paths = sorted(args.matching_dir.glob(f"*/matched_papers_*{args.input_suffix}.parquet"))
    if topic_paths:
        frames = [pd.read_parquet(path) for path in topic_paths]
        frames = [frame for frame in frames if not frame.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return load_matches(args.matching_dir, args.input_suffix)


def fallback_candidates_from_matches(matches: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_id",
        "target_topic",
        "target_topic_slug",
        "candidate_topic",
        "candidate_topic_norm",
        "candidate_topic_type",
    ]
    if matches.empty:
        return pd.DataFrame(columns=columns + ["source_count"])
    candidates = matches[columns].drop_duplicates(subset=["candidate_id"]).copy()
    candidates["source_count"] = None
    return candidates.reset_index(drop=True)


def clusters_to_frequency_candidates(clusters: pd.DataFrame) -> pd.DataFrame:
    if clusters.empty:
        return clusters
    rows = []
    for record in clusters.to_dict(orient="records"):
        candidate_topic = str(record.get("canonical_topic") or "").strip()
        rows.append(
            {
                "candidate_id": str(record.get("cluster_id") or ""),
                "target_topic": record.get("target_topic"),
                "target_topic_slug": record.get("target_topic_slug"),
                "candidate_topic": candidate_topic,
                "candidate_topic_norm": normalize_candidate_topic(candidate_topic),
                "candidate_topic_type": record.get("candidate_topic_type"),
                "source_count": record.get("source_paper_count"),
                "member_count": record.get("member_count"),
                "source_paper_count": record.get("source_paper_count"),
                "member_candidate_ids": record.get("member_candidate_ids"),
                "member_topics": record.get("member_topics"),
                "source_paper_ids": record.get("source_paper_ids"),
                "source_years": record.get("source_years"),
            }
        )
    return pd.DataFrame(rows)


def load_candidate_indexes(args: argparse.Namespace, matches: pd.DataFrame) -> pd.DataFrame:
    if args.topic:
        topics = load_topics(args.topics_json, args.topic)
    elif not matches.empty and "target_topic" in matches.columns:
        topics = {str(topic): {} for topic in sorted(matches["target_topic"].dropna().unique())}
    else:
        topics = load_topics(args.topics_json)

    frames = []
    stem = "candidate_clusters" if args.candidate_source == "cluster" else "candidate_index"
    for topic in topics:
        path = topic_candidate_path(
            args.dedup_dir,
            topic,
            args.dedup_suffix,
            candidate_topic_type=args.candidate_topic_type,
            stem=stem,
        )
        if path.exists():
            frame = read_candidate_json(path)
            if args.candidate_source == "cluster":
                frame = clusters_to_frequency_candidates(frame)
            frames.append(frame)
        elif args.topic:
            raise FileNotFoundError(f"Candidate {args.candidate_source} file not found: {path}")

    if frames:
        candidates = pd.concat(frames, ignore_index=True)
        candidates.drop_duplicates(subset=["candidate_id"], keep="first", inplace=True)
        return candidates.reset_index(drop=True)

    print("No candidate_dedup files found for selected topics; falling back to matched candidates only.")
    return fallback_candidates_from_matches(matches)


def matched_counts(matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(columns=["candidate_id", "year", "n_year"])
    return (
        matches.groupby(["candidate_id", "year"])["matched_paper_id"]
        .nunique()
        .reset_index(name="n_year")
    )


def compute_metrics(args: argparse.Namespace) -> pd.DataFrame:
    matches = load_matches_for_topics(args)
    candidates = load_candidate_indexes(args, matches)
    counts = matched_counts(matches)
    years = sorted(set(args.early_years + [args.later_year]))
    rows: list[dict[str, Any]] = []

    for _, cand in candidates.iterrows():
        topic_slug = str(cand["target_topic_slug"])
        candidate_id = str(cand["candidate_id"])
        row = {
            "candidate_id": candidate_id,
            "target_topic": cand.get("target_topic"),
            "target_topic_slug": topic_slug,
            "candidate_topic": cand.get("candidate_topic"),
            "candidate_topic_norm": cand.get("candidate_topic_norm"),
            "candidate_topic_type": cand.get("candidate_topic_type"),
            "candidate_source": args.candidate_source,
            "source_count": cand.get("source_count", 0),
            "member_count": cand.get("member_count"),
            "source_paper_count": cand.get("source_paper_count"),
            "member_candidate_ids": cand.get("member_candidate_ids"),
            "member_topics": cand.get("member_topics"),
            "source_paper_ids": cand.get("source_paper_ids"),
            "source_years": cand.get("source_years"),
        }
        candidate_topic = str(row.get("candidate_topic") or "")
        target_topic = str(row.get("target_topic") or "")
        target_named = is_target_named_candidate(candidate_topic, target_topic, topic_slug)
        late_target_era = is_late_target_era_candidate(candidate_topic, topic_slug)
        over_specific = is_over_specific_candidate(candidate_topic)
        row.update(
            {
                "candidate_word_count": word_count(candidate_topic),
                "is_target_named_candidate": target_named,
                "is_late_target_era_candidate": late_target_era,
                "is_over_specific_candidate": over_specific,
                "is_precursor_style_candidate": not (target_named or late_target_era or over_specific),
            }
        )
        for year in years:
            n_vals = counts[(counts["candidate_id"] == candidate_id) & (counts["year"] == year)]["n_year"]
            n_year = int(n_vals.iloc[0]) if not n_vals.empty else 0
            N_year = count_total_papers(args.papers_dir, topic_slug, year, args.papers_suffix)
            f_year = n_year / N_year if N_year else float("nan")
            row[f"n_{year}"] = n_year
            row[f"N_{year}"] = N_year
            row[f"f_{year}"] = f_year

        n_early = sum(row[f"n_{year}"] for year in args.early_years)
        N_early = sum(row[f"N_{year}"] for year in args.early_years)
        f_early = n_early / N_early if N_early else float("nan")
        n_later = row[f"n_{args.later_year}"]
        N_later = row[f"N_{args.later_year}"]
        f_later = n_later / N_later if N_later else float("nan")
        growth = f_later - f_early if pd.notna(f_later) and pd.notna(f_early) else float("nan")
        impact_denom = f_early + args.impact_epsilon if pd.notna(f_early) else float("nan")
        impact = growth * math.log(1.0 / impact_denom) if pd.notna(growth) else float("nan")

        row.update(
            {
                "n_early": n_early,
                "N_early": N_early,
                "f_early": f_early,
                "n_later": n_later,
                "N_later": N_later,
                "f_later": f_later,
                "growth": growth,
                "impact_epsilon": args.impact_epsilon,
                "impact": impact,
                "is_rare_early": bool(pd.notna(f_early) and f_early < args.f_early_max),
                "is_growing": bool(pd.notna(growth) and growth > 0),
                "has_early_support": bool(n_early >= args.min_early_papers),
                "min_early_papers": args.min_early_papers,
            }
        )
        row["is_candidate_weak_signal"] = (
            row["has_early_support"] and row["is_rare_early"] and row["is_growing"]
        )
        if args.exclude_target_named_candidates and row["is_target_named_candidate"]:
            row["is_candidate_weak_signal"] = False
        if args.exclude_late_target_era_candidates and row["is_late_target_era_candidate"]:
            row["is_candidate_weak_signal"] = False
        if args.exclude_over_specific_candidates and row["is_over_specific_candidate"]:
            row["is_candidate_weak_signal"] = False
        rows.append(row)

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        metrics.sort_values(["is_candidate_weak_signal", "impact"], ascending=[False, False], inplace=True)
    return metrics.reset_index(drop=True)


def write_markdown(metrics: pd.DataFrame, path: Path, early_years: list[int], later_year: int) -> None:
    lines = [
        "# Candidate Topic Frequency",
        "",
        f"Early years: {', '.join(map(str, early_years))}",
        f"Later year: {later_year}",
        "",
        "| Target topic | Candidate topic | Type | f_early | f_later | growth | impact | weak? |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    if not metrics.empty:
        for _, row in metrics.iterrows():
            lines.append(
                "| {target} | {candidate} | {typ} | {f_early:.6f} | {f_later:.6f} | "
                "{growth:.6f} | {impact:.6f} | {weak} |".format(
                    target=row.get("target_topic", ""),
                    candidate=row.get("candidate_topic", ""),
                    typ=row.get("candidate_topic_type", ""),
                    f_early=row["f_early"] if pd.notna(row["f_early"]) else float("nan"),
                    f_later=row["f_later"] if pd.notna(row["f_later"]) else float("nan"),
                    growth=row["growth"] if pd.notna(row["growth"]) else float("nan"),
                    impact=row["impact"] if pd.notna(row["impact"]) else float("nan"),
                    weak="yes" if row.get("is_candidate_weak_signal") else "no",
                )
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def weak_signal_subset(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or "is_candidate_weak_signal" not in metrics.columns:
        return metrics.copy()
    weak = metrics[metrics["is_candidate_weak_signal"]].copy()
    if not weak.empty:
        weak.sort_values("impact", ascending=False, inplace=True)
    return weak.reset_index(drop=True)


def dataframe_to_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if not isinstance(value, (list, dict)) and pd.isna(value):
                record[key] = None
    return records


def write_json(metrics: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(dataframe_to_json_records(metrics), fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def frequency_output_paths(output_dir: Path, topic_slug: str, output_suffix: str) -> dict[str, Path]:
    topic_dir = output_dir / topic_slug
    return {
        "frequency_parquet": topic_dir / f"candidate_frequency_{topic_slug}{output_suffix}.parquet",
        "frequency_json": topic_dir / f"candidate_frequency_{topic_slug}{output_suffix}.json",
        "frequency_md": topic_dir / f"candidate_frequency_{topic_slug}{output_suffix}.md",
        "weak_parquet": topic_dir / f"candidate_weak_signals_{topic_slug}{output_suffix}.parquet",
        "weak_json": topic_dir / f"candidate_weak_signals_{topic_slug}{output_suffix}.json",
        "weak_md": topic_dir / f"candidate_weak_signals_{topic_slug}{output_suffix}.md",
    }


def write_frequency_outputs(
    metrics: pd.DataFrame,
    output_dir: Path,
    output_suffix: str,
    early_years: list[int],
    later_year: int,
) -> None:
    if metrics.empty:
        topic_groups = [("empty", metrics)]
    else:
        topic_groups = list(metrics.groupby("target_topic_slug", sort=True))

    for topic_slug, topic_metrics in topic_groups:
        topic_metrics = topic_metrics.reset_index(drop=True)
        weak = weak_signal_subset(topic_metrics)
        paths = frequency_output_paths(output_dir, str(topic_slug), output_suffix)
        paths["frequency_parquet"].parent.mkdir(parents=True, exist_ok=True)
        topic_metrics.to_parquet(paths["frequency_parquet"], index=False)
        write_json(topic_metrics, paths["frequency_json"])
        write_markdown(topic_metrics, paths["frequency_md"], early_years, later_year)
        weak.to_parquet(paths["weak_parquet"], index=False)
        write_json(weak, paths["weak_json"])
        write_markdown(weak, paths["weak_md"], early_years, later_year)
        print(f"Saved metrics: {paths['frequency_parquet']}")
        print(f"Saved JSON: {paths['frequency_json']}")
        print(f"Saved markdown: {paths['frequency_md']}")
        print(f"Saved weak signals: {paths['weak_parquet']}")
        print(f"Saved weak signals JSON: {paths['weak_json']}")
        print(f"Saved weak signals markdown: {paths['weak_md']}")
        print(f"{topic_slug}: candidates evaluated={len(topic_metrics):,}, weak signals={len(weak):,}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(args)
    write_frequency_outputs(metrics, args.output_dir, args.output_suffix, args.early_years, args.later_year)
    print(f"Total candidates evaluated: {len(metrics):,}")
    if not metrics.empty:
        print(f"Total candidate weak signals: {int(metrics['is_candidate_weak_signal'].sum()):,}")


if __name__ == "__main__":
    main()
