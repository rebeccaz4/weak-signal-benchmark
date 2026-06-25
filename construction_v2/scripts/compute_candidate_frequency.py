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
DEFAULT_IMPACT_ALPHAS = [0.0, 0.3, 0.5, 0.7, 1.0]

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
        help="Minimum early-stage source papers required to call a candidate a weak signal.",
    )
    parser.add_argument("--impact-epsilon", type=float, default=1e-6)
    parser.add_argument(
        "--impact-alphas",
        type=float,
        nargs="+",
        default=DEFAULT_IMPACT_ALPHAS,
        help="Alpha values for impact_final = alpha * impact_original_norm + (1-alpha) * growth_impact_norm.",
    )
    parser.add_argument(
        "--primary-impact-alpha",
        type=float,
        default=0.5,
        help="Alpha variant used for sorting markdown and weak-signal outputs.",
    )
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


def alpha_label(alpha: float) -> str:
    label = f"{alpha:g}".replace(".", "_").replace("-", "neg_")
    return f"a{label}"


def impact_column(alpha: float) -> str:
    return f"impact_final_{alpha_label(alpha)}"


def is_survey_paper(title: Any) -> bool:
    return bool(re.search(r"\bsurvey\b", str(title or ""), flags=re.IGNORECASE))


def list_cell(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist") and not isinstance(value, str):
        parsed = value.tolist()
        return parsed if isinstance(parsed, list) else []
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else [value]
        except json.JSONDecodeError:
            return [value]
    return []


def papers_path(papers_dir: Path, topic_slug: str, year: int, suffix: str = "") -> Path:
    return papers_dir / topic_slug / f"papers_{topic_slug}_{year}{suffix}.parquet"


def read_paper_metadata(papers_dir: Path, topic_slug: str, year: int, suffix: str = "") -> pd.DataFrame:
    path = papers_path(papers_dir, topic_slug, year, suffix)
    if not path.exists():
        return pd.DataFrame(columns=["paperId", "title", "year", "is_survey"])
    df = pd.read_parquet(path)
    if "paperId" not in df.columns:
        return pd.DataFrame(columns=["paperId", "title", "year", "is_survey"])
    if "title" not in df.columns:
        df["title"] = ""
    df = df[["paperId", "title"]].dropna(subset=["paperId"]).copy()
    df["paperId"] = df["paperId"].astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["year"] = year
    df["is_survey"] = df["title"].map(is_survey_paper)
    df.drop_duplicates(subset=["paperId"], keep="first", inplace=True)
    return df.reset_index(drop=True)


def build_topic_paper_index(
    papers_dir: Path,
    topic_slug: str,
    years: list[int],
    suffix: str = "",
) -> tuple[dict[str, dict[str, Any]], dict[int, dict[str, int]], pd.DataFrame]:
    frames = [read_paper_metadata(papers_dir, topic_slug, year, suffix) for year in years]
    papers = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if papers.empty:
        return {}, {year: {"total": 0, "survey_excluded": 0, "non_survey": 0} for year in years}, papers

    paper_index = {
        str(row["paperId"]): {
            "year": int(row["year"]),
            "title": str(row.get("title") or ""),
            "is_survey": bool(row.get("is_survey")),
        }
        for _, row in papers.iterrows()
    }
    year_stats: dict[int, dict[str, int]] = {}
    for year in years:
        year_papers = papers[papers["year"] == year]
        total = int(year_papers["paperId"].nunique())
        survey = int(year_papers[year_papers["is_survey"]]["paperId"].nunique())
        year_stats[year] = {"total": total, "survey_excluded": survey, "non_survey": total - survey}
    survey_papers = papers[papers["is_survey"]].copy().reset_index(drop=True)
    return paper_index, year_stats, survey_papers


def normalize_series_to_unit(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    if numeric.empty:
        return numeric
    min_val = float(numeric.min())
    max_val = float(numeric.max())
    if math.isclose(max_val, min_val):
        return pd.Series([1.0 if max_val > 0 else 0.0] * len(numeric), index=numeric.index)
    return (numeric - min_val) / (max_val - min_val)


def fit_log_linear_trend(frequencies: list[float], epsilon: float) -> dict[str, float]:
    if not frequencies:
        return {"trend_slope": 0.0, "trend_intercept": 0.0, "trend_r2": 0.0, "growth_impact": 0.0}

    xs = list(range(len(frequencies)))
    ys = [math.log((freq if pd.notna(freq) else 0.0) + epsilon) for freq in frequencies]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    if ss_xx == 0:
        return {"trend_slope": 0.0, "trend_intercept": y_mean, "trend_r2": 0.0, "growth_impact": 0.0}

    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / ss_xx
    intercept = y_mean - slope * x_mean
    fitted = [intercept + slope * x for x in xs]
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - y_hat) ** 2 for y, y_hat in zip(ys, fitted))
    r2 = 0.0 if ss_tot == 0 else max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
    return {
        "trend_slope": slope,
        "trend_intercept": intercept,
        "trend_r2": r2,
        "growth_impact": r2 if slope > 0 else 0.0,
    }


def source_counts_by_year(
    cand: pd.Series,
    paper_index: dict[str, dict[str, Any]],
    early_years: list[int],
    candidate_id: str,
    candidate_topic: str,
) -> tuple[dict[int, int], dict[int, list[str]], list[dict[str, Any]], list[str]]:
    counts = {year: 0 for year in early_years}
    ids_by_year = {year: [] for year in early_years}
    exclusions: list[dict[str, Any]] = []
    missing_ids: list[str] = []

    for paper_id in sorted({str(pid) for pid in list_cell(cand.get("source_paper_ids")) if str(pid).strip()}):
        info = paper_index.get(paper_id)
        if not info:
            missing_ids.append(paper_id)
            continue
        year = int(info["year"])
        if year not in counts:
            continue
        if info.get("is_survey"):
            exclusions.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_topic": candidate_topic,
                    "paperId": paper_id,
                    "year": year,
                    "title": info.get("title", ""),
                    "excluded_reason": "title_contains_survey",
                    "exclusion_context": "source_paper",
                }
            )
            continue
        counts[year] += 1
        ids_by_year[year].append(paper_id)
    return counts, ids_by_year, exclusions, missing_ids


def later_count_for_candidate(
    matches: pd.DataFrame,
    candidate_id: str,
    later_year: int,
    paper_index: dict[str, dict[str, Any]],
    candidate_topic: str,
) -> tuple[int, list[dict[str, Any]]]:
    if matches.empty:
        return 0, []
    subset = matches[
        (matches["candidate_id"].astype(str) == candidate_id)
        & (matches["year"].astype(int) == later_year)
    ]
    exclusions: list[dict[str, Any]] = []
    counted_ids: set[str] = set()
    for paper_id in subset["matched_paper_id"].dropna().astype(str).unique():
        info = paper_index.get(paper_id)
        if info and info.get("is_survey"):
            exclusions.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_topic": candidate_topic,
                    "paperId": paper_id,
                    "year": later_year,
                    "title": info.get("title", ""),
                    "excluded_reason": "title_contains_survey",
                    "exclusion_context": "later_matched_paper",
                }
            )
            continue
        counted_ids.add(paper_id)
    return len(counted_ids), exclusions


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


def compute_metrics(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matches = load_matches_for_topics(args)
    candidates = load_candidate_indexes(args, matches)
    years = sorted(set(args.early_years + [args.later_year]))
    rows: list[dict[str, Any]] = []
    paper_indexes: dict[str, dict[str, dict[str, Any]]] = {}
    year_stats_by_topic: dict[str, dict[int, dict[str, int]]] = {}
    survey_paper_frames: list[pd.DataFrame] = []
    candidate_survey_exclusions: list[dict[str, Any]] = []
    topic_name_by_slug = (
        candidates.dropna(subset=["target_topic_slug"])
        .drop_duplicates(subset=["target_topic_slug"])
        .set_index("target_topic_slug")["target_topic"]
        .astype(str)
        .to_dict()
    )

    topic_slugs = sorted(candidates["target_topic_slug"].dropna().astype(str).unique())
    for topic_slug in topic_slugs:
        paper_index, year_stats, survey_papers = build_topic_paper_index(
            args.papers_dir,
            topic_slug,
            years,
            args.papers_suffix,
        )
        paper_indexes[topic_slug] = paper_index
        year_stats_by_topic[topic_slug] = year_stats
        if not survey_papers.empty:
            survey_papers = survey_papers.copy()
            survey_papers["target_topic"] = topic_name_by_slug.get(topic_slug)
            survey_papers["target_topic_slug"] = topic_slug
            survey_papers["excluded_reason"] = "title_contains_survey"
            survey_paper_frames.append(survey_papers)

    for _, cand in candidates.iterrows():
        topic_slug = str(cand["target_topic_slug"])
        candidate_id = str(cand["candidate_id"])
        paper_index = paper_indexes.get(topic_slug, {})
        year_stats = year_stats_by_topic.get(topic_slug, {})
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

        source_counts, source_ids_by_year, source_exclusions, missing_source_ids = source_counts_by_year(
            cand,
            paper_index,
            args.early_years,
            candidate_id,
            candidate_topic,
        )
        candidate_survey_exclusions.extend(source_exclusions)
        later_n, later_exclusions = later_count_for_candidate(
            matches,
            candidate_id,
            args.later_year,
            paper_index,
            candidate_topic,
        )
        for exclusion in source_exclusions + later_exclusions:
            exclusion["target_topic"] = row.get("target_topic")
            exclusion["target_topic_slug"] = topic_slug
        candidate_survey_exclusions.extend(later_exclusions)

        for year in years:
            if year in args.early_years:
                n_year = source_counts.get(year, 0)
                row[f"source_paper_ids_{year}"] = source_ids_by_year.get(year, [])
            else:
                n_year = later_n
            stats = year_stats.get(year, {"total": 0, "survey_excluded": 0, "non_survey": 0})
            N_year = stats["non_survey"]
            f_year = n_year / N_year if N_year else float("nan")
            row[f"n_{year}"] = n_year
            row[f"N_{year}"] = N_year
            row[f"N_total_{year}"] = stats["total"]
            row[f"N_survey_excluded_{year}"] = stats["survey_excluded"]
            row[f"f_{year}"] = f_year

        n_early = sum(row[f"n_{year}"] for year in args.early_years)
        N_early = sum(row[f"N_{year}"] for year in args.early_years)
        f_early = n_early / N_early if N_early else float("nan")
        n_later = row[f"n_{args.later_year}"]
        N_later = row[f"N_{args.later_year}"]
        f_later = n_later / N_later if N_later else float("nan")
        growth = f_later - f_early if pd.notna(f_later) and pd.notna(f_early) else float("nan")
        impact_denom = f_early + args.impact_epsilon if pd.notna(f_early) else float("nan")
        impact_original = growth * math.log(1.0 / impact_denom) if pd.notna(growth) else float("nan")
        trend = fit_log_linear_trend([row[f"f_{year}"] for year in args.early_years], args.impact_epsilon)
        excluded_source_ids = [record["paperId"] for record in source_exclusions]
        excluded_later_ids = [record["paperId"] for record in later_exclusions]

        row.update(
            {
                "n_early": n_early,
                "N_early": N_early,
                "N_total_early": sum(row[f"N_total_{year}"] for year in args.early_years),
                "N_survey_excluded_early": sum(row[f"N_survey_excluded_{year}"] for year in args.early_years),
                "f_early": f_early,
                "n_later": n_later,
                "N_later": N_later,
                "f_later": f_later,
                "growth": growth,
                "impact_epsilon": args.impact_epsilon,
                "impact_original": impact_original,
                "impact_original_clipped": max(0.0, impact_original) if pd.notna(impact_original) else 0.0,
                "impact": impact_original,
                "excluded_survey_source_paper_ids": excluded_source_ids,
                "excluded_survey_later_paper_ids": excluded_later_ids,
                "missing_source_paper_ids": missing_source_ids,
                "is_rare_early": bool(pd.notna(f_early) and f_early < args.f_early_max),
                "is_growing": bool(pd.notna(growth) and growth > 0),
                "has_early_support": bool(n_early >= args.min_early_papers),
                "min_early_papers": args.min_early_papers,
                **trend,
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
        metrics["impact_original_norm"] = metrics.groupby("target_topic_slug", group_keys=False)[
            "impact_original_clipped"
        ].transform(normalize_series_to_unit)
        metrics["growth_impact_norm"] = metrics.groupby("target_topic_slug", group_keys=False)[
            "growth_impact"
        ].transform(normalize_series_to_unit)
        for alpha in args.impact_alphas:
            col = impact_column(alpha)
            metrics[col] = (
                alpha * metrics["impact_original_norm"]
                + (1.0 - alpha) * metrics["growth_impact_norm"]
            )
        primary_col = impact_column(args.primary_impact_alpha)
        if primary_col not in metrics.columns:
            metrics[primary_col] = (
                args.primary_impact_alpha * metrics["impact_original_norm"]
                + (1.0 - args.primary_impact_alpha) * metrics["growth_impact_norm"]
            )
        metrics["impact_final"] = metrics[primary_col]
        metrics.sort_values(["is_candidate_weak_signal", "impact_final"], ascending=[False, False], inplace=True)

    survey_papers_df = pd.concat(survey_paper_frames, ignore_index=True) if survey_paper_frames else pd.DataFrame(
        columns=["paperId", "title", "year", "is_survey", "target_topic", "target_topic_slug", "excluded_reason"]
    )
    candidate_survey_exclusions_df = pd.DataFrame(candidate_survey_exclusions)
    return metrics.reset_index(drop=True), survey_papers_df, candidate_survey_exclusions_df


def write_markdown(metrics: pd.DataFrame, path: Path, early_years: list[int], later_year: int) -> None:
    lines = [
        "# Candidate Topic Frequency",
        "",
        f"Early years: {', '.join(map(str, early_years))}",
        f"Later year: {later_year}",
        "",
        "| Target topic | Candidate topic | Type | f_early | f_later | growth | impact_final | weak? |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    if not metrics.empty:
        for _, row in metrics.iterrows():
            lines.append(
                "| {target} | {candidate} | {typ} | {f_early:.6f} | {f_later:.6f} | "
                "{growth:.6f} | {impact_final:.6f} | {weak} |".format(
                    target=row.get("target_topic", ""),
                    candidate=row.get("candidate_topic", ""),
                    typ=row.get("candidate_topic_type", ""),
                    f_early=row["f_early"] if pd.notna(row["f_early"]) else float("nan"),
                    f_later=row["f_later"] if pd.notna(row["f_later"]) else float("nan"),
                    growth=row["growth"] if pd.notna(row["growth"]) else float("nan"),
                    impact_final=row["impact_final"] if pd.notna(row["impact_final"]) else float("nan"),
                    weak="yes" if row.get("is_candidate_weak_signal") else "no",
                )
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def weak_signal_subset(metrics: pd.DataFrame, score_col: str = "impact_final") -> pd.DataFrame:
    if metrics.empty or "is_candidate_weak_signal" not in metrics.columns:
        return metrics.copy()
    weak = metrics[metrics["is_candidate_weak_signal"]].copy()
    if not weak.empty and score_col in weak.columns:
        weak.sort_values(score_col, ascending=False, inplace=True)
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
        "frequency_xlsx": topic_dir / f"candidate_frequency_{topic_slug}{output_suffix}.xlsx",
        "weak_parquet": topic_dir / f"candidate_weak_signals_{topic_slug}{output_suffix}.parquet",
        "weak_json": topic_dir / f"candidate_weak_signals_{topic_slug}{output_suffix}.json",
        "weak_md": topic_dir / f"candidate_weak_signals_{topic_slug}{output_suffix}.md",
        "survey_parquet": topic_dir / f"excluded_survey_papers_{topic_slug}{output_suffix}.parquet",
        "survey_json": topic_dir / f"excluded_survey_papers_{topic_slug}{output_suffix}.json",
        "candidate_survey_parquet": topic_dir / f"candidate_survey_exclusions_{topic_slug}{output_suffix}.parquet",
        "candidate_survey_json": topic_dir / f"candidate_survey_exclusions_{topic_slug}{output_suffix}.json",
    }


def excel_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if out[column].map(lambda value: isinstance(value, (list, dict))).any():
            out[column] = out[column].map(
                lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            )
    return out


def write_excel_workbook(
    path: Path,
    topic_metrics: pd.DataFrame,
    survey_papers: pd.DataFrame,
    candidate_survey_exclusions: pd.DataFrame,
    impact_alphas: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(path) as writer:
            excel_safe_df(topic_metrics).to_excel(writer, sheet_name="all_candidates", index=False)
            for alpha in impact_alphas:
                col = impact_column(alpha)
                weak = weak_signal_subset(topic_metrics, col)
                sheet = f"weak_signals_{alpha_label(alpha)}"[:31]
                excel_safe_df(weak).to_excel(writer, sheet_name=sheet, index=False)
            excel_safe_df(survey_papers).to_excel(writer, sheet_name="excluded_survey_papers", index=False)
            excel_safe_df(candidate_survey_exclusions).to_excel(
                writer,
                sheet_name="candidate_survey_exclusions"[:31],
                index=False,
            )
    except ImportError as exc:
        raise RuntimeError("Install openpyxl or xlsxwriter to write Excel outputs.") from exc


def write_frequency_outputs(
    metrics: pd.DataFrame,
    survey_papers: pd.DataFrame,
    candidate_survey_exclusions: pd.DataFrame,
    output_dir: Path,
    output_suffix: str,
    early_years: list[int],
    later_year: int,
    impact_alphas: list[float],
) -> None:
    if metrics.empty:
        topic_groups = [("empty", metrics)]
    else:
        topic_groups = list(metrics.groupby("target_topic_slug", sort=True))

    for topic_slug, topic_metrics in topic_groups:
        topic_metrics = topic_metrics.reset_index(drop=True)
        weak = weak_signal_subset(topic_metrics, "impact_final")
        topic_survey = survey_papers[
            survey_papers.get("target_topic_slug", pd.Series(dtype=str)).astype(str) == str(topic_slug)
        ].reset_index(drop=True) if not survey_papers.empty else survey_papers.copy()
        topic_candidate_survey = candidate_survey_exclusions[
            candidate_survey_exclusions.get("target_topic_slug", pd.Series(dtype=str)).astype(str) == str(topic_slug)
        ].reset_index(drop=True) if not candidate_survey_exclusions.empty else candidate_survey_exclusions.copy()
        paths = frequency_output_paths(output_dir, str(topic_slug), output_suffix)
        paths["frequency_parquet"].parent.mkdir(parents=True, exist_ok=True)
        topic_metrics.to_parquet(paths["frequency_parquet"], index=False)
        write_json(topic_metrics, paths["frequency_json"])
        write_markdown(topic_metrics, paths["frequency_md"], early_years, later_year)
        write_excel_workbook(
            paths["frequency_xlsx"],
            topic_metrics,
            topic_survey,
            topic_candidate_survey,
            impact_alphas,
        )
        weak.to_parquet(paths["weak_parquet"], index=False)
        write_json(weak, paths["weak_json"])
        write_markdown(weak, paths["weak_md"], early_years, later_year)
        topic_survey.to_parquet(paths["survey_parquet"], index=False)
        write_json(topic_survey, paths["survey_json"])
        topic_candidate_survey.to_parquet(paths["candidate_survey_parquet"], index=False)
        write_json(topic_candidate_survey, paths["candidate_survey_json"])
        print(f"Saved metrics: {paths['frequency_parquet']}")
        print(f"Saved JSON: {paths['frequency_json']}")
        print(f"Saved markdown: {paths['frequency_md']}")
        print(f"Saved Excel: {paths['frequency_xlsx']}")
        print(f"Saved weak signals: {paths['weak_parquet']}")
        print(f"Saved weak signals JSON: {paths['weak_json']}")
        print(f"Saved weak signals markdown: {paths['weak_md']}")
        print(f"Saved excluded survey papers: {paths['survey_parquet']}")
        print(f"Saved candidate survey exclusions: {paths['candidate_survey_parquet']}")
        print(f"{topic_slug}: candidates evaluated={len(topic_metrics):,}, weak signals={len(weak):,}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics, survey_papers, candidate_survey_exclusions = compute_metrics(args)
    write_frequency_outputs(
        metrics,
        survey_papers,
        candidate_survey_exclusions,
        args.output_dir,
        args.output_suffix,
        args.early_years,
        args.later_year,
        args.impact_alphas,
    )
    print(f"Total candidates evaluated: {len(metrics):,}")
    if not metrics.empty:
        print(f"Total candidate weak signals: {int(metrics['is_candidate_weak_signal'].sum()):,}")


if __name__ == "__main__":
    main()
