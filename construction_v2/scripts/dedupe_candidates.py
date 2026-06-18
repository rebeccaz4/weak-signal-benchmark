#!/usr/bin/env python
"""Deduplicate extracted candidate topics into a candidate index.

Run examples:
  conda run -n osworld python construction_v2/scripts/dedupe_candidates.py
  conda run -n osworld python construction_v2/scripts/dedupe_candidates.py --topic "trustworthy AI"
  conda run -n osworld python construction_v2/scripts/dedupe_candidates.py --topic "trustworthy AI" --years 2019 2020 2021 2022 2023
  conda run -n osworld python construction_v2/scripts/dedupe_candidates.py --use-clustering --cluster-threshold 0.9
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CONSTRUCTION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TOPICS_JSON = DEFAULT_CONSTRUCTION_DIR / "topics.json"
DEFAULT_CANDIDATE_DIR = DEFAULT_CONSTRUCTION_DIR / "candidate_topics"
DEFAULT_OUTPUT_DIR = DEFAULT_CONSTRUCTION_DIR / "candidate_dedup"
DEFAULT_YEARS = [2019, 2020, 2021, 2022, 2023]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate extracted candidate topics into candidate_index.json."
    )
    parser.add_argument("--topics-json", type=Path, default=DEFAULT_TOPICS_JSON)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--topic", help="Process only one topic from topics.json.")
    parser.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    parser.add_argument("--candidate-text", choices=["topic", "topic-evidence"], default="topic")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--write-parquet", action="store_true")
    parser.add_argument("--use-clustering", action="store_true")
    parser.add_argument("--cluster-threshold", type=float, default=0.9)
    parser.add_argument("--cluster-embed-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--cluster-batch-size", type=int, default=128)
    parser.add_argument("--cluster-summary", action="store_true")
    return parser.parse_args()


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "item"


def normalize_candidate_topic(text: str) -> str:
    norm = text.lower().strip()
    norm = re.sub(r"\s+", " ", norm)
    norm = re.sub(r"[\s\.,;:]+$", "", norm)
    return norm


def candidate_id(target_topic_slug: str, topic_type: str, candidate_topic_norm: str) -> str:
    raw = f"{target_topic_slug}|{topic_type}|{candidate_topic_norm}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def cluster_id(target_topic_slug: str, topic_type: str, member_ids: list[str]) -> str:
    raw = f"{target_topic_slug}|{topic_type}|{'|'.join(sorted(member_ids))}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_topics(path: Path, selected_topic: str | None = None) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        topics = json.load(fp)
    if selected_topic:
        if selected_topic not in topics:
            available = ", ".join(sorted(topics))
            raise SystemExit(
                f"Topic not found in {path}: {selected_topic}\nAvailable topics: {available}"
            )
        return {selected_topic: topics[selected_topic]}
    return topics


def candidate_jsonl_path(candidate_dir: Path, topic: str, year: int) -> Path:
    topic_slug = slugify(topic)
    return candidate_dir / topic_slug / f"candidate_topics_{topic_slug}_{year}.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_candidate_index(
    candidate_dir: Path,
    topics: dict[str, dict[str, Any]],
    years: list[int],
    candidate_text_mode: str,
) -> pd.DataFrame:
    by_id: dict[str, dict[str, Any]] = {}
    raw_mentions = 0
    for target_topic in topics:
        target_slug = slugify(target_topic)
        for year in years:
            for record in read_jsonl(candidate_jsonl_path(candidate_dir, target_topic, year)):
                source_paper = record.get("paper") or {}
                source_paper_id = str(source_paper.get("paperId") or "")
                for item in record.get("candidate_topics") or []:
                    candidate_topic = str(item.get("topic") or "").strip()
                    topic_type = str(item.get("topic_type") or "").strip()
                    if not candidate_topic or topic_type not in {"problem-space", "solution-space"}:
                        continue
                    raw_mentions += 1
                    topic_norm = normalize_candidate_topic(candidate_topic)
                    cid = candidate_id(target_slug, topic_type, topic_norm)
                    evidence = str(item.get("evidence") or "").strip()
                    confidence = str(item.get("confidence") or "").strip()
                    if cid not in by_id:
                        match_text = candidate_topic
                        if candidate_text_mode == "topic-evidence" and evidence:
                            match_text = f"{candidate_topic}. {evidence}"
                        by_id[cid] = {
                            "candidate_id": cid,
                            "target_topic": target_topic,
                            "target_topic_slug": target_slug,
                            "candidate_topic": candidate_topic,
                            "candidate_topic_norm": topic_norm,
                            "candidate_topic_type": topic_type,
                            "match_text": match_text,
                            "source_paper_ids": [],
                            "source_years": [],
                            "source_evidence": [],
                            "source_confidences": [],
                            "raw_mentions": 0,
                        }
                    row = by_id[cid]
                    row["raw_mentions"] += 1
                    if source_paper_id and source_paper_id not in row["source_paper_ids"]:
                        row["source_paper_ids"].append(source_paper_id)
                    if year not in row["source_years"]:
                        row["source_years"].append(year)
                    if evidence and evidence not in row["source_evidence"]:
                        row["source_evidence"].append(evidence)
                    if confidence and confidence not in row["source_confidences"]:
                        row["source_confidences"].append(confidence)

    candidates = pd.DataFrame(by_id.values())
    if candidates.empty:
        print(f"Raw candidate mentions: {raw_mentions:,}")
        return candidates
    candidates["source_count"] = candidates["source_paper_ids"].apply(len)
    candidates.sort_values(["target_topic_slug", "candidate_topic_type", "candidate_topic_norm"], inplace=True)
    candidates = candidates.reset_index(drop=True)
    print(f"Raw candidate mentions: {raw_mentions:,}")
    print(f"Deduplicated candidates: {len(candidates):,}")
    return candidates


def dataframe_to_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if not isinstance(value, (list, dict)) and pd.isna(value):
                record[key] = None
    return records


def write_candidate_json(df: pd.DataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = dataframe_to_json_records(df)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(records, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def read_candidate_json(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fp:
        records = json.load(fp)
    return pd.DataFrame(records)


def topic_candidate_path(
    output_dir: Path | str,
    topic: str,
    output_suffix: str = "",
    candidate_topic_type: str = "all",
    stem: str = "candidate_index",
    extension: str = "json",
) -> Path:
    output_dir = Path(output_dir)
    topic_slug = slugify(topic)
    if candidate_topic_type == "all":
        return output_dir / topic_slug / f"{stem}_{topic_slug}{output_suffix}.{extension}"
    return (
        output_dir
        / topic_slug
        / candidate_topic_type
        / f"{stem}_{topic_slug}_{candidate_topic_type}{output_suffix}.{extension}"
    )


def write_topic_candidate_outputs(
    candidates: pd.DataFrame,
    output_dir: Path,
    output_suffix: str = "",
    write_parquet: bool = False,
) -> list[Path]:
    written: list[Path] = []
    for target_topic, topic_df in candidates.groupby("target_topic", sort=True):
        topic_path = topic_candidate_path(output_dir, str(target_topic), output_suffix)
        write_candidate_json(topic_df.reset_index(drop=True), topic_path)
        written.append(topic_path)
        print(f"Saved topic candidate JSON: {topic_path}")
        if write_parquet:
            parquet_path = topic_candidate_path(
                output_dir,
                str(target_topic),
                output_suffix,
                extension="parquet",
            )
            topic_df.reset_index(drop=True).to_parquet(parquet_path, index=False)
            written.append(parquet_path)
            print(f"Saved topic candidate parquet: {parquet_path}")

        for topic_type, type_df in topic_df.groupby("candidate_topic_type", sort=True):
            type_path = topic_candidate_path(
                output_dir,
                str(target_topic),
                output_suffix,
                candidate_topic_type=str(topic_type),
            )
            write_candidate_json(type_df.reset_index(drop=True), type_path)
            written.append(type_path)
            print(f"Saved {topic_type} candidate JSON: {type_path}")
            if write_parquet:
                type_parquet_path = topic_candidate_path(
                    output_dir,
                    str(target_topic),
                    output_suffix,
                    candidate_topic_type=str(topic_type),
                    extension="parquet",
                )
                type_df.reset_index(drop=True).to_parquet(type_parquet_path, index=False)
                written.append(type_parquet_path)
                print(f"Saved {topic_type} candidate parquet: {type_parquet_path}")
    return written


def choose_canonical_topic(group: pd.DataFrame) -> str:
    ranked = group.copy()
    ranked["topic_len"] = ranked["candidate_topic"].astype(str).str.len()
    ranked.sort_values(
        ["source_count", "raw_mentions", "topic_len", "candidate_topic_norm"],
        ascending=[False, False, True, True],
        inplace=True,
    )
    return str(ranked.iloc[0]["candidate_topic"])


def summarize_cluster(member_topics: list[str], canonical_topic: str, enabled: bool) -> str:
    if not enabled:
        return canonical_topic
    unique_topics = []
    for topic in member_topics:
        if topic not in unique_topics:
            unique_topics.append(topic)
    preview = "; ".join(unique_topics[:5])
    return f"{canonical_topic}. Related extracted phrasings: {preview}"


def semantic_cluster_candidates(
    candidates: pd.DataFrame,
    threshold: float,
    embed_model_name: str,
    batch_size: int,
    include_summary: bool,
) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer

    if candidates.empty:
        return candidates

    model = SentenceTransformer(embed_model_name)
    cluster_rows: list[dict[str, Any]] = []
    for (target_slug, topic_type), group in candidates.groupby(
        ["target_topic_slug", "candidate_topic_type"], sort=True
    ):
        group = group.reset_index(drop=True)
        texts = group["match_text"].fillna(group["candidate_topic"]).astype(str).tolist()
        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        norms = (embeddings ** 2).sum(axis=1, keepdims=True) ** 0.5
        norms[norms == 0] = 1.0
        sims = (embeddings / norms) @ (embeddings / norms).T

        parent = list(range(len(group)))

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if sims[i, j] >= threshold:
                    union(i, j)

        components: dict[int, list[int]] = {}
        for idx in range(len(group)):
            components.setdefault(find(idx), []).append(idx)

        for member_indexes in components.values():
            members = group.iloc[member_indexes].copy()
            member_ids = members["candidate_id"].astype(str).tolist()
            canonical_topic = choose_canonical_topic(members)
            member_topics = members["candidate_topic"].astype(str).tolist()
            source_paper_ids = sorted(
                {
                    str(paper_id)
                    for paper_ids in members["source_paper_ids"]
                    for paper_id in (paper_ids or [])
                }
            )
            source_years = sorted(
                {
                    int(year)
                    for years in members["source_years"]
                    for year in (years or [])
                }
            )
            cluster_rows.append(
                {
                    "cluster_id": cluster_id(str(target_slug), str(topic_type), member_ids),
                    "target_topic": members.iloc[0]["target_topic"],
                    "target_topic_slug": target_slug,
                    "candidate_topic_type": topic_type,
                    "canonical_topic": canonical_topic,
                    "summary": summarize_cluster(member_topics, canonical_topic, include_summary),
                    "member_count": len(members),
                    "source_paper_count": len(source_paper_ids),
                    "member_candidate_ids": member_ids,
                    "member_topics": member_topics,
                    "source_paper_ids": source_paper_ids,
                    "source_years": source_years,
                }
            )

    clusters = pd.DataFrame(cluster_rows)
    clusters.sort_values(["target_topic_slug", "candidate_topic_type", "canonical_topic"], inplace=True)
    clusters.reset_index(drop=True, inplace=True)
    print(f"Semantic clusters: {len(clusters):,}")
    return clusters


def main() -> None:
    args = parse_args()
    topics = load_topics(args.topics_json, args.topic)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidate_index(args.candidate_dir, topics, args.years, args.candidate_text)
    if candidates.empty:
        raise RuntimeError("No candidate topics found. Run extract_candidate.py first.")
    write_topic_candidate_outputs(
        candidates,
        output_dir=args.output_dir,
        output_suffix=args.output_suffix,
        write_parquet=args.write_parquet,
    )
    if args.use_clustering:
        clusters = semantic_cluster_candidates(
            candidates,
            threshold=args.cluster_threshold,
            embed_model_name=args.cluster_embed_model,
            batch_size=args.cluster_batch_size,
            include_summary=args.cluster_summary,
        )
        for target_topic, cluster_df in clusters.groupby("target_topic", sort=True):
            cluster_path = topic_candidate_path(
                args.output_dir,
                str(target_topic),
                args.output_suffix,
                stem="candidate_clusters",
            )
            write_candidate_json(cluster_df.reset_index(drop=True), cluster_path)
            print(f"Saved topic cluster JSON: {cluster_path}")
            if args.write_parquet:
                cluster_parquet_path = topic_candidate_path(
                    args.output_dir,
                    str(target_topic),
                    args.output_suffix,
                    stem="candidate_clusters",
                    extension="parquet",
                )
                cluster_df.reset_index(drop=True).to_parquet(cluster_parquet_path, index=False)
                print(f"Saved topic cluster parquet: {cluster_parquet_path}")


if __name__ == "__main__":
    main()
