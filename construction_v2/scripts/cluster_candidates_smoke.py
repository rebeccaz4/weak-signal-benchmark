#!/usr/bin/env python
"""Smoke-test semantic clustering for deduplicated candidate topics.

Default example:
  conda run -n osworld python construction_v2/scripts/cluster_candidates_smoke.py

This reads the 2019-only large-language-models deduped candidates, embeds the
candidate topic text, builds a KNN graph, unions edges above a cosine threshold,
and writes cluster JSON/Markdown for manual inspection.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm


DEFAULT_CONSTRUCTION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    DEFAULT_CONSTRUCTION_DIR
    / "candidate_dedup"
    / "large-language-models"
    / "candidate_index_large-language-models_2019_only.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_CONSTRUCTION_DIR / "candidate_dedup" / "large-language-models" / "cluster_smoke"
OFFICIAL_OPENAI_BASE_URL = "https://api.openai.com/v1"


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dedupe_candidates import slugify  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test KNN semantic clustering for candidates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-provider", choices=["openai", "local"], default="openai")
    parser.add_argument("--embed-model", default="text-embedding-3-large")
    parser.add_argument("--openai-base-url", default=OFFICIAL_OPENAI_BASE_URL)
    parser.add_argument("--openai-timeout", type=float, default=60)
    parser.add_argument("--openai-retries", type=int, default=4)
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--neighbors", type=int, default=50)
    parser.add_argument("--cluster-threshold", type=float, default=0.88)
    parser.add_argument("--prefix-absorb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefix-min-tokens", type=int, default=2)
    parser.add_argument("--candidate-topic-type", choices=["all", "problem-space", "solution-space"], default="all")
    parser.add_argument("--max-candidates", type=int, help="Optional cap for quick smoke tests.")
    parser.add_argument("--output-suffix", default="_knn_t0.88")
    return parser.parse_args()


def load_candidates(path: Path, candidate_topic_type: str, max_candidates: int | None) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as fp:
        rows = json.load(fp)
    df = pd.DataFrame(rows)
    if candidate_topic_type != "all":
        df = df[df["candidate_topic_type"] == candidate_topic_type].copy()
    df = df[df["candidate_topic"].fillna("").astype(str).str.strip().ne("")].copy()
    if max_candidates:
        df = df.head(max_candidates).copy()
    return df.reset_index(drop=True)


def cache_path(output_dir: Path, provider: str, model: str, input_path: Path, candidate_topic_type: str) -> Path:
    name = f"{input_path.stem}_{candidate_topic_type}_{provider}_{slugify(model)}.parquet"
    return output_dir / "embedding_cache" / name


def openai_embed_batch(client: OpenAI, model: str, texts: list[str], retries: int) -> list[list[float]]:
    kwargs: dict[str, Any] = {"model": model, "input": texts}
    for attempt in range(1, retries + 1):
        try:
            response = client.embeddings.create(**kwargs)
            ordered = sorted(response.data, key=lambda item: item.index)
            return [item.embedding for item in ordered]
        except Exception as exc:
            if attempt >= retries:
                raise
            sleep_seconds = min(2 ** (attempt - 1), 20)
            print(
                f"OpenAI embedding failed on attempt {attempt}/{retries}: "
                f"{type(exc).__name__}: {str(exc)[:200]}; retrying in {sleep_seconds}s"
            )
            time.sleep(sleep_seconds)
    raise RuntimeError("unreachable")


def embed_texts(args: argparse.Namespace, texts: list[str]) -> np.ndarray:
    path = cache_path(args.output_dir, args.embedding_provider, args.embed_model, args.input, args.candidate_topic_type)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        cache_df = pd.read_parquet(path)
    else:
        cache_df = pd.DataFrame(columns=["text", "embedding"])

    text_to_idx = {text: idx for idx, text in enumerate(cache_df["text"].tolist())}
    missing = [text for text in texts if text not in text_to_idx]
    if missing:
        new_rows = []
        if args.embedding_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("Set OPENAI_API_KEY to use OpenAI embeddings.")
            client = OpenAI(api_key=api_key, base_url=args.openai_base_url, timeout=args.openai_timeout)
            for start in tqdm(range(0, len(missing), args.embed_batch_size), desc="Embedding candidates", unit="batch"):
                batch = missing[start:start + args.embed_batch_size]
                embeddings = openai_embed_batch(client, args.embed_model, batch, args.openai_retries)
                for text, emb in zip(batch, embeddings):
                    new_rows.append({"text": text, "embedding": np.asarray(emb, dtype=np.float32).tolist()})
        else:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(args.embed_model)
            for start in tqdm(range(0, len(missing), args.embed_batch_size), desc="Embedding candidates", unit="batch"):
                batch = missing[start:start + args.embed_batch_size]
                embeddings = model.encode(batch, batch_size=args.embed_batch_size, show_progress_bar=False)
                for text, emb in zip(batch, embeddings):
                    new_rows.append({"text": text, "embedding": np.asarray(emb, dtype=np.float32).tolist()})
        cache_df = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
        cache_df.drop_duplicates(subset="text", keep="last", inplace=True)
        cache_df.reset_index(drop=True, inplace=True)
        cache_df.to_parquet(path, index=False)
        text_to_idx = {text: idx for idx, text in enumerate(cache_df["text"].tolist())}
        print(f"Saved embedding cache: {path}")
    else:
        print(f"All embeddings cached: {path}")

    embeddings = cache_df["embedding"].values
    return np.stack([np.asarray(embeddings[text_to_idx[text]], dtype=np.float32) for text in texts])


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def choose_canonical(members: pd.DataFrame) -> str:
    ranked = members.copy()
    ranked["topic_len"] = ranked["candidate_topic"].astype(str).str.len()
    ranked.sort_values(
        ["source_count", "raw_mentions", "topic_len", "candidate_topic_norm"],
        ascending=[False, False, True, True],
        inplace=True,
    )
    return str(ranked.iloc[0]["candidate_topic"])


def normalize_topic_text(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def is_prefix_extension(short_topic: str, long_topic: str, min_tokens: int) -> bool:
    short_norm = normalize_topic_text(short_topic)
    long_norm = normalize_topic_text(long_topic)
    if short_norm == long_norm:
        return False
    short_tokens = short_norm.split()
    if len(short_tokens) < min_tokens:
        return False
    if not long_norm.startswith(short_norm + " "):
        return False
    remainder = long_norm[len(short_norm):].strip()
    if not remainder:
        return False
    # Avoid absorbing coordinate topics such as "reasoning and planning".
    if remainder.startswith(("and ", "or ")):
        return False
    return True


def cluster_to_dataframe(cluster: dict[str, Any]) -> pd.DataFrame:
    rows = []
    source_paper_ids = cluster.get("source_paper_ids") or []
    source_years = cluster.get("source_years") or []
    for cid, topic in zip(cluster["member_candidate_ids"], cluster["member_topics"]):
        rows.append(
            {
                "candidate_id": cid,
                "candidate_topic": topic,
                "candidate_topic_norm": normalize_topic_text(topic),
                "source_count": cluster.get("source_paper_count", 0),
                "raw_mentions": 0,
                "source_paper_ids": source_paper_ids,
                "source_years": source_years,
            }
        )
    return pd.DataFrame(rows)


def rebuild_cluster_from_members(
    template: dict[str, Any],
    member_candidate_ids: list[str],
    member_topics: list[str],
    source_paper_ids: list[str],
    source_years: list[int],
) -> dict[str, Any]:
    member_df = pd.DataFrame(
        {
            "candidate_id": member_candidate_ids,
            "candidate_topic": member_topics,
            "candidate_topic_norm": [normalize_topic_text(topic) for topic in member_topics],
            "source_count": len(source_paper_ids),
            "raw_mentions": 0,
        }
    )
    rebuilt = dict(template)
    rebuilt["canonical_topic"] = choose_canonical(member_df)
    rebuilt["member_count"] = len(member_topics)
    rebuilt["member_candidate_ids"] = member_candidate_ids
    rebuilt["member_topics"] = member_topics
    rebuilt["source_paper_count"] = len(source_paper_ids)
    rebuilt["source_paper_ids"] = source_paper_ids
    rebuilt["source_years"] = source_years
    return rebuilt


def apply_prefix_absorb_to_group(clusters: list[dict[str, Any]], min_tokens: int) -> tuple[list[dict[str, Any]], int]:
    if not clusters:
        return clusters, 0

    parent = list(range(len(clusters)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(absorber: int, absorbed: int) -> None:
        absorber_root = find(absorber)
        absorbed_root = find(absorbed)
        if absorber_root != absorbed_root:
            parent[absorbed_root] = absorber_root

    canonical_topics = [str(cluster["canonical_topic"]) for cluster in clusters]
    merge_count = 0
    for short_idx, short_topic in enumerate(canonical_topics):
        for long_idx, long_topic in enumerate(canonical_topics):
            if short_idx == long_idx:
                continue
            if is_prefix_extension(short_topic, long_topic, min_tokens):
                union(short_idx, long_idx)
                merge_count += 1

    components: dict[int, list[int]] = {}
    for idx in range(len(clusters)):
        components.setdefault(find(idx), []).append(idx)

    absorbed_clusters = []
    for indexes in components.values():
        if len(indexes) == 1:
            absorbed_clusters.append(clusters[indexes[0]])
            continue
        template = clusters[indexes[0]]
        member_candidate_ids: list[str] = []
        member_topics: list[str] = []
        source_paper_ids = set()
        source_years = set()
        for idx in indexes:
            cluster = clusters[idx]
            member_candidate_ids.extend(cluster.get("member_candidate_ids") or [])
            member_topics.extend(cluster.get("member_topics") or [])
            source_paper_ids.update(cluster.get("source_paper_ids") or [])
            source_years.update(cluster.get("source_years") or [])
        dedup_member_topics = []
        dedup_member_ids = []
        seen_ids = set()
        for cid, topic in zip(member_candidate_ids, member_topics):
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            dedup_member_ids.append(cid)
            dedup_member_topics.append(topic)
        absorbed_clusters.append(
            rebuild_cluster_from_members(
                template,
                dedup_member_ids,
                dedup_member_topics,
                sorted(str(pid) for pid in source_paper_ids),
                sorted(int(year) for year in source_years),
            )
        )
    return absorbed_clusters, merge_count


def apply_prefix_absorb(clusters: list[dict[str, Any]], min_tokens: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cluster in clusters:
        key = (str(cluster["target_topic_slug"]), str(cluster["candidate_topic_type"]))
        grouped.setdefault(key, []).append(cluster)

    absorbed_all = []
    total_merges = 0
    for key, group_clusters in grouped.items():
        absorbed, merges = apply_prefix_absorb_to_group(group_clusters, min_tokens)
        total_merges += merges
        absorbed_all.extend(absorbed)
        print(f"{key[0]} / {key[1]}: prefix_absorb_edges={merges:,}, clusters_after_prefix={len(absorbed):,}")
    print(f"Prefix absorb edges total: {total_merges:,}")
    return absorbed_all


def cluster_group(group: pd.DataFrame, embeddings: np.ndarray, threshold: float, neighbors: int) -> list[dict[str, Any]]:
    if group.empty:
        return []
    n_neighbors = min(max(neighbors + 1, 2), len(group))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(embeddings)
    distances, indexes = nn.kneighbors(embeddings)

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

    edge_count = 0
    for row_idx, (row_distances, row_indexes) in enumerate(zip(distances, indexes)):
        for dist, neighbor_idx in zip(row_distances, row_indexes):
            if row_idx == neighbor_idx:
                continue
            similarity = 1.0 - float(dist)
            if similarity >= threshold:
                union(row_idx, int(neighbor_idx))
                edge_count += 1

    components: dict[int, list[int]] = {}
    for idx in range(len(group)):
        components.setdefault(find(idx), []).append(idx)

    clusters = []
    for member_indexes in components.values():
        members = group.iloc[member_indexes].copy()
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
        clusters.append(
            {
                "target_topic": members.iloc[0]["target_topic"],
                "target_topic_slug": members.iloc[0]["target_topic_slug"],
                "candidate_topic_type": members.iloc[0]["candidate_topic_type"],
                "canonical_topic": choose_canonical(members),
                "member_count": len(members),
                "member_candidate_ids": members["candidate_id"].astype(str).tolist(),
                "member_topics": member_topics,
                "source_paper_count": len(source_paper_ids),
                "source_paper_ids": source_paper_ids,
                "source_years": source_years,
            }
        )
    print(
        f"{group.iloc[0]['target_topic_slug']} / {group.iloc[0]['candidate_topic_type']}: "
        f"candidates={len(group):,}, edges={edge_count:,}, clusters={len(clusters):,}"
    )
    return clusters


def write_outputs(clusters: list[dict[str, Any]], args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"candidate_clusters_smoke{args.output_suffix}.json"
    md_path = args.output_dir / f"candidate_clusters_smoke{args.output_suffix}.md"
    clusters = sorted(clusters, key=lambda row: (row["member_count"], row["source_paper_count"]), reverse=True)
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(clusters, fp, ensure_ascii=False, indent=2)
        fp.write("\n")

    lines = [
        "# Candidate Cluster Smoke Test",
        "",
        f"Input: `{args.input}`",
        f"Embedding provider: `{args.embedding_provider}`",
        f"Embedding model: `{args.embed_model}`",
        f"Neighbors: `{args.neighbors}`",
        f"Threshold: `{args.cluster_threshold}`",
        f"Prefix absorb: `{args.prefix_absorb}`",
        f"Clusters: `{len(clusters)}`",
        "",
        "| # | Type | Members | Canonical topic | Member topics |",
        "|---:|---|---:|---|---|",
    ]
    for idx, cluster in enumerate(clusters[:200], start=1):
        members = "<br>".join(cluster["member_topics"][:12])
        if len(cluster["member_topics"]) > 12:
            members += f"<br>... +{len(cluster['member_topics']) - 12} more"
        lines.append(
            f"| {idx} | {cluster['candidate_topic_type']} | {cluster['member_count']} | "
            f"{cluster['canonical_topic']} | {members} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved clusters JSON: {json_path}")
    print(f"Saved clusters markdown: {md_path}")


def main() -> None:
    load_dotenv(DEFAULT_CONSTRUCTION_DIR / ".env")
    load_dotenv()
    args = parse_args()
    candidates = load_candidates(args.input, args.candidate_topic_type, args.max_candidates)
    if candidates.empty:
        raise RuntimeError(f"No candidates loaded from {args.input}")
    texts = candidates["candidate_topic"].astype(str).tolist()
    embeddings = normalize_matrix(embed_texts(args, texts))

    clusters: list[dict[str, Any]] = []
    for (_, topic_type), group in candidates.groupby(["target_topic_slug", "candidate_topic_type"], sort=True):
        group_indexes = group.index.to_numpy()
        clusters.extend(
            cluster_group(
                group.reset_index(drop=True),
                embeddings[group_indexes],
                threshold=args.cluster_threshold,
                neighbors=args.neighbors,
            )
        )
    if args.prefix_absorb:
        clusters = apply_prefix_absorb(clusters, args.prefix_min_tokens)
    write_outputs(clusters, args)


if __name__ == "__main__":
    main()
