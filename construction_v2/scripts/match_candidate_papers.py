#!/usr/bin/env python
"""Match extracted candidate topics to supporting papers.

This script reads candidate-topic JSONL files and fetched paper parquet files,
deduplicates candidates within each target topic, embeds candidates and papers,
and writes matched candidate-paper pairs.

Run examples:
  conda run -n osworld python construction_v2/scripts/match_candidate_papers.py --topic "trustworthy AI"
  conda run -n osworld python construction_v2/scripts/match_candidate_papers.py --topic "trustworthy AI" --years 2019 2020 2021 2022 2023 2024
  conda run -n osworld python construction_v2/scripts/match_candidate_papers.py --topic "trustworthy AI" --embedding-provider openai --embed-model text-embedding-3-large
  conda run -n osworld python construction_v2/scripts/match_candidate_papers.py --topic "trustworthy AI" --use-reranker --rerank-threshold 0.8 --output-suffix _rerank0.8
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
from tqdm.auto import tqdm


DEFAULT_CONSTRUCTION_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TOPICS_JSON = DEFAULT_CONSTRUCTION_DIR / "topics.json"
DEFAULT_CANDIDATE_DIR = DEFAULT_CONSTRUCTION_DIR / "candidate_topics"
DEFAULT_DEDUP_DIR = DEFAULT_CONSTRUCTION_DIR / "candidate_dedup"
DEFAULT_PAPERS_DIR = DEFAULT_CONSTRUCTION_DIR / "papers"
DEFAULT_OUTPUT_DIR = DEFAULT_CONSTRUCTION_DIR / "candidate_matching"
DEFAULT_YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
OFFICIAL_OPENAI_BASE_URL = "https://api.openai.com/v1"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dedupe_candidates import (  # noqa: E402
    load_candidate_index,
    load_topics,
    normalize_candidate_topic,
    read_candidate_json,
    slugify,
    topic_candidate_path,
    write_topic_candidate_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed candidate topics and papers, then find supporting papers."
    )
    parser.add_argument("--topics-json", type=Path, default=DEFAULT_TOPICS_JSON)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--dedup-dir", type=Path, default=DEFAULT_DEDUP_DIR)
    parser.add_argument("--papers-dir", type=Path, default=DEFAULT_PAPERS_DIR)
    parser.add_argument("--papers-suffix", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--topic", help="Process only one topic from topics.json.")
    parser.add_argument("--years", type=int, nargs="+", default=DEFAULT_YEARS)
    parser.add_argument("--similarity-threshold", type=float, default=0.7)
    parser.add_argument("--embedding-provider", choices=["local", "openai"], default="local")
    parser.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--embed-batch-size", type=int, default=128)
    parser.add_argument("--openai-embedding-retries", type=int, default=4)
    parser.add_argument("--openai-embedding-timeout", type=float, default=60)
    parser.add_argument("--openai-embedding-base-url", default=OFFICIAL_OPENAI_BASE_URL)
    parser.add_argument("--openai-embedding-dimensions", type=int)
    parser.add_argument("--candidate-text", choices=["topic", "topic-evidence"], default="topic")
    parser.add_argument(
        "--candidate-source",
        choices=["index", "cluster"],
        default="index",
        help="Use exact-deduped candidate_index files or clustered candidate_clusters files.",
    )
    parser.add_argument(
        "--candidate-topic-type",
        choices=["all", "problem-space", "solution-space"],
        default="all",
    )
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--rerank-threshold", type=float, default=0.8)
    parser.add_argument("--reranker-batch-size", type=int, default=32)
    parser.add_argument("--candidate-input-suffix", default="")
    parser.add_argument("--dedup-suffix", default="")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def papers_path(papers_dir: Path, topic: str, year: int, suffix: str = "") -> Path:
    topic_slug = slugify(topic)
    return papers_dir / topic_slug / f"papers_{topic_slug}_{year}{suffix}.parquet"


def load_papers(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "paperId" not in df.columns:
        return pd.DataFrame()
    df["paperId"] = df["paperId"].astype(str)
    if "abstract" in df.columns:
        df = df[df["abstract"].fillna("").astype(str).str.strip().ne("")].copy()
    else:
        return pd.DataFrame()
    df["paper_text"] = df["title"].fillna("").astype(str) + ". " + df["abstract"].fillna("").astype(str)
    return df.reset_index(drop=True)


def cache_path(output_dir: Path, topic_slug: str, name: str, provider: str, model: str) -> Path:
    return (
        output_dir
        / topic_slug
        / "embedding_cache"
        / f"{name}_{slugify(provider)}_{slugify(model)}.parquet"
    )


def dataframe_to_json_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.to_dict(orient="records")
    for record in records:
        for key, value in list(record.items()):
            if not isinstance(value, (list, dict)) and pd.isna(value):
                record[key] = None
    return records


def write_json_records(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(dataframe_to_json_records(df), fp, ensure_ascii=False, indent=2)
        fp.write("\n")


def clusters_to_match_candidates(clusters: pd.DataFrame, candidate_text_mode: str) -> pd.DataFrame:
    if clusters.empty:
        return clusters
    rows = []
    for record in clusters.to_dict(orient="records"):
        candidate_topic = str(record.get("canonical_topic") or "").strip()
        summary = str(record.get("summary") or "").strip()
        if candidate_text_mode == "topic-evidence" and summary and summary != candidate_topic:
            match_text = f"{candidate_topic}. {summary}"
        else:
            match_text = candidate_topic
        row = {
            "candidate_id": str(record.get("cluster_id") or ""),
            "target_topic": record.get("target_topic"),
            "target_topic_slug": record.get("target_topic_slug"),
            "candidate_topic": candidate_topic,
            "candidate_topic_norm": normalize_candidate_topic(candidate_topic),
            "candidate_topic_type": record.get("candidate_topic_type"),
            "match_text": match_text,
            "member_count": record.get("member_count"),
            "source_paper_count": record.get("source_paper_count"),
            "member_candidate_ids": record.get("member_candidate_ids"),
            "member_topics": record.get("member_topics"),
            "source_paper_ids": record.get("source_paper_ids"),
            "source_years": record.get("source_years"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def matched_output_path(output_dir: Path, topic: str, output_suffix: str, extension: str) -> Path:
    topic_slug = slugify(topic)
    return output_dir / topic_slug / f"matched_papers_{topic_slug}{output_suffix}.{extension}"


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
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return [value]
    return []


def match_row(
    *,
    cand: pd.Series,
    paper: pd.Series,
    year: int,
    cosine: float | None,
    rerank_score: float | None,
    match_method: str,
    candidate_source: str,
) -> dict[str, Any]:
    return {
        "candidate_id": cand["candidate_id"],
        "target_topic": cand["target_topic"],
        "target_topic_slug": cand["target_topic_slug"],
        "candidate_topic": cand["candidate_topic"],
        "candidate_topic_norm": cand["candidate_topic_norm"],
        "candidate_topic_type": cand["candidate_topic_type"],
        "year": year,
        "matched_paper_id": paper["paperId"],
        "matched_paper_title": paper.get("title", ""),
        "cosine": cosine,
        "rerank_score": rerank_score,
        "match_method": match_method,
        "candidate_source": candidate_source,
        "cluster_member_count": cand.get("member_count"),
        "cluster_source_paper_count": cand.get("source_paper_count"),
        "cluster_member_candidate_ids": cand.get("member_candidate_ids"),
        "cluster_member_topics": cand.get("member_topics"),
    }


class LocalEmbeddingModel:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, batch_size=batch_size, show_progress_bar=False),
            dtype=np.float32,
        )


class OpenAIEmbeddingModel:
    def __init__(
        self,
        model_name: str,
        base_url: str,
        timeout: float,
        retries: int,
        dimensions: int | None,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY to use --embedding-provider openai.")
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.model_name = model_name
        self.retries = retries
        self.dimensions = dimensions

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            kwargs: dict[str, Any] = {"model": self.model_name, "input": batch}
            if self.dimensions:
                kwargs["dimensions"] = self.dimensions
            for attempt in range(1, self.retries + 1):
                try:
                    response = self.client.embeddings.create(**kwargs)
                    ordered = sorted(response.data, key=lambda item: item.index)
                    vectors.extend([item.embedding for item in ordered])
                    break
                except Exception as exc:
                    if attempt >= self.retries:
                        raise
                    sleep_seconds = min(2 ** (attempt - 1), 20)
                    print(
                        f"OpenAI embedding failed on attempt {attempt}/{self.retries}: "
                        f"{type(exc).__name__}: {str(exc)[:200]}; retrying in {sleep_seconds}s"
                    )
                    time.sleep(sleep_seconds)
        return np.asarray(vectors, dtype=np.float32)


def embed_texts_cached(
    model: Any,
    texts: list[str],
    path: Path,
    batch_size: int,
    label: str,
) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        cache_df = pd.read_parquet(path)
    else:
        cache_df = pd.DataFrame(columns=["text", "embedding"])

    text_to_idx = {text: idx for idx, text in enumerate(cache_df["text"].tolist())}
    needed = [text for text in texts if text not in text_to_idx]
    if needed:
        new_rows = []
        for start in tqdm(range(0, len(needed), batch_size), desc=f"Embedding {label}", unit="batch"):
            batch = needed[start:start + batch_size]
            embeddings = model.encode(batch, batch_size=batch_size)
            for text, emb in zip(batch, embeddings):
                new_rows.append({"text": text, "embedding": np.asarray(emb, dtype=np.float32).tolist()})
        cache_df = pd.concat([cache_df, pd.DataFrame(new_rows)], ignore_index=True)
        cache_df.drop_duplicates(subset="text", keep="last", inplace=True)
        cache_df.reset_index(drop=True, inplace=True)
        cache_df.to_parquet(path, index=False)
        text_to_idx = {text: idx for idx, text in enumerate(cache_df["text"].tolist())}
    else:
        print(f"{label}: all embeddings cached.")

    embeddings = cache_df["embedding"].values
    return np.stack([np.asarray(embeddings[text_to_idx[text]], dtype=np.float32) for text in texts])


def normalize_matrix(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def load_reranker(model_name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name, max_length=512)


def load_embedding_model(model_name: str):
    return LocalEmbeddingModel(model_name)


def load_openai_embedding_model(args: argparse.Namespace) -> OpenAIEmbeddingModel:
    return OpenAIEmbeddingModel(
        model_name=args.embed_model,
        base_url=args.openai_embedding_base_url,
        timeout=args.openai_embedding_timeout,
        retries=args.openai_embedding_retries,
        dimensions=args.openai_embedding_dimensions,
    )


def load_candidates_for_topic(
    *,
    topic: str,
    args: argparse.Namespace,
    topics_for_dedupe: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    stem = "candidate_clusters" if args.candidate_source == "cluster" else "candidate_index"
    candidate_index_path = topic_candidate_path(
        args.dedup_dir,
        topic,
        args.dedup_suffix,
        candidate_topic_type=args.candidate_topic_type,
        stem=stem,
    )
    if candidate_index_path.exists() and not args.force:
        candidates = read_candidate_json(candidate_index_path)
        if args.candidate_source == "cluster":
            candidates = clusters_to_match_candidates(candidates, args.candidate_text)
        print(f"Loaded candidate {args.candidate_source}: {candidate_index_path}")
        return candidates

    if args.candidate_source == "cluster":
        raise RuntimeError(
            f"Clustered candidates not found: {candidate_index_path}. "
            "Run dedupe_candidates.py with --use-clustering first, or use --candidate-source index."
        )

    candidates = load_candidate_index(
        args.candidate_dir,
        topics_for_dedupe,
        args.years,
        args.candidate_text,
        args.candidate_input_suffix,
    )
    if candidates.empty:
        raise RuntimeError(f"No candidate topics found for {topic}. Run extract_candidate.py first.")
    write_topic_candidate_outputs(
        candidates,
        output_dir=args.dedup_dir,
        output_suffix=args.dedup_suffix,
        write_parquet=False,
    )
    if args.candidate_topic_type != "all":
        candidates = candidates[candidates["candidate_topic_type"] == args.candidate_topic_type].copy()
    print(f"Saved candidate index files under: {args.dedup_dir / slugify(topic)}")
    return candidates.reset_index(drop=True)


def rerank_pairs(reranker: Any, pairs: list[list[str]], batch_size: int) -> np.ndarray:
    scores = np.asarray(
        reranker.predict(pairs, batch_size=batch_size, show_progress_bar=True),
        dtype=np.float32,
    )
    if scores.size and (scores.min() < 0.0 or scores.max() > 1.0):
        scores = 1.0 / (1.0 + np.exp(-scores))
    return scores


def rerank_cache_file(output_dir: Path, topic_slug: str, year: int, suffix: str) -> Path:
    return output_dir / topic_slug / "rerank_cache" / f"rerank_{topic_slug}_{year}{suffix}.parquet"


def load_rerank_cache(path: Path) -> dict[tuple[str, str], float]:
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    return {
        (str(cid), str(pid)): float(score)
        for cid, pid, score in zip(df["candidate_id"], df["paper_id"], df["rerank_score"])
    }


def save_rerank_cache(path: Path, scores: dict[tuple[str, str], float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"candidate_id": cid, "paper_id": pid, "rerank_score": score}
        for (cid, pid), score in scores.items()
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def match_topic_year(
    *,
    topic: str,
    year: int,
    candidates: pd.DataFrame,
    papers_dir: Path,
    output_dir: Path,
    embed_model: Any,
    args: argparse.Namespace,
    reranker: Any | None,
) -> pd.DataFrame:
    topic_slug = slugify(topic)
    topic_candidates = candidates[candidates["target_topic"] == topic].reset_index(drop=True)
    papers = load_papers(papers_path(papers_dir, topic, year, args.papers_suffix))
    if topic_candidates.empty or papers.empty:
        print(f"{topic_slug}/{year}: candidates={len(topic_candidates)}, papers={len(papers)}; skipping.")
        return pd.DataFrame()

    paper_by_id = {str(row["paperId"]): row for _, row in papers.iterrows()}
    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    source_pair_count = 0
    for _, cand in topic_candidates.iterrows():
        for paper_id in list_cell(cand.get("source_paper_ids")):
            paper_id = str(paper_id)
            paper = paper_by_id.get(paper_id)
            if paper is None:
                continue
            key = (str(cand["candidate_id"]), paper_id)
            if key in rows_by_key:
                continue
            rows_by_key[key] = match_row(
                cand=cand,
                paper=paper,
                year=year,
                cosine=None,
                rerank_score=None,
                match_method="extraction_source",
                candidate_source=args.candidate_source,
            )
            source_pair_count += 1

    candidate_texts = topic_candidates["match_text"].astype(str).tolist()
    paper_texts = papers["paper_text"].astype(str).tolist()
    cand_emb = embed_texts_cached(
        embed_model,
        candidate_texts,
        cache_path(
            output_dir,
            topic_slug,
            f"candidates_{topic_slug}_{args.candidate_topic_type}",
            args.embedding_provider,
            args.embed_model,
        ),
        args.embed_batch_size,
        f"{topic_slug}/{args.candidate_topic_type} candidates",
    )
    paper_emb = embed_texts_cached(
        embed_model,
        paper_texts,
        cache_path(output_dir, topic_slug, f"papers_{topic_slug}_{year}", args.embedding_provider, args.embed_model),
        args.embed_batch_size,
        f"{topic_slug}/{year} papers",
    )
    cand_norm = normalize_matrix(cand_emb)
    paper_norm = normalize_matrix(paper_emb)
    sims = cand_norm @ paper_norm.T

    cand_idx, paper_idx = np.where(sims >= args.similarity_threshold)
    print(
        f"{topic_slug}/{year}: {len(cand_idx):,} cosine candidates, "
        f"{source_pair_count:,} extraction-source pairs."
    )
    if len(cand_idx) == 0:
        return pd.DataFrame(list(rows_by_key.values()))

    rerank_scores: dict[tuple[str, str], float] = {}
    if args.use_reranker and reranker is not None:
        cache_file = rerank_cache_file(output_dir, topic_slug, year, args.output_suffix)
        rerank_scores = load_rerank_cache(cache_file)
        pairs_to_score: list[list[str]] = []
        keys_to_score: list[tuple[str, str]] = []
        for ci, pi in zip(cand_idx, paper_idx):
            cand = topic_candidates.iloc[ci]
            paper = papers.iloc[pi]
            key = (str(cand["candidate_id"]), str(paper["paperId"]))
            if key in rerank_scores:
                continue
            pairs_to_score.append([str(cand["candidate_topic"]), str(paper["paper_text"])])
            keys_to_score.append(key)
        if pairs_to_score:
            print(f"{topic_slug}/{year}: reranking {len(pairs_to_score):,} new pairs.")
            scores = rerank_pairs(reranker, pairs_to_score, args.reranker_batch_size)
            for key, score in zip(keys_to_score, scores):
                rerank_scores[key] = float(score)
            save_rerank_cache(cache_file, rerank_scores)

    embedding_pair_count = 0
    for ci, pi in zip(cand_idx, paper_idx):
        cand = topic_candidates.iloc[ci]
        paper = papers.iloc[pi]
        cosine = float(sims[ci, pi])
        key = (str(cand["candidate_id"]), str(paper["paperId"]))
        rerank_score = rerank_scores.get(key)
        if args.use_reranker:
            is_match = rerank_score is not None and rerank_score >= args.rerank_threshold
        else:
            is_match = True
        if not is_match:
            continue
        method = "embedding+reranker" if args.use_reranker else "embedding"
        key = (str(cand["candidate_id"]), str(paper["paperId"]))
        if key in rows_by_key:
            rows_by_key[key]["cosine"] = cosine
            rows_by_key[key]["rerank_score"] = rerank_score
            if method not in str(rows_by_key[key]["match_method"]).split("+"):
                rows_by_key[key]["match_method"] = f"{rows_by_key[key]['match_method']}+{method}"
            continue
        rows_by_key[key] = match_row(
            cand=cand,
            paper=paper,
            year=year,
            cosine=cosine,
            rerank_score=rerank_score,
            match_method=method,
            candidate_source=args.candidate_source,
        )
        embedding_pair_count += 1
    print(f"{topic_slug}/{year}: added {embedding_pair_count:,} embedding-only pairs after dedupe.")
    return pd.DataFrame(list(rows_by_key.values()))


def main() -> None:
    load_dotenv(DEFAULT_CONSTRUCTION_DIR / ".env")
    load_dotenv()
    args = parse_args()
    topics = load_topics(args.topics_json, args.topic)
    args.dedup_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Embedding provider: {args.embedding_provider}")
    print(f"Candidate source: {args.candidate_source}")
    print(f"Loading embedding model: {args.embed_model}")
    if args.embedding_provider == "openai":
        embed_model = load_openai_embedding_model(args)
    else:
        embed_model = load_embedding_model(args.embed_model)
    reranker = None
    if args.use_reranker:
        print(f"Loading reranker: {args.reranker_model}")
        reranker = load_reranker(args.reranker_model)

    for topic in topics:
        topic_matches = []
        topic_candidates = load_candidates_for_topic(
            topic=topic,
            args=args,
            topics_for_dedupe={topic: topics[topic]},
        )
        print(
            f"{slugify(topic)}: loaded {len(topic_candidates):,} "
            f"{args.candidate_topic_type} deduplicated candidates."
        )
        for year in args.years:
            result = match_topic_year(
                topic=topic,
                year=year,
                candidates=topic_candidates,
                papers_dir=args.papers_dir,
                output_dir=args.output_dir,
                embed_model=embed_model,
                args=args,
                reranker=reranker,
            )
            if not result.empty:
                topic_matches.append(result)

        output_path = matched_output_path(args.output_dir, topic, args.output_suffix, "parquet")
        json_output_path = matched_output_path(args.output_dir, topic, args.output_suffix, "json")
        if topic_matches:
            matches = pd.concat(topic_matches, ignore_index=True)
        else:
            matches = pd.DataFrame()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        matches.to_parquet(output_path, index=False)
        write_json_records(matches, json_output_path)
        print(f"Saved {len(matches):,} matched pairs: {output_path}")
        print(f"Saved {len(matches):,} matched pairs JSON: {json_output_path}")


if __name__ == "__main__":
    main()
