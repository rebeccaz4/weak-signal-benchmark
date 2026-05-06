#!/usr/bin/env python
# coding: utf-8

# # 1. The Top-One Topics Workflow

# In[ ]:


# Cell 1 top-one filter – Imports, environment, and core paths
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT_TOPONE = globals().get("PROJECT_ROOT")
if PROJECT_ROOT_TOPONE is None:
  PROJECT_ROOT_TOPONE = next(
      (p for p in Path.cwd().resolve().parents if (p / "README.md").exists()),
      Path.cwd().resolve(),
  )

DATA_ROOT_TOPONE = PROJECT_ROOT_TOPONE / "data" / "backward_construction"
TOPICS_RAW_DIR_TOPONE = DATA_ROOT_TOPONE / "topics_raw"
TOPICS_PROCESSED_DIR_TOPONE = DATA_ROOT_TOPONE / "topics_processed"
METRICS_DIR_TOPONE = DATA_ROOT_TOPONE / "metrics"
TOPONE_TOPIC_EMBED_CACHE_DIR = TOPICS_PROCESSED_DIR_TOPONE / "topic_embedding_cache_top_1"
TOPONE_TOPIC_EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

for directory in (TOPICS_RAW_DIR_TOPONE, TOPICS_PROCESSED_DIR_TOPONE, METRICS_DIR_TOPONE):
  directory.mkdir(parents=True, exist_ok=True)

CANONICAL_YEAR_TOPONE = 2024
YEARS_AVAILABLE_TOPONE = list(range(CANONICAL_YEAR_TOPONE, 2019, -1))

print(f"Project root (top-one):  {PROJECT_ROOT_TOPONE}")
print(f"Raw topics dir (top-one):    {TOPICS_RAW_DIR_TOPONE}")
print(f"Processed dir (top-one): {TOPICS_PROCESSED_DIR_TOPONE}")
print(f"Metrics dir (top-one):   {METRICS_DIR_TOPONE}")
print(f"Canonical year (top-one): {CANONICAL_YEAR_TOPONE}")
print(f"Years available for alignment (top-one): {YEARS_AVAILABLE_TOPONE}")


# In[ ]:


# Cell 2 top-one filter – Load topic tables and summarize coverage
from typing import Any
from tqdm.auto import tqdm

TOPONE_SUFFIX = "top_1"
TOPONE_TAG = "topone_filtered_top_1"
TOPONE_TOPIC_COL = "topic_name"
BROADER_TOPIC_COL = "broader_topic"

def topone_raw_jsonl_path(year: int) -> Path:
  return TOPICS_RAW_DIR_TOPONE / f"topics_{year}_{TOPONE_TAG}.jsonl"

def topone_processed_parquet_path(year: int) -> Path:
  return TOPICS_PROCESSED_DIR_TOPONE / f"topics_{year}_{TOPONE_TAG}_paper_topics.parquet"

def _safe_source_query(record: Dict[str, Any]) -> str:
  prompt_meta = record.get("prompt_meta") or {}
  return (
      record.get("source_query")
      or prompt_meta.get("source_query")
      or prompt_meta.get("query")
      or ""
  )

def flatten_topone_raw(year: int) -> pd.DataFrame:
  path = topone_raw_jsonl_path(year)
  if not path.exists():
      raise FileNotFoundError(f"Top-one raw topics missing for {year}: {path}")

  rows: List[Dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as fp:
      for line in fp:
          line = line.strip()
          if not line:
              continue
          record = json.loads(line)
          paper_id = str(record.get("paperId") or "")
          topics = record.get("topics") or []
          for topic in topics:
              topic_name = str(topic.get("topic_name") or "").strip()
              if not topic_name:
                  continue
              rows.append(
                  {
                      "paperId": paper_id,
                      TOPONE_TOPIC_COL: topic_name,
                      BROADER_TOPIC_COL: str(topic.get("broader_topic") or "").strip(),
                      "anchor": str(topic.get("anchor") or "").strip(),
                      "confidence": topic.get("confidence"),
                      "justification": topic.get("justification"),
                      "year": record.get("year") or year,
                      "source_query": _safe_source_query(record),
                  }
              )
  df = pd.DataFrame(rows)
  if not df.empty:
      df["year"] = df["year"].fillna(year).astype(int)
  return df

def load_topone_topics(year: int) -> pd.DataFrame:
  processed_path = topone_processed_parquet_path(year)
  if processed_path.exists():
      df = pd.read_parquet(processed_path)
  else:
      df = flatten_topone_raw(year)
      if not df.empty:
          processed_path.parent.mkdir(parents=True, exist_ok=True)
          df.to_parquet(processed_path, index=False)

  if df.empty:
      return df

  df["paperId"] = df["paperId"].astype(str)
  df[TOPONE_TOPIC_COL] = df[TOPONE_TOPIC_COL].fillna("").astype(str).str.strip()
  df[BROADER_TOPIC_COL] = df[BROADER_TOPIC_COL].replace("", pd.NA)
  df = df[df[TOPONE_TOPIC_COL] != ""].copy()
  return df.reset_index(drop=True)

TOPONE_TOPICS: Dict[int, pd.DataFrame] = {}
topone_coverage_rows: List[Dict[str, Any]] = []

for year in YEARS_AVAILABLE_TOPONE:
  try:
      df = load_topone_topics(year)
  except FileNotFoundError as exc:
      print(f"[warn] {exc}")
      continue

  TOPONE_TOPICS[year] = df
  topone_coverage_rows.append(
      {
          "year": year,
          "rows": len(df),
          "papers_with_topics": df["paperId"].nunique(),
          "unique_topics": df[TOPONE_TOPIC_COL].nunique(),
      }
  )

topone_coverage_df = (
  pd.DataFrame(topone_coverage_rows)
  .set_index("year")
  .sort_index(ascending=False)
)
display(topone_coverage_df)


# In[ ]:


# Cell 3 top-one filter – Canonical-year topic expansion
CANONICAL_YEAR_TOPONE = 2024
TOPONE_TOPIC_COL = "topic_name"

if CANONICAL_YEAR_TOPONE not in TOPONE_TOPICS:
  raise RuntimeError(
      f"No top-one topics loaded for canonical year {CANONICAL_YEAR_TOPONE}. "
      "Re-run Cell 2 or confirm the processed parquet exists."
  )

TOPONE_CANONICAL_TOPICS = (
  TOPONE_TOPICS[CANONICAL_YEAR_TOPONE]
  .copy()
  .sort_values(["paperId", TOPONE_TOPIC_COL])
  .reset_index(drop=True)
)

if TOPONE_CANONICAL_TOPICS.empty:
  raise RuntimeError(
      f"Canonical topic table for {CANONICAL_YEAR_TOPONE} is empty."
  )

print(f"Canonical pool ({CANONICAL_YEAR_TOPONE}) topics: {len(TOPONE_CANONICAL_TOPICS):,} rows")
print(f"Papers represented: {TOPONE_CANONICAL_TOPICS['paperId'].nunique():,}")
print(f"Unique topic strings: {TOPONE_CANONICAL_TOPICS[TOPONE_TOPIC_COL].nunique():,}")


# In[ ]:


# Cell 4 top-one filter – Embed canonical top-one topics (with caching)
from openai import OpenAI
import hashlib
import time
from tqdm.auto import tqdm

EMBED_MODEL_TOPONE = os.getenv("BWD_TOPONE_EMBED_MODEL", os.getenv("BWD_EMBED_MODEL", "text-embedding-3-large"))
EMBED_BATCH_SIZE_TOPONE = int(os.getenv("BWD_TOPONE_EMBED_BATCH_SIZE", os.getenv("BWD_EMBED_BATCH_SIZE", "128")))
TOPONE_CANONICAL_EMBED_PATH = TOPICS_PROCESSED_DIR_TOPONE / f"canonical_{CANONICAL_YEAR_TOPONE}_embeddings_top_1.parquet"

client_topone_embed = OpenAI()

def normalize_topic_text(text: str) -> str:
  return " ".join(text.lower().split())

def fingerprint_topic(text: str) -> str:
  return hashlib.md5(text.encode("utf-8")).hexdigest()

def load_embedding_cache_topone(path: Path) -> pd.DataFrame:
  if not path.exists():
      return pd.DataFrame(columns=["topic_key", "topic_text", "fingerprint", "embedding", "created_at", "model"])
  df = pd.read_parquet(path)
  if not df.empty:
      df["embedding"] = df["embedding"].apply(lambda x: list(x) if not isinstance(x, list) else x)
  return df

def save_embedding_cache_topone(df: pd.DataFrame, path: Path) -> None:
  df.to_parquet(path, index=False)

def embed_text_batch_topone(texts: List[str]) -> List[List[float]]:
  response = client_topone_embed.embeddings.create(model=EMBED_MODEL_TOPONE, input=texts)
  return [item.embedding for item in response.data]

canonical_universe_topone = (
  TOPONE_CANONICAL_TOPICS.assign(topic_key=TOPONE_CANONICAL_TOPICS[TOPONE_TOPIC_COL].fillna("").map(normalize_topic_text))
  .loc[lambda df: df["topic_key"] != ""]
  .drop_duplicates(subset="topic_key", keep="first")
  [["topic_key", TOPONE_TOPIC_COL]]
  .rename(columns={TOPONE_TOPIC_COL: "topic_text"})
  .reset_index(drop=True)
)
canonical_universe_topone["fingerprint"] = canonical_universe_topone["topic_text"].map(
  lambda txt: fingerprint_topic(txt.strip().lower())
)

embedding_cache_topone = load_embedding_cache_topone(TOPONE_CANONICAL_EMBED_PATH)
cached_keys_topone = set(embedding_cache_topone["topic_key"]) if not embedding_cache_topone.empty else set()

to_embed_topone = canonical_universe_topone[~canonical_universe_topone["topic_key"].isin(cached_keys_topone)].reset_index(drop=True)

print(f"Canonical top-one topics needing embeddings: {len(to_embed_topone):,}")
if not to_embed_topone.empty:
  new_rows: List[Dict[str, Any]] = []
  iterator = range(0, len(to_embed_topone), EMBED_BATCH_SIZE_TOPONE)
  for start in tqdm(iterator, desc="Embedding top-one canonical topics", unit="batch"):
      batch = to_embed_topone.iloc[start : start + EMBED_BATCH_SIZE_TOPONE]
      texts = batch["topic_text"].tolist()
      embeddings = embed_text_batch_topone(texts)

      now = time.time()
      for (topic_key, topic_text, fingerprint), emb in zip(
          batch[["topic_key", "topic_text", "fingerprint"]].itertuples(index=False), embeddings
      ):
          new_rows.append(
              {
                  "topic_key": topic_key,
                  "topic_text": topic_text,
                  "fingerprint": fingerprint,
                  "embedding": emb,
                  "created_at": now,
                  "model": EMBED_MODEL_TOPONE,
              }
          )

  if new_rows:
      new_df = pd.DataFrame(new_rows)
      embedding_cache_topone = pd.concat([embedding_cache_topone, new_df], ignore_index=True)
      save_embedding_cache_topone(embedding_cache_topone, TOPONE_CANONICAL_EMBED_PATH)
      print(f"Cached {len(new_rows):,} new embeddings to {TOPONE_CANONICAL_EMBED_PATH}")
else:
  print("All top-one canonical topics already have cached embeddings.")

TOPONE_CANONICAL_EMBEDDINGS = (
  embedding_cache_topone.merge(canonical_universe_topone, on=["topic_key", "topic_text", "fingerprint"], how="inner")
  .drop_duplicates(subset="topic_key")
  .reset_index(drop=True)
)
print(f"Total top-one canonical embeddings cached: {len(TOPONE_CANONICAL_EMBEDDINGS):,}")


# In[ ]:


# Variables re-run cell after running Cell 5 the first time. This is to avoid running Cell 5 again if you do not want to.
TOPONE_CANONICAL_CLUSTER_PATH = TOPICS_PROCESSED_DIR_TOPONE / f"canonical_{CANONICAL_YEAR_TOPONE}_clusters_top_1.parquet"
TOPONE_CANONICAL_SIM_THRESHOLD = float(os.getenv("BWD_TOPONE_CANONICAL_SIM_THRESHOLD", os.getenv("BWD_CANONICAL_SIM_THRESHOLD", "0.8")))
TOPONE_CANONICAL_KNN = int(os.getenv("BWD_TOPONE_CANONICAL_KNN", os.getenv("BWD_CANONICAL_KNN", "50")))
TOPONE_CANONICAL_KNN_BATCH = int(os.getenv("BWD_TOPONE_CANONICAL_KNN_BATCH", os.getenv("BWD_CANONICAL_KNN_BATCH", "2000")))


# In[ ]:


# Cell 5 top-one filter – Cluster canonical topics into stable IDs (KNN + union-find)
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

TOPONE_CANONICAL_CLUSTER_PATH = TOPICS_PROCESSED_DIR_TOPONE / f"canonical_{CANONICAL_YEAR_TOPONE}_clusters_top_1.parquet"
TOPONE_CANONICAL_SIM_THRESHOLD = float(os.getenv("BWD_TOPONE_CANONICAL_SIM_THRESHOLD", os.getenv("BWD_CANONICAL_SIM_THRESHOLD", "0.8")))
TOPONE_CANONICAL_KNN = int(os.getenv("BWD_TOPONE_CANONICAL_KNN", os.getenv("BWD_CANONICAL_KNN", "50")))
TOPONE_CANONICAL_KNN_BATCH = int(os.getenv("BWD_TOPONE_CANONICAL_KNN_BATCH", os.getenv("BWD_CANONICAL_KNN_BATCH", "2000")))

print(f"Top-one clustering cosine threshold: {TOPONE_CANONICAL_SIM_THRESHOLD}")
print(f"KNN neighbors per topic: {TOPONE_CANONICAL_KNN}")
print(f"KNN query batch size: {TOPONE_CANONICAL_KNN_BATCH}")

emb_matrix = np.vstack(
  TOPONE_CANONICAL_EMBEDDINGS["embedding"].apply(lambda vec: np.asarray(vec, dtype=np.float32))
)

norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
norms[norms == 0] = 1.0
normalized = emb_matrix / norms
n_topics = normalized.shape[0]
print(f"Normalized embedding matrix shape: {normalized.shape}")

nbrs = NearestNeighbors(
  metric="cosine",
  algorithm="brute",
  n_neighbors=min(TOPONE_CANONICAL_KNN + 1, n_topics),
  n_jobs=-1,
)
nbrs.fit(normalized)

class UnionFind:
  def __init__(self, n: int) -> None:
      self.parent = np.arange(n, dtype=np.int32)
      self.rank = np.zeros(n, dtype=np.int8)

  def find(self, x: int) -> int:
      parent = self.parent
      while parent[x] != x:
          parent[x] = parent[parent[x]]
          x = parent[x]
      return x

  def union(self, x: int, y: int) -> None:
      root_x, root_y = self.find(x), self.find(y)
      if root_x == root_y:
          return
      if self.rank[root_x] < self.rank[root_y]:
          self.parent[root_x] = root_y
      elif self.rank[root_x] > self.rank[root_y]:
          self.parent[root_y] = root_x
      else:
          self.parent[root_y] = root_x
          self.rank[root_x] += 1

uf = UnionFind(n_topics)

for start in tqdm(range(0, n_topics, TOPONE_CANONICAL_KNN_BATCH), desc="Linking top-one topics", unit="chunk"):
  end = min(start + TOPONE_CANONICAL_KNN_BATCH, n_topics)
  distances, indices = nbrs.kneighbors(normalized[start:end], return_distance=True)
  for local_idx, (row_dist, row_idx) in enumerate(zip(distances, indices)):
      src = start + local_idx
      for dist_val, tgt in zip(row_dist[1:], row_idx[1:]):
          similarity = 1.0 - dist_val
          if similarity >= TOPONE_CANONICAL_SIM_THRESHOLD:
              uf.union(src, tgt)

roots = np.array([uf.find(i) for i in range(n_topics)], dtype=np.int32)
unique_roots, cluster_ids = np.unique(roots, return_inverse=True)
print(f"Identified top-one clusters: {len(unique_roots):,}")

TOPONE_CANONICAL_CLUSTERS = TOPONE_CANONICAL_EMBEDDINGS.copy()
TOPONE_CANONICAL_CLUSTERS["cluster_id"] = cluster_ids
TOPONE_CANONICAL_CLUSTERS["cluster_size"] = (
  TOPONE_CANONICAL_CLUSTERS.groupby("cluster_id")["cluster_id"].transform("count")
)

cluster_sizes = (
  TOPONE_CANONICAL_CLUSTERS["cluster_size"]
  .value_counts()
  .sort_index()
  .reset_index(name="num_clusters")
  .rename(columns={"index": "cluster_size"})
)
print("\nCluster size distribution (cluster_size → #clusters):")
display(cluster_sizes.head(20))

TOPONE_CANONICAL_CLUSTERS.to_parquet(TOPONE_CANONICAL_CLUSTER_PATH, index=False)
print(f"Saved top-one canonical clusters to {TOPONE_CANONICAL_CLUSTER_PATH}")


# In[ ]:


# Cell 6 top-one filter – Summarize multi-topic clusters with gpt-5-mini (resumable + progress bar)
from openai import OpenAI
from tqdm.auto import tqdm
import time
import json
from datetime import datetime

TOPONE_CLUSTER_LABELS_PATH = TOPICS_PROCESSED_DIR_TOPONE / f"canonical_{CANONICAL_YEAR_TOPONE}_cluster_labels_top_1.parquet"
TOPONE_FAILED_LABELS_PATH = TOPICS_PROCESSED_DIR_TOPONE / f"canonical_{CANONICAL_YEAR_TOPONE}_cluster_label_failures_top_1.jsonl"
LLM_MODEL_TOPONE = os.getenv("BWD_TOPONE_CANONICAL_LABEL_MODEL", os.getenv("BWD_CANONICAL_LABEL_MODEL", "gpt-5-mini"))
LLM_SLEEP_SECONDS_TOPONE = float(os.getenv("BWD_TOPONE_CANONICAL_LABEL_SLEEP", os.getenv("BWD_CANONICAL_LABEL_SLEEP", "0.5")))
LLM_FLUSH_EVERY_TOPONE = int(os.getenv("BWD_TOPONE_CANONICAL_LABEL_FLUSH", os.getenv("BWD_CANONICAL_LABEL_FLUSH", "25")))

client_topone_label = OpenAI()

TOPONE_CANONICAL_CLUSTERS = TOPONE_CANONICAL_CLUSTERS if "TOPONE_CANONICAL_CLUSTERS" in globals() else pd.read_parquet(TOPONE_CANONICAL_CLUSTER_PATH)
if "TOPONE_CANONICAL_TOPICS" not in globals():
  TOPONE_CANONICAL_TOPICS = TOPONE_TOPICS[CANONICAL_YEAR_TOPONE].copy()

cluster_with_topics_topone = TOPONE_CANONICAL_CLUSTERS.merge(
  TOPONE_CANONICAL_TOPICS[[TOPONE_TOPIC_COL, "anchor", "paperId"]],
  left_on="topic_text",
  right_on=TOPONE_TOPIC_COL,
  how="left",
).drop(columns=[TOPONE_TOPIC_COL])

cluster_groups_topone = (
  cluster_with_topics_topone.groupby("cluster_id")
  .agg(
      cluster_size=("topic_text", "size"),
      topic_texts=("topic_text", lambda s: list(dict.fromkeys(s))),
      anchors=("anchor", lambda s: sorted({a for a in s if pd.notna(a) and a})),
      paper_ids=("paperId", lambda s: sorted({str(p) for p in s if pd.notna(p)})),
  )
  .reset_index()
)

# Only multi-topic clusters need LLM summaries; singletons are already “summarized”
multitopic_topone = cluster_groups_topone[cluster_groups_topone["cluster_size"] > 1].copy()
print(f"Multi-topic clusters requiring LLM labels: {len(multitopic_topone):,}")

if TOPONE_CLUSTER_LABELS_PATH.exists():
  TOPONE_CLUSTER_LABELS = pd.read_parquet(TOPONE_CLUSTER_LABELS_PATH)
else:
  TOPONE_CLUSTER_LABELS = pd.DataFrame(columns=["cluster_id", "cluster_size", "representative_label", "reasoning", "model", "created_at"])

processed_ids_topone = set(TOPONE_CLUSTER_LABELS["cluster_id"]) if not TOPONE_CLUSTER_LABELS.empty else set()
to_label_topone = multitopic_topone[~multitopic_topone["cluster_id"].isin(processed_ids_topone)].reset_index(drop=True)
print(f"Clusters pending summarization: {len(to_label_topone):,}")

def build_summary_messages_topone(cluster_row: pd.Series) -> list[dict]:
  topics = cluster_row["topic_texts"]
  topic_list = "\n".join(f"{idx + 1}. {topic}" for idx, topic in enumerate(topics))
  anchor_text = ", ".join(cluster_row["anchors"]) if cluster_row["anchors"] else "none supplied"

  instructions = (
      "You are an expert research scientist with deep knowledge across ML/AI domains with an IQ of 160.\n"
      "Given multiple related topic variants, craft a single canonical label for a weak-signal benchmark.\n"
      "The label must be concise (≤ 10 words), specific, and capture the common idea across all variants.\n"
      "You must not copy any variant verbatim.\n"
      "Return a JSON object with:\n"
      "- representative_label: the merged topic title.\n"
      "- reasoning: one short sentence explaining why this label captures the variants.\n"
  )

  return [
      {"role": "system", "content": "You create precise, human-readable names for clusters of related ML/AI topics."},
      {
          "role": "user",
          "content": (
              f"{instructions}\n"
              f"Cluster size: {cluster_row['cluster_size']}\n"
              f"Anchors observed: {anchor_text}\n"
              f"Topic variants:\n{topic_list}"
          ),
      },
  ]

new_rows: list[dict] = []
failure_count = 0

if not to_label_topone.empty:
  for _, row in tqdm(to_label_topone.iterrows(), total=len(to_label_topone), desc="Summarizing clusters"):
      messages = build_summary_messages_topone(row)
      try:
          response = client_topone_label.chat.completions.create(
              model=LLM_MODEL_TOPONE,
              messages=messages,
              response_format={"type": "json_object"},
          )
          payload = json.loads(response.choices[0].message.content)
          new_rows.append(
              {
                  "cluster_id": row["cluster_id"],
                  "cluster_size": row["cluster_size"],
                  "representative_label": payload.get("representative_label"),
                  "reasoning": payload.get("reasoning"),
                  "model": LLM_MODEL_TOPONE,
                  "created_at": datetime.utcnow().isoformat(),
              }
          )
      except Exception as exc:  # noqa: BLE001
          failure_count += 1
          with TOPONE_FAILED_LABELS_PATH.open("a", encoding="utf-8") as fh:
              fh.write(json.dumps({"cluster_id": row["cluster_id"], "error": str(exc)}) + "\n")
          continue

      if new_rows and (len(new_rows) % LLM_FLUSH_EVERY_TOPONE == 0):
          append_df = pd.DataFrame(new_rows)
          TOPONE_CLUSTER_LABELS = pd.concat([TOPONE_CLUSTER_LABELS, append_df], ignore_index=True)
          TOPONE_CLUSTER_LABELS.to_parquet(TOPONE_CLUSTER_LABELS_PATH, index=False)
          new_rows = []
          time.sleep(LLM_SLEEP_SECONDS_TOPONE)

  if new_rows:
      append_df = pd.DataFrame(new_rows)
      TOPONE_CLUSTER_LABELS = pd.concat([TOPONE_CLUSTER_LABELS, append_df], ignore_index=True)
      TOPONE_CLUSTER_LABELS.to_parquet(TOPONE_CLUSTER_LABELS_PATH, index=False)

print(f"Completed labeling. Failures: {failure_count}")


# In[ ]:


# Cell 7 top-one filter – Build canonical topic inventory table (clustered + labeled)
TOPONE_VOCAB_PATH = TOPICS_PROCESSED_DIR_TOPONE / f"canonical_{CANONICAL_YEAR_TOPONE}_vocabulary_top_1.parquet"
TOPONE_CLUSTER_PATH = TOPICS_PROCESSED_DIR_TOPONE / f"canonical_{CANONICAL_YEAR_TOPONE}_clusters_top_1.parquet"
TOPONE_CLUSTER_LABELS_PATH = TOPICS_PROCESSED_DIR_TOPONE / f"canonical_{CANONICAL_YEAR_TOPONE}_cluster_labels_top_1.parquet"

TOPONE_CANONICAL_CLUSTERS = TOPONE_CANONICAL_CLUSTERS if "TOPONE_CANONICAL_CLUSTERS" in globals() else pd.read_parquet(TOPONE_CLUSTER_PATH)
cluster_with_topics_topone = TOPONE_CANONICAL_CLUSTERS.merge(
  TOPONE_CANONICAL_TOPICS[[TOPONE_TOPIC_COL, "anchor", "paperId"]],
  left_on="topic_text",
  right_on=TOPONE_TOPIC_COL,
  how="left",
).drop(columns=[TOPONE_TOPIC_COL])

cluster_groups_topone = (
  cluster_with_topics_topone.groupby("cluster_id")
  .agg(
      cluster_size=("topic_text", "size"),
      topic_texts=("topic_text", lambda s: list(dict.fromkeys(s))),
      anchors=("anchor", lambda s: sorted({a for a in s if pd.notna(a) and a})),
      paper_ids=("paperId", lambda s: sorted({str(p) for p in s if pd.notna(p)})),
  )
  .reset_index()
)

if TOPONE_VOCAB_PATH.exists():
  topone_vocab_df = pd.read_parquet(TOPONE_VOCAB_PATH)
  print(f"Loaded existing top-one vocabulary ({len(topone_vocab_df):,} entries).")
else:
  if "TOPONE_CLUSTER_LABELS" not in globals():
      if TOPONE_CLUSTER_LABELS_PATH.exists():
          TOPONE_CLUSTER_LABELS = pd.read_parquet(TOPONE_CLUSTER_LABELS_PATH)
      else:
          TOPONE_CLUSTER_LABELS = pd.DataFrame(columns=["cluster_id", "cluster_size", "representative_label", "reasoning", "model", "created_at"])

  vocab_df_topone = (
      cluster_groups_topone.merge(
          TOPONE_CLUSTER_LABELS,
          on=["cluster_id", "cluster_size"],
          how="left",
      )
      .rename(columns={"topic_texts": "topic_variants"})
      .reset_index(drop=True)
  )

  singleton_mask = vocab_df_topone["cluster_size"] == 1
  if singleton_mask.any():
      original_topics = (
          TOPONE_CANONICAL_CLUSTERS.sort_values("cluster_id")
          .drop_duplicates(subset="cluster_id")
          [["cluster_id", "topic_text"]]
          .rename(columns={"topic_text": "original_topic"})
      )
      vocab_df_topone = vocab_df_topone.merge(original_topics, on="cluster_id", how="left")
      fill_count = vocab_df_topone.loc[singleton_mask, "representative_label"].isna().sum()
      if fill_count:
          vocab_df_topone.loc[
              singleton_mask & vocab_df_topone["representative_label"].isna(),
              "representative_label",
          ] = vocab_df_topone.loc[
              singleton_mask & vocab_df_topone["representative_label"].isna(),
              "original_topic",
          ]
          print(f"[info] Filled representative labels for {fill_count} singleton clusters using original topics.")
      vocab_df_topone.drop(columns=["original_topic"], inplace=True)

  missing_labels = vocab_df_topone["representative_label"].isna().sum()
  if missing_labels:
      print(f"[warn] {missing_labels} cluster(s) have no representative_label; check failures file.")

  topone_vocab_df = vocab_df_topone[
      [
          "cluster_id",
          "cluster_size",
          "representative_label",
          "reasoning",
          "anchors",
          "topic_variants",
          "paper_ids",
      ]
  ]

  topone_vocab_df.to_parquet(TOPONE_VOCAB_PATH, index=False)
  TOPONE_CANONICAL_CLUSTERS.to_parquet(TOPONE_CLUSTER_PATH, index=False)
  print(f"Saved top-one canonical clusters to {TOPONE_CLUSTER_PATH} ({len(TOPONE_CANONICAL_CLUSTERS):,} rows).")
  print(f"Saved top-one vocabulary to {TOPONE_VOCAB_PATH} ({len(topone_vocab_df):,} clusters).")

print(f"Top-one vocabulary rows: {len(topone_vocab_df):,}")
display(topone_vocab_df.head(5))


# In[ ]:


# Cell 8 top-one filter – Map subsequent-year top-one topics onto the canonical vocabulary
# Running Cell 8 once is fine. You don't have to run Cell 8 again when you perform analyses.
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tqdm.auto import tqdm

TOPONE_MAPPING_DIR = TOPICS_PROCESSED_DIR_TOPONE / "aligned_topics"
TOPONE_MAPPING_DIR.mkdir(parents=True, exist_ok=True)

MATCH_MODEL_TOPONE = os.getenv("BWD_TOPONE_MATCH_MODEL", os.getenv("BWD_MATCH_EMBED_MODEL", "text-embedding-3-large"))
MATCH_BATCH_SIZE_TOPONE = int(os.getenv("BWD_TOPONE_MATCH_BATCH_SIZE", os.getenv("BWD_MATCH_BATCH_SIZE", "512")))
MATCH_THRESHOLD_TOPONE = float(os.getenv("BWD_TOPONE_MATCH_THRESHOLD", os.getenv("BWD_MATCH_THRESHOLD", "0.70")))
MATCH_CHUNK_SIZE_TOPONE = int(os.getenv("BWD_TOPONE_MATCH_CHUNK_SIZE", os.getenv("BWD_MATCH_CHUNK_SIZE", "2000")))

TOPONE_CANONICAL_CLUSTERS = TOPONE_CANONICAL_CLUSTERS if "TOPONE_CANONICAL_CLUSTERS" in globals() else pd.read_parquet(
  TOPICS_PROCESSED_DIR_TOPONE / f"canonical_{CANONICAL_YEAR_TOPONE}_clusters_top_1.parquet"
)

if "TOPONE_CANONICAL_EMBEDDINGS" not in globals():
  TOPONE_CANONICAL_EMBEDDINGS = pd.read_parquet(
      TOPICS_PROCESSED_DIR_TOPONE / f"canonical_{CANONICAL_YEAR_TOPONE}_embeddings_top_1.parquet"
  )

topone_vocab_df = topone_vocab_df if "topone_vocab_df" in globals() else pd.read_parquet(
  TOPICS_PROCESSED_DIR_TOPONE / f"canonical_{CANONICAL_YEAR_TOPONE}_vocabulary_top_1.parquet"
)

cluster_label_lookup_topone = topone_vocab_df.set_index("cluster_id")["representative_label"].to_dict()
cluster_anchor_lookup_topone = topone_vocab_df.set_index("cluster_id")["anchors"].to_dict()

canonical_matrix_topone = np.vstack(
  TOPONE_CANONICAL_CLUSTERS["embedding"].apply(lambda vec: np.asarray(vec, dtype=np.float32))
)
print(f"Top-one canonical embedding matrix: {canonical_matrix_topone.shape}")

nn_topone = NearestNeighbors(metric="cosine", n_neighbors=5, algorithm="brute", n_jobs=-1)
nn_topone.fit(canonical_matrix_topone)

client_topone_match = OpenAI()

def topone_topic_cache_path(year: int) -> Path:
  return TOPICS_PROCESSED_DIR_TOPONE / "aligned_topics" / f"topic_embeddings_{year}_top_1.parquet"

def load_topic_embedding_cache_topone(year: int) -> pd.DataFrame:
  path = topone_topic_cache_path(year)
  if not path.exists():
      return pd.DataFrame(columns=["topic", "embedding"])
  df = pd.read_parquet(path)
  if not df.empty:
      df["embedding"] = df["embedding"].apply(lambda x: list(x) if not isinstance(x, list) else x)
  return df

def save_topic_embedding_cache_topone(year: int, df: pd.DataFrame) -> None:
  df.to_parquet(topone_topic_cache_path(year), index=False)

def embed_missing_topics_topone(topics: List[str], cache_df: pd.DataFrame) -> pd.DataFrame:
  cached_topics = set(cache_df["topic"]) if not cache_df.empty else set()
  needed = [t for t in topics if t not in cached_topics]
  if not needed:
      return cache_df

  new_rows: List[Dict[str, Any]] = []
  for start in range(0, len(needed), MATCH_BATCH_SIZE_TOPONE):
      batch = needed[start : start + MATCH_BATCH_SIZE_TOPONE]
      resp = client_topone_match.embeddings.create(model=MATCH_MODEL_TOPONE, input=batch)
      for topic, vec in zip(batch, resp.data):
          new_rows.append({"topic": topic, "embedding": vec.embedding})

  if new_rows:
      add_df = pd.DataFrame(new_rows)
      cache_df = pd.concat([cache_df, add_df], ignore_index=True)
      save_topic_embedding_cache_topone(year, cache_df)

  return cache_df

def load_topone_topic_pairs(year: int) -> pd.DataFrame:
  df = TOPONE_TOPICS.get(year)
  if df is None or df.empty:
      return pd.DataFrame(columns=["paperId", "topic", "pair_key"])
  topics_df = (
      df[["paperId", TOPONE_TOPIC_COL]]
      .rename(columns={TOPONE_TOPIC_COL: "topic"})
      .copy()
  )
  topics_df["topic"] = topics_df["topic"].astype(str).str.strip()
  topics_df = topics_df[topics_df["topic"] != ""]
  topics_df["pair_key"] = topics_df["paperId"].astype(str) + "::" + topics_df["topic"]
  return topics_df.reset_index(drop=True)

for year in YEARS_AVAILABLE_TOPONE:
  if year == CANONICAL_YEAR_TOPONE:
      print(f"Skipping canonical year {year} (already represented).")
      continue

  topic_pairs = load_topone_topic_pairs(year)
  if topic_pairs.empty:
      print(f"{year}: no top-one topics to map; skipping.")
      continue

  aligned_path = TOPONE_MAPPING_DIR / f"topic_alignment_{year}_to_{CANONICAL_YEAR_TOPONE}_top_1.parquet"
  unmapped_path = TOPONE_MAPPING_DIR / f"topic_alignment_unmapped_{year}_to_{CANONICAL_YEAR_TOPONE}_top_1.parquet"

  aligned_df = pd.read_parquet(aligned_path) if aligned_path.exists() else pd.DataFrame()
  unmapped_df = pd.read_parquet(unmapped_path) if unmapped_path.exists() else pd.DataFrame()

  aligned_keys = set(
      aligned_df.assign(pair_key=lambda df: df["paperId"].astype(str) + "::" + df["topic"])["pair_key"]
  ) if not aligned_df.empty else set()

  topic_pairs = topic_pairs[~topic_pairs["pair_key"].isin(aligned_keys)].reset_index(drop=True)
  if topic_pairs.empty:
      print(f"{year}: all top-one topics already aligned; skipping.")
      continue

  cache_df = load_topic_embedding_cache_topone(year)
  pending_aligned: List[Dict[str, Any]] = []
  pending_unmapped: List[Dict[str, Any]] = []

  progress = tqdm(
      range(0, len(topic_pairs), MATCH_CHUNK_SIZE_TOPONE),
      desc=f"Aligning top-one topics for {year}",
      unit="chunk",
  )

  for start in progress:
      chunk = topic_pairs.iloc[start : start + MATCH_CHUNK_SIZE_TOPONE]
      needed_topics = chunk["topic"].unique().tolist()
      cache_df = embed_missing_topics_topone(needed_topics, cache_df)
      cache_map = dict(zip(cache_df["topic"], cache_df["embedding"]))

      emb_matrix = np.vstack([np.asarray(cache_map[topic], dtype=np.float32) for topic in chunk["topic"]])
      distances, indices = nn_topone.kneighbors(emb_matrix, n_neighbors=5)
      similarities = 1.0 - distances

      for idx, row in enumerate(chunk.itertuples(index=False)):
          sim_row = similarities[idx]
          idx_row = indices[idx]
          mask = sim_row >= MATCH_THRESHOLD_TOPONE

          if not np.any(mask):
              best_cluster = int(TOPONE_CANONICAL_CLUSTERS.iloc[idx_row[0]]["cluster_id"])
              pending_unmapped.append(
                  {
                      "paperId": row.paperId,
                      "topic": row.topic,
                      "best_similarity": float(sim_row[0]),
                      "best_cluster_id": best_cluster,
                      "threshold": MATCH_THRESHOLD_TOPONE,
                      "pair_key": row.pair_key,
                  }
              )
              continue

          cluster_idx = idx_row[mask][0]
          best_cluster = int(TOPONE_CANONICAL_CLUSTERS.iloc[cluster_idx]["cluster_id"])
          pending_aligned.append(
              {
                  "paperId": row.paperId,
                  "topic": row.topic,
                  "cluster_id": best_cluster,
                  "similarity": float(sim_row[mask][0]),
                  "representative_label": cluster_label_lookup_topone.get(best_cluster),
                  "anchors": cluster_anchor_lookup_topone.get(best_cluster, []),
                  "pair_key": row.pair_key,
              }
          )

      if pending_aligned:
          aligned_df = pd.concat([aligned_df, pd.DataFrame(pending_aligned)], ignore_index=True)
          aligned_df.drop_duplicates(subset="pair_key", keep="last", inplace=True)
          aligned_df.drop(columns="pair_key").to_parquet(aligned_path, index=False)
          pending_aligned = []

      if pending_unmapped:
          unmapped_df = pd.concat([unmapped_df, pd.DataFrame(pending_unmapped)], ignore_index=True)
          unmapped_df.drop_duplicates(subset="pair_key", keep="last", inplace=True)
          unmapped_df.drop(columns="pair_key").to_parquet(unmapped_path, index=False)
          pending_unmapped = []

      save_topic_embedding_cache_topone(year, cache_df)

  progress.close()
  print(f"Completed mapping for {year}: aligned {len(aligned_df)}, unmapped {len(unmapped_df)}.")

print("\nTop-one alignment complete for all available years.")


# In[ ]:


# Variables re-run cell after running Cell 9 the first time. You don't have to run Cell 9 again if you don't want to.
TOPONE_CLUSTER_PATH = TOPICS_PROCESSED_DIR_TOPONE / f"canonical_{CANONICAL_YEAR_TOPONE}_clusters_top_1.parquet"
TOPONE_VOCAB_PATH = TOPICS_PROCESSED_DIR_TOPONE / f"canonical_{CANONICAL_YEAR_TOPONE}_vocabulary_top_1.parquet"
TOPONE_MAPPING_DIR = TOPICS_PROCESSED_DIR_TOPONE / "aligned_topics"
TOPONE_METRICS_PATH = METRICS_DIR_TOPONE / f"canonical_metrics_{CANONICAL_YEAR_TOPONE}_top_1.parquet"
TOPONE_ANCHOR_METRICS_PATH = METRICS_DIR_TOPONE / f"canonical_anchor_metrics_{CANONICAL_YEAR_TOPONE}_top_1.parquet"


# In[ ]:


# Cell 9 top-one filter – Compute backward frequency, decline, and impact metrics
import numpy as np

TOPONE_CLUSTER_PATH = TOPICS_PROCESSED_DIR_TOPONE / f"canonical_{CANONICAL_YEAR_TOPONE}_clusters_top_1.parquet"
TOPONE_VOCAB_PATH = TOPICS_PROCESSED_DIR_TOPONE / f"canonical_{CANONICAL_YEAR_TOPONE}_vocabulary_top_1.parquet"
TOPONE_MAPPING_DIR = TOPICS_PROCESSED_DIR_TOPONE / "aligned_topics"
TOPONE_METRICS_PATH = METRICS_DIR_TOPONE / f"canonical_metrics_{CANONICAL_YEAR_TOPONE}_top_1.parquet"
TOPONE_ANCHOR_METRICS_PATH = METRICS_DIR_TOPONE / f"canonical_anchor_metrics_{CANONICAL_YEAR_TOPONE}_top_1.parquet"

# Load clusters/vocab
TOPONE_CANONICAL_CLUSTERS = TOPONE_CANONICAL_CLUSTERS if "TOPONE_CANONICAL_CLUSTERS" in globals() else pd.read_parquet(TOPONE_CLUSTER_PATH)
topone_vocab_df = topone_vocab_df if "topone_vocab_df" in globals() else pd.read_parquet(TOPONE_VOCAB_PATH)

def _normalize_topic_text(text: str) -> str:
  return " ".join(str(text).lower().split())

def _normalize_anchors_val(value) -> tuple:
  import numpy as np
  if value is None or (isinstance(value, float) and pd.isna(value)):
      return ()
  if isinstance(value, np.ndarray):
      value = value.tolist()
  if isinstance(value, (list, tuple)):
      return tuple(a.strip() for a in value if str(a).strip())
  text = str(value).strip()
  return (text,) if text else ()

# Aggregate aligned frames
aligned_frames_topone = []
for fpath in sorted(TOPONE_MAPPING_DIR.glob(f"topic_alignment_*_to_{CANONICAL_YEAR_TOPONE}_top_1.parquet")):
  if "unmapped" in fpath.name:
      continue
  year_str = fpath.name.split("_")[2]
  try:
      year = int(year_str)
  except ValueError:
      print(f"[warn] Could not parse year from {fpath.name}; skipping.")
      continue

  df = pd.read_parquet(fpath)
  if df.empty:
      continue

  totals = TOPONE_TOPICS[year]["paperId"].nunique()
  counts = df.groupby("cluster_id")["paperId"].nunique().reset_index(name="matched_papers")
  counts["year"] = year
  counts["total_papers"] = totals
  counts["rarity"] = counts["matched_papers"] / totals if totals else 0.0
  aligned_frames_topone.append(counts)

# Baseline mapping with normalized keys
baseline_counts = TOPONE_CANONICAL_TOPICS.copy()
baseline_counts["topic_key"] = baseline_counts[TOPONE_TOPIC_COL].map(_normalize_topic_text)

cluster_keys = TOPONE_CANONICAL_CLUSTERS.copy()
cluster_keys["topic_key"] = cluster_keys["topic_text"].map(_normalize_topic_text)
cluster_keys = cluster_keys[["topic_key", "cluster_id"]]

baseline_counts = baseline_counts.merge(cluster_keys, on="topic_key", how="left")

missing = baseline_counts["cluster_id"].isna().sum()
if missing:
  print(f"[warn] {missing} canonical topics could not be mapped after normalization; dropping them from metrics.")
  baseline_counts = baseline_counts.dropna(subset=["cluster_id"]).copy()

baseline_counts = baseline_counts.groupby("cluster_id")["paperId"].nunique().reset_index(name="matched_papers")
baseline_counts["year"] = CANONICAL_YEAR_TOPONE
baseline_totals = TOPONE_CANONICAL_TOPICS["paperId"].nunique()
baseline_counts["total_papers"] = baseline_totals
baseline_counts["rarity"] = baseline_counts["matched_papers"] / baseline_totals if baseline_totals else 0.0

rarity_df_topone = pd.concat([baseline_counts, *aligned_frames_topone], ignore_index=True)
print(f"Rarity table rows (top-one): {len(rarity_df_topone):,}")

metrics_topone = []
for cluster_id, cluster_rows in rarity_df_topone.groupby("cluster_id"):
  cluster_rows = cluster_rows.sort_values("year")
  baseline_year = cluster_rows["year"].max()
  rarity_baseline = cluster_rows.loc[cluster_rows["year"] == baseline_year, "rarity"].iloc[0]

  for _, row in cluster_rows.iterrows():
      year = row["year"]
      rarity_t = row["rarity"]
      matched = row["matched_papers"]
      total = row["total_papers"]
      decline = rarity_t - rarity_baseline if year != baseline_year else 0.0
      impact_score = 0.0 if rarity_t <= 0 else decline * np.log(1.0 / rarity_baseline)

      metrics_topone.append(
          {
              "cluster_id": cluster_id,
              "year": year,
              "matched_papers": matched,
              "total_papers": total,
              "rarity": rarity_t,
              "decline": decline,
              "impact_score": impact_score,
              "impact_magnitude": abs(impact_score),
          }
      )

metrics_df_topone = pd.DataFrame(metrics_topone)
metrics_df_topone = metrics_df_topone.merge(
  topone_vocab_df[["cluster_id", "representative_label", "anchors"]],
  on="cluster_id",
  how="left",
)
metrics_df_topone.to_parquet(TOPONE_METRICS_PATH, index=False)
print(f"Saved top-one topic-level metrics to {TOPONE_METRICS_PATH}")

anchor_records_topone = []
for year, year_rows in metrics_df_topone.groupby("year"):
  total = year_rows["total_papers"].max()
  grouped = year_rows.groupby(year_rows["anchors"].apply(_normalize_anchors_val))

  for anchors, anchor_group in grouped:
      anchors_list = list(anchors)
      matched_total = anchor_group["matched_papers"].sum()
      rarity = matched_total / total if total else 0.0
      decline = anchor_group["decline"].mean()
      impact = anchor_group["impact_score"].mean()
      magnitude = anchor_group["impact_magnitude"].mean()

      anchor_records_topone.append(
          {
              "year": year,
              "anchors": anchors_list,
              "matched_papers": matched_total,
              "total_papers": total,
              "rarity": rarity,
              "decline": decline,
              "impact_score": impact,
              "impact_magnitude": magnitude,
          }
      )

anchor_df_topone = pd.DataFrame(anchor_records_topone)
anchor_df_topone.to_parquet(TOPONE_ANCHOR_METRICS_PATH, index=False)
print(f"Saved top-one anchor-level metrics to {TOPONE_ANCHOR_METRICS_PATH}")

display(metrics_df_topone.sort_values(["year", "impact_magnitude"], ascending=[False, False]).head(10))


# In[ ]:


# Cell 10 top-one filter – Plot top 5 backward top-one signals filtered by anchor (with handles)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

anchor_target_topone = "vertical federated learning"  # ← set anchor
year_to_plot_topone = 2020                      # ← set non-canonical year

metrics_df_topone = pd.read_parquet(TOPONE_METRICS_PATH)

if year_to_plot_topone == CANONICAL_YEAR_TOPONE:
  raise ValueError("year_to_plot_topone must be a non-canonical year (e.g., 2023).")

def normalize_anchors_topone(value):
  if value is None or (isinstance(value, float) and pd.isna(value)):
      return []
  if isinstance(value, list):
      return value
  if isinstance(value, tuple):
      return list(value)
  if isinstance(value, np.ndarray):
      return value.tolist()
  return [value]

subset = metrics_df_topone[
  (metrics_df_topone["year"] == year_to_plot_topone) & (metrics_df_topone["decline"] < 0)
].copy()
subset = subset.dropna(subset=["representative_label"])

subset["anchor_list"] = subset["anchors"].apply(normalize_anchors_topone)
subset = subset[subset["anchor_list"].apply(lambda lst: anchor_target_topone in lst)]

if subset.empty:
  print(f"No top-one topics for anchor '{anchor_target_topone}' in year {year_to_plot_topone}.")
else:
  subset.sort_values("impact_magnitude", ascending=False, inplace=True)
  top_k = subset.head(5).copy()

  if not top_k.empty:
      max_mag = top_k["impact_magnitude"].max() or 1.0
      top_k["impact_normalized"] = 100.0 * top_k["impact_magnitude"] / max_mag
  else:
      top_k["impact_normalized"] = 0.0

  top_k["label_with_handle"] = top_k.apply(
      lambda row: f"{row['representative_label']}", axis=1
  )

  plt.figure(figsize=(12, 8))
  sns.barplot(
      data=top_k,
      x="impact_normalized",
      y="label_with_handle",
      color="orange",
  )
  plt.xlabel("Normalized impact (0–100)")
  plt.ylabel("")
  plt.title(
      f"Backward Signals — Anchor '{anchor_target_topone}' — Year {year_to_plot_topone} vs {CANONICAL_YEAR_TOPONE}"
  )
  plt.tight_layout()
  plt.show()


# In[ ]:


# Cell 11 top-one filter – Plot % growth in paper counts from target year → 2024 for top-5 signals
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

anchor_target_topone = "trustworthy AI"  # ← set anchor
year_to_plot_topone = 2020                      # ← set non-canonical year

metrics_df_topone = pd.read_parquet(TOPONE_METRICS_PATH)

if year_to_plot_topone == CANONICAL_YEAR_TOPONE:
  raise ValueError("year_to_plot_topone must be non-canonical (e.g., 2023).")

def normalize_anchors_topone(value):
  if value is None or (isinstance(value, float) and pd.isna(value)):
      return []
  if isinstance(value, list):
      return value
  if isinstance(value, tuple):
      return list(value)
  if isinstance(value, np.ndarray):
      return value.tolist()
  return [value]

# Select top 5 by impact magnitude (decline < 0) for the anchor/year
subset = metrics_df_topone[
  (metrics_df_topone["year"] == year_to_plot_topone) & (metrics_df_topone["decline"] < 0)
].copy()
subset = subset.dropna(subset=["representative_label"])
subset["anchor_list"] = subset["anchors"].apply(normalize_anchors_topone)
subset = subset[subset["anchor_list"].apply(lambda lst: anchor_target_topone in lst)]
subset.sort_values("impact_magnitude", ascending=False, inplace=True)
top_k = subset.head(5).copy()

if top_k.empty:
  print(f"No top-one topics for anchor '{anchor_target_topone}' in year {year_to_plot_topone}.")
else:
  def papers_for(cluster_id, year):
      row = metrics_df_topone[(metrics_df_topone["cluster_id"] == cluster_id) & (metrics_df_topone["year"] == year)]
      if row.empty:
          return 0.0
      return float(row["matched_papers"].iloc[0])

  growth_rows = []
  dropped_zero = 0
  for _, row in top_k.iterrows():
      start = papers_for(row["cluster_id"], year_to_plot_topone)
      end = papers_for(row["cluster_id"], CANONICAL_YEAR_TOPONE)
      if start <= 0:
          dropped_zero += 1
          continue
      growth_pct = ((end - start) / start) * 100.0
      growth_rows.append(
          {
              "cluster_id": int(row["cluster_id"]),
              "label_with_handle": f"{row['representative_label']}",
              "start_papers": start,
              "end_papers": end,
              "growth_pct": growth_pct,
          }
      )

  if dropped_zero:
      print(f"[info] Skipped {dropped_zero} item(s) with zero papers in {year_to_plot_topone} (cannot compute growth).")

  if not growth_rows:
      print("No rows to plot after zero-baseline filtering.")
  else:
      df_growth = pd.DataFrame(growth_rows)
      df_growth.sort_values("growth_pct", ascending=False, inplace=True)

      plt.figure(figsize=(12, 8))
      sns.barplot(
          data=df_growth,
          x="growth_pct",
          y="label_with_handle",
          color="orange",
      )
      plt.xlabel(f"% growth in matched papers ({year_to_plot_topone} → {CANONICAL_YEAR_TOPONE})")
      plt.ylabel("")
      plt.title(
          f"Backward Signals — Anchor '{anchor_target_topone}' — % Growth in Papers\n"
          f"{year_to_plot_topone} → {CANONICAL_YEAR_TOPONE}"
      )
      plt.tight_layout()
      plt.show()

