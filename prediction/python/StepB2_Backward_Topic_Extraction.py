#!/usr/bin/env python
# coding: utf-8

# # 1. The Top-One Level EU Signal Workflow

# In[ ]:


# Cell 1 top-one filter – Imports, paths, and EU anchors (core-tech workflow)
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()

PROJECT_ROOT = next(
  (p for p in Path.cwd().resolve().parents if (p / "README.md").exists()),
  Path.cwd().resolve(),
)

DATA_ROOT = PROJECT_ROOT / "data" / "backward_construction"
MANIFEST_DIR = DATA_ROOT / "manifests"
TOPICS_RAW_DIR = DATA_ROOT / "topics_raw"
TOPICS_PROCESSED_DIR = DATA_ROOT / "topics_processed"

for directory in (TOPICS_RAW_DIR, TOPICS_PROCESSED_DIR):
  directory.mkdir(parents=True, exist_ok=True)

BASELINE_YEAR_TOPONE = 2024
MIN_YEAR_TOPONE = 2020
YEARS_TOPONE = list(range(BASELINE_YEAR_TOPONE, MIN_YEAR_TOPONE - 1, -1))

EU_EXTERNAL_SIGNALS_TOPONE = [
  "Artificial Intelligence of Things",
  "asynchronous federated learning",
  "attention mechanisms in CNN",
  "decentralized federated learning",
  "epistemic AI",
  "evolutionary Neural Architecture Search",
  "explainable AI",
  "federated deep learning",
  "federated machine learning",
  "federated reinforcement learning",
  "human AI interface",
  "human centric AI",
  "large language models",
  "machine unlearning",
  "masked face recognition",
  "masked language model",
  "multimodal AI",
  "multimodal hate speech detection",
  "privacy-preserving machine learning",
  "scientific machine learning",
  "self supervised learning CNN",
  "tiny machine learning",
  "trustworthy AI",
  "trustworthy machine learning",
  "vertical federated learning",
]

#FILTERED_SUFFIX_TOPONE = "filtered_top_1"
#MANIFEST_TEMPLATE_TOPONE = f"semantic_scholar_{{year}}_{FILTERED_SUFFIX_TOPONE}.parquet"
MANIFEST_TEMPLATE_TOPONE = f"semantic_scholar_{{year}}.parquet"


# In[ ]:


# Cell 2 top-one filter – Sanity check of paths and top-one anchor list
print(f"Project root: {PROJECT_ROOT}")
print(f"Data root:    {DATA_ROOT}")
print(f"Manifests:    {MANIFEST_DIR}  | exists={MANIFEST_DIR.exists()}")
print(f"Topics raw:   {TOPICS_RAW_DIR}  | exists={TOPICS_RAW_DIR.exists()}")
print(f"Topics proc:  {TOPICS_PROCESSED_DIR}  | exists={TOPICS_PROCESSED_DIR.exists()}")
print(f"Years (baseline → earliest): {YEARS_TOPONE}")
print(f"EU anchor count (top-one):   {len(EU_EXTERNAL_SIGNALS_TOPONE)}")
print(f"Manifest template:           {MANIFEST_TEMPLATE_TOPONE}")


# In[ ]:


# Cell 3 top-one filter – Path helpers for manifests, raw outputs, and processed tables
def manifest_path_topone(year: int) -> Path:
  return MANIFEST_DIR / MANIFEST_TEMPLATE_TOPONE.format(year=year)


def raw_output_path_topone(year: int) -> Path:
  return TOPICS_RAW_DIR / f"topics_{year}_topone_filtered_top_1.jsonl"


def raw_meta_path_topone(year: int) -> Path:
  return TOPICS_RAW_DIR / f"topics_{year}_topone_filtered_top_1_meta.json"


def state_path_topone(year: int) -> Path:
  return TOPICS_RAW_DIR / f"topics_{year}_topone_filtered_top_1_state.json"


def processed_topics_path_topone(year: int) -> Path:
  return TOPICS_PROCESSED_DIR / f"topics_{year}_topone_filtered_top_1_paper_topics.parquet"


# In[ ]:


# Cell 4 top-one filter – Existing data/status snapshot per year
status_rows_topone: List[Dict[str, Any]] = []

for year in YEARS_TOPONE:
  row: Dict[str, Any] = {"year": year}

  manifest_file = manifest_path_topone(year)
  if manifest_file.exists():
      try:
          manifest_df = pd.read_parquet(manifest_file, columns=["paperId"])
          row["manifest_records"] = len(manifest_df)
      except Exception as exc:
          row["manifest_records"] = None
          row["manifest_error"] = str(exc)
  else:
      row["manifest_records"] = 0

  raw_file = raw_output_path_topone(year)
  row["topics_raw_exists"] = raw_file.exists()

  processed_file = processed_topics_path_topone(year)
  row["topics_processed_exists"] = processed_file.exists()

  state_file = state_path_topone(year)
  row["state_exists"] = state_file.exists()

  status_rows_topone.append(row)

status_df_topone = (
  pd.DataFrame(status_rows_topone)
  .set_index("year")
  .sort_index(ascending=False)
)
status_df_topone


# In[ ]:


# Cell 5 top-one filter – Manifest loading helper (filtered corpus, non-empty abstracts)
def load_manifest_topone(year: int, columns: Optional[List[str]] = None) -> pd.DataFrame:
  """
  Load the per-year filtered manifest (citation/venue qualified) and keep only papers with
  non-empty abstracts.
  """
  path = manifest_path_topone(year)
  if not path.exists():
      raise FileNotFoundError(f"Filtered manifest not found for {year}: {path}")

  default_cols = [
      "paperId",
      "title",
      "abstract",
      "year",
      "venue",
      "publicationDate",
      "fieldsOfStudy",
      "authors",
      "source_query",
      "query_year",
  ]
  cols_to_load = columns if columns is not None else default_cols

  df = pd.read_parquet(path, columns=cols_to_load).copy()
  df["paperId"] = df["paperId"].astype(str)

  if "abstract" not in df.columns:
      raise ValueError(f"Filtered manifest for {year} is missing an 'abstract' column.")

  df = df[df["abstract"].fillna("").str.strip() != ""].reset_index(drop=True)
  return df


# In[ ]:


# Cell 6 top-one filter – Preview filtered manifest sample for a chosen year
preview_year_topone = 2024  # adjust as needed

try:
  preview_df_topone = load_manifest_topone(
      preview_year_topone,
      columns=["paperId", "title", "abstract"],
  )
  print(f"{preview_year_topone} filtered manifest count (non-empty abstracts): {len(preview_df_topone):,}")
  display(preview_df_topone.head(3))
except FileNotFoundError as exc:
  print(exc)


# In[ ]:


# Cell 7 top-one filter – LLM client configuration & extraction parameters (temperature optional)
from openai import OpenAI
import tiktoken

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
  raise RuntimeError(
      "Missing OPENAI_API_KEY. Populate it in your environment or .env before running StepB2."
  )

client_topone = OpenAI()

TOPIC_MODEL_TOPONE = os.getenv(
  "BWD_TOPIC_MODEL_TOPONE",
  os.getenv("TOPIC_MODEL_TOPONE", "gpt-5-mini"),
)

temp_env_topone = (os.getenv("BWD_TOPIC_TEMPERATURE_TOPONE") or "").strip()
TOPIC_TEMPERATURE_TOPONE = float(temp_env_topone) if temp_env_topone else None

MAX_TOPICS_PER_PAPER_TOPONE = int(os.getenv("BWD_MAX_TOPICS_PER_PAPER_TOPONE", "2"))
MAX_OUTPUT_TOKENS_TOPONE = int(os.getenv("BWD_TOPIC_MAX_OUTPUT_TOKENS_TOPONE", "1000"))
MAX_INPUT_TOKENS_TOPONE = int(os.getenv("BWD_TOPIC_MAX_INPUT_TOKENS_TOPONE", "3000"))
REQUEST_THROTTLE_SECONDS_TOPONE = float(os.getenv("BWD_TOPIC_THROTTLE_SECONDS_TOPONE", "0.0"))
MAX_LLM_RETRIES_TOPONE = int(os.getenv("BWD_TOPIC_MAX_RETRIES_TOPONE", "5"))

try:
  ENCODER_TOPONE = tiktoken.encoding_for_model(TOPIC_MODEL_TOPONE)
except Exception:
  ENCODER_TOPONE = tiktoken.get_encoding("cl100k_base")

print(f"Model: {TOPIC_MODEL_TOPONE}")
print(f"Max topics per paper: {MAX_TOPICS_PER_PAPER_TOPONE}")
print(f"Max input tokens: {MAX_INPUT_TOKENS_TOPONE}, max output tokens: {MAX_OUTPUT_TOKENS_TOPONE}")
if TOPIC_TEMPERATURE_TOPONE is None:
  print("Temperature: default (model-defined)")
else:
  print(f"Temperature override: {TOPIC_TEMPERATURE_TOPONE}")


# In[ ]:


# Cell 8 top-one filter – Prompt builder and token estimation (macro EU template, core-tech emphasis)
ANCHOR_LIST_TOPONE = "\n".join(f"- {anchor}" for anchor in EU_EXTERNAL_SIGNALS_TOPONE)

SYSTEM_PROMPT_TOPONE = """You are an expert analyst of frontier research with the highest IQ in the world.
Given an abstract and a list of broad EU mainframe signal topics (anchors), identify up to {k} fine-grained topics
that (a) are explicitly supported by the abstract and (b) align with exactly one anchor.
Emit a topic only if you are confident in the match; otherwise return no topics.
Return only JSON: {{"topics": [{{...}}, ...]}}.
Each topic must include: topic_name, justification, anchor, confidence, broader_topic.
"""

USER_PROMPT_TEMPLATE_TOPONE = """Paper metadata:
- Title: {title}
- Paper ID: {paper_id}
- Year: {year}
- Venue: {venue}
- Source query: {source_query}

Abstract:
{abstract}

EU Anchors:
{anchors}

Requirements:
- Output a JSON object with a "topics" array (may be empty).
- Each element must include: "topic_name", "justification", "anchor", "confidence", "broader_topic".
- For every fine-grained topic, produce exactly one "broader_topic" string that best generalizes it. 
- Emit a topic only when confidence is high.
- "anchor" must match exactly one item from the EU anchor list.
- Justifications must be true, accurate, and concise (ideally one sentence referencing the abstract).
- You MUST NOT invent anchors or topics unsupported by the abstract.
"""

def build_prompt_topone(paper: Dict[str, Any]) -> Dict[str, str]:
  system_prompt = SYSTEM_PROMPT_TOPONE.format(k=MAX_TOPICS_PER_PAPER_TOPONE)
  user_prompt = USER_PROMPT_TEMPLATE_TOPONE.format(
      title=paper.get("title") or "Unknown title",
      paper_id=paper.get("paperId"),
      year=paper.get("year"),
      venue=paper.get("venue") or "Unknown venue",
      source_query=paper.get("source_query") or "N/A",
      abstract=paper.get("abstract"),
      anchors=ANCHOR_LIST_TOPONE,
  )
  return {"system": system_prompt, "user": user_prompt}

def count_tokens_topone(text: str) -> int:
  return len(ENCODER_TOPONE.encode(text))

def paper_token_estimate_topone(paper: Dict[str, Any]) -> int:
  prompts = build_prompt_topone(paper)
  return count_tokens_topone(prompts["system"]) + count_tokens_topone(prompts["user"])


# In[ ]:


# Cell 9 top-one filter – Parse & validate LLM response (≤10-word direction-level topics)
import re

ANCHOR_SET_TOPONE = {anchor for anchor in EU_EXTERNAL_SIGNALS_TOPONE}

def _safe_json_loads_topone(text: str) -> Dict[str, Any] | None:
  if not text:
      return None
  try:
      return json.loads(text)
  except json.JSONDecodeError:
      match = re.search(r"\{[\s\S]*\}", text)
      if match:
          try:
              return json.loads(match.group(0))
          except json.JSONDecodeError:
              return None
  return None

def _normalize_confidence_topone(value: Any) -> str:
  if value is None:
      return ""
  if isinstance(value, (int, float)):
      return "high" if float(value) >= 0.8 else "medium" if float(value) >= 0.5 else "low"
  text = str(value).strip().lower()
  if text in {"high", "medium", "low"}:
      return text
  if text in {"very high", "certain", "confident"}:
      return "high"
  if text in {"uncertain", "unsure", "maybe"}:
      return "low"
  return text

def _word_count(text: str) -> int:
  return len([tok for tok in text.strip().split() if tok])

def parse_topics_response_topone(raw_text: str) -> List[Dict[str, Any]]:
  payload = _safe_json_loads_topone(raw_text)
  if not payload:
      return []

  topics = payload.get("topics")
  if not isinstance(topics, list):
      return []

  cleaned: List[Dict[str, Any]] = []
  for entry in topics:
      if not isinstance(entry, dict):
          continue

      topic_name = str(entry.get("topic_name") or "").strip()
      anchor = str(entry.get("anchor") or "").strip()
      justification = str(entry.get("justification") or "").strip()
      broader_topic = str(entry.get("broader_topic") or "").strip()
      confidence = _normalize_confidence_topone(entry.get("confidence"))

      if not topic_name or not anchor or not justification or not broader_topic:
          continue
      if _word_count(topic_name) > 10:
          continue
      if anchor not in ANCHOR_SET_TOPONE:
          continue
      if confidence not in {"high", "medium"}:
          continue

      cleaned.append(
          {
              "topic_name": topic_name,
              "anchor": anchor,
              "justification": justification,
              "confidence": confidence,
              "broader_topic": broader_topic,
          }
      )

  if len(cleaned) > MAX_TOPICS_PER_PAPER_TOPONE:
      cleaned = cleaned[:MAX_TOPICS_PER_PAPER_TOPONE]

  return cleaned


# In[ ]:


# Cell 10 top-one filter – Prompt preparation with token-aware truncation
def truncate_text_to_tokens_topone(text: str, max_tokens: int) -> Dict[str, Any]:
  tokens = ENCODER_TOPONE.encode(text or "")
  original_len = len(tokens)
  if original_len <= max_tokens:
      return {
          "text": text,
          "used_tokens": original_len,
          "original_tokens": original_len,
          "was_truncated": False,
      }

  truncated_tokens = tokens[:max_tokens]
  truncated_text = ENCODER_TOPONE.decode(truncated_tokens)
  return {
      "text": truncated_text,
      "used_tokens": len(truncated_tokens),
      "original_tokens": original_len,
      "was_truncated": True,
  }

def prepare_messages_topone(paper: Dict[str, Any]) -> Dict[str, Any]:
  abstract = paper.get("abstract") or ""
  trunc_info = truncate_text_to_tokens_topone(abstract, MAX_INPUT_TOKENS_TOPONE)

  paper_copy = dict(paper)
  paper_copy["abstract"] = trunc_info["text"]

  prompts = build_prompt_topone(paper_copy)
  total_tokens = (
      count_tokens_topone(prompts["system"])
      + count_tokens_topone(prompts["user"])
  )

  return {
      "messages": [
          {"role": "system", "content": prompts["system"]},
          {"role": "user", "content": prompts["user"]},
      ],
      "prompt_meta": {
          "abstract_tokens_used": trunc_info["used_tokens"],
          "abstract_tokens_original": trunc_info["original_tokens"],
          "abstract_truncated": trunc_info["was_truncated"],
          "total_prompt_tokens": total_tokens,
      },
  }


# In[ ]:


# Cell 11 top-one filter – Concurrency & resumable output helpers
import threading

MAX_CONCURRENT_REQUESTS_TOPONE = int(
  os.getenv("BWD_TOPIC_MAX_CONCURRENT_TOPONE", os.getenv("BWD_TOPIC_MAX_CONCURRENT", "8"))
)
SAVE_BATCH_SIZE_TOPONE = int(
  os.getenv("BWD_TOPIC_SAVE_BATCH_SIZE_TOPONE", os.getenv("BWD_TOPIC_SAVE_BATCH_SIZE", "100"))
)

write_lock_topone = threading.Lock()

def load_existing_raw_topone(year: int) -> List[Dict[str, Any]]:
  path = raw_output_path_topone(year)
  if not path.exists():
      return []
  records: List[Dict[str, Any]] = []
  with path.open("r", encoding="utf-8") as fp:
      for line in fp:
          line = line.strip()
          if not line:
              continue
          try:
              records.append(json.loads(line))
          except json.JSONDecodeError:
              continue
  return records

def existing_processed_ids_topone(year: int) -> set:
  return {rec.get("paperId") for rec in load_existing_raw_topone(year) if rec.get("paperId")}

def append_raw_records_topone(year: int, records: List[Dict[str, Any]]) -> None:
  if not records:
      return
  path = raw_output_path_topone(year)
  with write_lock_topone:
      with path.open("a", encoding="utf-8") as fp:
          for rec in records:
              fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

def load_state_metadata_topone(year: int) -> Dict[str, Any]:
  path = state_path_topone(year)
  if not path.exists():
      return {}
  try:
      return json.loads(path.read_text(encoding="utf-8"))
  except json.JSONDecodeError:
      return {}

def save_state_metadata_topone(year: int, payload: Dict[str, Any]) -> None:
  path = state_path_topone(year)
  path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# In[ ]:


# Cell 12 top-one filter – LLM request wrapper & worker function (adaptive tokens + skip on cap)
import math
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

def call_llm_topone(messages: List[Dict[str, str]]) -> str:
  attempt = 0
  max_tokens = MAX_OUTPUT_TOKENS_TOPONE
  max_tokens_cap = int(
      os.getenv(
          "BWD_TOPIC_MAX_OUTPUT_CAP_TOPONE",
          str(MAX_OUTPUT_TOKENS_TOPONE * 4),
      )
  )

  while True:
      attempt += 1
      try:
          kwargs = {
              "model": TOPIC_MODEL_TOPONE,
              "messages": messages,
              "max_completion_tokens": max_tokens,
              "response_format": {"type": "json_object"},
          }
          if TOPIC_TEMPERATURE_TOPONE is not None:
              kwargs["temperature"] = TOPIC_TEMPERATURE_TOPONE

          response = client_topone.chat.completions.create(**kwargs)
          text = response.choices[0].message.content or ""
          return text.strip()

      except Exception as exc:
          message = str(exc)

          hit_limit = (
              "max_tokens" in message
              and ("limit was reached" in message or "max_tokens" in message)
          )
          if hit_limit:
              if max_tokens < max_tokens_cap:
                  max_tokens = min(max_tokens + 1000, max_tokens_cap)
                  print(
                      f"[warn] LLM hit completion limit; increasing max tokens to {max_tokens} "
                      f"and retrying (attempt {attempt})."
                  )
                  sleep(min(5 * attempt, 30))
                  continue
              else:
                  raise RuntimeError(
                      f"LLM max token limit reached even at cap ({max_tokens_cap}); skipping."
                  ) from exc

          if attempt >= MAX_LLM_RETRIES_TOPONE:
              raise RuntimeError(f"LLM request failed after {attempt} attempts: {exc}") from exc

          backoff = min(5 * attempt, 30)
          print(f"[warn] LLM error (attempt {attempt}): {exc}. Retrying in {backoff}s...")
          sleep(backoff)

def process_single_paper_topone(paper: Dict[str, Any]) -> Dict[str, Any]:
  paper_id = paper.get("paperId")
  messages_bundle = prepare_messages_topone(paper)
  raw_text = call_llm_topone(messages_bundle["messages"])
  parsed_topics = parse_topics_response_topone(raw_text)

  result = {
      "paperId": paper_id,
      "topics": parsed_topics,
      "prompt_meta": messages_bundle["prompt_meta"],
      "model": TOPIC_MODEL_TOPONE,
  }
  if REQUEST_THROTTLE_SECONDS_TOPONE > 0:
      sleep(REQUEST_THROTTLE_SECONDS_TOPONE)
  return result


# In[ ]:


# Cell 13 top-one filter – Year-level extraction orchestrator (concurrent, resumable)
def extract_topics_for_year_topone(year: int, *, force: bool = False) -> Dict[str, Any]:
  manifest_df = load_manifest_topone(year)
  total_papers = len(manifest_df)
  print(f"[{year}] Filtered papers with non-empty abstracts: {total_papers:,}")

  processed_ids: set[str] = set()
  state_meta: Dict[str, Any] = {}

  if not force:
      processed_ids = existing_processed_ids_topone(year)
      state_meta = load_state_metadata_topone(year)

  if force:
      processed_ids.clear()
      state_meta = {}
      raw_path = raw_output_path_topone(year)
      if raw_path.exists():
          backup = raw_path.with_suffix(f".jsonl.bak.{int(time.time())}")
          raw_path.rename(backup)
          print(f"[{year}] Existing raw output moved to {backup}")
      state_path_topone(year).unlink(missing_ok=True)

  remaining_df = manifest_df[~manifest_df["paperId"].isin(processed_ids)].reset_index(drop=True)
  remaining = len(remaining_df)
  if remaining == 0:
      print(f"[{year}] No remaining papers to process.")
      return {
          "year": year,
          "total": total_papers,
          "already_processed": len(processed_ids),
          "processed_now": 0,
      }

  print(f"[{year}] Already processed: {len(processed_ids):,} | Remaining: {remaining:,}")
  progress = tqdm(
      total=total_papers,
      initial=len(processed_ids),
      desc=f"{year} top-one extraction",
      unit="paper",
  )

  buffer: List[Dict[str, Any]] = []

  def submit_tasks(executor: ThreadPoolExecutor):
      future_to_idx = {}
      for _, row in remaining_df.iterrows():
          paper = row.to_dict()
          future = executor.submit(process_single_paper_topone, paper)
          future_to_idx[future] = paper["paperId"]
      return future_to_idx

  processed_now = 0
  errors: List[Dict[str, Any]] = []

  try:
      with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS_TOPONE) as executor:
          future_map = submit_tasks(executor)
          for future in as_completed(future_map):
              paper_id = future_map[future]
              try:
                  result = future.result()
                  buffer.append(result)
                  processed_now += 1
                  progress.update(1)
              except Exception as exc:
                  errors.append({"paperId": paper_id, "error": str(exc)})
                  progress.write(f"[error] Failed for paper {paper_id}: {exc}")

              if len(buffer) >= SAVE_BATCH_SIZE_TOPONE:
                  append_raw_records_topone(year, buffer)
                  buffer.clear()
                  save_state_metadata_topone(
                      year,
                      {
                          "last_processed_paper": paper_id,
                          "timestamp": time.time(),
                          "processed_count": len(processed_ids) + processed_now,
                      },
                  )

  finally:
      progress.close()
      if buffer:
          append_raw_records_topone(year, buffer)
          buffer.clear()
      save_state_metadata_topone(
          year,
          {
              "last_processed_paper": None,
              "timestamp": time.time(),
              "processed_count": len(processed_ids) + processed_now,
          },
      )

  print(f"[{year}] Top-one extraction finished. Newly processed: {processed_now:,}")
  if errors:
      print(f"[{year}] Encountered {len(errors)} errors. See error list for details.")
  return {
      "year": year,
      "total": total_papers,
      "already_processed": len(processed_ids),
      "processed_now": processed_now,
      "errors": errors,
  }


# In[ ]:


# Preview (top-one filter) – dry run on a random sample (no persistence, flattened topics)
preview_year_topone = 2024
preview_n_topone = 3
preview_seed_topone = None  # set an int for reproducible sampling

sample_df_topone = (
  load_manifest_topone(preview_year_topone)
  .sample(n=preview_n_topone, random_state=preview_seed_topone)
  .reset_index(drop=True)
)

records_topone = []
print(f"Previewing {preview_n_topone} random papers from {preview_year_topone} (no results will be saved).")

for _, row in sample_df_topone.iterrows():
  paper = row.to_dict()
  result = process_single_paper_topone(paper)

  topics = result["topics"] or []
  if not topics:
      records_topone.append(
          {
              "paperId": paper["paperId"],
              "title": paper["title"],
              "topic_name": "",
              "anchor": "",
              "justification": "",
              "confidence": "",
              "abstract_tokens": result["prompt_meta"]["abstract_tokens_used"],
              "abstract_truncated": result["prompt_meta"]["abstract_truncated"],
          }
      )
  else:
      for topic in topics:
          records_topone.append(
              {
                  "paperId": paper["paperId"],
                  "title": paper["title"],
                  "topic_name": topic["topic_name"],
                  "anchor": topic["anchor"],
                  "justification": topic["justification"],
                  "confidence": topic["confidence"],
                  "abstract_tokens": result["prompt_meta"]["abstract_tokens_used"],
                  "abstract_truncated": result["prompt_meta"]["abstract_truncated"],
              }
          )

pd.DataFrame(records_topone)


# In[ ]:


# Cell 15 top-one filter – Batch request/response directories
BATCH_DIR_TOPONE = TOPICS_RAW_DIR / "batch_requests_topone_filtered_top_1"
BATCH_RESULTS_DIR_TOPONE = TOPICS_RAW_DIR / "batch_results_topone_filtered_top_1"
BATCH_DIR_TOPONE.mkdir(parents=True, exist_ok=True)
BATCH_RESULTS_DIR_TOPONE.mkdir(parents=True, exist_ok=True)

print(f"Batch request dir (top-one):  {BATCH_DIR_TOPONE}")
print(f"Batch results dir (top-one):  {BATCH_RESULTS_DIR_TOPONE}")


# In[ ]:


# Cell 16 top-one filter – Build batch request JSONL files (filtered corpus, resumable, fixed chunk IDs)
import math
import re

BATCH_CHUNK_SIZE_TOPONE = int(
  os.getenv("BWD_BATCH_CHUNK_SIZE_TOPONE", os.getenv("BWD_BATCH_CHUNK_SIZE", "2000"))
)

def manifest_with_chunks_topone(year: int) -> pd.DataFrame:
  df = load_manifest_topone(year).reset_index(drop=True)
  df["paperId"] = df["paperId"].astype(str)
  df["chunk_idx"] = df.index // BATCH_CHUNK_SIZE_TOPONE
  return df

def existing_processed_ids_topone_batch(year: int) -> set:
  raw_file = raw_output_path_topone(year)
  if not raw_file.exists():
      return set()

  ids: set[str] = set()
  with raw_file.open("r", encoding="utf-8") as fp:
      for line in fp:
          if not line.strip():
              continue
          try:
              rec = json.loads(line)
          except json.JSONDecodeError:
              continue
          pid = rec.get("paperId")
          if pid:
              ids.add(str(pid))
  return ids

def build_batch_files_topone(year: int, *, overwrite: bool = False) -> List[Path]:
  manifest_df = manifest_with_chunks_topone(year)
  total_chunks = manifest_df["chunk_idx"].max() + 1

  processed_ids = existing_processed_ids_topone_batch(year)
  pending_df = manifest_df[~manifest_df["paperId"].isin(processed_ids)]
  pending_chunks = sorted(pending_df["chunk_idx"].unique())

  metadata_path = BATCH_DIR_TOPONE / f"batch_{year}_topone_filtered_top_1_manifest.json"

  if overwrite:
      for path in BATCH_DIR_TOPONE.glob(f"batch_{year}_topone_filtered_top_1_*.jsonl"):
          path.unlink()
      if metadata_path.exists():
          metadata_path.unlink()

  existing_meta: Dict[str, Dict[str, Any]] = {}
  if metadata_path.exists():
      try:
          existing_meta = {
              Path(entry["request_file"]).name: entry
              for entry in json.loads(metadata_path.read_text(encoding="utf-8"))
          }
      except json.JSONDecodeError:
          existing_meta = {}

  new_files: List[Path] = []

  for chunk_idx in pending_chunks:
      chunk = pending_df[pending_df["chunk_idx"] == chunk_idx]
      filename = f"batch_{year}_topone_filtered_top_1_{chunk_idx:04d}.jsonl"
      batch_path = BATCH_DIR_TOPONE / filename

      if batch_path.exists():
          batch_path.unlink()

      records = []
      for _, row in chunk.iterrows():
          paper = row.to_dict()
          prepared = prepare_messages_topone(paper)
          record = {
              "custom_id": f"{year}_{paper['paperId']}",
              "method": "POST",
              "url": "/v1/chat/completions",
              "body": {
                  "model": TOPIC_MODEL_TOPONE,
                  "messages": prepared["messages"],
                  "max_completion_tokens": MAX_OUTPUT_TOKENS_TOPONE,
                  "response_format": {"type": "json_object"},
              },
          }
          if TOPIC_TEMPERATURE_TOPONE is not None:
              record["body"]["temperature"] = TOPIC_TEMPERATURE_TOPONE
          records.append(record)

      with batch_path.open("w", encoding="utf-8") as fp:
          for record in records:
              fp.write(json.dumps(record, ensure_ascii=False) + "\n")

      existing_meta[filename] = {
          "year": year,
          "chunk_idx": int(chunk_idx),
          "request_file": str(batch_path),
          "created_at": time.time(),
          "n_requests": len(records),
      }
      new_files.append(batch_path)

  metadata_entries = [
      existing_meta[name]
      for name in sorted(existing_meta.keys())
  ]
  metadata_path.write_text(
      json.dumps(metadata_entries, ensure_ascii=False, indent=2),
      encoding="utf-8",
  )

  print(f"[{year}] Created {len(new_files)} new top-one batch request file(s).")
  print(f"[{year}] Total chunk slots: {total_chunks}")
  print(f"[{year}] Pending chunk indices: {pending_chunks}")
  return [
      BATCH_DIR_TOPONE / f"batch_{year}_topone_filtered_top_1_{idx:04d}.jsonl"
      for idx in range(total_chunks)
      if (BATCH_DIR_TOPONE / f"batch_{year}_topone_filtered_top_1_{idx:04d}.jsonl").exists()
  ]


# In[ ]:


# Cell 17 top-one filter – Submit top-one batch jobs to OpenAI
BATCH_STATUS_DIR_TOPONE = TOPICS_RAW_DIR / "batch_status_topone_filtered_top_1"
BATCH_STATUS_DIR_TOPONE.mkdir(parents=True, exist_ok=True)

def submit_batch_request_topone(request_path: Path) -> Dict[str, Any]:
  if not request_path.exists():
      raise FileNotFoundError(f"Request file not found: {request_path}")

  print(f"Uploading {request_path.name} ({request_path.stat().st_size / 1024:.1f} KB)...")
  with request_path.open("rb") as fp:
      upload = client_topone.files.create(
          file=fp,
          purpose="batch",
      )

  print(f"Creating batch job for {request_path.name} (file_id={upload.id})...")
  batch = client_topone.batches.create(
      input_file_id=upload.id,
      endpoint="/v1/chat/completions",
      completion_window="24h",
  )

  job_record = {
      "request_file": str(request_path),
      "file_id": upload.id,
      "batch_id": batch.id,
      "status": batch.status,
      "created_at": batch.created_at,
  }

  status_path = BATCH_STATUS_DIR_TOPONE / f"{request_path.stem}_status.json"
  status_path.write_text(json.dumps(job_record, indent=2), encoding="utf-8")

  print(f"Batch job created: id={batch.id}, status={batch.status}")
  print(f"Status written to {status_path}")
  return job_record


# In[ ]:


# Preview – inspect pending batch request files (no submission)
# You can run this cell to check if you want to check if any pending batch request files exist. This only checks existence and does not submit anything.
preview_year_topone_batch = 2020
preview_lines_topone = 3

request_files_topone = sorted(
  BATCH_DIR_TOPONE.glob(f"batch_{preview_year_topone_batch}_topone_filtered_top_1_*.jsonl")
)

if not request_files_topone:
  print(f"No top-one batch request files found for {preview_year_topone_batch} in {BATCH_DIR_TOPONE}.")
else:
  print(f"Found {len(request_files_topone)} top-one batch request file(s) for {preview_year_topone_batch}:")
  for path in request_files_topone:
      size_kb = path.stat().st_size / 1024
      with path.open("r", encoding="utf-8") as fp:
          sample_lines = [next(fp).strip() for _ in range(preview_lines_topone)]
      print(f"\n{path.name} | size: {size_kb:.1f} KB | first {preview_lines_topone} lines:")
      for line in sample_lines:
          print(f"  {line}")


# In[ ]:


# Cell 18 top-one filter – Monitor top-one batch jobs and download results
def list_batch_status_files_topone(year: Optional[int] = None) -> List[Path]:
  pattern = f"batch_{year}_topone_filtered_top_1_*_status.json" if year else "batch_*_topone_filtered_top_1_*_status.json"
  return sorted(BATCH_STATUS_DIR_TOPONE.glob(pattern))

def refresh_batch_status_topone(status_path: Path) -> Dict[str, Any]:
  record = json.loads(status_path.read_text(encoding="utf-8"))
  batch_id = record["batch_id"]
  batch = client_topone.batches.retrieve(batch_id)

  counts = getattr(batch, "request_counts", None)
  if counts is not None:
      counts = {
          "total": getattr(counts, "total", None),
          "completed": getattr(counts, "completed", None),
          "failed": getattr(counts, "failed", None),
          "in_progress": getattr(counts, "in_progress", None),
      }

  record.update(
      {
          "status": batch.status,
          "output_file_id": getattr(batch, "output_file_id", None),
          "error_file_id": getattr(batch, "error_file_id", None),
          "request_counts": counts,
          "last_refreshed": time.time(),
      }
  )
  status_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
  return record

def download_batch_output_topone(batch_record: Dict[str, Any], overwrite: bool = False) -> Optional[Path]:
  output_file_id = batch_record.get("output_file_id")
  if not output_file_id:
      print(f"Batch {batch_record['batch_id']} has no output_file_id yet.")
      return None

  output_path = BATCH_RESULTS_DIR_TOPONE / f"{batch_record['batch_id']}_output.jsonl"
  if output_path.exists() and not overwrite:
      print(f"Output already exists for batch {batch_record['batch_id']}: {output_path}")
      return output_path

  print(f"Downloading output for batch {batch_record['batch_id']}...")
  content = client_topone.files.content(output_file_id)
  output_path.write_bytes(content.read())
  print(f"Saved output to {output_path}")
  return output_path

def monitor_batches_topone(year: Optional[int] = None, download_complete: bool = True) -> List[Dict[str, Any]]:
  status_files = list_batch_status_files_topone(year)
  if not status_files:
      print("No top-one batch status files found.")
      return []

  records = []
  for status_path in status_files:
      record = refresh_batch_status_topone(status_path)
      records.append(record)
      print(
          f"{status_path.name} | status={record['status']} | "
          f"output_file_id={record.get('output_file_id')} | "
          f"errors={record.get('error_file_id')}"
      )
      if download_complete and record["status"] == "completed":
          download_batch_output_topone(record)

  return records


# In[ ]:


# Cell 19 top-one filter – Parse batch outputs into raw topics JSONL
def parse_batch_output_file_topone(year: int, output_path: Path, *, overwrite: bool = False) -> Dict[str, Any]:
  if not output_path.exists():
      raise FileNotFoundError(f"Batch output not found: {output_path}")

  manifest_df = load_manifest_topone(year)
  manifest_map = {row["paperId"]: row.to_dict() for _, row in manifest_df.iterrows()}

  existing_ids = existing_processed_ids_topone(year)
  new_records: List[Dict[str, Any]] = []
  skipped_ids: List[str] = []
  errors: List[Dict[str, Any]] = []

  with output_path.open("r", encoding="utf-8") as fp:
      for line in fp:
          record = json.loads(line)
          custom_id = record.get("custom_id")
          if not custom_id:
              continue

          _, paper_id = custom_id.split("_", 1)
          if paper_id in existing_ids and not overwrite:
              skipped_ids.append(paper_id)
              continue

          response = record.get("response", {})
          error = record.get("error")
          if error:
              errors.append({"paperId": paper_id, "error": error})
              continue

          choices = response.get("body", {}).get("choices", [])
          if not choices:
              errors.append({"paperId": paper_id, "error": "No choices in response"})
              continue

          message_content = choices[0].get("message", {}).get("content", "")
          parsed_topics = parse_topics_response_topone(message_content)

          paper_meta = manifest_map.get(paper_id)
          if not paper_meta:
              errors.append({"paperId": paper_id, "error": "Paper not found in manifest"})
              continue

          prepared = prepare_messages_topone(paper_meta)
          prompt_meta = prepared["prompt_meta"]

          new_records.append(
              {
                  "paperId": paper_id,
                  "topics": parsed_topics,
                  "prompt_meta": prompt_meta,
                  "model": TOPIC_MODEL_TOPONE,
              }
          )
          existing_ids.add(paper_id)

  append_raw_records_topone(year, new_records)

  state_payload = {
      "last_batch_output": str(output_path),
      "records_saved": len(new_records),
      "skipped": len(skipped_ids),
      "errors": errors,
      "timestamp": time.time(),
  }
  save_state_metadata_topone(year, state_payload)

  print(
      f"[{year}] Parsed {len(new_records)} top-one records from {output_path.name} "
      f"(skipped {len(skipped_ids)} duplicates, {len(errors)} errors)."
  )
  return {
      "saved": len(new_records),
      "skipped": skipped_ids,
      "errors": errors,
      "output_file": str(output_path),
  }


# # The following single cell is for partial parsing of batched outputs of a single year (e.g. 2024, 2023, etc.) -- the reason is that some batched requests were completed while some failed, so you can choose to clean up the completed ones first

# In[ ]:


# Parse all completed top-one batch outputs for a single year (e.g. 2024, 2023, 2022, etc.)
records_topone = monitor_batches_topone(2024, download_complete=True)  # refresh + download
completed_topone = [r for r in records_topone if r["status"] == "completed"]

if not completed_topone:
  print("No completed top-one batches yet.")
else:
  summaries_topone = []
  for rec in tqdm(completed_topone, desc="Parsing top-one outputs", unit="batch"):
      output_path = BATCH_RESULTS_DIR_TOPONE / f"{rec['batch_id']}_output.jsonl"
      if not output_path.exists():
          print(f"[warn] Missing top-one output file for {rec['batch_id']}")
          continue
      summary = parse_batch_output_file_topone(2024, output_path)
      summaries_topone.append(summary)

  total_saved = sum(s["saved"] for s in summaries_topone)
  total_skipped = sum(len(s["skipped"]) for s in summaries_topone)
  total_errors = sum(len(s["errors"]) for s in summaries_topone)
  print(f"Saved {total_saved} new top-one records; skipped {total_skipped} duplicates; {total_errors} errors logged.")


# # -------------------------------------------------------------------------------

# In[ ]:


# Cell 21 top-one filter – Resubmit failed batches sequentially with per-paper progress
# You can use this cell to submit any failed batches 
target_year_topone = 2024
poll_interval_seconds_topone = 60   # seconds between status polls
parse_outputs_topone = True         # set False to resubmit only, without parsing

DONE_STATUSES_TOPONE = {"completed", "failed", "cancelled"}

def silent_download_batch_output_topone(batch_record: Dict[str, Any]) -> Optional[Path]:
  """Download a batch result without extra printing; reuse existing file if present."""
  output_file_id = batch_record.get("output_file_id")
  if not output_file_id:
      return None
  output_path = BATCH_RESULTS_DIR_TOPONE / f"{batch_record['batch_id']}_output.jsonl"
  if output_path.exists():
      return output_path
  content = client_topone.files.content(output_file_id)
  output_path.write_bytes(content.read())
  return output_path

def count_request_lines_topone(request_path: Path) -> int:
  with request_path.open("r", encoding="utf-8") as fp:
      return sum(1 for line in fp if line.strip())

failed_status_paths_topone = [
  status_path
  for status_path in list_batch_status_files_topone(target_year_topone)
  if json.loads(status_path.read_text(encoding="utf-8")).get("status") == "failed"
]

if not failed_status_paths_topone:
  print(f"No failed top-one batches for {target_year_topone}.")
else:
  total_saved_topone = total_skipped_topone = total_errors_topone = 0
  missing_requests_topone: List[Path] = []
  total_batches_topone = len(failed_status_paths_topone)

  for idx, status_path in enumerate(failed_status_paths_topone, start=1):
      original_record = json.loads(status_path.read_text(encoding="utf-8"))
      request_path = Path(original_record["request_file"])
      if not request_path.exists():
          missing_requests_topone.append(request_path)
          continue

      total_requests = (
          (original_record.get("request_counts") or {}).get("total")
          or count_request_lines_topone(request_path)
      )

      backup_suffix = status_path.suffix + f".retry_{int(time.time())}"
      status_path.rename(status_path.with_suffix(backup_suffix))

      job = submit_batch_request_topone(request_path)
      batch_id = job["batch_id"]
      status_file = BATCH_STATUS_DIR_TOPONE / f"{request_path.stem}_status.json"

      progress = tqdm(
          total=total_requests,
          desc=f"Top-one batch {idx}/{total_batches_topone}",
          unit="paper",
          leave=False,
      )
      latest_record: Optional[Dict[str, Any]] = None
      completed_count = 0

      while True:
          time.sleep(poll_interval_seconds_topone)
          if not status_file.exists():
              continue
          latest_record = refresh_batch_status_topone(status_file)
          counts = latest_record.get("request_counts") or {}
          completed = counts.get("completed") or 0
          failed = counts.get("failed") or 0
          finished = min(total_requests, completed + failed)
          if finished > completed_count:
              progress.update(finished - completed_count)
              completed_count = finished
          progress.refresh()
          if latest_record["status"] in DONE_STATUSES_TOPONE:
              break

      progress.close()

      if parse_outputs_topone and latest_record and latest_record["status"] == "completed":
          output_path = BATCH_RESULTS_DIR_TOPONE / f"{batch_id}_output.jsonl"
          if not output_path.exists():
              output_path = silent_download_batch_output_topone(latest_record)
          if output_path is not None:
              summary = parse_batch_output_file_topone(target_year_topone, output_path)
              total_saved_topone += summary["saved"]
              total_skipped_topone += len(summary["skipped"])
              total_errors_topone += len(summary["errors"])

  if parse_outputs_topone:
      print(
          f"Totals for {target_year_topone}: saved={total_saved_topone}, "
          f"skipped={total_skipped_topone}, errors={total_errors_topone}"
      )
  if missing_requests_topone:
      print("The following top-one request files were missing and were not resubmitted:")
      for path in missing_requests_topone:
          print(f"  {path}")


# In[ ]:


# Cell 22 top-one filter – Integrated batch workflow for a selected year
from collections import Counter

target_year_topone = 2020          # ← change this to any year 2015–2024
overwrite_requests_topone = False  # set True to rebuild JSONLs from scratch
wave_size_topone = 1               # number of batch files to submit at once
poll_interval_seconds_topone = 30  # seconds between status polls
parse_outputs_topone = True        # set False to submit/monitor only
parse_existing_completed_topone = False  # parse completed batches before new submissions
skip_cancelled_batches_topone = True     # skip batches marked cancelled/cancelling
max_waves_per_run_topone = 30            # stop after this many waves

DONE_STATUSES_TOPONE = {"completed", "failed", "cancelled"}
CANCEL_STATUSES_TOPONE = {"cancelled", "cancelling"}

def count_request_lines_topone_workflow(request_path: Path) -> int:
  with request_path.open("r", encoding="utf-8") as fp:
      return sum(1 for line in fp if line.strip())

print(f"=== Top-one batch workflow for {target_year_topone} ===")

request_files_topone = build_batch_files_topone(
  target_year_topone,
  overwrite=overwrite_requests_topone,
)
print(f"{len(request_files_topone)} top-one request file(s) present in {BATCH_DIR_TOPONE}.")

status_records_topone: List[Dict[str, Any]] = []
for status_path in list_batch_status_files_topone(target_year_topone):
  try:
      status_records_topone.append(refresh_batch_status_topone(status_path))
  except Exception as exc:
      print(f"[warn] Could not refresh {status_path.name}: {exc}")

status_lookup_topone = {Path(rec["request_file"]).name: rec for rec in status_records_topone}

parsed_batches_topone: set[str] = set()
total_saved_topone = total_skipped_topone = total_errors_topone = 0

if parse_outputs_topone and parse_existing_completed_topone:
  existing_completed = [rec for rec in status_records_topone if rec.get("status") == "completed"]
  if existing_completed:
      print(f"Parsing {len(existing_completed)} previously completed top-one batch(es) before new submissions.")
      for rec in tqdm(existing_completed, desc="Parsing existing top-one", unit="batch"):
          batch_id = rec["batch_id"]
          if batch_id in parsed_batches_topone:
              continue
          output_path = download_batch_output_topone(rec)
          if output_path is None:
              print(f"[warn] Missing output for {batch_id}; skipping.")
              continue
          summary = parse_batch_output_file_topone(target_year_topone, output_path)
          parsed_batches_topone.add(batch_id)
          total_saved_topone += summary["saved"]
          total_skipped_topone += len(summary["skipped"])
          total_errors_topone += len(summary["errors"])
      print("Finished parsing previously completed top-one batches.\n")

active_statuses = {"validating", "in_progress", "processing"}
pending_files_topone: List[Path] = []
for path in request_files_topone:
  rec = status_lookup_topone.get(path.name)
  if rec is None:
      pending_files_topone.append(path)
      continue

  status = rec.get("status")
  if status == "completed":
      print(f"- {path.name}: already completed; no submission needed.")
  elif status in active_statuses:
      print(f"- {path.name}: currently {status}; skipping this run.")
  elif skip_cancelled_batches_topone and status in CANCEL_STATUSES_TOPONE:
      print(f"- {path.name}: marked {status}; skipping (delete status file to retry).")
  else:
      pending_files_topone.append(path)

if not pending_files_topone:
  print("No top-one batch request files need submission right now.")
else:
  print(f"{len(pending_files_topone)} top-one batch request file(s) pending submission.")

remaining_failed_batches_topone: List[Dict[str, Any]] = []
for wave_idx in range(0, len(pending_files_topone), wave_size_topone):
  wave_number = wave_idx // wave_size_topone + 1
  wave_paths = pending_files_topone[wave_idx : wave_idx + wave_size_topone]
  if not wave_paths:
      continue

  print("\n" + "=" * 90)
  print(f"Top-one wave {wave_number}: submitting {len(wave_paths)} batch file(s)")

  submitted_topone: Dict[str, Dict[str, Any]] = {}
  for path in wave_paths:
      status_path = BATCH_STATUS_DIR_TOPONE / f"{path.stem}_status.json"
      if status_path.exists():
          backup = status_path.with_suffix(status_path.suffix + f".retry_{int(time.time())}")
          status_path.rename(backup)

      job = submit_batch_request_topone(path)
      total_requests = count_request_lines_topone_workflow(path)

      submitted_topone[job["batch_id"]] = {
          "status_path": BATCH_STATUS_DIR_TOPONE / f"{path.stem}_status.json",
          "request_path": path,
          "total_requests": total_requests,
      }

  progress_bars_topone: Dict[str, tqdm] = {}
  completed_counts_topone: Dict[str, int] = {}
  wave_records_topone: List[Dict[str, Any]] = []

  try:
      for batch_id, meta in submitted_topone.items():
          progress_bars_topone[batch_id] = tqdm(
              total=meta["total_requests"],
              desc=f"{meta['request_path'].stem}",
              unit="paper",
              leave=False,
          )
          completed_counts_topone[batch_id] = 0

      while True:
          time.sleep(poll_interval_seconds_topone)
          wave_records_topone = []
          for batch_id, meta in submitted_topone.items():
              rec = refresh_batch_status_topone(meta["status_path"])
              wave_records_topone.append(rec)

              counts = rec.get("request_counts") or {}
              finished = (counts.get("completed") or 0) + (counts.get("failed") or 0)
              total_requests = meta["total_requests"] or 0
              finished = min(total_requests, finished)

              prev = completed_counts_topone.get(batch_id, 0)
              delta = finished - prev
              if delta > 0:
                  progress_bars_topone[batch_id].update(delta)
                  completed_counts_topone[batch_id] = finished
                  progress_bars_topone[batch_id].refresh()

          if all(rec["status"] in DONE_STATUSES_TOPONE for rec in wave_records_topone):
              break
  finally:
      for bar in progress_bars_topone.values():
          bar.close()

  if parse_outputs_topone:
      completed = [rec for rec in wave_records_topone if rec["status"] == "completed"]
      if completed:
          for rec in tqdm(completed, desc=f"Parsing top-one wave {wave_number}", unit="batch"):
              batch_id = rec["batch_id"]
              if batch_id in parsed_batches_topone:
                  continue
              output_path = download_batch_output_topone(rec)
              if output_path is None:
                  print(f"[warn] Output missing for {batch_id}; skipping parse.")
                  continue
              summary = parse_batch_output_file_topone(target_year_topone, output_path)
              parsed_batches_topone.add(batch_id)
              total_saved_topone += summary["saved"]
              total_skipped_topone += len(summary["skipped"])
              total_errors_topone += len(summary["errors"])

  failed = [rec for rec in wave_records_topone if rec["status"] == "failed"]
  if failed:
      print(f"Top-one wave {wave_number}: {len(failed)} batch(es) failed (see status files for details).")
      remaining_failed_batches_topone.extend(failed)

  if wave_number >= max_waves_per_run_topone:
      print(f"\nStopping after wave {wave_number} (max_waves_per_run={max_waves_per_run_topone}).")
      break

print("\n" + "=" * 90)
print(f"Finished top-one batch workflow for {target_year_topone}.")
print(f"  New top-one records saved this run: {total_saved_topone}")
print(f"  Top-one duplicates skipped:        {total_skipped_topone}")
print(f"  Top-one per-record errors logged:  {total_errors_topone}")

if remaining_failed_batches_topone:
  print(f"\n{len(remaining_failed_batches_topone)} top-one batch(es) still failed (likely token limit).")
  print("Rerun this cell later—after in-progress batches finish—to resubmit them.")
else:
  print("\nAll submitted top-one batches completed successfully.")


# In[ ]:


# Cell 23 top-one filter – Cancel all active top-one batch jobs for a target year
target_year_topone_cancel = 2023

active_statuses_topone = {"validating", "in_progress", "processing", "queued"}
records_topone_cancel = monitor_batches_topone(target_year_topone_cancel, download_complete=False)

active_topone = [rec for rec in records_topone_cancel if rec["status"] in active_statuses_topone]
if not active_topone:
  print(f"No active top-one batch jobs found for {target_year_topone_cancel}.")
else:
  print(f"Cancelling {len(active_topone)} active top-one batch job(s) for {target_year_topone_cancel}...")
  for rec in active_topone:
      batch_id = rec["batch_id"]
      try:
          client_topone.batches.cancel(batch_id)
          print(f"  Cancelled top-one batch {batch_id}")
      except Exception as exc:
          print(f"  [warn] Unable to cancel top-one batch {batch_id}: {exc}")

  # Refresh to confirm everything stopped
  records_topone_cancel = monitor_batches_topone(target_year_topone_cancel, download_complete=False)
  still_active_topone = [rec for rec in records_topone_cancel if rec["status"] in active_statuses_topone]
  if still_active_topone:
      print(f"[warn] {len(still_active_topone)} top-one batch(es) still report active status:")
      for rec in still_active_topone:
          print(f"  {rec['batch_id']} → {rec['status']}")
  else:
      print("All top-one batch jobs are now completed, failed, or cancelled.")


# In[ ]:


# Cell 24 top-one filter – Check whether a given year has active top-one batch jobs (silent)
check_year_topone = 2021  # change to any top-one year you want to inspect

status_files_topone = list_batch_status_files_topone(check_year_topone)
if not status_files_topone:
  print(f"{check_year_topone}: No top-one batch status files found (no batches submitted yet).")
else:
  active_statuses_topone = {"validating", "in_progress", "processing", "queued"}
  active_count_topone = 0
  for status_path in status_files_topone:
      record = refresh_batch_status_topone(status_path)  # silent refresh
      if record.get("status") in active_statuses_topone:
          active_count_topone += 1

  if active_count_topone:
      print(f"{check_year_topone}: {active_count_topone} active top-one batch job(s).")
  else:
      print(f"{check_year_topone}: No active top-one batch jobs.")


# In[ ]:


# Cell 26 top-one filter – Parse partially completed top-one batch
# If some instances failed for a given batch and you still want to parse whatever succeeded, you can use this cell.
target_year_topone_partial = 2023
batch_filename_topone = "batch_2023_topone_filtered_top_1_0056.jsonl"  # change per batch

status_path_topone = BATCH_STATUS_DIR_TOPONE / f"{Path(batch_filename_topone).stem}_status.json"
if not status_path_topone.exists():
  raise FileNotFoundError(f"Status snapshot not found: {status_path_topone}")

record_topone = json.loads(status_path_topone.read_text(encoding="utf-8"))
batch_id_topone = record_topone["batch_id"]
status_topone = record_topone["status"]
print(f"Initial top-one status: {status_topone}")

if status_topone not in DONE_STATUSES_TOPONE:
  print("Cancelling top-one batch so we can grab partial output…")
  client_topone.batches.cancel(batch_id_topone)
  status_topone = "cancelling"

def try_download_topone(rec, *, attempts=4, wait_seconds=5):
  for i in range(attempts):
      rec = refresh_batch_status_topone(status_path_topone)
      status = rec["status"]
      print(f"Attempt {i+1}: status={status}")
      if rec.get("output_file_id"):
          return download_batch_output_topone(rec, overwrite=True), rec
      time.sleep(wait_seconds)
  return None, rec

output_path_topone, record_topone = try_download_topone(record_topone)
if not output_path_topone:
  print("Top-one output file not available yet; rerun this cell later.")
else:
  summary = parse_batch_output_file_topone(target_year_topone_partial, output_path_topone)
  print(f"Saved {summary['saved']} top-one records; skipped {len(summary['skipped'])}; errors {len(summary['errors'])}")

if record_topone.get("error_file_id"):
  err_path = BATCH_RESULTS_DIR_TOPONE / f"{batch_id_topone}_errors.jsonl"
  err_path.write_bytes(client_topone.files.content(record_topone["error_file_id"]).read())
  print(f"Downloaded top-one error log to {err_path}")


# In[ ]:


# Cell 27 -- build batch files for a year
build_batch_files_topone(2020, overwrite=True)


# In[ ]:


# Cell 28 -- monitor batch files status for a year
monitor_batches_topone(2021, download_complete=False)

