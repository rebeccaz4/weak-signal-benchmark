#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Cell 1 – Imports, paths, and core config
from __future__ import annotations

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = next(
  (p for p in Path.cwd().resolve().parents if (p / "README.md").exists()),
  Path.cwd().resolve(),
)

PROBLEM_DIR = PROJECT_ROOT / "Problem-space signals"
SOLUTION_DIR = PROJECT_ROOT / "Solution-space signals"
APPLICATION_DIR = PROJECT_ROOT / "Application-focused signals"

print(f"Project root:     {PROJECT_ROOT}")
print(f"Problem-space:    {PROBLEM_DIR} | exists={PROBLEM_DIR.exists()}")
print(f"Solution-space:   {SOLUTION_DIR} | exists={SOLUTION_DIR.exists()}")
print(f"Application:      {APPLICATION_DIR} | exists={APPLICATION_DIR.exists()}")

SEED = int(os.getenv("BWD_LLM_JUDGE_SEED", "27"))
random.seed(SEED)

SAMPLES_PER_SOURCE = 5
print(f"Seed: {SEED}")
print(f"Samples per source: {SAMPLES_PER_SOURCE}")


# # Problem-Space Signals Evaluations

# ## 2020-2022

# In[21]:


# Cell 2 – Problem-space (2020–2022)
problem_2020_2022 = [
    "Inverse‑scaling strong prior failures where bigger models can’t override their own prior",
    "Knowledge-guided scientific review generation with oracle pre-training",
    "Learning from rationales – when optimizing explanations diverges from optimizing answers",
    "RLHF-trained helpful and harmless assistants and their side effects on general NLP capabilities",
    "Document-level scientific information extraction for structured experimental representations"
]

print(f"Problem-space 2020–2022 selections: {len(problem_2020_2022)}")
for item in problem_2020_2022:
  print(f"  - {item}")


# In[24]:


# Cell 3 – Dimensions + BERTrend/WISDOM pools + seeded sampling
dimensions = {
  "actionability": "Actionability is the extent to which people in the relevant field(s) can take specific next-step research action(s) after reading the signals.",
  "specificity": "Specificity is the extent to which people in the relevant field(s) can understand what detailed problem(s) or research direction(s) warrant further investigation.",
  "novelty": "Novelty is the extent to which the signal represents a genuinely emerging or non-obvious research direction.",
}

# BERTrend example topics from paper (7)
bertrend_pool = [
  "coronavirus / virus (NYT)",
  "impeachment",
  "Taal volcano",
  "American football teams",
  "attention models / transformers",
  "ImageNet",
  "LSTM / RNNs",
]

# WISDOM ground-truth topics (18)
wisdom_pool = [
  "WSNs, Data and Applications",
  "Canadian Navy",
  "Underwater Fish Tracking",
  "Australia’s Anti-Submarine Warfare Program",
  "Maritime Aircraft",
  "Oceanography",
  "China’s Military Modernization",
  "Underwater Imaging",
  "Mobile Anchor in USNs",
  "Sea Floor Characterization",
  "Coap Congestion Control Scheme for UWSNs",
  "Energy Efficiency in UASNs",
  "Restoration and Enhancement of Underwater Image",
  "Low-Power UWSNs",
  "SDRT",
  "Submarine Laser Based Remote Imaging and Tracking",
  "Subsea Cable",
  "Sensor Fault Detection",
]

# Seeded sampling (seed already set in Cell 1)
bertrend_sample = random.sample(bertrend_pool, k=5)
wisdom_sample = random.sample(wisdom_pool, k=5)

print("Problem-space 2020–2022 (your pipeline):")
for s in problem_2020_2022:
  print(f"  - {s}")

print("\nBERTrend sample (5):")
for s in bertrend_sample:
  print(f"  - {s}")

print("\nWISDOM sample (5):")
for s in wisdom_sample:
  print(f"  - {s}")


# In[25]:


# Cell 4 – Build evaluation payload + specialty-agnostic prompt templates
eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": problem_2020_2022,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample,
  },
}

SYSTEM_PROMPT = """You are a frontier researcher that is familiar with all specialty fields and have the highest IQ in the world. 
You are also a careful, fair, and specialty-agnostic evaluator of research signals.
You must judge signals without favoring any domain; treat all fields as equally important.
Focus on the general qualities of each signal rather than your familiarity with its topic area.
"""

USER_PROMPT_TEMPLATE = """You will evaluate research signals from three different sources.
Your scoring must be specialty-agnostic: do NOT reward or penalize a signal based on domain familiarity or perceived prestige.
Instead, judge each signal on general clarity and usefulness for advancing research in its own field.

Scoring dimensions (1–10):
1) Actionability: {actionability}
2) Specificity: {specificity}
3) Novelty: {novelty}

Return ONLY valid JSON with this schema:
{{
"results": [
  {{
    "source": "<source_key>",
    "signal": "<signal text>",
    "scores": {{
      "actionability": <1-10 int>,
      "specificity": <1-10 int>,
      "novelty": <1-10 int>
    }},
    "rationale": "<one concise sentence>"
  }}
]
}}

Signals to evaluate:
{signals_block}
"""

def format_signals_block(payload: dict) -> str:
  lines = []
  for key, group in payload.items():
      lines.append(f"[{key}] {group['label']}")
      for idx, sig in enumerate(group["signals"], start=1):
          lines.append(f"  {idx}. {sig}")
      lines.append("")  # blank line between groups
  return "\n".join(lines).strip()

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[4]:


# Cell 5 – OpenAI client + robust judge with debug + fallback retry
from openai import OpenAI
import time
import re

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
  raise RuntimeError("Missing OPENAI_API_KEY. Please set it in your environment.")

MODEL_ID = os.getenv("BWD_LLM_JUDGE_MODEL", "gpt-5-mini")
TEMPERATURE = float(os.getenv("BWD_LLM_JUDGE_TEMPERATURE", "1"))
MAX_COMPLETION_TOKENS = int(os.getenv("BWD_LLM_JUDGE_MAX_TOKENS", "3000"))
MAX_RETRIES = int(os.getenv("BWD_LLM_JUDGE_MAX_RETRIES", "4"))
RETRY_BACKOFF = float(os.getenv("BWD_LLM_JUDGE_RETRY_BACKOFF", "2.0"))

client = OpenAI()

def _safe_json_loads(text: str) -> Optional[dict]:
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

def _extract_debug(resp) -> str:
  try:
      choice = resp.choices[0] if resp.choices else None
      finish = getattr(choice, "finish_reason", None) if choice else None
      msg = getattr(choice, "message", None) if choice else None
      refusal = getattr(msg, "refusal", None) if msg else None
      return f"finish_reason={finish}, refusal={refusal}"
  except Exception:
      return "debug_unavailable"

def run_judge_once(system_prompt: str, user_prompt: str) -> dict:
  attempt = 0
  while True:
      attempt += 1
      try:
          resp = client.chat.completions.create(
              model=MODEL_ID,
              messages=[
                  {"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt},
              ],
              temperature=TEMPERATURE,
              max_completion_tokens=MAX_COMPLETION_TOKENS,
              response_format={"type": "json_object"},
          )

          if not resp.choices:
              raise ValueError("No choices returned from model.")

          text = (resp.choices[0].message.content or "").strip()
          if not text:
              dbg = _extract_debug(resp)
              raise ValueError(f"Empty response content from model ({dbg}).")

          payload = _safe_json_loads(text)
          if payload is None:
              raise ValueError("Could not parse JSON from model response.")
          return payload

      except Exception as exc:
          if attempt >= MAX_RETRIES:
              raise RuntimeError(f"LLM judge failed after {attempt} attempts: {exc}") from exc
          sleep_s = RETRY_BACKOFF * attempt
          print(f"[warn] LLM error (attempt {attempt}): {exc}. Retrying in {sleep_s:.1f}s...")
          time.sleep(sleep_s)

print(f"Model: {MODEL_ID} | temperature={TEMPERATURE} | max_tokens={MAX_COMPLETION_TOKENS}")


# In[5]:


from tqdm.auto import tqdm


# In[27]:


# Cell 6 – Run judge 5 times with progress bar (no saving)

N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[28]:


# Cell 7 – Flatten raw runs into a table
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[29]:


# Cell 8 – Normalize source labels + compute mean/std by group
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower()
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

# Mean/std across ALL runs per group and dimension
summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[31]:


# Cell 9 – Save summary table to Excel (Problem-space 2020-2022)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Problem-space 2020-2022"

# Write new workbook with this sheet
with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 2023-2024

# In[32]:


# Cell 10 – Problem-space (2023–2024)
problem_2023_2024 = [
    "Recasting preference alignment as direct optimization without an RL loop",
    "User‑pleasing sycophancy as an RLHF‑induced misalignment pattern",
    "Task taxonomies that decouple long context from mere retrieval difficulty",
    "Designing co-writing assistance around explicit scaffolding levels rather than generic suggestions",
    "Treating foundation-model parametric knowledge as an explicit hypothesis‑mining substrate"
]

print(f"Problem-space 2023–2024 selections: {len(problem_2023_2024)}")
for item in problem_2023_2024:
  print(f"  - {item}")


# In[33]:


# Cell 11 – Problem-space 2023–2024: sample BERTrend/WISDOM + build payload/prompt
# New random samples (RNG state continues from prior sampling)
bertrend_sample_2 = random.sample(bertrend_pool, k=5)
wisdom_sample_2 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": problem_2023_2024,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_2,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_2,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[34]:


# Cell 12 – Run judge 5 times (Problem-space 2023–2024)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[35]:


# Cell 13 – Flatten raw runs into a table (Problem-space 2023–2024)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[36]:


# Cell 14 – Summary stats by group (Problem-space 2023–2024)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower()
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[37]:


# Cell 15 – Append summary to Excel (Problem-space 2023-2024)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"
sheet_name = "Problem-space 2023-2024"

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# # Solution-Space Signals Evaluations

# ## 2020-2022

# In[2]:


# Cell 16 – Solution-space (2020–2022)
solution_2020_2022 = [
    "Answer-validated self-training for chain-of-thought reasoning in language models",
    "KL‑regularized RLHF objectives that tie policies to a reference language model",
    "Reward-model judges trained from pairwise human preferences",
    "Frozen-language-model interface via image prefix embeddings",
    "Unsupervised contrastive pretraining for domain-general dense retrievers"
]

print(f"Solution-space 2020–2022 selections: {len(solution_2020_2022)}")
for item in solution_2020_2022:
  print(f"  - {item}")


# In[7]:


# Cell 17 – Solution-space 2020–2022: sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_3 = random.sample(bertrend_pool, k=5)
wisdom_sample_3 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": solution_2020_2022,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_3,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_3,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[11]:


# Cell 18 – Run judge 5 times (Solution-space 2020–2022)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[12]:


# Cell 19 – Flatten raw runs into a table (Solution-space 2020–2022)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[17]:


# Cell 20 – Summary stats by group (Solution-space 2020–2022) [fixed normalization]
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower()
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[19]:


# Cell 21 – Append summary to Excel (Solution-space 2020-2022)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"
sheet_name = "Solution-space 2020-2022"

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 2023-2024

# In[7]:


# Cell 22 – Solution-space (2023–2024)
solution_2023_2024 = [
    "LLM-generated executable dense reward functions for sparse environments",
    "PPO-style continual RLHF across evolving domains (CPPO)",
    "Training-free supernet-style subnet search for efficient LLMs",
    "Functional‑kernel families for relative positional embeddings",
    "Training‑free hierarchical context merging of prompt chunks inside the transformer"
]

print(f"Solution-space 2023–2024 selections: {len(solution_2023_2024)}")
for item in solution_2023_2024:
  print(f"  - {item}")


# In[8]:


# Cell 23 – Solution-space 2023–2024: sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_4 = random.sample(bertrend_pool, k=5)
wisdom_sample_4 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": solution_2023_2024,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_4,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_4,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[11]:


# Cell 24 – Run judge 5 times (Solution-space 2023–2024)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[12]:


# Cell 25 – Flatten raw runs into a table (Solution-space 2023–2024)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[13]:


# Cell 26 – Summary stats by group (Solution-space 2023–2024)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")  # remove brackets if present
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[15]:


# Cell 27 – Append summary to Excel (Solution-space 2023-2024)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"
sheet_name = "Solution-space 2023-2024"

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# # Application-Focused Signals Evaluations

# ## 1. Artificial Intelligence of Things

# In[17]:


# Cell 28 – Application-focused (Artificial Intelligence of Things)
app_aiot = [
    "Cloud-edge assisted visual SLAM for resource limited robots",
    "AIoT license plate recognition for tolling and enforcement",
    "resource-efficient embedded speech emotion recognition",
    "Edge AIoT real-time landslide monitoring and early warning",
    "AIoT UAV and robotic systems for landmine detection",
]

print(f"Application-focused 'Artificial Intelligence of Things' selections: {len(app_aiot)}")
for item in app_aiot:
  print(f"  - {item}")


# In[18]:


# Cell 29 – Application-focused (Artificial Intelligence of Things): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_5 = random.sample(bertrend_pool, k=5)
wisdom_sample_5 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_aiot,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_5,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_5,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[19]:


# Cell 30 – Run judge 5 times (Application-focused: Artificial Intelligence of Things)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[20]:


# Cell 31 – Flatten raw runs into a table (Application-focused: AIoT)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[21]:


# Cell 32 – Summary stats by group (Application-focused: AIoT)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[25]:


# Cell 33 – Append summary to Excel (Application-focused) with sheet-name truncation
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Artificial Intelligence of Things"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 2. Attention Mechanisms in CNN

# In[26]:


# Cell 34 – Application-focused (Attention Mechanisms in CNN)
app_attention_cnn = [
    "Attention-guided CNN-RNN hybrids for stock time-series forecasting",
    "Swin attention-enhanced YOLO backbones for detection",
    "Attention-enhanced CNN with bidirectional RNN for weather and power forecasting",
    "Attention-enhanced convolutional models for EEG seizure analysis",
    "Attention-guided multiple instance learning for medical image analysis"
]

print(f"Application-focused 'Attention Mechanisms in CNN' selections: {len(app_attention_cnn)}")
for item in app_attention_cnn:
  print(f"  - {item}")


# In[27]:


# Cell 35 – Application-focused (Attention Mechanisms in CNN): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_6 = random.sample(bertrend_pool, k=5)
wisdom_sample_6 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_attention_cnn,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_6,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_6,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[28]:


# Cell 36 – Run judge 5 times (Application-focused: Attention Mechanisms in CNN)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[29]:


# Cell 37 – Flatten raw runs into a table (Attention Mechanisms in CNN)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[30]:


# Cell 38 – Summary stats by group (Attention Mechanisms in CNN)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[31]:


# Cell 39 – Append summary to Excel (Attention Mechanisms in CNN)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Attention Mechanisms in CNN"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 3. Decentralized Federated Learning

# In[32]:


# Cell 40 – Application-focused (Decentralized Federated Learning)
app_decentralized_fl = [
    "Decentralized federated learning for radio source management",
    "Byzantine-resilient aggregation and optimization for heterogeneous federated/decentralized systems",
    "Adaptive participant selection, incentives, and fairness in federated learning"
]

print(f"Application-focused 'Decentralized Federated Learning' selections: {len(app_decentralized_fl)}")
for item in app_decentralized_fl:
  print(f"  - {item}")


# In[33]:


# Cell 41 – Application-focused (Decentralized Federated Learning): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_7 = random.sample(bertrend_pool, k=5)
wisdom_sample_7 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_decentralized_fl,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_7,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_7,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[34]:


# Cell 42 – Run judge 5 times (Application-focused: Decentralized Federated Learning)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[35]:


# Cell 43 – Flatten raw runs into a table (Decentralized Federated Learning)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 13})")


# In[36]:


# Cell 44 – Summary stats by group (Decentralized Federated Learning)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[37]:


# Cell 45 – Append summary to Excel (Decentralized Federated Learning)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Decentralized Federated Learning"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 4. Epistemic AI

# In[38]:


# Cell 46 – Application-focused (Epistemic AI)
app_epistemic_ai = [
    "Estimating epistemic model uncertainty with MC dropout",
    "Calibrated, distribution-free conformal methods for predictive uncertainty",
    "AI epistemic constraints on creativity and novel research",
    "Posterior uncertainty calibration and propagation for earthquake-source inversion",
    "Epistemic-uncertainty-driven human-in-the-loop data acquisition",
]

print(f"Application-focused 'Epistemic AI' selections: {len(app_epistemic_ai)}")
for item in app_epistemic_ai:
  print(f"  - {item}")


# In[39]:


# Cell 47 – Application-focused (Epistemic AI): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_8 = random.sample(bertrend_pool, k=5)
wisdom_sample_8 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_epistemic_ai,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_8,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_8,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[40]:


# Cell 48 – Run judge 5 times (Application-focused: Epistemic AI)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[41]:


# Cell 49 – Flatten raw runs into a table (Epistemic AI)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[42]:


# Cell 50 – Summary stats by group (Epistemic AI)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[43]:


# Cell 51 – Append summary to Excel (Epistemic AI)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Epistemic AI"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 5. Evolutionary Neural Architecture Search

# In[6]:


# Cell 52 – Application-focused (Evolutionary Neural Architecture Search)
app_evolutionary_nas = [
    "Evolutionary neural architecture search for PV power forecasting",
    "Reinforcement learning-guided QNN architecture design",
    "Swarm-based optimization for ANN architecture and hyperparameter search",
    "Evolutionary design of compact, robust interpretable trees",
    "Neural intrusion detectors evolved with genetic algorithms"
]

print(f"Application-focused 'Evolutionary Neural Architecture Search' selections: {len(app_evolutionary_nas)}")
for item in app_evolutionary_nas:
  print(f"  - {item}")


# In[7]:


# Cell 53 – Application-focused (Evolutionary Neural Architecture Search): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_9 = random.sample(bertrend_pool, k=5)
wisdom_sample_9 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_evolutionary_nas,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_9,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_9,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[8]:


# Cell 54 – Run judge 5 times (Application-focused: Evolutionary NAS)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[9]:


# Cell 55 – Flatten raw runs into a table (Evolutionary NAS)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[10]:


# Cell 56 – Summary stats by group (Evolutionary NAS)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[12]:


# Cell 57 – Append summary to Excel (Evolutionary Neural Architecture Search)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Evolutionary Neural Architecture Search"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 6. Explainable AI

# In[13]:


# Cell 58 – Application-focused (Explainable AI)
app_explainable_ai = [
    "Boruta-driven feature importance for model interpretability",
    "Explainable credit-risk models for underserved microfinance clients",
    "Human-centered personalized diet and health recommender",
    "Concrete mix variable importance for compressive strength",
    "SHAP-driven exaplainability and feature selection for venture success"
]

print(f"Application-focused 'Explainable AI' selections: {len(app_explainable_ai)}")
for item in app_explainable_ai:
  print(f"  - {item}")


# In[14]:


# Cell 59 – Application-focused (Explainable AI): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_10 = random.sample(bertrend_pool, k=5)
wisdom_sample_10 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_explainable_ai,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_10,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_10,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[15]:


# Cell 60 – Run judge 5 times (Application-focused: Explainable AI)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[16]:


# Cell 61 – Flatten raw runs into a table (Explainable AI)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[17]:


# Cell 62 – Summary stats by group (Explainable AI)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[18]:


# Cell 63 – Append summary to Excel (Explainable AI)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Explainable AI"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 7. Federated Deep Learning

# In[19]:


# Cell 64 – Application-focused (Federated Deep Learning)
app_federated_deep_learning = [
    "Federated DNNs for RSRP estimation and handover optimization",
    "Decentralized federated learning for radio resource management"
]

print(f"Application-focused 'Federated Deep Learning' selections: {len(app_federated_deep_learning)}")
for item in app_federated_deep_learning:
  print(f"  - {item}")


# In[20]:


# Cell 65 – Application-focused (Federated Deep Learning): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_11 = random.sample(bertrend_pool, k=5)
wisdom_sample_11 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_federated_deep_learning,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_11,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_11,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[21]:


# Cell 66 – Run judge 5 times (Application-focused: Federated Deep Learning)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[22]:


# Cell 67 – Flatten raw runs into a table (Federated Deep Learning)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 12})")


# In[23]:


# Cell 68 – Summary stats by group (Federated Deep Learning)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[24]:


# Cell 69 – Append summary to Excel (Federated Deep Learning)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Federated Deep Learning"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 8. Federated Machine Learning

# In[25]:


# Cell 70 – Application-focused (Federated Machine Learning)
app_federated_ml = [
    "Cost-aware customized federated models for energy demand forecasting",
    "Byzantine-resilient aggregation and optimization for heterogeneous federated/decentralized systems",
    "Client-side gradient reconstruction threats in federated ML",
    "Membership privacy leakage in federated and transfer learning",
    "Privacy-preserving federated SVM anomaly detection"
]

print(f"Application-focused 'Federated Machine Learning' selections: {len(app_federated_ml)}")
for item in app_federated_ml:
  print(f"  - {item}")


# In[26]:


# Cell 71 – Application-focused (Federated Machine Learning): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_12 = random.sample(bertrend_pool, k=5)
wisdom_sample_12 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_federated_ml,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_12,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_12,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[27]:


# Cell 72 – Run judge 5 times (Application-focused: Federated Machine Learning)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[28]:


# Cell 73 – Flatten raw runs into a table (Federated Machine Learning)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[29]:


# Cell 74 – Summary stats by group (Federated Machine Learning)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[30]:


# Cell 75 – Append summary to Excel (Federated Machine Learning)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Federated Machine Learning"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 9. Human AI Interface

# In[31]:


# Cell 76 – Application-focused (Human AI Interface)
app_human_ai_interface = [
    "Human-AI collaborative summarization for scientific and clinical texts",
    "Mobile human-AI interface for remote aquarium monitoring and control",
    "Human teleoperation interface for dual-arm bimanual manipulation",
    "Real-time facial-emotion-aware music personalization interface",
    "Human-centered personalized skincare recommendation interfaces"
]

print(f"Application-focused 'Human AI Interface' selections: {len(app_human_ai_interface)}")
for item in app_human_ai_interface:
  print(f"  - {item}")


# In[32]:


# Cell 77 – Application-focused (Human AI Interface): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_13 = random.sample(bertrend_pool, k=5)
wisdom_sample_13 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_human_ai_interface,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_13,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_13,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[33]:


# Cell 78 – Run judge 5 times (Application-focused: Human AI Interface)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[34]:


# Cell 79 – Flatten raw runs into a table (Human AI Interface)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[35]:


# Cell 80 – Summary stats by group (Human AI Interface)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[36]:


# Cell 81 – Append summary to Excel (Human AI Interface)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Human AI Interface"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 10. Human Centric AI

# In[37]:


# Cell 82 – Application-focused (Human Centric AI)
app_human_centric_ai = [
    "Human-AI interfaces and collaborative adoption in logistics",
    "E-commerce customer experience sentiment modeling",
    "Human-centered trustworthy gradient boosting for individualized obesity risk",
    "Assessing readability and acccessibility of LLM-authored patient materials",
    "Human-centered AI shaping consumer preferences and purchases"
]

print(f"Application-focused 'Human Centric AI' selections: {len(app_human_centric_ai)}")
for item in app_human_centric_ai:
  print(f"  - {item}")


# In[38]:


# Cell 83 – Application-focused (Human Centric AI): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_14 = random.sample(bertrend_pool, k=5)
wisdom_sample_14 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_human_centric_ai,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_14,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_14,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[39]:


# Cell 84 – Run judge 5 times (Application-focused: Human Centric AI)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[40]:


# Cell 85 – Flatten raw runs into a table (Human Centric AI)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[41]:


# Cell 86 – Summary stats by group (Human Centric AI)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[42]:


# Cell 87 – Append summary to Excel (Human Centric AI)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Human Centric AI"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 11. Large Language Models

# In[43]:


# Cell 88 – Application-focused (Large Language Models)
app_large_language_models = [
    "Benchmarking LLM healthcare chatbots for clinical advice and education",
    "LLM-extracted financial sentiment for market prediction and trading",
    "Multimodal vision-language controllers for embodied robotics",
    "Fine-tuned transformer LMs for text-based depression detection",
    "LLM-driven document summarization, evaluation, and derivative content generation",
]

print(f"Application-focused 'Large Language Models' selections: {len(app_large_language_models)}")
for item in app_large_language_models:
  print(f"  - {item}")


# In[44]:


# Cell 89 – Application-focused (Large Language Models): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_15 = random.sample(bertrend_pool, k=5)
wisdom_sample_15 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_large_language_models,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_15,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_15,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[45]:


# Cell 90 – Run judge 5 times (Application-focused: Large Language Models)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[46]:


# Cell 91 – Flatten raw runs into a table (Large Language Models)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[47]:


# Cell 92 – Summary stats by group (Large Language Models)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[48]:


# Cell 93 – Append summary to Excel (Large Language Models)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Large Language Models"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 12. Masked Face Recognition

# In[50]:


# Cell 94 – Application-focused (Masked Face Recognition)
app_masked_face_recognition = [
    "Haar-cascade based mask-wearing classifier"
]

print(f"Application-focused 'Masked Face Recognition' selections: {len(app_masked_face_recognition)}")
for item in app_masked_face_recognition:
  print(f"  - {item}")


# In[51]:


# Cell 95 – Application-focused (Masked Face Recognition): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_16 = random.sample(bertrend_pool, k=5)
wisdom_sample_16 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_masked_face_recognition,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_16,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_16,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[52]:


# Cell 96 – Run judge 5 times (Application-focused: Masked Face Recognition)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[53]:


# Cell 97 – Flatten raw runs into a table (Masked Face Recognition)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 11})")


# In[54]:


# Cell 98 – Summary stats by group (Masked Face Recognition)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[55]:


# Cell 99 – Append summary to Excel (Masked Face Recognition)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Masked Face Recognition"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 13. Masked Language Model

# In[56]:


# Cell 100 – Application-focused (Masked Language Model)
app_masked_language_model = [
    "Cloze-trained BERT models for Bangla text classification",
    "RoBERTa masked-language models for multilingual clinical EHR NLP",
    "Masked-LM initialized relation extraction, including biomedical n-ary events",
    "Fine-tuned BERT family masked-models for downstream text tasks",
    "Contrastive Siamese fine-tuning of masked LMs for sentence vectors"
]

print(f"Application-focused 'Masked Language Model' selections: {len(app_masked_language_model)}")
for item in app_masked_language_model:
  print(f"  - {item}")


# In[57]:


# Cell 101 – Application-focused (Masked Language Model): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_17 = random.sample(bertrend_pool, k=5)
wisdom_sample_17 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_masked_language_model,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_17,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_17,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[58]:


# Cell 102 – Run judge 5 times (Application-focused: Masked Language Model)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[59]:


# Cell 103 – Flatten raw runs into a table (Masked Language Model)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[60]:


# Cell 104 – Summary stats by group (Masked Language Model)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[61]:


# Cell 105 – Append summary to Excel (Masked Language Model)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Masked Language Model"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 14. Multimodal AI

# In[62]:


# Cell 106 – Application-focused (Multimodal AI)
app_multimodal_ai = [
    "Multimodal AI sensing for food freshness, quality, contamination",
    "Multimodal reinforcement learning for human-centered portfolio optimization",
    "Chest X-ray and clinical data multimodal diagnosis for TB and pneumonia",
    "Multimodal AI deepfakes and synthetic media for political manipulation",
    "Multimodal cardiac deep learning from ECG, imaging, and clinical data"
]

print(f"Application-focused 'Multimodal AI' selections: {len(app_multimodal_ai)}")
for item in app_multimodal_ai:
  print(f"  - {item}")


# In[63]:


# Cell 107 – Application-focused (Multimodal AI): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_18 = random.sample(bertrend_pool, k=5)
wisdom_sample_18 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_multimodal_ai,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_18,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_18,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[64]:


# Cell 108 – Run judge 5 times (Application-focused: Multimodal AI)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[65]:


# Cell 109 – Flatten raw runs into a table (Multimodal AI)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[66]:


# Cell 110 – Summary stats by group (Multimodal AI)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[67]:


# Cell 111 – Append summary to Excel (Multimodal AI)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Multimodal AI"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 15. Multimodal Hate Speech Detection

# In[6]:


# Cell 112 – Application-focused (Multimodal Hate Speech Detection)
app_multimodal_hate_speech = [
    "Multimodal abuse and hate detection across media and networks"
]

print(f"Application-focused 'Multimodal Hate Speech Detection' selections: {len(app_multimodal_hate_speech)}")
for item in app_multimodal_hate_speech:
  print(f"  - {item}")


# In[7]:


# Cell 113 – Application-focused (Multimodal Hate Speech Detection): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_19 = random.sample(bertrend_pool, k=5)
wisdom_sample_19 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_multimodal_hate_speech,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_19,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_19,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[8]:


# Cell 114 – Run judge 5 times (Application-focused: Multimodal Hate Speech Detection)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[9]:


# Cell 115 – Flatten raw runs into a table (Multimodal Hate Speech Detection)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 11})")


# In[10]:


# Cell 116 – Summary stats by group (Multimodal Hate Speech Detection)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[11]:


# Cell 117 – Append summary to Excel (Multimodal Hate Speech Detection)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Multimodal Hate Speech Detection"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 16. Privacy-Preserving Machine Learning

# In[12]:


# Cell 118 – Application-focused (Privacy-Preserving Machine Learning)
app_privacy_preserving_ml = [
    "Privacy-aware autoencoder embeddings for secure data sharing",
    "On-device and federated private recommendation models",
    "Privacy-enhanced biometric authentication, template protection, and decentralized verification",
    "Privacy-aware federated forecasting of cross-silo traffic and load",
    "Differential privacy for commercial analytics and fraud detection"
]

print(f"Application-focused 'Privacy-Preserving Machine Learning' selections: {len(app_privacy_preserving_ml)}")
for item in app_privacy_preserving_ml:
  print(f"  - {item}")


# In[13]:


# Cell 119 – Application-focused (Privacy-Preserving ML): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_20 = random.sample(bertrend_pool, k=5)
wisdom_sample_20 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_privacy_preserving_ml,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_20,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_20,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[14]:


# Cell 120 – Run judge 5 times (Application-focused: Privacy-Preserving ML)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[15]:


# Cell 121 – Flatten raw runs into a table (Privacy-Preserving ML)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[16]:


# Cell 122 – Summary stats by group (Privacy-Preserving ML)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[17]:


# Cell 123 – Append summary to Excel (Privacy-Preserving ML)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Privacy-Preserving ML"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 17. Scientific Machine Learning

# In[18]:


# Cell 124 – Application-focused (Scientific Machine Learning)
app_scientific_ml = [
    "ML prediction of mechanical properties for concrete and cementitious materials",
    "AI-enhanced data assimilation for numerical weather prediction",
    "Neural network-based forward and inverse kinematics for robot arms",
    "ML biomarker and prognostic-signature discovery in IBD and GI cancers",
    "Physics-constrained neural solvers for scientific and engineering PDEs"
]

print(f"Application-focused 'Scientific Machine Learning' selections: {len(app_scientific_ml)}")
for item in app_scientific_ml:
  print(f"  - {item}")


# In[19]:


# Cell 125 – Application-focused (Scientific ML): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_21 = random.sample(bertrend_pool, k=5)
wisdom_sample_21 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_scientific_ml,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_21,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_21,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[20]:


# Cell 126 – Run judge 5 times (Application-focused: Scientific ML)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[21]:


# Cell 127 – Flatten raw runs into a table (Scientific ML)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[22]:


# Cell 128 – Summary stats by group (Scientific ML)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[23]:


# Cell 129 – Append summary to Excel (Scientific ML)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Scientific Machine Learning"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 18. Self Supervised Learning CNN

# In[24]:


# Cell 130 – Application-focused (Self Supervised Learning CNN)
app_self_supervised_cnn = [
    "Self-supervision for few-shot image adaptation",
    "Self-supervised CNNs for visual traversability prediction and mapping",
    "Attention-augmented convolutional autoencoder for acoustic anomaly detection",
    "No-reference image quality evaluation using self-supervised CNNs",
    "Self-supervised convolutional models for on-device visual anomaly detection"
]

print(f"Application-focused 'Self Supervised Learning CNN' selections: {len(app_self_supervised_cnn)}")
for item in app_self_supervised_cnn:
  print(f"  - {item}")


# In[25]:


# Cell 131 – Application-focused (Self Supervised Learning CNN): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_22 = random.sample(bertrend_pool, k=5)
wisdom_sample_22 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_self_supervised_cnn,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_22,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_22,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[26]:


# Cell 132 – Run judge 5 times (Application-focused: Self Supervised Learning CNN)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[27]:


# Cell 133 – Flatten raw runs into a table (Self Supervised Learning CNN)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[28]:


# Cell 134 – Summary stats by group (Self Supervised Learning CNN)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[29]:


# Cell 135 – Append summary to Excel (Self Supervised Learning CNN)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Self Supervised Learning CNN"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 19. Tiny Machine Learning

# In[30]:


# Cell 136 – Application-focused (Tiny Machine Learning)
app_tiny_ml = [
    "TinyML visual surface-defect inspection on constrained devices",
    "Tiny-YOLO lightweight UAV vehicle and obstacle detection",
    "Distilling large vision language models into compact 2D/3D architectures",
    "Real-time edge YOLO-based pothole detection and segmentation",
    "On-device TinyML for closed-loop implant neural modulation"
]

print(f"Application-focused 'Tiny Machine Learning' selections: {len(app_tiny_ml)}")
for item in app_tiny_ml:
  print(f"  - {item}")


# In[31]:


# Cell 137 – Application-focused (Tiny Machine Learning): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_23 = random.sample(bertrend_pool, k=5)
wisdom_sample_23 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_tiny_ml,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_23,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_23,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[32]:


# Cell 138 – Run judge 5 times (Application-focused: Tiny Machine Learning)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[33]:


# Cell 139 – Flatten raw runs into a table (Tiny Machine Learning)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[34]:


# Cell 140 – Summary stats by group (Tiny Machine Learning)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[35]:


# Cell 141 – Append summary to Excel (Tiny Machine Learning)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Tiny Machine Learning"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 20. Trustworthy AI

# In[36]:


# Cell 142 – Application-focused (Trustworthy AI)
app_trustworthy_ai = [
    "Ethical trust and transparency standards for AI virtual influencers",
    "Fair, reliable AI for real estate valuations and transactions",
    "AI influence on clinician-patient trust and empathy",
    "LLM safety, reliability, and ethics evaluation in high-risk sectors",
    "Stakeholder trust and adoption of ML clinical decision-support"
]

print(f"Application-focused 'Trustworthy AI' selections: {len(app_trustworthy_ai)}")
for item in app_trustworthy_ai:
  print(f"  - {item}")


# In[37]:


# Cell 143 – Application-focused (Trustworthy AI): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_24 = random.sample(bertrend_pool, k=5)
wisdom_sample_24 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_trustworthy_ai,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_24,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_24,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[38]:


# Cell 144 – Run judge 5 times (Application-focused: Trustworthy AI)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[39]:


# Cell 145 – Flatten raw runs into a table (Trustworthy AI)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[40]:


# Cell 146 – Summary stats by group (Trustworthy AI)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[41]:


# Cell 147 – Append summary to Excel (Trustworthy AI)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Trustworthy AI"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# ## 21. Trustworthy Machine Learning

# In[42]:


# Cell 148 – Application-focused (Trustworthy Machine Learning)
app_trustworthy_ml = [
    "Robust trustworthy ensemble methods for transactional fraud detection",
    "Trustworthy stacked-ensemble models for landslide susceptibility mapping",
    "Human-centered trustworthy gradient boosting for individualized obesity risk",
    "Trustworthy ML ensembles for power-grid stability prediction",
    "Isolation forest anomaly detection for operational system monitoring"
]

print(f"Application-focused 'Trustworthy Machine Learning' selections: {len(app_trustworthy_ml)}")
for item in app_trustworthy_ml:
  print(f"  - {item}")


# In[43]:


# Cell 149 – Application-focused (Trustworthy Machine Learning): sample BERTrend/WISDOM + build payload/prompt
bertrend_sample_25 = random.sample(bertrend_pool, k=5)
wisdom_sample_25 = random.sample(wisdom_pool, k=5)

eval_payload = {
  "your_pipeline": {
      "label": "Your backward pipeline (ML/NLP)",
      "signals": app_trustworthy_ml,
  },
  "bertrend": {
      "label": "BERTrend (news/social science)",
      "signals": bertrend_sample_25,
  },
  "wisdom": {
      "label": "WISDOM (underwater sensing)",
      "signals": wisdom_sample_25,
  },
}

signals_block = format_signals_block(eval_payload)

user_prompt = USER_PROMPT_TEMPLATE.format(
  actionability=dimensions["actionability"],
  specificity=dimensions["specificity"],
  novelty=dimensions["novelty"],
  signals_block=signals_block,
)

print("Signals block preview:\n")
print(signals_block)


# In[44]:


# Cell 150 – Run judge 5 times (Application-focused: Trustworthy Machine Learning)
N_RUNS = 5
raw_runs = []

for i in tqdm(range(1, N_RUNS + 1), desc="LLM judge runs"):
  result = run_judge_once(SYSTEM_PROMPT, user_prompt)
  raw_runs.append(result)

print(f"Collected {len(raw_runs)} runs.")
print(f"Last run items: {len(raw_runs[-1].get('results', [])) if raw_runs else 0}")


# In[45]:


# Cell 151 – Flatten raw runs into a table (Trustworthy Machine Learning)
def flatten_runs(runs: list) -> pd.DataFrame:
  rows = []
  for run_idx, payload in enumerate(runs, start=1):
      for rec in payload.get("results", []):
          rows.append(
              {
                  "run": run_idx,
                  "source": rec.get("source"),
                  "signal": rec.get("signal"),
                  "actionability": rec.get("scores", {}).get("actionability"),
                  "specificity": rec.get("scores", {}).get("specificity"),
                  "novelty": rec.get("scores", {}).get("novelty"),
                  "rationale": rec.get("rationale"),
              }
          )
  return pd.DataFrame(rows)

results_df = flatten_runs(raw_runs)

print(results_df.head(10))
print(f"\nRows: {len(results_df)} (expected {N_RUNS * 15})")


# In[46]:


# Cell 152 – Summary stats by group (Trustworthy Machine Learning)
def normalize_source(src: str) -> str:
  if src is None:
      return "unknown"
  s = str(src).lower().strip()
  s = s.strip("[]")
  if s.startswith("your_pipeline"):
      return "your_pipeline"
  if s.startswith("bertrend") or s.startswith("bertrand"):
      return "bertrend"
  if s.startswith("wisdom"):
      return "wisdom"
  return "unknown"

results_df["source_group"] = results_df["source"].apply(normalize_source)

summary = (
  results_df
  .groupby("source_group")[["actionability", "specificity", "novelty"]]
  .agg(["mean", "std"])
  .round(3)
)

print(summary)


# In[47]:


# Cell 153 – Append summary to Excel (Trustworthy Machine Learning)
from pathlib import Path

EVAL_DIR = PROJECT_ROOT / "Evaluations"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_XLSX = EVAL_DIR / "llm_judge_summary.xlsx"

sheet_name = "Trustworthy Machine Learning"
if len(sheet_name) > 31:
  sheet_name = sheet_name[:31]

if EVAL_XLSX.exists():
  mode = "a"
else:
  mode = "w"

with pd.ExcelWriter(EVAL_XLSX, engine="openpyxl", mode=mode, if_sheet_exists="replace") as writer:
  summary.to_excel(writer, sheet_name=sheet_name)

print(f"Saved summary to {EVAL_XLSX} | sheet='{sheet_name}'")


# In[ ]:




