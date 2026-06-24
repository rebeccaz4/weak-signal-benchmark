# Weak Signal Prediction Guide

This directory contains eight independent prediction scripts, one per model or
method, that generate weak-signal predictions (prediction only — no evaluation)
for the mainframe topics of all **13 research domains** used in the benchmark.

The prediction window is fixed at **2023-2024**, and each topic is predicted in
both the **problem-space** and the **solution-space**.

All scripts share a single prompt module (`prediction_prompts.py`) to guarantee
identical instructions across models.

## Scripts

| Script | Model / method | Local GPU |
|---|---|---|
| `DR_Tulu_eval.py` | DR-Tulu (Deep Research agent + MCP + vLLM) | Yes |
| `Tongyi_eval.py` | Tongyi DeepResearch (ReAct agent + vLLM) | Yes |
| `qwen3_8B_rag.py` | Qwen3-8B + LlamaIndex RAG | Yes |
| `qwen3_30B_rag.py` | Qwen3-30B-A3B AWQ 4-bit + LlamaIndex RAG | Yes |
| `gemini_3_flash.py` | Gemini 3 Flash (API) | No |
| `gpt_5_4_chat.py` | GPT-5.4-chat (Azure OpenAI API) | No |
| `deepseek_r1_0528.py` | DeepSeek-R1-0528 (DeepSeek API) | No |
| `qwen3_5_397b.py` | Qwen3.5-397B-A17B (DashScope API) | No |

## Domains

Topics are loaded from `../../construction/weak_signals_by_domain.json` via
`mainframe_topics.py`.

| Domain | Topics |
|---|---:|
| Advanced materials and advanced manufacturing | 38 |
| Aerospace | 11 |
| Mobility and Transport | 13 |
| Digital twins | 9 |
| Artificial intelligence & Machine learning | 25 |
| Information and Communication Technologies | 20 |
| Medical imaging | 9 |
| Therapeutics and Biotechnologies | 19 |
| e-Health | 15 |
| Environment and agriculture | 17 |
| Energy | 18 |
| Quantum and Cryptography | 28 |
| Natural Language Processing | 32 |

## Common CLI arguments

Every script accepts:

| Argument | Description | Example |
|---|---|---|
| `--domain` | Optional. One or more domain names. Omit to run all 13. | `--domain "Aerospace" "Energy"` |
| `--spaces` | Optional. `problem` / `solution`. Default: run both. | `--spaces problem` |
| `--output-dir` | Output root directory (required). | `--output-dir ./outputs` |
| `--seed` | Random seed. | `--seed 42` |

Notes:

- **Year range** is fixed at `2023-2024`; no flag is needed.
- **Skip existing**: each script checks the output directory before running and
  skips work that already has results, so interrupted runs can safely resume.
- `mainframe_topics.py` exposes `TOPICS_BY_DOMAIN` and `ALL_DOMAINS`; when
  `--domain` is not given, every domain and all of its topics are iterated.

## Output layout

All scripts write to:

```
{output-dir}/
  {model_name}/              # e.g. dr_tulu, tongyi, qwen3_8b_rag, gemini_3_flash
    {domain_slug}/           # e.g. aerospace, energy, natural_language_processing
      {topic_slug}/          # lowercase + underscore slug of the topic
        {space}/             # problem / solution
          2023_2024/
            response_latest.txt        # raw model output
            response_{timestamp}.txt   # timestamped backup
            signals_latest.json        # extracted signal list
            signals_{timestamp}.json   # timestamped backup
```

### `signals_latest.json`

```json
{
  "domain": "Aerospace",
  "space": "problem",
  "mainframe_topic": "6G in space",
  "year_range": "2023-2024",
  "timestamp": "20260327T120000Z",
  "signals": ["signal name 1", "signal name 2", "..."]
}
```

### Model JSON schema

Every prompt asks the model to return:

```json
{
  "weak_signals": [
    {
      "signal": "weak signal name",
      "what_it_was": "1-2 sentences describing what it was, including the year",
      "why_weak_signal": "1-2 sentences on why it was a weak signal for the topic",
      "references": [
        {"title": "paper title", "year": 2023, "url": "paper URL"}
      ]
    }
  ]
}
```

---

## Environment setup

Below is a clean-room setup using conda.

### 1. Create and activate the conda environment

```bash
# Python 3.10 is recommended for vLLM and llama-index compatibility
conda create -n weak-signal python=3.10 -y
conda activate weak-signal
```

### 2. Install shared dependencies

```bash
pip install python-dotenv requests
```

### 3. Install per-script extras

Install only what you need:

```bash
# GPT-5.4-chat / evaluation scripts / StepB series (require openai)
pip install openai

# Gemini 3 Flash
pip install google-genai

# Qwen3-8B RAG / Qwen3-30B RAG (require openai + llama-index)
pip install openai llama-index llama-index-embeddings-huggingface

# StepB evaluation pipeline (data processing + plotting)
pip install pandas tqdm tiktoken numpy scikit-learn matplotlib seaborn

# DR-Tulu / Tongyi (require a local GPU + vLLM)
pip install vllm
```

### 4. Clone external repositories (only for DR-Tulu / Tongyi)

```bash
cd weak-signal-benchmark

# DR-Tulu
git clone https://github.com/rlresearch/dr-tulu.git dr-tulu-main
cd dr-tulu-main && pip install -e . && cd ..

# Tongyi-DeepResearch
git clone https://github.com/QwenLM/Tongyi-DeepResearch.git Tongyi-DeepResearch-main
cd Tongyi-DeepResearch-main && pip install -e . && cd ..
```

### 5. Install everything at once (optional)

```bash
pip install \
    python-dotenv requests \
    openai google-genai \
    llama-index llama-index-embeddings-huggingface \
    pandas tqdm tiktoken numpy scikit-learn matplotlib seaborn \
    vllm
```

### 6. Configure environment variables

Each script loads `.env` via `python-dotenv`. Create `prediction/python/.env`:

```bash
cat > .env << 'EOF'
# Semantic Scholar (DR-Tulu / Tongyi / RAG scripts)
SEMANTIC_SCHOLAR_API_KEY=your_s2_key

# Gemini
GEMINI_API_KEY=your_gemini_key

# Azure OpenAI / GPT-5.4-chat
OPENAI_API_KEY=your_azure_openai_key

# DeepSeek
DEEPSEEK_API_KEY=your_deepseek_key

# DashScope (Qwen3.5)
DASHSCOPE_API_KEY=your_dashscope_key
EOF
```

> `gpt_5_4_chat.py` targets Azure OpenAI, so `OPENAI_API_KEY` must hold an
> Azure key. If a standard OpenAI key is present in `.env`, authentication can
> fail — override with `--openai-api-key` on the command line when needed.

### 7. Verify the installation

```bash
python3 -c "
import dotenv, requests, json, re, pathlib
print('OK: core deps')

try:
    import openai; print('OK: openai')
except ImportError: print('MISSING: openai')

try:
    import google.genai; print('OK: google-genai')
except ImportError: print('MISSING: google-genai')

try:
    import llama_index; print('OK: llama-index')
except ImportError: print('MISSING: llama-index')

try:
    import pandas, tqdm, numpy, sklearn; print('OK: data stack')
except ImportError: print('MISSING: data stack')
"
```

### Per-script dependency cheat-sheet

| Script | Extra dependencies |
|---|---|
| `DR_Tulu_eval.py` | `requests`; DR-Tulu repo (ships vLLM + MCP server) |
| `Tongyi_eval.py` | Tongyi-DeepResearch repo (ships vLLM) |
| `qwen3_8B_rag.py` | `openai`, `llama-index`, `llama-index-embeddings-huggingface` |
| `qwen3_30B_rag.py` | Same as `qwen3_8B_rag.py` |
| `gemini_3_flash.py` | `google-genai` |
| `gpt_5_4_chat.py` | `openai` |
| `deepseek_r1_0528.py` | `openai` |
| `qwen3_5_397b.py` | `openai` |
| `evaluate_signals.py` | `openai`, `pandas` |
| `StepB1`–`StepB4` | `openai`, `pandas`, `tqdm`, `tiktoken`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn` |

---

## Running vLLM services

These four scripts require a local vLLM server:

| Script | Model | Default port | GPUs | Default utilisation |
|---|---|---|---|---|
| `DR_Tulu_eval.py` | `rl-research/DR-Tulu-8B` | 30001 | 1 | 0.7 |
| `Tongyi_eval.py` | `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B` | 6001 | 2 (TP=2) | 0.9 |
| `qwen3_8B_rag.py` | `Qwen/Qwen3-8B` | 6003 | 1 | 0.7 |
| `qwen3_30B_rag.py` | `stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ` | 6004 | 1 (AWQ 4-bit) | 0.9 |

### Isolate GPUs with `CUDA_VISIBLE_DEVICES`

When running more than one vLLM instance on the same host, always isolate GPUs
with `CUDA_VISIBLE_DEVICES`. vLLM claims every visible GPU by default, so
omitting this will cause a later-launched server to starve an earlier one and
crash its engine core, even if the two use different ports.

### Recommended: start vLLM manually, then use `--skip-vllm-start`

Starting vLLM in its own terminal makes logs readable and simplifies
debugging. Each server gets its own shell.

**Example — DR-Tulu (GPU 0) + Qwen3-8B (GPU 1) side by side:**

```bash
# Terminal 1: DR-Tulu vLLM (GPU 0, port 30001)
conda activate vllm018
CUDA_VISIBLE_DEVICES=0 python -u -m vllm.entrypoints.openai.api_server \
    --model rl-research/DR-Tulu-8B \
    --host 127.0.0.1 --port 30001 \
    --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.8

# Terminal 2: Qwen3-8B vLLM (GPU 1, port 6003)
conda activate vllm018
CUDA_VISIBLE_DEVICES=1 python -u -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --host 127.0.0.1 --port 6003 \
    --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.6

# Terminal 3: DR-Tulu prediction (reuses the vLLM above)
conda activate weak-signal
python3 DR_Tulu_eval.py --output-dir ./outputs \
    --dr-tulu-dir "$REPO_ROOT/dr-tulu-main" \
    --skip-vllm-start

# Terminal 4: Qwen3-8B prediction (reuses the vLLM above)
conda activate weak-signal
python3 qwen3_8B_rag.py --output-dir ./outputs --skip-vllm-start
```

**Example — Tongyi with tensor parallelism (2 GPUs, port 6001):**

```bash
# Terminal 1: Tongyi vLLM (2 GPUs)
conda activate vllm018
CUDA_VISIBLE_DEVICES=0,1 python -u -m vllm.entrypoints.openai.api_server \
    --model Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
    --host 127.0.0.1 --port 6001 \
    --dtype bfloat16 --max-model-len 32768 \
    --gpu-memory-utilization 0.8 --tensor-parallel-size 2 --enforce-eager

# Terminal 2: prediction
conda activate weak-signal
python3 Tongyi_eval.py --output-dir ./outputs \
    --tongyi-dir "$REPO_ROOT/Tongyi-DeepResearch-main" \
    --skip-vllm-start
```

**Example — Qwen3-30B AWQ 4-bit (1 GPU, port 6004):**

```bash
# Terminal 1: Qwen3-30B AWQ vLLM (single GPU)
conda activate vllm018
CUDA_VISIBLE_DEVICES=0 python -u -m vllm.entrypoints.openai.api_server \
    --model stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ \
    --host 127.0.0.1 --port 6004 \
    --dtype auto --max-model-len 32768 \
    --gpu-memory-utilization 0.6 --enforce-eager

# Terminal 2: prediction
conda activate weak-signal
python3 qwen3_30B_rag.py --output-dir ./outputs --skip-vllm-start
```

### Common GPU layouts (2 × L40S 48 GB)

| Layout | GPU 0 | GPU 1 | Notes |
|---|---|---|---|
| DR-Tulu + Qwen3-8B | DR-Tulu (30001) | Qwen3-8B (6003) | One model per GPU |
| Tongyi TP=2 | Tongyi | Tongyi | Exclusive use of both GPUs |
| Qwen3-30B AWQ | Qwen3-30B AWQ (6004) | free | Single-GPU model |
| DR-Tulu solo | DR-Tulu (30001) | free | — |
| Qwen3-8B solo | Qwen3-8B (6003) | free | — |

### Troubleshooting vLLM

```bash
# Running vLLM processes
ps aux | grep "vllm.entrypoints"

# Port usage
lsof -i :6003
lsof -i :30001

# Kill a specific vLLM instance
pkill -f "vllm.entrypoints.*--port 6003"

# Kill every vLLM instance
pkill -f "vllm.entrypoints"

# GPU state
nvidia-smi
```

---

## Per-script details

### 1. `DR_Tulu_eval.py`

DR-Tulu is a Deep Research agent that uses an MCP server to query Semantic
Scholar, with vLLM for inference. The script:

1. Patches the Semantic Scholar API file to inject a year cutoff
   (`S2_CUTOFF_YEAR=2022`).
2. Starts the MCP server, the vLLM server and the DR-Tulu agent.
3. Sends prediction requests to the DR-Tulu HTTP endpoint.

**Prerequisites:**

```bash
cd weak-signal-benchmark
git clone https://github.com/rlresearch/dr-tulu.git dr-tulu-main
cd dr-tulu-main && pip install -e . && cd ..
```

**Run:**

```bash
# Every domain and both spaces
python3 DR_Tulu_eval.py \
    --output-dir ./outputs \
    --dr-tulu-dir /path/to/weak-signal-benchmark/dr-tulu-main

# Single domain
python3 DR_Tulu_eval.py \
    --domain "e-Health" \
    --output-dir ./outputs \
    --dr-tulu-dir /path/to/weak-signal-benchmark/dr-tulu-main

# Solution space only
python3 DR_Tulu_eval.py \
    --domain "Aerospace" \
    --spaces solution \
    --output-dir ./outputs \
    --dr-tulu-dir ./dr-tulu-main
```

**DR-Tulu-specific arguments:**

| Argument | Default | Description |
|---|---|---|
| `--dr-tulu-dir` | (required) | Path to the DR-Tulu repository |
| `--dr-tulu-model` | `rl-research/DR-Tulu-8B` | vLLM model name |
| `--dr-tulu-port` | 8080 | DR-Tulu agent HTTP port |
| `--vllm-port` | 30001 | vLLM server port |
| `--mcp-port` | 8000 | MCP server port |
| `--request-timeout` | 1800 | Request timeout (seconds) |
| `--gpu-memory-utilization` | 0.7 | vLLM GPU memory utilisation |
| `--max-model-len` | 32768 | vLLM max sequence length |

---

### 2. `Tongyi_eval.py`

Tongyi DeepResearch is a multi-turn ReAct agent. The script invokes
`run_multi_react.py` as a subprocess and writes outputs as JSONL. It:

1. Patches four files inside the Tongyi repo to inject the S2 cutoff and the
   correct API config.
2. Starts the vLLM server.
3. Runs ReAct inference via subprocess.

**Prerequisites:**

```bash
cd weak-signal-benchmark
git clone https://github.com/QwenLM/Tongyi-DeepResearch.git Tongyi-DeepResearch-main
cd Tongyi-DeepResearch-main && pip install -e . && cd ..
```

**Run:**

```bash
# All domains, both spaces
python3 Tongyi_eval.py \
    --output-dir ./outputs \
    --tongyi-dir /path/to/weak-signal-benchmark/Tongyi-DeepResearch-main

# Single domain
python3 Tongyi_eval.py \
    --output-dir ./outputs \
    --domain "Natural Language Processing" \
    --tongyi-dir /path/to/weak-signal-benchmark/Tongyi-DeepResearch-main
```

**Tongyi-specific arguments:**

| Argument | Default | Description |
|---|---|---|
| `--tongyi-dir` | (required) | Path to the Tongyi-DeepResearch repository |
| `--tongyi-model` | `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B` | vLLM model name |
| `--vllm-host` | `127.0.0.1` | vLLM host |
| `--vllm-port` | 6001 | vLLM port |
| `--tensor-parallel` | 2 | Tensor parallelism |
| `--max-model-len` | 32768 | Max sequence length |
| `--gpu-memory-utilization` | 0.9 | GPU memory utilisation |
| `--skip-vllm-start` | false | Reuse an already-running vLLM |
| `--temperature` | 0.7 | Sampling temperature |
| `--top-p` | 0.95 | Top-p sampling |
| `--presence-penalty` | 1.1 | Presence penalty |
| `--s2-api-key` | env `S2_API_KEY` | Semantic Scholar API key |

---

### 3. `qwen3_8B_rag.py`

Qwen3-8B with a LlamaIndex RAG pipeline: retrieve papers from Semantic Scholar,
rerank them with LlamaIndex, and inject the resulting evidence block into the
prompt.

**Install:**

```bash
pip install openai llama-index llama-index-embeddings-huggingface
```

**Start vLLM manually (recommended):**

```bash
# Terminal 1: vLLM (single GPU)
conda activate vllm018
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --host 127.0.0.1 --port 6003 \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.7

# Terminal 2: prediction
conda activate weak-signal
python3 qwen3_8B_rag.py --output-dir ./outputs --skip-vllm-start
```

**Run (auto-start vLLM):**

```bash
# All domains (topic name is used as the S2 retrieval query)
CUDA_VISIBLE_DEVICES=1 python3 qwen3_8B_rag.py --output-dir ./outputs

# Single domain
python3 qwen3_8B_rag.py \
    --domain "Natural Language Processing" \
    --output-dir ./outputs

# Custom retrieval queries (shared across topics)
python3 qwen3_8B_rag.py \
    --domain "Medical imaging" \
    --spaces solution \
    --output-dir ./outputs \
    --retrieval-queries "medical imaging deep learning" "radiology AI"
```

**Qwen3-8B RAG-specific arguments:**

| Argument | Default | Description |
|---|---|---|
| `--retrieval-queries` | topic name is used if omitted | S2 retrieval queries (nargs=+) |
| `--qwen-model` | `Qwen/Qwen3-8B` | vLLM model name |
| `--vllm-host` | `127.0.0.1` | vLLM host |
| `--vllm-port` | 6003 | vLLM port |
| `--tensor-parallel` | 1 | Tensor parallelism |
| `--temperature` | 0.7 | Sampling temperature |
| `--max-tokens` | 32768 | Max generated tokens |
| `--skip-vllm-start` | false | Reuse an already-running vLLM |
| `--rag-top-k` | 30 | Top-k papers after reranking |
| `--s2-max-total` | 10000 | S2 retrieval page cap |
| `--s2-page-size` | 100 | S2 page size |
| `--s2-api-key` | env `S2_API_KEY` | Semantic Scholar API key |

---

### 4. `qwen3_30B_rag.py`

Same pipeline as `qwen3_8B_rag.py`, but defaults to the Qwen3-30B-A3B AWQ 4-bit
quantised build (~5 GB, fits on a single GPU).

**Start vLLM manually (recommended):**

```bash
# Single GPU, AWQ 4-bit
conda activate vllm018
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ \
    --host 127.0.0.1 --port 6004 \
    --dtype auto \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.3

# In a second terminal
conda activate weak-signal
python3 qwen3_30B_rag.py --domain "Quantum and Cryptography" --output-dir ./outputs
```

**Run (auto-start vLLM):**

```bash
# All domains
python3 qwen3_30B_rag.py --output-dir ./outputs

# Single domain and space
python3 qwen3_30B_rag.py \
    --domain "Quantum and Cryptography" \
    --spaces problem \
    --output-dir ./outputs
```

**Differences vs the 8B variant:**

| Argument | 8B default | 30B default |
|---|---|---|
| `--qwen-model` | `Qwen/Qwen3-8B` | `stelterlab/Qwen3-30B-A3B-Instruct-2507-AWQ` |
| `--vllm-port` | 6003 | 6004 |
| `--tensor-parallel` | 1 | 1 |
| quantization | none | AWQ 4-bit |

---

### 5. `gemini_3_flash.py`

Pure API call to Google Gemini 3 Flash. No local GPU required.

**Install:**

```bash
pip install google-genai
```

**Run:**

```bash
# All 13 domains
python3 gemini_3_flash.py --output-dir ./outputs

# Two domains
python3 gemini_3_flash.py \
    --domain "Aerospace" "Energy" \
    --output-dir ./outputs

# Solution space only
python3 gemini_3_flash.py \
    --domain "Aerospace" \
    --spaces solution \
    --output-dir ./outputs
```

**Gemini-specific arguments:**

| Argument | Default | Description |
|---|---|---|
| `--gemini-model` | `gemini-3-flash-preview` | Gemini model name |
| `--gemini-api-key` | env `GEMINI_API_KEY` | Gemini API key |
| `--temperature` | 0.6 | Sampling temperature |
| `--max-output-tokens` | 32768 | Max output tokens |
| `--max-retries` | 4 | Max retries |
| `--retry-backoff` | 2.0 | Retry backoff factor |

---

### 6. `gpt_5_4_chat.py`

Azure OpenAI call to GPT-5.4-chat. No local GPU required.

**Install:**

```bash
pip install openai
```

**Run:**

```bash
# All 13 domains
python3 gpt_5_4_chat.py --output-dir ./outputs

# Single domain
python3 gpt_5_4_chat.py \
    --domain "Digital twins" \
    --output-dir ./outputs

# Problem space only
python3 gpt_5_4_chat.py \
    --domain "Digital twins" \
    --spaces problem \
    --output-dir ./outputs
```

**GPT-5.4-chat-specific arguments:**

| Argument | Default | Description |
|---|---|---|
| `--openai-model` | `gpt-5.4-chat` | Azure deployment name |
| `--openai-api-key` | Azure key | Azure OpenAI API key |
| `--azure-endpoint` | Azure endpoint | Azure endpoint URL |
| `--api-version` | `2025-04-01-preview` | Azure API version |
| `--temperature` | 1.0 | Sampling temperature |
| `--max-tokens` | 32768 | Max generated tokens |
| `--max-retries` | 4 | Max retries |
| `--retry-backoff` | 2.0 | Retry backoff factor |

---

### 7. `deepseek_r1_0528.py`

DeepSeek-R1-0528 through the DeepSeek API (OpenAI-compatible). No local GPU
required.

**Install:**

```bash
pip install openai
```

**Configure:** set `DEEPSEEK_API_KEY` in `.env`.

**Run:**

```bash
# All 13 domains
CUDA_VISIBLE_DEVICES=0 python3 deepseek_r1_0528.py --output-dir ./outputs

# Two domains
python3 deepseek_r1_0528.py \
    --domain "Aerospace" "Energy" \
    --output-dir ./outputs

# Problem space only
python3 deepseek_r1_0528.py \
    --domain "Natural Language Processing" \
    --spaces problem \
    --output-dir ./outputs
```

**DeepSeek-specific arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model` | `deepseek-r1-0528` | DeepSeek model name |
| `--api-key` | env `DEEPSEEK_API_KEY` | DeepSeek API key |
| `--base-url` | `https://api.deepseek.com/v1` | API base URL |
| `--temperature` | 1.0 | Sampling temperature (1.0 recommended for R1) |
| `--max-tokens` | 8192 | Max generated tokens |
| `--max-retries` | 6 | Max retries |
| `--retry-backoff` | 2.0 | Retry backoff factor |

---

### 8. `qwen3_5_397b.py`

Qwen3.5-397B-A17B through the DashScope OpenAI-compatible endpoint. No local
GPU required.

**Install:**

```bash
pip install openai
```

**Configure:** set `DASHSCOPE_API_KEY` in `.env`.

**Run:**

```bash
# All 13 domains
python3 qwen3_5_397b.py --output-dir ./outputs

# Two domains
python3 qwen3_5_397b.py \
    --domain "Quantum and Cryptography" "Natural Language Processing" \
    --output-dir ./outputs

# Solution space only
python3 qwen3_5_397b.py \
    --domain "Natural Language Processing" \
    --spaces solution \
    --output-dir ./outputs
```

**Qwen3.5-specific arguments:**

| Argument | Default | Description |
|---|---|---|
| `--model` | `qwen3.5-397b-a17b` | DashScope model name |
| `--api-key` | env `DASHSCOPE_API_KEY` | DashScope API key |
| `--base-url` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | API base URL |
| `--temperature` | 0.7 | Sampling temperature |
| `--max-tokens` | 8192 | Max generated tokens |
| `--max-retries` | 6 | Max retries |
| `--retry-backoff` | 2.0 | Retry backoff factor |

---

## Batch runs

**Simplest — omit `--domain` to iterate over all 13 domains:**

```bash
# All domains × problem + solution (~254 topics × 2 spaces = 508 runs)
python3 gpt_5_4_chat.py --output-dir ./outputs

# All domains, problem space only
python3 gemini_3_flash.py --spaces problem --output-dir ./outputs
```

**Multiple named domains:**

```bash
python3 gpt_5_4_chat.py \
    --domain "Aerospace" "Energy" "Digital twins" \
    --output-dir ./outputs
```

**Single domain, solution space only:**

```bash
python3 gemini_3_flash.py \
    --domain "Natural Language Processing" \
    --spaces solution \
    --output-dir ./outputs
```

All scripts support safe resumption: if the target output directory already
contains results, the corresponding run is skipped. Interrupted runs can be
restarted without losing progress.

## Signal spaces

| Space | Meaning |
|---|---|
| `problem` | Problem-space weak signals — previously-unknown research problems |
| `solution` | Solution-space weak signals — previously-unknown solutions or methods |
