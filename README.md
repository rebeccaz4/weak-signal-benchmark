# Weak-Signal Benchmark (BackTrend)

This repository contains the full pipeline and benchmark suite used in the
**BackTrend** study on weak-signal prediction in scientific and technological
research. It covers end-to-end construction of ground-truth weak signals from
bibliographic corpora, post-hoc verification against later publication activity,
prediction with a diverse set of LLM-based and agentic systems, and a four-way
evaluation protocol.

The dataset of verified weak signals is mirrored on Hugging Face:
[`rebeccazzzz/BackTrend`](https://huggingface.co/datasets/rebeccazzzz/BackTrend).

## Repository layout

```
weak-signal-benchmark/
├── construction/       # Step 1 — build candidate weak signals from per-domain paper sets
├── verification/       # Step 2 — verify signals via Semantic Scholar retrieval + frequency dynamics
├── prediction/         # Step 3 — model predictions (agents, RAG, API-only LLMs)
├── evaluation/         # Step 4 — four evaluation settings (set/signal × BERTScore/LLM)
├── ablation/           # Ablation experiments (retrieval budget, human vs judge kappa)
├── src/wsb/            # `wsb` package: LLM-as-a-judge evaluation library
├── scripts/            # CLI entry points (run_evaluation.py, prompt experiments)
├── tools/              # Utility scripts (output integrity checks, reports)
├── configs/            # YAML evaluation configs (templates per model)
├── benchmark_before/   # Legacy signal source materials (pptx/docx decks)
├── pyproject.toml      # Python project + dependencies
└── Makefile            # Shorthand for install / test / evaluate
```

## Pipeline overview

The end-to-end pipeline runs in four ordered stages:

```
construction  →  verification  →  prediction  →  evaluation
```

1. **Construction** (`construction/`) — for each of 13 domains, fetch the
   relevant mainframe topics and generate candidate weak signals grouped by
   `problem` / `solution` direction. Outputs per-topic
   `result_latest.json` files that form the ground truth used downstream.
2. **Verification** (`verification/`) — for each signal, retrieve Semantic
   Scholar papers per domain per year, embed signal text with
   `text-embedding-3-large`, and compute how often the signal matches abstracts
   in early (2023–2024) vs later (2025) windows. A signal is **verified** when
   it was rare early on and grew later. See
   [`verification/README.md`](verification/README.md) for the full formula.
3. **Prediction** (`prediction/python/`) — run eight models / methods
   (DR-Tulu agent, Tongyi DeepResearch, Qwen3 RAG (8B / 30B), Gemini 3 Flash,
   GPT-5.4, DeepSeek-R1, Qwen3.5-397B) in parallel on the verified topics
   for the 2023–2024 window. See
   [`prediction/python/prediction_guide.md`](prediction/python/prediction_guide.md).
4. **Evaluation** (`evaluation/`) — score every model prediction against the
   ground-truth signals under four settings: set-level BERTScore, set-level
   LLM judgement, signal-level BERTScore, signal-level LLM judgement. See
   [`evaluation/evaluation_guide.md`](evaluation/evaluation_guide.md).

`ablation/` adds orthogonal experiments: varying the retrieval / rollout budget
on a fixed 50-topic sample, and measuring Cohen's kappa between human and LLM
judges.

## Installation

Requires **Python 3.10+** (3.10 is the version pinned by the `weak-signal`
conda environment used in the paper, which is needed for vLLM and llama-index
compatibility — see [`prediction/python/prediction_guide.md`](prediction/python/prediction_guide.md)).

```bash
pip install -e ".[dev]"
```

Core dependencies include: `pandas`, `pyarrow`, `scikit-learn`, `openai`,
`tiktoken`, `rapidfuzz`, `aiohttp`, `requests`, `matplotlib`, `seaborn`,
`jupyterlab`. The prediction scripts add model-specific extras; see the
per-script cheat sheet in `prediction/python/prediction_guide.md`.

Or use the Makefile shortcut:

```bash
make install                                    # pip install -e ".[dev]"
make evaluate CONFIG=configs/dr_tulu.yaml       # run a single eval config
make evaluate-all                               # run scripts/run_all.sh
```

### External agent codebases (only for DR-Tulu / Tongyi prediction)

Two predictors run as external agent frameworks. Clone them at the repo root
*before* running the matching prediction scripts. Both directories are
gitignored so they live alongside the rest of the repo without bloating it:

```bash
cd weak-signal-benchmark

# DR-Tulu (used by prediction/python/DR_Tulu_eval.py)
git clone https://github.com/rlresearch/dr-tulu.git dr-tulu-main
pip install -e ./dr-tulu-main

# Tongyi-DeepResearch (used by prediction/python/Tongyi_eval.py and ablation/tongyi_budget_runner.py)
git clone https://github.com/QwenLM/Tongyi-DeepResearch.git Tongyi-DeepResearch-main
pip install -e ./Tongyi-DeepResearch-main
```

Both repositories ship their own vLLM dependency. See
[`prediction/python/prediction_guide.md`](prediction/python/prediction_guide.md)
for GPU placement, port assignments, and patch behaviour.

## Data

The verified weak-signal benchmark is published on Hugging Face:
[`rebeccazzzz/BackTrend`](https://huggingface.co/datasets/rebeccazzzz/BackTrend).

```bash
# Option A — Hugging Face CLI
pip install -U "huggingface_hub[cli]"
huggingface-cli download rebeccazzzz/BackTrend \
    --repo-type dataset --local-dir ./data/backtrend

# Option B — datasets library
python -c "from datasets import load_dataset; load_dataset('rebeccazzzz/BackTrend')"
```

Place verification outputs (`signal_results.parquet`, `verified_signals.json`,
etc.) under `verification/data/` if you want to skip re-running stages 1-3.
Construction ground-truth files
(`construction/outputs/<domain>/<topic>/<direction>/result_latest.json`)
must exist locally for evaluation and ablation scripts; reproduce them with
`python construction/run_all_domains.py` or download from the Hugging Face
release.

## Environment variables

Create a `.env` in the repo root (or in `prediction/python/`) with the keys
needed by the components you intend to run:

```bash
# Paper retrieval (used by verification/ and by RAG / agentic predictors)
SEMANTIC_SCHOLAR_API_KEY=...

# LLM embeddings and judges
OPENAI_API_KEY=...

# Optional, per-prediction-script
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=...
DASHSCOPE_API_KEY=...

# Optional, for the signal-level LLM judge used in setting 4
IKUNCODE_API_KEY=...
```

No secrets are committed — `.env` is gitignored.

## Running components

### Construction

```bash
python construction/run_all_domains.py
# or, topic-scoped:
python construction/run_topic_weak_signals.py --domain "Aerospace"
```

Outputs land under `construction/outputs/` (gitignored).

### Verification

Run the three stages sequentially from `verification/`:

```bash
cd verification/
python data_acquisition.py      # 1. Fetch S2 papers per (domain, year)
python topic_extraction.py        # 2. Embed signals and abstracts, score similarity
python frequency_dynamics.py    # 3. Compute f_early, f_later, Decline, Impact
```

Optional: `python sample_matched_papers.py` samples matched
`(signal, paper)` pairs for manual inspection.

### Prediction

See [`prediction/python/prediction_guide.md`](prediction/python/prediction_guide.md)
for per-model setup (vLLM placement, GPU isolation, API keys). Example:

```bash
cd prediction/python/
python3 gpt_5_4_chat.py --output-dir ./outputs
python3 qwen3_8B_rag.py --domain "Aerospace" --output-dir ./outputs
```

### Evaluation

The `wsb` package drives the four-setting evaluation from YAML configs. Pick
a tracked template under `configs/` (or copy to `configs/local_*.yaml` for
your overrides — those are gitignored), then:

```bash
# Full sweep from a config
python scripts/run_evaluation.py --config configs/dr_tulu.yaml

# Single evaluation
python scripts/run_evaluation.py --config configs/dr_tulu.yaml --eval-index 0

# Override defaults
python scripts/run_evaluation.py --config configs/dr_tulu.yaml --n-runs 5 --batch-size 3
```

Minimal YAML:

```yaml
model_name: dr_tulu
judge_model: gpt-5-mini
temperature: 1.0
n_runs: 10
seed: 27
output_xlsx: Evaluations/llm_judge_summary_dr_tulu.xlsx
evaluations:
  - signal_type: problem
    year_bucket: 2020-2022
    topic: Reward Type - Process or Outcome
    ground_truth: data/ground_truth/problem/2020-2022/reward_type_process_or_outcome.json
    external: data/external/dr_tulu/problem/2020-2022/reward_type_process_or_outcome.json
```

The four settings live in `evaluation/`:

| Setting | Name | Method |
|---|---|---|
| 1 | Set-level BERTScore | Concatenate signals → BERTScore on two strings |
| 2 | Set-level LLM | LLM returns set-level precision/recall → compute F1 |
| 3 | Signal-level BERTScore | Pairwise BERTScore F1 matrix → greedy max |
| 4 | Signal-level LLM | Per-signal binary LLM match → count-based P/R |

Run the full evaluation sweep with `python evaluation/run_all.py`. See
[`evaluation/evaluation_guide.md`](evaluation/evaluation_guide.md).

### Ablation

Experiment 1 (retrieval/rollout budget on 50 sampled topics) and experiment 3
(human vs LLM judge agreement via Cohen's kappa) are documented in
[`ablation/README.md`](ablation/README.md). Example:

```bash
python -m ablation.topic_sampler --target-total 50 --seed 42
python -m ablation.rag_budget_runner --family qwen3_8b_rag --top-k 10 30 50
python -m ablation.evaluate_budget_variants
python -m ablation.compute_kappa
```

## Data policy

Large artefacts and run-time outputs are never committed:

- `verification/paper/` — Semantic Scholar dumps, parquet caches, embeddings
- `verification/data/` — intermediate data
- `construction/outputs/`, `construction/logs/`
- `prediction/python/outputs/`
- `evaluation/results/`, `tools/reports/`
- `ablation/outputs/`, `ablation/results/`, `ablation/annotation_pack/`, `ablation/data/`
- `configs/`, all `*.log`, `validation_output.txt`

See `.gitignore` for the complete list. Upload logic for the public dataset
lives in `verification/export_signal_results.py`.

## Reproducibility

- Every stage is a self-contained Python entry point — no notebook state.
- Network access is required for Semantic Scholar and LLM API calls.
- API snapshots and model versions drift over time; record the exact model
  identifiers (`GPT-5.4`, `DeepSeek-R1-0528`, etc.) when reporting results.
- The verified signals released on Hugging Face are frozen to the version used
  in the paper.

## License

Released under the [MIT License](LICENSE).
