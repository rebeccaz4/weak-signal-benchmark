# Weak-Signal Ablation

This directory contains non-invasive wrappers for ablation experiments.

## Scope
- Experiment 1: retrieval / rollout budget ablation on a fixed 50-topic benchmark
- Experiment 3: human vs LLM judge agreement with `Cohen's kappa`
- Extra note: "Web Search rounds vs DeepResearch quality" is not yet a standalone script here; see the evaluation design section below

## Rules
- No existing benchmark files are modified.
- New predictions are written to `ablation/outputs/`.
- New evaluation tables and summaries are written to `ablation/results/`.

## Main Scripts
- `topic_sampler.py`: build the proportionally sampled 50-topic benchmark list.
- `build_sample50_bundle.py`: build a sample20-style GT/prediction bundle.
- `rag_budget_runner.py`: run or reuse `top-k=10/30/50` RAG variants for `qwen3_8b_rag` and `qwen3_30b_awq_rag`.
- `tongyi_budget_runner.py`: run Tongyi `google_scholar tool-call budget = 0/1/3/8` variants on the labeled 20-topic benchmark.
- `evaluate_budget_variants.py`: score ablation outputs with the original metrics.
- `summarize_budget_results.py`: aggregate experiment 1 results into budget-level tables.
- `judge_replay_runner.py`: replay prediction-level LLM judge labels.
- `compute_kappa.py`: compute `Cohen's kappa`.

## Before You Start

Run all commands from the repo root:

```bash
cd /path/to/weak-signal-benchmark   # the directory you cloned this repo into
```

### Minimum prerequisites

1. Activate the Python environment used by the prediction scripts.
2. Make sure the base benchmark assets already exist:
   - `construction/outputs/.../result_latest.json`
   - `prediction/python/outputs/.../signals_latest.json`
3. Make sure `.env` is configured:
   - `prediction/python/.env`: at least `SEMANTIC_SCHOLAR_API_KEY`
   - `ablation/.env`: optional override for ablation scripts
   - for LLM evaluation / judge replay: `IKUNCODE_API_KEY`
4. If you will run Tongyi ablation, make sure `Tongyi-DeepResearch-main/` is cloned and installable.
5. If you will run local vLLM models, make sure the target GPU and port are available.

For a full environment setup, refer to `prediction/python/prediction_guide.md`.

## Artifacts Produced By Each Step

- `ablation/data/topic_benchmark_50.json`: the fixed 50-topic benchmark list
- `ablation/data/topic_benchmark_50.md`: a human-readable summary of the benchmark list
- `ablation/data/sampled_50_topics_gt_vs_all_models.json`: GT + original model predictions bundle
- `ablation/outputs/*`: new ablation predictions
- `ablation/results/budget_eval_all.csv`: raw per-topic evaluation table
- `ablation/results/budget_summary.csv`: aggregated experiment 1 summary
- `ablation/results/judge_raw_decisions.jsonl`: replayed LLM judge binary decisions
- `ablation/results/cohen_kappa.json`: final experiment 3 metric

## Experiment 1

Experiment 1 compares budget variants on the same fixed 50-topic benchmark:

- RAG models: `top-k = 10 / 30 / 50`
- Tongyi: `google_scholar tool-call budget = 0 / 1 / 3 / 8` on the labeled 20-topic benchmark

### Step 1: Build the fixed 50-topic benchmark

```bash
python -m ablation.topic_sampler --target-total 50 --seed 42
```

Expected outputs:

- `ablation/data/topic_benchmark_50.json`
- `ablation/data/topic_benchmark_50.md`

This step fails if some domain does not have enough topics with GT plus all required baseline model outputs.

### Step 2: Run Qwen3-8B RAG budget variants

```bash
python -m ablation.rag_budget_runner \
  --family qwen3_8b_rag \
  --top-k 10 30 50
```

Expected outputs:

- `ablation/outputs/qwen3_8b_rag_k10/...`
- `ablation/outputs/qwen3_8b_rag_k30/...`
- `ablation/outputs/qwen3_8b_rag_k50/...`

Notes:

- `top-k=30` is reused from existing baseline outputs when available.
- Missing variants are generated under `ablation/outputs/`.
- Existing output directories are skipped automatically.

### Step 3: Run Qwen3-30B-AWQ RAG budget variants

```bash
python -m ablation.rag_budget_runner \
  --family qwen3_30b_awq_rag \
  --top-k 10 30 50
```

Expected outputs:

- `ablation/outputs/qwen3_30b_awq_rag_k10/...`
- `ablation/outputs/qwen3_30b_awq_rag_k30/...`
- `ablation/outputs/qwen3_30b_awq_rag_k50/...`

If you already have a 30B vLLM server running, add `--skip-vllm-start`.

### Step 4: Run Tongyi google_scholar-budget variants

```bash
python -m ablation.tongyi_budget_runner \
  --tongyi-dir ./Tongyi-DeepResearch-main
```

Expected outputs:

- `ablation/outputs/tongyi_gsb0/...`
- `ablation/outputs/tongyi_gsb1/...`
- `ablation/outputs/tongyi_gsb3/...`
- `ablation/outputs/tongyi_gsb8/...`

Notes:

- Tongyi uses `ablation/data/topic_benchmark_tongyi_20.json`.
- Qwen RAG experiments continue using `ablation/data/topic_benchmark_50.json`.
- The wrapper patches Tongyi to use only `google_scholar` plus `PythonInterpreter`, disables generic web search tools, and enforces a per-run `google_scholar` tool-call budget.
- Query counts are saved as auxiliary analysis metadata and are not the control variable.

### Step 5: Evaluate all generated variants

Run the original four evaluation settings:

```bash
python -m ablation.evaluate_budget_variants \
  --settings 1 2 3 4
```

Expected outputs:

- `ablation/results/budget_eval_all.csv`
- `ablation/results/budget_eval_summary.md`

Important:

- Settings `2` and `4` require `IKUNCODE_API_KEY` or `--api-key`.
- The script reads predictions from `ablation/outputs/`.

### Step 6: Aggregate experiment 1 tables

```bash
python -m ablation.summarize_budget_results
```

Expected outputs:

- `ablation/results/budget_summary.csv`
- `ablation/results/budget_summary.md`

These are the final experiment 1 tables you would usually cite in the paper.

## Experiment 3

Experiment 3 measures agreement between human labels and LLM-as-a-judge labels using `Cohen's kappa`.

### Step 1: Reuse the curated 20-topic human judgments

Use the already finished unified human labels:

Required file:

- `evaluation/results/sampled_20_topics_gt_vs_all_models_zh_judged_unified.json`

This file is the source of truth for human labels in experiment 3.

### Step 2: Reuse the matching 20-topic GT/prediction bundle

Use the original sample20 bundle that matches the unified judgments:

Required file:

- `evaluation/results/sampled_20_topics_gt_vs_all_models.json`

This bundle contains the GT plus original model predictions that the machine judge will replay against.

### Step 3: Replay LLM judge labels at prediction level

```bash
python -m ablation.judge_replay_runner \
  --bundle-json evaluation/results/sampled_20_topics_gt_vs_all_models.json \
  --judge-model gpt-5.4 \
  --api-key "$IKUNCODE_API_KEY"
```

Expected output:

- `ablation/results/judge_raw_decisions.jsonl`

This step produces one binary judge label per prediction, which is what `compute_kappa.py` needs.

### Step 4: Compute final Cohen's kappa

```bash
python -m ablation.compute_kappa \
  --human-judgments-json evaluation/results/sampled_20_topics_gt_vs_all_models_zh_judged_unified.json \
  --judge-jsonl ablation/results/judge_raw_decisions.jsonl
```

Expected outputs:

- `ablation/results/cohen_kappa.json`
- `ablation/results/cohen_kappa.md`

These are the final experiment 3 results.

## Full Command Checklist

If you want the shortest from-zero command list, it is:

```bash
cd /path/to/weak-signal-benchmark

python -m ablation.topic_sampler --target-total 50 --seed 42

python -m ablation.rag_budget_runner --family qwen3_8b_rag --top-k 10 30 50
python -m ablation.rag_budget_runner --family qwen3_30b_awq_rag --top-k 10 30 50
python -m ablation.tongyi_budget_runner --tongyi-dir ./Tongyi-DeepResearch-main --rollout-count 1 3 5

python -m ablation.evaluate_budget_variants --settings 1 2 3 4
python -m ablation.summarize_budget_results

python -m ablation.judge_replay_runner --bundle-json evaluation/results/sampled_20_topics_gt_vs_all_models.json --judge-model gpt-5.4 --api-key "$IKUNCODE_API_KEY"
python -m ablation.compute_kappa --human-judgments-json evaluation/results/sampled_20_topics_gt_vs_all_models_zh_judged_unified.json --judge-jsonl ablation/results/judge_raw_decisions.jsonl
```

## How To Evaluate "Web Search Rounds" For DeepResearch

### The core issue

If the agent does not expose a hard parameter like `max_search_rounds`, then "Web Search rounds" is not a controlled budget variable. In that case, you should not claim a strict causal ablation of "set rounds = 1/3/5". What you can evaluate is:

- observed search usage vs output quality
- or a new patched budget variable that explicitly limits search calls

### What is controllable today in this repo

- Tongyi ablation now controls `google_scholar` tool-call budget, while query counts are retained only as auxiliary analysis metadata.
- RAG ablation controls retrieval `top-k`, not search-call count.
- DR-Tulu currently fixes `search_tool_name=s2-only` and disables browse, but does not expose a `max_search_rounds` flag in this ablation directory.

So if your question is specifically "Web Search rounds influence on DeepResearch quality", the current codebase supports only an observational analysis unless you patch the agent.

### Recommended evaluation design when you cannot control search count

1. Freeze everything else:
   - same benchmark topics
   - same model
   - same prompt
   - same year cutoff
   - same evaluation metric

2. Record the actual search usage for each sample:
   - for DR-Tulu, inspect `prediction/python/outputs/dr_tulu/.../signals_latest.json`
   - some outputs already contain `metadata.total_tool_calls`
   - because `use_browse_agent=false` and the search tool is `s2-only`, this is a useful proxy for search effort, though it may still include non-search tool calls
   - for Tongyi, parse `tongyi_runner_outputs/**/iter1.jsonl` or patch the wrapper to count `google_scholar` tool calls and save that count into the final JSON

3. Turn actual counts into analysis buckets:
   - `0-1`
   - `2-3`
   - `4-5`
   - `6+`

4. Evaluate quality within each bucket using the same metrics as experiment 1:
   - `set_bertscore`
   - `set_llm`
   - `signal_bertscore`
   - `signal_llm`

5. Report both quality and cost:
   - mean F1 per bucket
   - average observed search calls per bucket
   - marginal gain from extra search usage

### What to call this experiment

If the agent decides its own search count, name the experiment something like:

- "observed search usage vs quality"
- "actual tool-call count vs quality"
- "search-effort analysis"

Do not call it a strict "search rounds ablation" unless you actually enforce the budget.

### If you want a true causal ablation

You need to patch the agent wrapper so each run enforces a hard cap, for example:

- `max_google_scholar_calls = 1 / 3 / 5`
- or `max_total_tool_calls = 1 / 3 / 5`

Then rerun the same fixed benchmark and evaluate exactly as in experiment 1.

In other words:

- no hard cap -> observational analysis
- hard cap -> true ablation
