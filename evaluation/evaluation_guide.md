# Evaluation Guide

## Overview

This folder contains code to evaluate model-predicted weak signals against ground-truth weak signals across 4 settings:

| Setting | Name | Method |
|---------|------|--------|
| 1 | Set-level BERTScore | Concatenate all signals per set → BERTScore on two strings |
| 2 | Set-level LLM | Give both sets to LLM → LLM returns set-level precision/recall → compute F1 |
| 3 | Signal-level BERTScore | Pairwise BERTScore F1 matrix → greedy max aggregation |
| 4 | Signal-level LLM | Per-signal binary LLM judgment → count-based P/R |

All evaluation is computed within a single `(topic, direction)` pair.

---

## File Structure

```
evaluation/
├── loaders.py           # Load GT and predicted signals from disk
├── bertscore_eval.py    # Settings 1 and 3
├── llm_set_eval.py      # Setting 2
├── llm_signal_eval.py   # Setting 4
├── run_all.py           # Main runner (use this)
├── results/             # CSV outputs saved here
└── evaluation_guide.md  # This file
```

---

## Setup

Activate the environment:
```bash
conda activate weak-signal
```

Verify bert-score is installed:
```bash
python -c "import bert_score; print('ok')"
```

If not installed:
```bash
pip install bert-score
```

Ensure your `.env` has `IKUNCODE_API_KEY` set (required for settings 2 and 4):
```
IKUNCODE_API_KEY=sk-...
```

---

## Running Evaluations

All commands should be run from the repo root:
```bash
cd /path/to/weak-signal-benchmark   # the directory you cloned this repo into
```

### Settings 1 & 3 only (BERTScore, no API needed)
```bash
python evaluation/run_all.py --settings 1 3
```

### Settings 2 & 4 only (LLM judge)
```bash
python evaluation/run_all.py --settings 2 4
```

### All 4 settings
```bash
python evaluation/run_all.py --settings 1 2 3 4 --domains aerospace mobility
```

### Specific models only
```bash
python evaluation/run_all.py --settings 1 3 --models qwen3.5_397b gpt_5_4_chat
```

### Specific domains only
```bash
python evaluation/run_all.py --settings 1 3 --domains aerospace natural_language_processing
```
You can also pass display names with quotes, for example:
```bash
python evaluation/run_all.py --settings 1 3 --domains "Mobility and Transport"
```

### One direction only
```bash
python evaluation/run_all.py --settings 1 3 --directions problem
```

### Default incremental continuation
```bash
python evaluation/run_all.py --settings 1 2 3 4 --skip-existing
```
By default, each domain keeps its own persistent master CSV at `evaluation/results/<domain>/eval_all.csv`. When you rerun evaluation, completed `(model, domain, topic, direction, setting)` rows already present in that domain's master CSV are skipped automatically.

Each run also creates per-domain timestamped CSVs for newly computed rows only:

- `evaluation/results/<domain>/eval_{timestamp}.csv`
- `evaluation/results/<domain>/eval_all.csv`

Use `--no-skip-existing` if you want to force recomputation of rows that already exist.

### Custom LLM settings
```bash
python evaluation/run_all.py --settings 2 4 \
    --judge-model gpt-5.4 \
    --base-url "https://api.ikuncode.cc/v1" \
    --n-runs 5
```

---

## Output Format

Results are saved per domain:

- `evaluation/results/<domain>/eval_{timestamp}.csv`: rows newly computed in the current run for that domain
- `evaluation/results/<domain>/eval_all.csv`: persistent master table for that domain, incrementally appended across runs

Both CSV types use these columns:

| Column | Description |
|--------|-------------|
| `model` | Prediction model name (e.g., `qwen3.5_397b`) |
| `domain` | Domain slug (e.g., `aerospace`) |
| `topic` | Topic slug (e.g., `urban_air_mobility`) |
| `direction` | `problem` or `solution` |
| `setting` | `set_bertscore`, `set_llm`, `signal_bertscore`, `signal_llm` |
| `precision` | Precision score (mean for LLM settings) |
| `recall` | Recall score (mean for LLM settings) |
| `f1` | F1 score (mean for LLM settings) |
| `precision_std` | Std of precision across runs (LLM settings only) |
| `recall_std` | Std of recall across runs (LLM settings only) |
| `f1_std` | Std of F1 across runs (LLM settings only) |
| `n_pred` | Number of predicted signals |
| `n_gt` | Number of ground truth signals |
| `n_runs` | Number of LLM judge runs (LLM settings only) |

A summary table (mean P/R/F1 per model × setting) is printed to the terminal at the end.

---

## Data Sources

- **Ground truth**: `construction/outputs/{domain}/{topic}/{direction}/result_latest.json`
  - Extracts: `data["result"]["weak_signals"][i]["signal"]`
- **Predictions**: `prediction/python/outputs/{model}/{domain}/{topic}/{direction}/2023_2024/signals_latest.json`
  - Extracts: `data["signals"]`

Available prediction models:
- `deepseek_r1_0528`
- `dr_tulu`
- `gpt_5_4_chat`
- `qwen3_30b_awq_rag`
- `qwen3.5_397b`
- `qwen3_8b_rag`
- `tongyi`

---

## Notes

- **BERTScore** uses `roberta-large` with `rescale_with_baseline=False`. The first run downloads the model (~500MB) and caches it automatically.
- **LLM settings** use `IKUNCODE_API_KEY` with `https://api.ikuncode.cc/v1` and `User-Agent: Mozilla/5.0`.
- Default `--n-runs` is 5 for LLM settings (use 10 for publication-quality results).
- `run_all.py` automatically loads `.env` from the repo root and `prediction/python/.env` if present.
- Results are written incrementally to both the per-run CSV and the per-domain `eval_all.csv`, so completed rows are preserved if a run stops midway.
