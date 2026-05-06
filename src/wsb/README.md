# `wsb` — Weak Signal Benchmark evaluation library

Pure-Python package that backs the four-setting evaluation. The CLI entry
points in [`scripts/`](../../scripts/) and [`evaluation/run_all.py`](../../evaluation/run_all.py)
are thin wrappers around this package.

Installed automatically by `pip install -e .` from the repo root (declared in
`pyproject.toml` under `[[tool.poetry.packages]]`).

## Subpackages

| Module | Purpose |
|---|---|
| `wsb.config` | Project-level defaults: `PROJECT_ROOT`, `DEFAULT_JUDGE_MODEL`, `DEFAULT_N_RUNS`, `DEFAULT_SEED`, `GEMINI_BASE_URL`. |
| `wsb.io.signals` | Load ground-truth and predicted weak signals from disk into uniform records. |
| `wsb.io.excel` | Aggregate run results into the per-model Excel summary used in the paper. |
| `wsb.evaluate.prompts` | The pairwise judge prompt template (`PAIRWISE_SYSTEM_PROMPT`, `build_pairwise_prompt`). |
| `wsb.evaluate.judge` | `run_evaluation(...)` — single-config LLM-as-a-judge pass with retries. |
| `wsb.evaluate.batch` | `run_batch_evaluation(...)` — multi-config sweep with per-evaluation seeds. |
| `wsb.evaluate.metrics` | `compute_metrics`, `flatten_metric_runs`, `compute_summary`. |
| `wsb.evaluate.cost` | Token / dollar accounting for OpenAI-compatible judges. |

## Minimal use

```python
from wsb.evaluate import build_pairwise_prompt, run_evaluation

prompt = build_pairwise_prompt(
    ground_truth=["signal A", "signal B"],
    predictions=["signal A'", "signal C"],
    signal_type="problem",
)
result = run_evaluation(
    judge_model="gpt-5-mini",
    prompt=prompt,
    n_runs=10,
    seed=27,
)
```

For full configuration semantics see
[`scripts/run_evaluation.py`](../../scripts/run_evaluation.py) and the
template configs in [`configs/`](../../configs/).
