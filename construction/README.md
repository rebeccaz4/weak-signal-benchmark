# Construction (Step 1)

Generate ground-truth weak-signal candidates for each `(domain, topic, direction)`
triple. Outputs feed every downstream stage (verification, prediction,
evaluation, ablation).

## Inputs

- `weak_signals_by_domain.json` — committed source of truth listing every
  domain and its mainframe topics. Edit this file to add or remove topics
  before running construction.

## Entry points

| Script | Scope |
|---|---|
| `run_all_domains.py` | Sequential over domains; topics inside each domain run in parallel. Use this for a full sweep. |
| `run_domain_weak_signals.py` | One domain end-to-end (all topics × `problem` / `solution`). |
| `run_topic_weak_signals.py` | A single `(domain, topic, direction)` cell — useful for retries. |
| `direction_prompts.py` | Shared prompts (`problem` / `solution`) used by every runner. |
| `validate_reference_titles.py` | Sanity-check that referenced paper titles in `result_latest.json` exist on Semantic Scholar. |
| `fill_empty_references.py` | Backfill missing `references` arrays after a partial run. |
| `backfill_nlp.py` | One-off backfill targeted at the NLP domain. |

All runners share the same output layout described below; missing cells are
generated and complete cells are skipped, so interrupted runs resume safely.

## Output layout

```
construction/outputs/
  <domain_slug>/
    <topic_slug>/
      problem/result_latest.json
      problem/result_<timestamp>.json
      solution/result_latest.json
      solution/result_<timestamp>.json
construction/logs/
  <domain_slug>/<topic_slug>__<direction>.log
```

Each `result_latest.json` carries `metadata` (domain, topic, direction, year
range, prompt and model identifiers) and a `result.weak_signals` list of
`{signal, what_it_was, why_weak_signal, references}` entries.

## Typical commands

```bash
# Full sweep (skip cells that are already complete)
python construction/run_all_domains.py --skip-existing

# Single domain
python construction/run_domain_weak_signals.py --domain "Aerospace"

# Single topic (single direction)
python construction/run_topic_weak_signals.py \
    --domain "Aerospace" --topic "Hyperloop transportation" --direction problem
```

## Required environment variables

- `OPENAI_API_KEY` (or the routed key configured via `--base-url`) — the
  weak-signal generator runs through an OpenAI-compatible endpoint.
- `IKUNCODE_API_KEY` if you keep the default `--base-url
  https://api.ikuncode.cc/v1`.
