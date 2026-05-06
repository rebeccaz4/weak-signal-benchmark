# Weak-Signal Verification Pipeline

This directory implements a three-step pipeline that **verifies** weak signals produced by the construction step. It measures whether each signal was genuinely rare during 2023-2024 and grew in prominence by 2025.

## Overview

```
domain_descriptions.py    Wikipedia-sourced descriptions for each domain (used as S2 queries)
data_acquisition.py       Step 1 – Fetch papers from Semantic Scholar per (domain, year)
topic_extraction.py         Step 2 – Embed signals & abstracts, compute cosine similarity
frequency_dynamics.py     Step 3 – Compute f_early, f_later, Decline, Impact; filter
sample_matched_papers.py  (Optional) Sample matched (signal, paper) pairs for manual inspection
```

### Formulas

For each weak signal:


| Symbol    | Definition                                                                                              |
| --------- | ------------------------------------------------------------------------------------------------------- |
| `n_year`  | Number of papers in the signal's domain whose abstract similarity with the signal exceeds the threshold |
| `N_year`  | Total number of papers retrieved for that domain in year Y                                              |
| `f_early` | `(n_2023 + n_2024) / (N_2023 + N_2024)`                                                                 |
| `f_later` | `n_2025 / N_2025`                                                                                       |
| `Decline` | `f_later - f_early`                                                                                     |
| `Impact`  | `Decline * log(1 / f_early)`                                                                            |


A signal is **verified** when:

- `f_early < F_EARLY_MAX` (default 0.1) — the signal was rare in the early window
- `Decline > 0` — the signal grew from the early to later window

### Similarity matching

For each weak signal, the matching text is `signal` + `what_it_was` (both fields from `construction/outputs/*/result_latest.json`). This text is embedded alongside paper abstracts using OpenAI `text-embedding-3-large`, and cosine similarity is computed. Papers above the threshold (default 0.5) count toward `n_year`.

## Prerequisites

```bash
pip install pandas requests python-dotenv openai tqdm numpy scikit-learn matplotlib seaborn
```

Environment variables (set in `.env` or shell):


| Variable                   | Required | Description                                |
| -------------------------- | -------- | ------------------------------------------ |
| `SEMANTIC_SCHOLAR_API_KEY` | Yes      | Semantic Scholar API key (or `S2_API_KEY`) |
| `OPENAI_API_KEY`           | Yes      | OpenAI API key for embeddings              |


## How to run

Run the three steps **sequentially** from the `verification/` directory:

```bash
cd verification/

# Step 1: Fetch papers from Semantic Scholar (2023, 2024, 2025) per domain
python data_acquisition.py

# Step 2: Embed weak signals & paper abstracts, compute similarity matches
python topic_extraction.py

# Step 3: Compute frequency metrics and filter verified signals
python frequency_dynamics.py

# Optional: Sample matched (signal, paper) pairs for human review
python sample_matched_papers.py
```

### Sampling matched papers for manual inspection

After Step 2 has cached embeddings for signals and abstracts, `sample_matched_papers.py` recomputes cosine similarity, picks papers above a threshold, and randomly samples a few per signal/year so you can eyeball whether matches look sensible. Output is a single JSON file with `signal`, `what_it_was`, `similarity`, `paper_title`, `paper_abstract`.

```bash
# Default: threshold 0.5, 3 papers per (signal, year), seed 42
python sample_matched_papers.py

# Match Step 2's threshold (IMPORTANT: use the same value you ran topic_extraction.py with,
# otherwise sampled papers won't correspond to the n_year counts)
python sample_matched_papers.py --similarity-threshold 0.7

# More samples, specific domains, custom output path
python sample_matched_papers.py \
  --n-per-signal 5 \
  --domains "Aerospace" "Energy" \
  --output data/verification/matching/samples_aero_energy.json

# With a custom data dir (e.g. the domain-only experiment)
python sample_matched_papers.py \
  --data-dir paper/verification_domain_only \
  --similarity-threshold 0.7
```

> ⚠️ The default threshold here (`0.5`) differs from `topic_extraction.py`'s CLI default (`0.7`). Pass `--similarity-threshold` to keep them aligned — otherwise the sampled pairs are drawn from a different match set than the `n_year` counts in `matching_results.parquet`.

### Comparing multiple thresholds without overwriting results

Both `topic_extraction.py` and `frequency_dynamics.py` accept `--output-suffix`, so you can keep outputs for different thresholds side-by-side. Embeddings are cached and re-used regardless of threshold — only similarity comparison and metric computation re-run.

```bash
# Run at threshold 0.5 → matching_results_t0.5.parquet
python topic_extraction.py --similarity-threshold 0.5 --output-suffix _t0.5
python frequency_dynamics.py --output-suffix _t0.5
#   → verification_metrics_t0.5.parquet, verified_signals_t0.5.parquet

# Run at threshold 0.7 → matching_results_t0.7.parquet (coexists with _t0.5 files)
python topic_extraction.py --similarity-threshold 0.7 --output-suffix _t0.7
python frequency_dynamics.py --output-suffix _t0.7

# Sample from each match set into separate JSONs
python sample_matched_papers.py --similarity-threshold 0.5 \
  --output data/verification/matching/sampled_matches_t0.5.json
python sample_matched_papers.py --similarity-threshold 0.7 \
  --output data/verification/matching/sampled_matches_t0.7.json


```

The suffix is a free-form string — `_t0.5`, `_high`, `_exp1` all work. The suffix passed to `frequency_dynamics.py` must match the one used in `topic_extraction.py` so it can find the right input file.

### Trying `signal`-only embedding (without `what_it_was`)

By default the matching text is `signal + ". " + what_it_was`. The `what_it_was` field sometimes contains generic phrasing that widens recall to unrelated papers. `--signal-only` embeds just `signal` instead, using a separate cache file (`signals_<slug>_signal_only.parquet`) so your original embeddings stay intact.

```bash
# Run matching with signal-only text; keep outputs separate via suffix
python topic_extraction.py \
  --signal-only \
  --similarity-threshold 0.7 \
  --output-suffix _signal_only_t0.7 \
  --data-dir paper/verification_domain_only 

python frequency_dynamics.py --output-suffix _signal_only_t0.7

# Sample from signal-only cache for inspection
python sample_matched_papers.py \
  --signal-only \
  --similarity-threshold 0.7 \
  --output data/verification/matching/sampled_matches_signal_only.json

python topic_extraction.py --signal-only --similarity-threshold 0.7 --output-suffix _signal_only_t0.7 --domain "Natural Language Processing" --data-dir paper/verification_domain_only

python sample_matched_papers.py --signal-only --similarity-threshold 0.7 --output paper/verification/matching/sampled_matches_signal_only.json --domain "Natural Language Processing" --data-dir paper/verification_domain_only
```

Note: abstract embeddings are unaffected by this flag — only signal embeddings are recomputed (typically only a few hundred signals, so it's cheap).

### Two-stage matching with a reranker

Once you've picked a rerank threshold via `calibrate_rerank.py` (e.g. 0.8), enable `--use-reranker` on `topic_extraction.py` so `n_year` counts only papers that pass **both** cosine recall AND reranker precision.

What happens per (domain, year):

1. Cosine filter using cached embeddings → candidate papers with `cosine ≥ --similarity-threshold`
2. Cross-encoder scores each (signal, paper) pair → cached by `(signal_id, paper_id)` in `rerank_cache/rerank_<slug>_<year>[<suffix>].parquet`
3. Count only pairs with `cosine ≥ threshold AND rerank ≥ --rerank-threshold` into `n_year`
4. A `n_year_cosine` column is also written to `matching_results` for side-by-side comparison

```bash
python topic_extraction.py \
  --data-dir paper/verification_domain_only \
  --signal-only \
  --similarity-threshold 0.7 \
  --use-reranker \
  --rerank-threshold 0.8 \
  --output-suffix _rerank0.8_aug
```

**Rerank score cache.** Scores are keyed by `(signal_id, paper_id)`. Tuning `--rerank-threshold` afterwards is free — the cache is hit and only the threshold comparison re-runs. But switching `--reranker-model` or toggling `--rerank-signal-only` changes what the cache *means*, so pass `--rerank-cache-suffix` to keep separate caches (e.g. `--rerank-cache-suffix _v2_full` vs `_v2_signal_only`).

```bash
# Retune threshold without re-scoring pairs
python topic_extraction.py \
  --data-dir paper/verification_domain_only \
  --signal-only \
  --use-reranker \
  --rerank-threshold 0.7 \
  --output-suffix _rerank0.7

# Switch to a different reranker or text mode — use a fresh cache suffix
python topic_extraction.py \
  --data-dir paper/verification_domain_only \
  --signal-only \
  --use-reranker \
  --reranker-model BAAI/bge-reranker-large \
  --rerank-cache-suffix _large \
  --output-suffix _rerank_large
```

Requirements: GPU strongly recommended; `pip install sentence-transformers`.

## Command-line arguments

All three scripts support `argparse`. Use `--help` to see full options.

### `data_acquisition.py`


| Arg                              | Description                                                   |
| -------------------------------- | ------------------------------------------------------------- |
| `--domains "Aerospace" "Energy"` | Only fetch specified domains (default: all)                   |
| `--domain-only`                  | Search by domain name only, without appending the description |
| `--force`                        | Force re-fetch, backing up existing manifests                 |
| `--years 2024 2025`              | Override year range (default: 2023 2024 2025)                 |
| `--max-papers 500`               | Max papers per (domain, year) (default: unlimited)            |
| `--page-size 50`                 | S2 API page size (default: 100)                               |
| `--language English`             | Language filter (default: English)                            |
| `--data-dir path/to/dir`         | Custom output data directory                                  |
| `--list-domains`                 | Print all available domain names and exit                     |


### `topic_extraction.py`


| Arg                                    | Description                                                 |
| -------------------------------------- | ----------------------------------------------------------- |
| `--domains "Aerospace" "Energy"`       | Only process specified domains (default: all with signals)  |
| `--similarity-threshold 0.5`           | Cosine similarity threshold (default: 0.5)                  |
| `--embed-model text-embedding-3-large` | OpenAI embedding model                                      |
| `--embed-batch-size 128`               | Embedding batch size                                        |
| `--data-dir path/to/dir`               | Override data directory                                     |
| `--output-suffix _t0.5`                | Append suffix to `matching_results` filename (default: '')  |
| `--signal-only`                        | Embed only `signal` (no `what_it_was`); uses separate cache |
| `--use-reranker`                       | Two-stage pipeline: cosine recall + cross-encoder rerank    |
| `--reranker-model MODEL`               | Cross-encoder (default: `BAAI/bge-reranker-v2-m3`)          |
| `--rerank-threshold 0.8`               | Min rerank score to count a paper (default: 0.8)            |
| `--reranker-batch-size 32`             | Reranker batch size (default: 32)                           |
| `--rerank-signal-only`                 | Feed only `signal` to reranker (default: full match_text)   |
| `--rerank-cache-suffix _v2`            | Suffix for rerank score cache files                         |
| `--list-domains`                       | Print all available domain names and exit                   |


### `frequency_dynamics.py`


| Arg                      | Description                                                               |
| ------------------------ | ------------------------------------------------------------------------- |
| `--f-early-max 0.1`      | f_early upper-bound threshold (default: 0.1)                              |
| `--data-dir path/to/dir` | Override data directory                                                   |
| `--output-suffix _t0.5`  | Suffix for matching/metrics/verified files (must match topic_extraction.py) |


### `sample_matched_papers.py`


| Arg                              | Description                                                                  |
| -------------------------------- | ---------------------------------------------------------------------------- |
| `--n-per-signal 3`               | Number of matched papers to sample per signal per year (default: 3)          |
| `--similarity-threshold 0.5`     | Cosine similarity threshold (default: env `VER_SIMILARITY_THRESHOLD` or 0.5) |
| `--seed 42`                      | Random seed for sampling (default: 42)                                       |
| `--domains "Aerospace" "Energy"` | Only sample for specified domains (default: all)                             |
| `--data-dir path/to/dir`         | Override data directory                                                      |
| `--output path/to/file.json`     | Output JSON path (default: `<data-dir>/matching/sampled_matches.json`)       |
| `--signal-only`                  | Read from the signal-only embedding cache (must match topic_extraction.py)     |
| `--use-reranker`                 | Only sample papers that also pass the rerank threshold                       |
| `--rerank-threshold 0.8`         | Min rerank score (default: 0.8). Only used with `--use-reranker`.            |
| `--rerank-cache-suffix _v2`      | Must match `topic_extraction.py --rerank-cache-suffix`                         |


### Example: domain-only experiment for a single domain

Use `--domain-only` to search Semantic Scholar with domain names only (without the Wikipedia description), and `--data-dir` to keep results separate:

```bash
# Fetch with domain name only, into a separate directory
python data_acquisition.py \
  --domain-only \
  --max-papers 30000 \
  --data-dir paper/verification_domain_only

# Run matching on that data
python topic_extraction.py \
  --data-dir paper/verification_domain_only

# Compute metrics
python frequency_dynamics.py \
  --data-dir paper/verification_domain_only
```

### Output structure

```
data/verification/
  manifests/              Per-(domain, year) paper Parquet files
  raw/                    Raw S2 API JSONL logs + state files
  embedding_cache/        Cached embeddings for signals and abstracts
  matching/
    matching_results.parquet   Per-(signal, year) n_year and N_year
  matching/
    sampled_matches.json           (Optional) sampled (signal, paper) pairs for manual review
  metrics/
    verification_metrics.parquet   All signals with f_early, f_later, Decline, Impact
    verified_signals.parquet       Filtered signals passing both constraints
    top_signals.png                Bar chart of top verified signals
    domain_summary.png             Per-domain verified vs. total counts
```

## Configuration

Most parameters are now configurable via command-line arguments (see above). The following environment variables are still supported as fallback defaults or for low-level tuning:


| Variable               | Default | Description                                            |
| ---------------------- | ------- | ------------------------------------------------------ |
| `VER_THROTTLE_SECONDS` | `0.2`   | Delay between S2 API requests                          |
| `VER_REQUEST_TIMEOUT`  | `30`    | S2 API request timeout (seconds)                       |
| `VER_MAX_RETRIES`      | `5`     | Max retries on transient S2 errors                     |
| `VER_RETRY_BACKOFF`    | `1.5`   | Retry backoff multiplier                               |
| `VER_SIM_CHUNK_SIZE`   | `5000`  | Chunk size for similarity computation (memory control) |


## Notes

- **Resumable**: Both Step 1 and Step 2 cache intermediate results. Re-running skips completed work.
- **Weak signals are loaded from** `construction/outputs/**/result_latest.json`. Each signal's `signal` and `what_it_was` fields are concatenated for embedding.
- **Domain descriptions** in `domain_descriptions.py` are appended to domain names when querying S2 to improve retrieval coverage. Descriptions are sourced from Wikipedia. Use `--domain-only` in `data_acquisition.py` to search with domain names only (without descriptions).

