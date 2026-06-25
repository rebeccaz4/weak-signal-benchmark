# Construction v2

This folder contains the weak-signal construction pipeline. The example below runs the pipeline for the target topic `large language models`.

## Requirements

Install the project dependencies from the repository root:

```bash
pip install -e .
```

The scripts read optional credentials from environment variables or `construction_v2/.env`:

```bash
SEMANTIC_SCHOLAR_API_KEY=...
OPENAI_API_KEY=...
```

`SEMANTIC_SCHOLAR_API_KEY` is used for paper retrieval and reference-adoption matching. `OPENAI_API_KEY` is required only for OpenAI-based candidate extraction or semantic clustering.

## Pipeline

Run all commands from the repository root.

### 1. Fetch target-topic papers

Fetch the 2019-2024 paper pools for the target topic:

```bash
python construction_v2/scripts/fetch_papers.py \
  --topic "large language models" \
  --years 2019 2020 2021 2022 2023 2024
```

Outputs:

```text
construction_v2/papers/large-language-models/papers_large-language-models_{year}.parquet
```

### 2. Extract candidate topics

Extract problem-space and solution-space candidate topics from the 2019-2023 papers:

```bash
python construction_v2/scripts/extract_candidate.py \
  --topic "large language models" \
  --years 2019 2020 2021 2022 2023 \
  --provider openai
```

Outputs:

```text
construction_v2/candidate_topics/large-language-models/candidate_topics_large-language-models_{year}.jsonl
```

### 3. Deduplicate and cluster candidates

Deduplicate candidate labels and cluster semantically similar candidates:

```bash
python construction_v2/scripts/dedupe_candidates.py \
  --topic "large language models" \
  --years 2019 2020 2021 2022 2023 \
  --use-clustering \
  --cluster-provider openai \
  --cluster-threshold 0.85 \
  --output-suffix _cluster_t0.85
```

Outputs:

```text
construction_v2/candidate_dedup/large-language-models/candidate_index_large-language-models_cluster_t0.85.json
construction_v2/candidate_dedup/large-language-models/candidate_clusters_large-language-models_cluster_t0.85.json
```

### 4. Match later adoption through references

Fetch references from 2024 target-topic papers and match them to candidate source papers:

```bash
python construction_v2/scripts/match_candidate_reference_adoption.py \
  --topic "large language models" \
  --candidate-source cluster \
  --dedup-suffix _cluster_t0.85 \
  --output-suffix _cluster_t0.85_reference_adoption \
  --batch-size 500
```

Outputs:

```text
construction_v2/candidate_matching/large-language-models/matched_papers_large-language-models_cluster_t0.85_reference_adoption.parquet
construction_v2/candidate_matching/large-language-models/matched_papers_large-language-models_cluster_t0.85_reference_adoption.json
```

### 5. Compute frequencies and weak-signal scores

Compute source-grounded early frequencies, 2024 reference-adoption frequencies, trend scores, and final impact scores:

```bash
python construction_v2/scripts/compute_candidate_frequency.py \
  --topic "large language models" \
  --candidate-source cluster \
  --dedup-suffix _cluster_t0.85 \
  --input-suffix _cluster_t0.85_reference_adoption \
  --output-suffix _cluster_t0.85_reference_adoption_precursor \
  --exclude-target-named-candidates \
  --exclude-late-target-era-candidates \
  --exclude-over-specific-candidates
```

Outputs:

```text
construction_v2/candidate_frequency/large-language-models/candidate_frequency_large-language-models_cluster_t0.85_reference_adoption_precursor.*
construction_v2/candidate_frequency/large-language-models/candidate_weak_signals_large-language-models_cluster_t0.85_reference_adoption_precursor.*
construction_v2/candidate_frequency/large-language-models/excluded_survey_papers_large-language-models_cluster_t0.85_reference_adoption_precursor.*
construction_v2/candidate_frequency/large-language-models/candidate_survey_exclusions_large-language-models_cluster_t0.85_reference_adoption_precursor.*
```

The Excel output contains one sheet for all candidates, one weak-signal sheet per alpha variant, and survey-exclusion audit sheets.

## Metrics

For each candidate topic and early year `y` in `2019-2023`:

```text
n_y = number of unique non-survey source papers in year y
N_y = total number of non-survey papers in the target-topic paper pool in year y
f_y = n_y / N_y
```

Early-period frequency:

```text
n_early = total number of unique non-survey source papers in 2019-2023
N_early = total number of non-survey papers in the 2019-2023 target-topic paper pool
f_early = n_early / N_early
```

Later-period adoption:

```text
n_later = number of unique non-survey 2024 target-topic papers that cite the candidate's early source papers
N_later = total number of non-survey 2024 target-topic papers
f_later = n_later / N_later
```

Scoring:

```text
growth = f_later - f_early
impact_original = growth * log(1 / (f_early + epsilon))
```

A log-linear trend is fitted to `f_2019` through `f_2023`. If the trend slope is positive, `growth_impact` is the trend `R^2`; otherwise it is `0`.

```text
impact_final = alpha * impact_original_norm + (1 - alpha) * growth_impact_norm
```

Survey papers are excluded when their title contains `survey`, case-insensitively.
