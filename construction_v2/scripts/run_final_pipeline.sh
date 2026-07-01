#!/usr/bin/env bash
set -euo pipefail

TOPIC=""
SPACE="all"
MODE="core"
OUTPUT_DIR=""
CONDA_ENV="osworld"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'USAGE'
Usage:
  bash construction_v2/scripts/run_final_pipeline.sh --topic "large language models" [--space all|problem-space|solution-space] [--mode core|strict] [--output-dir PATH]

Options:
  --topic      Target topic name from construction_v2/topics.json. Required.
  --space      Candidate space to process: all, problem-space, or solution-space. Default: all.
  --mode       Final gate mode: core or strict. Default: core.
  --output-dir Final results directory. Default: construction_v2/final_results_<mode>.
  --conda-env  Conda environment name. Default: osworld.
USAGE
}

step() {
  echo
  echo "==> $1"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --topic)
      TOPIC="${2:-}"
      shift 2
      ;;
    --space)
      SPACE="${2:-all}"
      shift 2
      ;;
    --mode)
      MODE="${2:-core}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --conda-env)
      CONDA_ENV="${2:-osworld}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${TOPIC}" ]]; then
  echo "--topic is required." >&2
  usage >&2
  exit 2
fi

if [[ "${SPACE}" != "all" && "${SPACE}" != "problem-space" && "${SPACE}" != "solution-space" ]]; then
  echo "--space must be one of: all, problem-space, solution-space." >&2
  exit 2
fi

if [[ "${MODE}" != "core" && "${MODE}" != "strict" ]]; then
  echo "--mode must be one of: core, strict." >&2
  exit 2
fi

if [[ -z "${OUTPUT_DIR}" ]]; then
  OUTPUT_DIR="construction_v2/final_results_${MODE}"
fi

cd "${ROOT_DIR}"

run_py() {
  conda run -n "${CONDA_ENV}" python "$@"
}

step "Step 1/5: Fetch target-topic papers for 2019-2024"
run_py construction_v2/scripts/fetch_papers.py \
  --topic "${TOPIC}"

step "Step 2/5: Extract problem-space and solution-space candidate topics from 2019-2023 papers"
run_py construction_v2/scripts/extract_candidate.py \
  --topic "${TOPIC}"

step "Step 3/5: Simple dedup candidates, then cluster them with threshold 0.85"
run_py construction_v2/scripts/dedupe_candidates.py \
  --topic "${TOPIC}" \
  --candidate-topic-type "${SPACE}" \
  --use-clustering \
  --cluster-threshold 0.85 \
  --cluster-provider openrouter \
  --cluster-embed-model openai/text-embedding-3-large \
  --output-suffix _cluster_t0.85

step "Step 4/5: Match 2024 reference adoption and write reference_match outputs"
run_py construction_v2/scripts/match_candidate_reference_adoption.py \
  --topic "${TOPIC}" \
  --candidate-topic-type "${SPACE}" \
  --candidate-source cluster \
  --dedup-suffix _cluster_t0.85 \
  --output-suffix _cluster_t0.85

step "Step 5/5: Compute final onset scores, apply hard gates, and write final_results"
run_py construction_v2/scripts/compute_final_results.py \
  --topic "${TOPIC}" \
  --candidate-topic-type "${SPACE}" \
  --candidate-source cluster \
  --dedup-suffix _cluster_t0.85 \
  --matching-suffix _cluster_t0.85 \
  --mode "${MODE}" \
  --output-dir "${OUTPUT_DIR}"

echo
echo "Done. Mode=${MODE}. Results are under ${OUTPUT_DIR}."
