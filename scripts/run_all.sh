#!/usr/bin/env bash
# Run all model evaluations sequentially.
# Usage: bash scripts/run_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

for config in "$PROJECT_ROOT"/configs/*.yaml; do
    echo ""
    echo "=========================================="
    echo "Running: $(basename "$config")"
    echo "=========================================="
    python "$SCRIPT_DIR/run_evaluation.py" --config "$config"
done

echo ""
echo "All evaluations complete."
