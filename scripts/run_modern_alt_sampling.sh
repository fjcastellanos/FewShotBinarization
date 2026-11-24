#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"
# Alternatives to oversampling requested by the reviewer, run as independent
# experiments: masking + centered larger context, and masking + random output
# masking. Compare against EXPERIMENT=full (masking + oversampling).
# Inherits all configuration variables from scripts/run_modern_single.sh.
# SAMPLINGS can be "centered_context", "output_masking" or both.
SAMPLINGS=${SAMPLINGS:-"centered_context output_masking"}
CONTEXT_MIN=${CONTEXT_MIN:-1.0}
CONTEXT_MAX=${CONTEXT_MAX:-2.0}
MASK_MIN=${MASK_MIN:-0.0}
MASK_MAX=${MASK_MAX:-0.95}
MASK_BLOCK=${MASK_BLOCK:-32}
BASE_EXTRA_ARGS=${EXTRA_ARGS:-}
MODEL=${MODEL:-segformer_b0}
for sampling in $SAMPLINGS; do
  case "$sampling" in
    centered_context) params="--context-min $CONTEXT_MIN --context-max $CONTEXT_MAX" ;;
    output_masking) params="--mask-min $MASK_MIN --mask-max $MASK_MAX --mask-block $MASK_BLOCK" ;;
    *) echo "Invalid sampling: $sampling" >&2; exit 2 ;;
  esac
  echo "=== Running masking + $sampling ($MODEL) ==="
  EXPERIMENT=masking \
  EXTRA_ARGS="$BASE_EXTRA_ARGS --sampling $sampling $params" \
  RESULTS_FILE="${RESULTS_DIR:-results/modern/${MODEL}}/masking_${sampling}.txt" \
  bash scripts/run_modern_single.sh
done
