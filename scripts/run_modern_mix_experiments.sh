#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"
# Inherits all configuration variables from scripts/run_modern_single.sh.
# MIX_STRATEGIES can be "mixup", "cutmix" or "mixup cutmix".
MIX_STRATEGIES=${MIX_STRATEGIES:-"mixup cutmix"}
MIXUP_MIN=${MIXUP_MIN:-0.25}
MIXUP_MAX=${MIXUP_MAX:-0.75}
BASE_EXTRA_ARGS=${EXTRA_ARGS:-}
MODEL=${MODEL:-segformer_b0}
EXPERIMENT=${EXPERIMENT:-full}
for strategy in $MIX_STRATEGIES; do
  case "$strategy" in mixup|cutmix) ;; *) echo "Invalid strategy: $strategy" >&2; exit 2 ;; esac
  echo "=== Running $strategy ($MODEL / $EXPERIMENT) ==="
  EXTRA_ARGS="$BASE_EXTRA_ARGS --mix-strategy $strategy --mixup-min $MIXUP_MIN --mixup-max $MIXUP_MAX" \
  RESULTS_FILE="${RESULTS_DIR:-results/modern/${MODEL}}/${EXPERIMENT}_${strategy}.txt" \
  bash scripts/run_modern_single.sh
done
