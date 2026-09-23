#!/bin/bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

# Fair 3-condition comparison. All conditions use the same page(s), annotation
# budget, number of virtual crops, augmentation settings and optimizer budget.
# Only oversampling and masked loss change.
MODEL=${MODEL:-segformer_b0}

for EXPERIMENT in baseline oversampling full; do
  echo "=================================================================="
  echo "MODEL=$MODEL EXPERIMENT=$EXPERIMENT"
  echo "=================================================================="
  MODEL="$MODEL" EXPERIMENT="$EXPERIMENT" bash scripts/run_modern_single.sh
done
