#!/bin/bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

# Optional complete 2x2 ablation:
# baseline      : no masking, no oversampling
# oversampling  : oversampling only
# masking       : masking only
# full          : masking + oversampling
MODEL=${MODEL:-segformer_b0}

for EXPERIMENT in baseline oversampling masking full; do
  echo "=================================================================="
  echo "MODEL=$MODEL EXPERIMENT=$EXPERIMENT"
  echo "=================================================================="
  MODEL="$MODEL" EXPERIMENT="$EXPERIMENT" bash scripts/run_modern_single.sh
done
