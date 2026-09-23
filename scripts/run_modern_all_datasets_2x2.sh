#!/bin/bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

# Seven datasets used by the project.
DATASETS=${DATASETS:-"Bickley Dibco Einsiedeln ISOS Palm PHI Salzinnes"}
MODEL=${MODEL:-segformer_b0}

for DATASET in $DATASETS; do
  echo "##################################################################"
  echo "DATASET=$DATASET MODEL=$MODEL"
  echo "##################################################################"
  DATASET_NAME="$DATASET" \
  SRC_TRAIN="datasets/$DATASET/train/SRC" \
  GT_TRAIN="datasets/$DATASET/train/GT" \
  SRC_TEST="datasets/$DATASET/test/SRC" \
  GT_TEST="datasets/$DATASET/test/GT" \
  MODEL="$MODEL" \
  bash scripts/run_modern_2x2_ablation.sh
done
