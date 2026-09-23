#!/bin/bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

MODEL=${MODEL:-segformer_b0}
EXPERIMENT=${EXPERIMENT:-full}
SRC_TRAIN=${SRC_TRAIN:-datasets/Dibco/train/SRC}
GT_TRAIN=${GT_TRAIN:-datasets/Dibco/train/GT}
SRC_TEST=${SRC_TEST:-datasets/Dibco/test/SRC}
GT_TEST=${GT_TEST:-datasets/Dibco/test/GT}
DATASET_NAME=${DATASET_NAME:-}
SEED=${SEED:-1}
PAGES_TRAIN=${PAGES_TRAIN:-1}
VAL_PAGES=${VAL_PAGES:-1}
ANNOTATED_PATCHES=${ANNOTATED_PATCHES:-1}
NPATCHES=${NPATCHES:-1024}
PATCH=${PATCH:-256}
INK_RATE=${INK_RATE:-0.02}
EPOCHS=${EPOCHS:-50}
PATIENCE=${PATIENCE:-10}
MIN_DELTA=${MIN_DELTA:-0.0001}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LR=${LR:-0.00006}
OVERLAP=${OVERLAP:-64}
AUG=${AUG:-random}
RESULTS_FILE=${RESULTS_FILE:-results/modern/${MODEL}/${EXPERIMENT}.txt}
EXTRA_ARGS=${EXTRA_ARGS:-}

read -r -a AUG_ARGS <<< "$AUG"
read -r -a EXTRA_ARRAY <<< "$EXTRA_ARGS"

ARGS=(
  -db_train_src "$SRC_TRAIN"
  -db_train_gt "$GT_TRAIN"
  -db_test_src "$SRC_TEST"
  -db_test_gt "$GT_TEST"
  -pages_train "$PAGES_TRAIN"
  --val-pages "$VAL_PAGES"
  -n_annotated_patches "$ANNOTATED_PATCHES"
  -npatches "$NPATCHES"
  -window_w "$PATCH" -window_h "$PATCH"
  -ink_rate "$INK_RATE"
  -aug "${AUG_ARGS[@]}"
  --model "$MODEL"
  --experiment "$EXPERIMENT"
  --seed "$SEED"
  -b "$BATCH_SIZE"
  --grad-accum "$GRAD_ACCUM"
  --lr "$LR"
  -e "$EPOCHS"
  --patience "$PATIENCE"
  --min-delta "$MIN_DELTA"
  --overlap "$OVERLAP"
)

if [[ -n "$DATASET_NAME" ]]; then
  ARGS+=(--dataset-name "$DATASET_NAME")
fi
if [[ -n "$RESULTS_FILE" && "$RESULTS_FILE" != "none" ]]; then
  ARGS+=(-res "$RESULTS_FILE")
fi
ARGS+=("${EXTRA_ARRAY[@]}")

python -u main_modern.py "${ARGS[@]}"
