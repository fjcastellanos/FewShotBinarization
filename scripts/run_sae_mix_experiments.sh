#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATASET:=Dibco}"
: "${DATA_ROOT:=datasets}"
: "${SAE_STRATEGIES:=mixup cutmix}"
: "${MIXUP_MIN:=0.25}"
: "${MIXUP_MAX:=0.75}"
: "${NPATCHES:=128}"
: "${N_ANNOTATED_PATCHES:=1}"
: "${PAGES_TRAIN:=1}"
: "${INK_RATE:=0.02}"
: "${EPOCHS:=200}"
: "${BATCH_SIZE:=32}"
: "${GPU:=0}"
: "${EXTRA_ARGS:=}"
mkdir -p logs/sae_mix
for strategy in ${SAE_STRATEGIES}; do
  case "$strategy" in mixup|cutmix) ;; *) echo "Unknown strategy: $strategy" >&2; exit 2;; esac
  echo "SAE ${strategy} -- dataset=${DATASET}"
  python -u main.py \
    -db_train_src "${DATA_ROOT}/${DATASET}/train/SRC" \
    -db_train_gt "${DATA_ROOT}/${DATASET}/train/GT" \
    -aug random -npatches "$NPATCHES" -n_annotated_patches "$N_ANNOTATED_PATCHES" \
    -pages_train "$PAGES_TRAIN" -ink_rate "$INK_RATE" \
    -e "$EPOCHS" -b "$BATCH_SIZE" -gpu "$GPU" \
    --sae_mix "$strategy" --mixup_min "$MIXUP_MIN" --mixup_max "$MIXUP_MAX" \
    ${EXTRA_ARGS} 2>&1 | tee "logs/sae_mix/${DATASET}_${strategy}.log"
done
