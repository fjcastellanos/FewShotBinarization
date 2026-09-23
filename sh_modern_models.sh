#!/bin/bash
MODERN_PYTHON="${MODERN_PYTHON:-/opt/conda/envs/modern/bin/python}"

# Cache de Hugging Face en una ruta escribible
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$(pwd)/.cache}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME}/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

mkdir -p "${HUGGINGFACE_HUB_CACHE}"
mkdir -p "${TRANSFORMERS_CACHE}"


set -euo pipefail

# Train/test/cross-test ONE selected modern model on the seven document datasets.
#
# Usage:
#   ./sh_modern_models.sh train segformer_b0
#   ./sh_modern_models.sh test segformer_b2
#   ./sh_modern_models.sh cross deeplabv3_resnet50
#   ./sh_modern_models.sh all segformer_doc_b3
#
# Defaults: 1 training page, 1 validation page, full method (masking + oversampling).
# Override any setting from the shell, for example:
#   MODEL=segformer_b2 EPOCHS=30 ./sh_modern_models.sh train
#   GPU=1 BATCH_SIZE=2 GRAD_ACCUM=16 ./sh_modern_models.sh train

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$ROOT_DIR"

ACTION="${1:-help}"
GPU="${GPU:-0}"
MODEL="${2:-${MODEL:-segformer_b0}}"
VALID_MODELS="segformer_b0 segformer_b1 segformer_b2 segformer_doc_b3 deeplabv3_resnet50 deeplabv3_resnet101"
DATASETS="${DATASETS:-Bickley Dibco Einsiedeln ISOS Palm PHI Salzinnes}"
CROSS_TARGETS="${CROSS_TARGETS:-$DATASETS}"

EXPERIMENT="${EXPERIMENT:-full}"
PAGES_TRAIN="${PAGES_TRAIN:-1}"
VAL_PAGES="${VAL_PAGES:-1}"
ANNOTATED_PATCHES="${ANNOTATED_PATCHES:-8}"
NPATCHES="${NPATCHES:-1024}"
PATCH="${PATCH:-256}"
INK_RATE="${INK_RATE:-0.02}"
EPOCHS="${EPOCHS:-50}"
PATIENCE="${PATIENCE:-10}"
MIN_DELTA="${MIN_DELTA:-0.0001}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-0.00006}"
OVERLAP="${OVERLAP:-64}"
AUG="${AUG:-random flipH flipV rot scale}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export CUDA_VISIBLE_DEVICES="$GPU"

usage() {
    cat <<EOF
Usage: $0 {train|test|cross|all} [model]

Selected model:
  $MODEL

Valid models:
  $VALID_MODELS

Datasets by default:
  $DATASETS

Main settings:
  PAGES_TRAIN=$PAGES_TRAIN
  VAL_PAGES=$VAL_PAGES
  ANNOTATED_PATCHES=$ANNOTATED_PATCHES
  NPATCHES=$NPATCHES
  EPOCHS=$EPOCHS
  BATCH_SIZE=$BATCH_SIZE
  GRAD_ACCUM=$GRAD_ACCUM
  EXPERIMENT=$EXPERIMENT
  GPU=$GPU
EOF
}

run_one() {
    local mode="$1"
    local model="$2"
    local source="$3"
    local target="$4"

    local train_src="datasets/${source}/train/SRC"
    local train_gt="datasets/${source}/train/GT"
    local test_src="datasets/${target}/test/SRC"
    local test_gt="datasets/${target}/test/GT"
    local result_file="results/modern/${model}/${EXPERIMENT}_${mode}.txt"
    local log_dir="logs/modern/${mode}/${model}"
    local log_file="${log_dir}/${source}__to__${target}.log"

    mkdir -p "$log_dir" "$(dirname "$result_file")"

    read -r -a aug_args <<< "$AUG"
    read -r -a extra_args <<< "$EXTRA_ARGS"

    args=(
        -db_train_src "$train_src"
        -db_train_gt "$train_gt"
        -pages_train "$PAGES_TRAIN"
        --val-pages "$VAL_PAGES"
        -n_annotated_patches "$ANNOTATED_PATCHES"
        -npatches "$NPATCHES"
        -window_w "$PATCH"
        -window_h "$PATCH"
        -ink_rate "$INK_RATE"
        -aug "${aug_args[@]}"
        --model "$model"
        --experiment "$EXPERIMENT"
        --dataset-name "$source"
        -e "$EPOCHS"
        --patience "$PATIENCE"
        --min-delta "$MIN_DELTA"
        -b "$BATCH_SIZE"
        --grad-accum "$GRAD_ACCUM"
        --lr "$LR"
        --overlap "$OVERLAP"
        -res "$result_file"
    )

    if [[ "$mode" == "test" || "$mode" == "cross" ]]; then
        args+=(
            -db_test_src "$test_src"
            -db_test_gt "$test_gt"
            --test
        )
    fi

    if ((${#extra_args[@]})); then
        args+=("${extra_args[@]}")
    fi

    echo "=================================================================="
    echo "mode=$mode model=$model source=$source target=$target"
    echo "pages=$PAGES_TRAIN annotations=$ANNOTATED_PATCHES npatches=$NPATCHES"
    echo "log=$log_file"
    echo "=================================================================="

    "${MODERN_PYTHON}" -u main_modern.py "${args[@]}" 2>&1 | tee "$log_file"
}

validate_model() {
    case " $VALID_MODELS " in
        *" $MODEL "*) ;;
        *)
            echo "Unknown model: $MODEL" >&2
            echo "Valid models: $VALID_MODELS" >&2
            exit 2
            ;;
    esac
}

train_models() {
    local dataset
    for dataset in $DATASETS; do
        run_one train "$MODEL" "$dataset" "$dataset"
    done
}

test_models() {
    local dataset
    for dataset in $DATASETS; do
        run_one test "$MODEL" "$dataset" "$dataset"
    done
}

cross_test_models() {
    local source target
    for source in $DATASETS; do
        for target in $CROSS_TARGETS; do
            run_one cross "$MODEL" "$source" "$target"
        done
    done
}

validate_model

case "$ACTION" in
    train)
        train_models
        ;;
    test)
        test_models
        ;;
    cross)
        cross_test_models
        ;;
    all)
        train_models
        test_models
        cross_test_models
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "Unknown action: $ACTION" >&2
        usage >&2
        exit 2
        ;;
esac
