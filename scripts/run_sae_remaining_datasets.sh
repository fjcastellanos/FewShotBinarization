#!/usr/bin/env bash
# Run the original one-page SAE MixUp/CutMix runner across datasets.
# Runs datasets sequentially in each Docker; use one instance per GPU/container.
set -euo pipefail
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"
DATASETS="Dibco Einsiedeln Palm PHI Salzinnes Bickley"
RUN_ARGS=()
while (($#)); do
    case "$1" in
        --datasets)
            if (($# < 2)); then echo 'ERROR: --datasets requires comma-separated names' >&2; exit 2; fi
            DATASETS="${2//,/ }"
            shift 2
            ;;
        --help|-h)
            cat <<'HELP'
Usage: bash scripts/run_sae_remaining_datasets.sh [--datasets NAME1,NAME2] [runner options]
Defaults to Bickley,Dibco,Einsiedeln,Palm,PHI,Salzinnes (ISOS already run).
Pass through original runner options: --jobs cutmix:1,mixup:8 --gpus 0 --dry-run --force etc.
One GPU is normally visible as 0 inside an isolated Docker.
Examples:
  bash scripts/run_sae_remaining_datasets.sh --jobs cutmix:1,cutmix:8,cutmix:32 --gpus 0
  bash scripts/run_sae_remaining_datasets.sh --datasets Bickley,Dibco --jobs mixup:1 --gpus 0
  bash scripts/run_sae_remaining_datasets.sh --datasets ISOS,PHI --jobs all --gpus 0
HELP
            exit 0
            ;;
        --dataset)
            echo 'ERROR: use --datasets in this wrapper (not --dataset).' >&2
            exit 2
            ;;
        *) RUN_ARGS+=("$1"); shift ;;
    esac
done
if [[ -z "${DATASETS// /}" ]]; then echo 'ERROR: empty --datasets' >&2; exit 2; fi
FAILED=0
for dataset in $DATASETS; do
    case "$dataset" in Bickley|Dibco|Einsiedeln|ISOS|Palm|PHI|Salzinnes) ;; *) echo "ERROR: unrecognized dataset '$dataset'" >&2; exit 2;; esac
    echo "========== SAE dataset: $dataset =========="
    if ! bash scripts/run_sae_parallel_1page.sh --dataset "$dataset" "${RUN_ARGS[@]}"; then
        echo "ERROR: dataset '$dataset' failed (see its logs). Continuing." >&2
        FAILED=1
    fi
done
exit "$FAILED"
