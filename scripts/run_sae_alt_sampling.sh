#!/usr/bin/env bash
# Train + test the SAE with the two reviewer-requested alternatives to
# oversampling (independent experiments, never combined with oversampling):
#   centered_context : fixed annotated patch, random centred context 1x-2x, resized to 256
#   output_masking   : fixed annotated patch, 0-95% of its labels masked in random blocks
# Same settings and sample budget as the over_noaug baseline (-npatches 1024,
# -aug none). Runs in parallel, one job per GPU (default GPUs 0,1), with a
# shared queue. Results: results/sae_baselines_1page/<dataset>/ and the summary
# results/sae_baselines_1page/alt_sampling_summary.csv
#
# Usage:
#   bash scripts/run_sae_alt_sampling.sh                      # everything on GPUs 0,1
#   bash scripts/run_sae_alt_sampling.sh --datasets Dibco,PHI --lambdas 1,8
#   bash scripts/run_sae_alt_sampling.sh --gpus 1             # a single GPU
#   bash scripts/run_sae_alt_sampling.sh --dry-run
# Finished runs are skipped on restart (use --force to repeat them).
set -euo pipefail
ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"
exec python3 scripts/run_sae_alt_sampling_parallel.py "$@"
