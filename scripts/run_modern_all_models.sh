#!/bin/bash
set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT_DIR"

# SegFormer B0/B1/B2 are transformer models with ImageNet-pretrained encoders.
# The document-specific B3 is included as a transfer-learning reference.
# DeepLabV3 R50/R101 are modern CNN baselines with ImageNet-pretrained backbones.
MODELS=${MODELS:-"segformer_b0 segformer_b1 segformer_b2 segformer_doc_b3 deeplabv3_resnet50 deeplabv3_resnet101"}

for MODEL in $MODELS; do
  MODEL="$MODEL" bash scripts/run_modern_ablation.sh
done
