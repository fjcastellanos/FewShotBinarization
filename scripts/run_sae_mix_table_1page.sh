#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATASET:=ISOS}"
: "${DATA_ROOT:=datasets}"
: "${NPATCHES:=1024}"
: "${EPOCHS:=200}"
: "${BATCH_SIZE:=32}"
: "${GPU:=0}"
: "${INK_RATE:=0.02}"
: "${MIXUP_MIN:=0.25}"
: "${MIXUP_MAX:=0.75}"
: "${PYTHON:=python}"
OUT="results/sae_mix_one_page_${DATASET}"
mkdir -p "$OUT"
SRC="$DATA_ROOT/$DATASET"
for p in train/SRC train/GT test/SRC test/GT; do
  if [[ ! -d "$SRC/$p" ]]; then echo "Missing dataset directory: $SRC/$p" >&2; exit 1; fi
done
CSV="$OUT/results.csv"
printf 'method,lambda,f1_percent,log\n' > "$CSV"
for method in cutmix mixup; do
  for lambda in 1 8 32; do
    BASE="$OUT/${method}_lambda${lambda}"
    common=( -db_train_src "$SRC/train/SRC" -db_train_gt "$SRC/train/GT"
      -db_test_src "$SRC/test/SRC" -db_test_gt "$SRC/test/GT"
      -aug random flipH flipV rot scale -window_w 256 -window_h 256
      -l 4 -f 32 -k 3 -drop 0.2 -pages_train 1
      -npatches "$NPATCHES" -n_annotated_patches "$lambda"
      -ink_rate "$INK_RATE" -e "$EPOCHS" -b "$BATCH_SIZE"
      -gpu "$GPU" --sae_mix "$method"
      --mixup_min "$MIXUP_MIN" --mixup_max "$MIXUP_MAX" )
    echo "Training: $method lambda=$lambda"
    "$PYTHON" -u main.py "${common[@]}" > "${BASE}_train.log" 2>&1
    echo "Evaluating: $method lambda=$lambda"
    "$PYTHON" -u main.py "${common[@]}" --test > "${BASE}_test.log" 2>&1
    score="$("$PYTHON" - "${BASE}_test.log" <<'PY'
import re,sys
s=open(sys.argv[1],encoding='utf8').read()
match=re.findall(r'Results:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)',s)
if not match:
    sys.exit('F-score not found in '+sys.argv[1])
v=float(match[-1]); print(f'{v*100:.2f}' if v <= 1 else f'{v:.2f}')
PY
)"
    printf '%s,%s,%s,%s\n' "$method" "$lambda" "$score" "${BASE}_test.log" >> "$CSV"
    echo "F-score: $score%"
  done
done
"$PYTHON" - "$CSV" "$OUT/table_rows.tex" <<'PY'
import csv,sys
with open(sys.argv[1],newline='') as f: d={(r['method'],int(r['lambda'])):r['f1_percent'] for r in csv.DictReader(f)}
with open(sys.argv[2],'w') as f:
    for method,title in [('cutmix','Over. + CutMix'),('mixup','Over. + MixUp')]:
        nums=['%.1f'%float(d[(method,n)]) for n in (1,8,32)]
        f.write(title+' & '+' & '.join(nums)+r' \\'+'\n')
print('CSV:',sys.argv[1]);print('LaTeX:',sys.argv[2])
PY
