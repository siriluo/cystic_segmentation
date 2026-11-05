#!/usr/bin/env bash
# 3-class: CBD vs cystic duct vs everything else as background. Same split as 8-class.
set -euo pipefail
cd "$(dirname "$0")/.."

EXP="${EXP:-experiments/ducts_only}"
DATA="${DATA:-data/processed}"
PY="${PY:-python3}"
SPLIT="${SPLIT:-experiments/train_test/split.json}"

mkdir -p "$EXP"

if [[ ! -f "$SPLIT" ]]; then
  $PY scripts/make_splits.py \
    --data-root "$DATA" \
    --out "$EXP/split.json" \
    --unit case \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42
  SPLIT="$EXP/split.json"
fi

$PY scripts/train.py \
  --data-root "$DATA" \
  --experiment-dir "$EXP" \
  --split-json "$SPLIT" \
  --task ducts-only \
  --class-weights \
  --duct-crop-prob 0.7 \
  --batch-size 16 \
  --epochs 50 \
  --lr 1e-4 \
  --split case \
  --num-workers 4 \
  --patience 12

$PY scripts/evaluate_split.py \
  --data-root "$DATA" \
  --split-json "$EXP/split.json" \
  --checkpoint "$EXP/deeplabv3_best.pth" \
  --task ducts-only \
  --subset test \
  --out "$EXP/test_metrics.json" \
  --csv "$EXP/test_per_image.csv" \
  --vis-dir "$EXP/test_overlays" \
  --num-vis 8 \
  --num-workers 4

$PY scripts/evaluate_split.py \
  --data-root "$DATA" \
  --split-json "$EXP/split.json" \
  --checkpoint "$EXP/deeplabv3_best.pth" \
  --task ducts-only \
  --subset val \
  --out "$EXP/val_metrics.json" \
  --csv "$EXP/val_per_image.csv" \
  --vis-dir "$EXP/val_overlays" \
  --num-vis 8 \
  --num-workers 4

echo "Done. Test metrics: $EXP/test_metrics.json"
