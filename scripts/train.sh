#!/usr/bin/env bash
# Train on data/processed, then score the held-out test patients.
set -euo pipefail
cd "$(dirname "$0")/.."

EXP="${EXP:-experiments/train_test}"
DATA="${DATA:-data/processed}"
PY="${PY:-python3}"

mkdir -p "$EXP"

$PY scripts/make_splits.py \
  --data-root "$DATA" \
  --out "$EXP/split.json" \
  --unit case \
  --val-ratio 0.15 \
  --test-ratio 0.15 \
  --seed 42

$PY scripts/train.py \
  --data-root "$DATA" \
  --experiment-dir "$EXP" \
  --split-json "$EXP/split.json" \
  --batch-size 16 \
  --epochs 50 \
  --lr 1e-4 \
  --split case \
  --class-weights \
  --num-workers 4 \
  --patience 10

$PY scripts/evaluate_split.py \
  --data-root "$DATA" \
  --split-json "$EXP/split.json" \
  --checkpoint "$EXP/deeplabv3_best.pth" \
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
  --subset val \
  --out "$EXP/val_metrics.json" \
  --csv "$EXP/val_per_image.csv" \
  --vis-dir "$EXP/val_overlays" \
  --num-vis 8 \
  --num-workers 4

echo "Done. Test metrics: $EXP/test_metrics.json"
