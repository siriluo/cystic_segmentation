#!/usr/bin/env bash
# Overlay the trained 8-class model on chunked_video.mp4
set -euo pipefail
cd "$(dirname "$0")/.."

module load ffmpeg/6.1.1 2>/dev/null || true

PY="${PY:-python3}"
VIDEO="${VIDEO:-chunked_video.mp4}"
CKPT="${CKPT:-experiments/train_test/deeplabv3_best.pth}"
OUT="${OUT:-outputs/chunked_video_segmented.mp4}"
TASK="${TASK:-landmarks}"

mkdir -p "$(dirname "$OUT")"

$PY scripts/infer.py \
  --video "$VIDEO" \
  --checkpoint "$CKPT" \
  --task "$TASK" \
  --output "$OUT" \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-4}" \
  ${MAX_FRAMES:+--max-frames "$MAX_FRAMES"}
