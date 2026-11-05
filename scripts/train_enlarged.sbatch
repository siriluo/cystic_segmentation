#!/bin/bash
#SBATCH --job-name=cystic-seg
#SBATCH --account=bbka-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/enlarged/slurm_%j.out
#SBATCH --error=experiments/enlarged/slurm_%j.err

set -euo pipefail
cd /projects/bbka/shirui/cystic_segmentation
mkdir -p experiments/enlarged

export PYTHONUNBUFFERED=1
PY=/sw/user/python/miniforge3-pytorch-2.10.0/bin/python3

echo "host=$(hostname) date=$(date)"
$PY -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

$PY scripts/train.py \
  --data-root data/processed \
  --experiment-dir experiments/enlarged \
  --batch-size 16 \
  --epochs 50 \
  --lr 1e-4 \
  --split case \
  --class-weights \
  --num-workers 4 \
  --patience 10

$PY scripts/validate.py \
  --data-root data/processed \
  --checkpoint experiments/enlarged/deeplabv3_best.pth \
  --split-json experiments/enlarged/split.json \
  --out experiments/enlarged/metrics.json \
  --num-workers 4

echo "done $(date)"
