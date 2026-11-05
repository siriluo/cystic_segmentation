#!/bin/bash
#SBATCH --job-name=cystic-3c
#SBATCH --account=bbka-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/ducts_only/slurm_%j.out
#SBATCH --error=experiments/ducts_only/slurm_%j.err

set -euo pipefail
cd /projects/bbka/shirui/cystic_segmentation
mkdir -p experiments/ducts_only

export PYTHONUNBUFFERED=1
export PY=/sw/user/python/miniforge3-pytorch-2.10.0/bin/python3
export EXP=experiments/ducts_only
export DATA=data/processed
export SPLIT=experiments/train_test/split.json

echo "host=$(hostname) date=$(date)"
$PY -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

bash scripts/train_ducts_only.sh
