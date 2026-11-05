#!/bin/bash
#SBATCH --job-name=cystic-bd
#SBATCH --account=bbka-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=experiments/bile_ducts/slurm_%j.out
#SBATCH --error=experiments/bile_ducts/slurm_%j.err

set -euo pipefail
cd /projects/bbka/shirui/cystic_segmentation
mkdir -p experiments/bile_ducts

export PYTHONUNBUFFERED=1
export PY=/sw/user/python/miniforge3-pytorch-2.10.0/bin/python3
export EXP=experiments/bile_ducts
export DATA=data/processed
export SPLIT=experiments/train_test/split.json

echo "host=$(hostname) date=$(date)"
$PY -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

bash scripts/train_bile_ducts.sh
