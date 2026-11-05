#!/bin/bash
#SBATCH --job-name=cystic-inf
#SBATCH --account=bbka-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/infer_%j.out
#SBATCH --error=outputs/infer_%j.err

set -euo pipefail
cd /projects/bbka/shirui/cystic_segmentation
mkdir -p outputs

module load ffmpeg/6.1.1
export PYTHONUNBUFFERED=1
export PY=/sw/user/python/miniforge3-pytorch-2.10.0/bin/python3

echo "host=$(hostname) date=$(date)"
which ffmpeg
$PY -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

bash scripts/infer.sh
