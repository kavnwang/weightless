#!/bin/bash
#SBATCH --job-name=weightless
#SBATCH --partition=cluster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

cd /home/ubuntu/weightless
mkdir -p logs checkpoints

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "=== $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Use pixi environment
export PATH="/home/ubuntu/weightless/.pixi/envs/default/bin:$PATH"

# Run training with all args passed through
python train.py "$@"
