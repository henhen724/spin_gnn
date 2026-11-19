#!/bin/bash
#SBATCH --job-name=heads105
#SBATCH --partition=sganguli
#SBATCH --array=0-3
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/runs/stageheads_2026-11-19/logs/probe_%A_%a.out

# Stage HEADS (105) -- attention probe over all 80 surviving checkpoints.
# Inference only: nothing is trained, checkpoints are opened read-only.
# CPU-only by design (one sequence at a time, batch size 1), so no -G.
# 4 shards x ~20 runs x 3 densities x 200 instances ~= 25 min per task.

set -e

module purge
module load devel math python/3.12.1
module load py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
module load py-numba/0.60.0_py312 py-scipy/1.12.0_py312

cd /home/users/hshunt/spin_gnn

python3 scripts/stageheads_probe.py \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --nshards "${SLURM_ARRAY_TASK_COUNT}" \
    --n-inst 200
