#!/bin/bash
#SBATCH --job-name=abl106
#SBATCH --partition=sganguli
#SBATCH --array=0-11
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/runs/stageablate_2026-11-19/logs/abl_%A_%a.out

# Stage ABLATE (106) -- attention ablation over the 18 gate-passing runs plus
# 25 below-gate cells at their lowest seed.
# Inference only: nothing is trained, checkpoints are opened read-only.
# CPU-only by design (one sequence at a time, batch size 1), so no -G.
# 12 shards x ~3.6 runs x 3 densities x 200 held-out instances x 15 conditions.
# Pilot: the deepest run (L=12 full) costs 2.44 s per instance for all 15
# conditions, so ~24 min for its 600 instances; a shard of 4 deep runs is the
# worst case at ~100 min, inside the 4 h limit.

set -e

module purge
module load devel math python/3.12.1
module load py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
module load py-numba/0.60.0_py312 py-scipy/1.12.0_py312

cd /home/users/hshunt/spin_gnn

python3 scripts/stageablate_probe.py \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --nshards "${SLURM_ARRAY_TASK_COUNT}"
