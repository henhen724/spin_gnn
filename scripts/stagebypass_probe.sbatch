#!/bin/bash
#SBATCH --job-name=byp107
#SBATCH --partition=sganguli
#SBATCH --array=0-7
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/runs/stagebypass_2026-11-19/logs/byp_%A_%a.out

# Stage BYPASS (107) -- the five section-level ablations on the 36 runs that
# stage ABLATE never probed.  Inference only: nothing is trained, checkpoints
# are opened read-only.
# CPU-only by design (one sequence at a time, batch size 1), so no -G.
# 8 shards x 4.5 runs x 3 densities x 200 held-out instances x 5 conditions.
# Pilot: r76 (L=16 tied k4, the decisive run) cost 5.0 s for 3 instances
# including model load and the check forwards.  Scaling ABLATE's measured
# throughput by conditions and runs puts the whole stage near 55 min, so a
# shard is ~7 min and 2 h is a wide margin.

set -e

module purge
module load devel math python/3.12.1
module load py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
module load py-numba/0.60.0_py312 py-scipy/1.12.0_py312

cd /home/users/hshunt/spin_gnn

python3 scripts/stagebypass_probe.py \
    --shard "${SLURM_ARRAY_TASK_ID}" \
    --nshards "${SLURM_ARRAY_TASK_COUNT}"
