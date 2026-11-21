#!/bin/bash
#SBATCH --job-name=sud2Nsplit
#SBATCH --partition=sganguli
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/runs/stagensplit_2026-11-20/slurm-%A_%a.out
#SBATCH --array=0-0

# Stage NSPLIT -- per-section NLL at N=96 and N=192, to split the N=192 excess
# into a deduction part (zero data floor, pure excess) and a floor-model part.
# Inference only, CPU, no training, no checkpoint written.
#
# Pass NSPLIT_ARGS="--n96-only" to run just the DN3 cross-check cell, which
# exists before the N=192 checkpoints do.

set -u
module purge --force
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312

OUT=/scratch/users/hshunt/spin_gnn/runs/stagensplit_2026-11-20
mkdir -p "$OUT"
cd /home/users/hshunt/spin_gnn

python3 -u scripts/stagensplit_probe.py \
    --task "${SLURM_ARRAY_TASK_ID:-0}" \
    --ntasks "${SLURM_ARRAY_TASK_COUNT:-1}" \
    --out "$OUT" ${NSPLIT_ARGS:-}
