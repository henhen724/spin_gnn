#!/bin/bash
#SBATCH --job-name=widthladder
#SBATCH --partition=sganguli
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/runs/widthladder_2026-11-20/slurm-%j.out

# Round-width statistics on an N ladder 48..1020.  Instance combinatorics only
# -- DD run on the puzzle, no model, no checkpoint, no GPU.  For slide 23 of
# the round-width deck.

set -u
module purge --force
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312

OUT=/scratch/users/hshunt/spin_gnn/runs/widthladder_2026-11-20
mkdir -p "$OUT"
cd /home/users/hshunt/spin_gnn

python3 -u scripts/widthladder_probe.py --n-per-cell 300 --out-dir "$OUT"
