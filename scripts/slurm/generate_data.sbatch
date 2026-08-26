#!/bin/bash
#SBATCH -J spin_gnn_gen
#SBATCH -p normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH -o %x-%j.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Pure-numpy simulation of the kinetic Ising disorder model; CPU-only.
# Usage: sbatch generate_data.sbatch [extra args passed to simulate.py]

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"

OUT_DIR="${SCRATCH}/spin_gnn/data"
mkdir -p "$OUT_DIR"

python -m spin_gnn.simulate \
    --out-dir "$OUT_DIR" \
    --n-spins 200 \
    --context-length 4 \
    --beta 0.5 \
    --t-steps 512 \
    --n-realizations 8 \
    "$@"
