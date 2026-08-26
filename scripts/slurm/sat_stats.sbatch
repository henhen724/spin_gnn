#!/bin/bash
#SBATCH -J spin_gnn_sat_stats
#SBATCH -p normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4GB
#SBATCH -o %x-%j.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Random-k-SAT unit-propagation-solver statistics; pure numpy, CPU-only.
# Usage: sbatch sat_stats.sbatch [extra args passed to sat_stats.py]

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"

OUT_DIR="${SCRATCH}/spin_gnn/sat_stats"
mkdir -p "$OUT_DIR"

python -m spin_gnn.sat_stats \
    --n-vars 24 \
    --k 3 \
    --alpha 4.2 \
    --n-instances 200 \
    --out-dir "$OUT_DIR" \
    "$@"
