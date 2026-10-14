#!/bin/bash
#SBATCH -J spin_gnn_sudoku_dd
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH -o %x-%j.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Greedy hint-removal paths for DD Sudoku on random regular graphs
# (spin_gnn.sudoku_sweep); pure numpy + numba, CPU-only, parallel over a
# process pool. Already-finished (q,d,N,seed) files are skipped, so this is
# safe to resubmit after a timeout.
# Usage: sbatch sudoku_dd_sweep.sbatch [extra args passed to sudoku_sweep.py]

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
python -c "import numba" 2>/dev/null || pip install numba
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_dd_results"
mkdir -p "$OUT_DIR"

python -u -m spin_gnn.sudoku_sweep \
    --degrees 4 5 6 7 8 \
    --seeds 24 \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --out-dir "$OUT_DIR" \
    "$@"
