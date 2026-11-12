#!/bin/bash
#SBATCH --job-name=fastslowMeas
#SBATCH --partition=sganguli
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/sudoku_fastslow_2026-11-12/measure_%j.out
#
# Re-measure the ensemble side of the fast/slow transition so the phases deck
# can draw figures instead of quoting finding 43's archived scalars. No model is
# loaded and nothing is trained: this is the TASK, not the solver.
#
# A --quick pass runs first into a throwaway dir. It exercises every code path
# with small sizes, so an API mistake costs twenty seconds here instead of
# surfacing after the N=17196 paths have run. The full pass only starts if it
# exits 0.
#
# Two hours because the fall paths in the slow phase are the slow part by
# construction: failed removals cascade, and N=17196 reaches T ~ 873 rounds. The
# script dumps its JSON after every size, so a wall-clock kill still leaves
# every size measured up to that point on disk.

set -u
cd /home/users/hshunt/spin_gnn || exit 1
OUT=/scratch/users/hshunt/spin_gnn/sudoku_fastslow_2026-11-12
mkdir -p "$OUT"

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
            py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
            py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR=/scratch/users/hshunt/numba_cache

python3 -m py_compile scripts/fastslow_measure.py || { echo "py_compile FAILED"; exit 1; }

echo
echo "=== quick pass (shape check) ==="
python3 -u -m scripts.fastslow_measure --out "$OUT/quick" --quick || {
    echo "QUICK PASS FAILED -- not running the full measurement"; exit 1; }

echo
echo "=== full pass ==="
python3 -u -m scripts.fastslow_measure --out "$OUT"
echo "full pass rc=$?"

echo
echo "end $(date '+%F %T')"
