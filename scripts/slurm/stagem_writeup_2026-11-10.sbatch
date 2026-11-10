#!/bin/bash
#SBATCH --job-name=stagemWriteup
#SBATCH --partition=sganguli
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagemWriteup_%j.out
#
# RENDER the stage M deck and results notebook. Does NOT commit and does NOT
# upload: the PNGs have to be LOOKED AT first, and reading them back is what
# caught six defects in the stage N deck that no exit code would have. Commit and
# upload follow, as a separate step, once the pages are known to be right.
#
# In a job, not on a login node: page 11 generates an N=96 instance and solves it
# with the DD solver, and the notebook re-runs the registered scorer as a
# subprocess over every N=96 cell on disk.
#
# NO VENV -- $SCRATCH/spin_gnn_venv was purged. Modules only, and matplotlib
# needs MPLCONFIGDIR somewhere writable or it warns on every figure.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
    py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
    py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export MPLCONFIGDIR="${SCRATCH}/mplcache"
export PYTHONUNBUFFERED=1

NB=notebooks/sudoku_n96_stagem_2026-11-09_results.ipynb
COMMIT=$(git rev-parse --short HEAD)

echo "host=$(hostname)  job=$SLURM_JOB_ID  commit=$COMMIT"
echo "start $(date '+%F %T')"

echo
echo "=== STEP 1/3  the deck (13 pages, a PNG each) ==========================="
python3 -u -m notebooks.make_stagem_deck --commit "$COMMIT"
DECK=$?
echo "deck rc=$DECK"
if [ "$DECK" -ne 0 ]; then
    echo "FAILING CLOSED: no deck, so nothing to read back and nothing to ship."
    exit 1
fi

echo
echo "=== STEP 2/3  build the notebook from source cells ======================"
python3 -u notebooks/build_stagem_nb.py
echo "build rc=$?"

echo
echo "=== STEP 3/3  execute it (every number is printed by a cell) ============"
python3 -u -m notebooks.exec_nb "$NB"
echo "exec rc=$?"

echo
ls -la notebooks/stagem/ 2>/dev/null | head -20
echo
echo "READ EVERY PNG at notebooks/stagem/page1..13.png BEFORE committing or"
echo "uploading. This job deliberately did neither."
echo "end $(date '+%F %T')"
