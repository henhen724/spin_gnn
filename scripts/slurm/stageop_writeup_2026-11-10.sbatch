#!/bin/bash
#SBATCH --job-name=stageopWriteup
#SBATCH --partition=sganguli
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stageopWriteup_%j.out
#
# RENDER the stage O + stage P deck and results notebook. ONE deck for the two
# stages, because they are the only two ways to make the standing cell simpler:
# smaller at the same arrangement (O), or the same size with one block instead
# of sixteen (P). Separating them would hide that the two failures are of
# DIFFERENT kinds -- OA measured a level 0.001 past the free line, P1 produced
# no architecture result at all because it never left the plateau.
#
# Does NOT commit and does NOT upload. The PNGs have to be LOOKED AT first --
# that read-back caught five defects in the stage M deck, six in stage N's and
# three in stage Q+R's, none of which had an exit code. Commit and upload follow
# as a separate step.
#
# In a job, not on a login node: the mandatory trace page generates an N=48
# instance and solves it with the DD solver, the sufficiency page fits a PAVA
# curve over twelve scan directories, the notebook re-runs both registered
# scorers as subprocesses, and the whole thing renders 16 figures.
#
# WHAT FAILS CLOSED HERE. main() exits non-zero if any cell is short of 3 seeds,
# if P1's trainable count equals its untied twin's (which would mean the sharing
# did not happen and the whole stage is a mislabelled rerun), or if J2's count
# has drifted from the standing minimum's 2,504,890. A deck comparing against a
# previous best has to be sure which cell that best is.
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

NB=notebooks/sudoku_n48_stageop_2026-11-10_results.ipynb
COMMIT=$(git rev-parse --short HEAD)

echo "host=$(hostname)  job=$SLURM_JOB_ID  commit=$COMMIT"
echo "start $(date '+%F %T')"

echo
echo "=== STEP 1/3  the deck (16 pages, a PNG each) ==========================="
python3 -u -m notebooks.make_stageop_deck --commit "$COMMIT"
DECK=$?
echo "deck rc=$DECK"
if [ "$DECK" -ne 0 ]; then
    echo "FAILING CLOSED: no deck, so nothing to read back and nothing to ship."
    exit 1
fi

echo
echo "=== STEP 2/3  build the notebook from source cells ======================"
python3 -u notebooks/build_stageop_nb.py
echo "build rc=$?"

echo
echo "=== STEP 3/3  execute it (every number is printed by a cell) ============"
python3 -u -m notebooks.exec_nb "$NB"
echo "exec rc=$?"

echo
ls -la notebooks/stageop/ 2>/dev/null | head -22
echo
echo "READ EVERY PNG at notebooks/stageop/page1..16.png BEFORE committing or"
echo "uploading. This job deliberately did neither. Watch page 8 (inverted y"
echo "axis, text near the limit), page 11 (bar labels at +-0.002 against the"
echo "axis limits) and page 9 (three seeds finishing within 0.0075 of each"
echo "other -- the coincident-label trap page 17 of stage Q+R hit)."
echo "end $(date '+%F %T')"
