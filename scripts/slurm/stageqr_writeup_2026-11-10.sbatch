#!/bin/bash
#SBATCH --job-name=stageqrWriteup
#SBATCH --partition=sganguli
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stageqrWriteup_%j.out
#
# RENDER the stage Q + stage R deck and results notebook. ONE deck for the two
# stages, because they are two halves of one question: Q edits a GIVEN hint and
# R edits a fact the model DERIVED, and either alone reads as a narrower claim
# than the pair supports.
#
# Does NOT commit and does NOT upload. The PNGs have to be LOOKED AT first --
# that read-back caught five defects in the stage M deck and six in stage N's,
# none of which had an exit code, including a page whose bold line contradicted
# the page before it. Commit and upload follow as a separate step.
#
# In a job, not on a login node: the two counterfactual pages REGENERATE case #1
# of each experiment from the arms' own --seed-start (torch, the DD solver, 80
# instances) and the notebook re-runs both registered scorers as subprocesses.
#
# WHAT FAILS CLOSED HERE, and why that matters more than usual on this deck: the
# counterfactual pages assert that the regenerated case matches row 1 of the
# arm's behaviour CSV on every field it records. If the instance stream ever
# drifts, the deck exits instead of illustrating a case that was never run. A
# deck about counterfactuals showing the wrong counterfactual would be worse
# than no deck.
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

NB=notebooks/sudoku_n48_stageqr_2026-11-10_results.ipynb
COMMIT=$(git rev-parse --short HEAD)

echo "host=$(hostname)  job=$SLURM_JOB_ID  commit=$COMMIT"
echo "start $(date '+%F %T')"

echo
echo "=== STEP 1/3  the deck (20 pages, a PNG each) ==========================="
python3 -u -m notebooks.make_stageqr_deck --commit "$COMMIT"
DECK=$?
echo "deck rc=$DECK"
if [ "$DECK" -ne 0 ]; then
    echo "FAILING CLOSED: no deck, so nothing to read back and nothing to ship."
    exit 1
fi

echo
echo "=== STEP 2/3  build the notebook from source cells ======================"
python3 -u notebooks/build_stageqr_nb.py
echo "build rc=$?"

echo
echo "=== STEP 3/3  execute it (every number is printed by a cell) ============"
python3 -u -m notebooks.exec_nb "$NB"
echo "exec rc=$?"

echo
ls -la notebooks/stageqr/ 2>/dev/null | head -24
echo
echo "READ EVERY PNG at notebooks/stageqr/page1..20.png BEFORE committing or"
echo "uploading. This job deliberately did neither."
echo "end $(date '+%F %T')"
