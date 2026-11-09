#!/bin/bash
#SBATCH --job-name=stagekWriteup2
#SBATCH --partition=sganguli
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagekWriteup2_%j.out
#
# Stage K write-up, second pass: rebuild the deck and the notebook after three
# presentation fixes that reading the rendered pages back as PNGs found.
#
# No number, no branch and no verdict changed -- see the commit message. The
# scorer is NOT re-run as its own step: it is untouched, its verdict is already
# in stagekWriteup_51699439.out, and step 3 re-runs it as a subprocess anyway.
#
# Step 1 is the dry run, repeated because the frontier page's prose is now a
# three-way branch and kb_text's crossed branch a two-way one; the observed data
# only exercises one path through each, and the other paths are exactly where an
# unconditional sentence became wrong the first time.
#
# Same resources and the same reason as 51699439: CPU-only small-file I/O across
# every N=48 run directory with a 400-sample bootstrap, which belongs in a job
# (rule 1) and which wedged when it was tried on a login node.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 \
    py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312 \
    py-matplotlib/3.8.3_py312 2>/dev/null

export NUMBA_CACHE_DIR=$SCRATCH/numba_cache
export MPLCONFIGDIR=$SCRATCH/mplcache

COMMIT=$(git rev-parse --short HEAD)
NB=notebooks/sudoku_n48_stagek_2026-11-09_results.ipynb

echo "host=$(hostname)  job=$SLURM_JOB_ID  commit=$COMMIT"
echo "start $(date '+%F %T')"

echo "=== STEP 1/4  branch dry run ==========================================="
python3 -u /scratch/users/hshunt/stagek_deck_branches.py
echo "step1 rc=$?"

echo "=== STEP 2/4  deck ====================================================="
python3 -u -m notebooks.make_stagek_deck --commit "$COMMIT"
echo "step2 rc=$?"

echo "=== STEP 3/4  notebook source ==========================================="
python3 -u notebooks/build_stagek_nb.py
echo "step3 rc=$?"

echo "=== STEP 4/4  execute notebook =========================================="
python3 -u -m notebooks.exec_nb "$NB"
echo "step4 rc=$?"

echo "end $(date '+%F %T')"
