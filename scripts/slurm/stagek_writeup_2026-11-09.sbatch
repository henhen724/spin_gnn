#!/bin/bash
#SBATCH --job-name=stagekWriteup
#SBATCH --partition=sganguli
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagekWriteup_%j.out
#
# Stage K write-up, run as a job rather than on a login node.
#
# The first attempt at scoring ran `python3 -m scripts.stagek_score` on
# sh02-ln02 and wedged: 11m32s elapsed, 00:00:00 of CPU, 12 MB RSS, state S in
# cl_sync_io_wait, holding two $SCRATCH files open that a fresh `head` read
# back instantly. It was making no progress, so it was stopped. Whatever that
# was, this analysis is sustained small-file I/O across every N=48 run
# directory this programme has produced (nine sufficiency dirs plus E/G/H/I/J/K)
# with a 400-sample bootstrap per cell, which belongs in a job under Sherlock
# rule 1 regardless. CPU only -- no model is loaded, only CSVs and config.json.
#
# All four steps in one job because they re-read the same tree and the notebook
# re-runs the scorer as a subprocess; splitting them would pay that I/O cost
# four times. python3 -u throughout so the log streams instead of buffering,
# which is the other thing that made the login-node attempt opaque.

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

echo "=== STEP 1/4  scorer ==================================================="
python3 -u -m scripts.stagek_score
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
