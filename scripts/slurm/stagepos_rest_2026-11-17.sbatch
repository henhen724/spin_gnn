#!/bin/bash
#
# STAGE POS, JOB 2 OF 2: THE OTHER 73 RUNS. NOTHING IS TRAINED.
#
# Same two measurements as job 1 -- A, teacher-forced next-token accuracy at
# every reasoning position; B, greedy rollout with per-pair node validity,
# colour correctness, first-error index and hazard -- on every successful
# (cell, seed) that hshunt did not name by hand. 80 runs in total, 7 in job 1,
# 73 here, and the two sets are DISJOINT by construction: job 1 is
# POS_RUNS=seven, this is POS_RUNS=rest, and the ten shards cover `rest`
# exactly once (checked before submission, no duplicates).
#
# SHARDING IS ROUND ROBIN, not contiguous. stagepos_cells sorts the runs by
# depth and then width, so contiguous task 9 would draw eight L=16 cells while
# task 0 drew eight L=3 cells and the array's wall clock would be set by the
# unlucky task. POS_SHARD="i/10" takes every tenth run, which puts a mix of
# depths in each: 7 or 8 runs per task, every task spanning L=3..16 or
# L=6..16.
#
# WHY IT DEPENDS ON JOB 1. Not for the result -- for the instances. All 80 runs
# share one instance distribution, so the 14 x 400 eval instances are built
# ONCE by job 1 into runs/stagepos_2026-11-17/instances/ and every task here
# loads the same pickles. That is what makes all 80 curves comparable point by
# point rather than merely drawn from the same generator, and it is why this is
# afterok and not a bare submission. The cache writer is an atomic os.replace,
# so even a race between tasks cannot tear a file.
#
# THE POPULATION IS 79 AND THE SCRIPT REFUSES ANY OTHER NUMBER. Every
# (cell, seed) from acc_vs_solve-style collection over the 15 ALL_DIRS scan
# directories whose solve rate at rho = 0.50 is nonzero: 20 full, 40 no_mlp,
# 11 tied_no_mlp, 5 linear, 3 rf_mlp, depths 3 to 16. All three depth-1 runs
# drop out, as hshunt expected, and so do the depth-2 ones. The eighth cell
# hshunt named, full L=2 d256 s0, fails the filter and is measured in job 1
# flagged in79=False.
#
# PREDICTIONS AND FALSIFIERS ARE REGISTERED IN
# runs/stagepos_2026-11-17/REGISTERED.md, committed before job 1 was submitted.
# Nothing here changes them and no verdict is printed.
#
# TIME. Job 1 measures the per-run cost on the same hardware; 4 hours for 7 or
# 8 runs is heavy margin on a few minutes each, chosen so a slow node or a busy
# GPU cannot silently truncate a shard. %5 keeps at most five tasks resident so
# this does not take the whole partition.
#
#SBATCH --job-name=stageposRest
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-9%5
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%A_%a.out

set -uo pipefail
cd /home/users/hshunt/spin_gnn

export POS_OUT="/scratch/users/hshunt/spin_gnn/runs/stagepos_2026-11-17"
export POS_RUNS="rest"
export POS_SHARD="${SLURM_ARRAY_TASK_ID}/10"
export POS_MEASURE="AB"
export POS_BATCH="64"

echo "host=$(hostname)  job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  shard=${POS_SHARD}"
echo "start $(date '+%F %T')"
git rev-parse --short HEAD

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo
echo "################ SHARD ${POS_SHARD}, A and B  $(date '+%T') ################"
python3 scripts/stagepos_score.py
echo "exit=$?"

echo
echo "end $(date '+%F %T')"
