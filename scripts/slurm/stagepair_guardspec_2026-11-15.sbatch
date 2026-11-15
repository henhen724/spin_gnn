#!/bin/bash
#
# Is stage PAIR's guard targeted, or is the lambda=1 model just fragile?
#
# WHY. 52452867's P4 came out 0.1087 against the 0.10 bar, falsifying my
# registered PRED_GUARD_FIRES = False -- but the clean-emission denominator fell
# 656 -> 506 on the same arm, so part of that could be a general loss of
# emission confidence rather than a guard. The conditional placebo analogue,
# P(placebo stops emitting v | clean emits v), is the control the first probe
# printed no denominator for. Marginals cannot settle it: clean 506 -> corrupt
# 479 is a net -27 while the guard counted 55 turn-offs, so 28 turned ON.
#
# EVAL ONLY. 52452867 kept both warm checkpoints, so nothing is retrained: three
# models, the same registered case list rebuilt from the same seeds, and the
# per-case rows written to disk so no later question needs a GPU at all. The
# guard column reproducing 0.0015/0.0031/0.1087 is the check that the rebuilt
# case list is the same one.
#
# NOT A STAGE and nothing registered changes: no new term, no new threshold, no
# bar re-read. It adds one conditional that should have been in the first probe.
#
# TIME. Case list from 1,400 instances (~1 min) + 3 probes of 1,000 cases x 3
# forwards (~6). 0:40 against that ~7.
#
#SBATCH --job-name=stagepairGuardSpec
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%j.out

set -uo pipefail
cd /home/users/hshunt/spin_gnn

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"
git rev-parse --short HEAD

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# No `set -e`: the per-arm table is the point even if a later arm dies.
python3 scripts/stagepair_guardspec.py
echo "guardspec exit=$?"

echo
echo "end $(date '+%F %T')"
