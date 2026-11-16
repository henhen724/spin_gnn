#!/bin/bash
#
# The placebo-corrected guard excess on seeds 1 and 2, plus seed 0 as a
# regression test on the GUARD_SEED refactor.
#
# WHY. 52456376 replicated the predicate result on three bodies (corrupt 0.8710
# / 0.8620 / 0.8710, placebo 0.4360 / 0.4450 / 0.4420) but its RAW guard
# conditionals are 0.1087, 0.2072 and 0.1632 -- all three over the 0.10 bar, and
# seeds 1 and 2 well over seed 0. Finding 83 showed that on seed 0 most of that
# raw rate is churn: the placebo, which leaves v forced, also stops the emission
# 0.0672 of the time, leaving a corrected excess of 0.0415 [0.0108, 0.0722],
# entirely below the bar. Whether that correction holds on the other two bodies
# decides whether the guard claim survives at all, and the pairprobe does not
# measure it -- only the per-case rows do.
#
# The prediction is registered in the script's docstring before the numbers:
# excess fraction 0.25-0.55 of the raw rate on both seeds, CI lower bound above
# zero on both.
#
# EVAL ONLY. No training, no new checkpoints, same registered 1,000-case list at
# seed stream 920,000,000 for every seed (DATA_SEED stays 777 on purpose, so the
# per-seed rates are comparable). Nothing registered changes.
#
# SEED 0 LAST, AND INTO ITS OWN DIRECTORY. GUARD_OUT_TAG=_recheck keeps finding
# 83's rows_*.jsonl untouched while re-running the same code path, so a
# bit-for-bit match is the check that parameterizing the script changed nothing.
#
# TIME. 52455457 was 1:25 for one seed, nearly all of it the CPU case-list
# build. Three is ~6 min; 40 is the margin.
#
#SBATCH --job-name=stagepairGuardSeeds
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

for s in 1 2; do
  echo
  echo "################ GUARD_SEED=$s  $(date '+%T') ################"
  GUARD_SEED=$s python3 scripts/stagepair_guardspec.py
  echo "seed $s exit=$?"
done

echo
echo "################ GUARD_SEED=0 regression recheck  $(date '+%T') ########"
GUARD_SEED=0 GUARD_OUT_TAG=_recheck python3 scripts/stagepair_guardspec.py
echo "seed 0 recheck exit=$?"

echo
echo "end $(date '+%F %T')"
