#!/bin/bash
#
# Stage PAIR seeds 1 and 2: does finding 82 replicate on a different body?
#
# WHY. 52452867 measured corrupt 0.8710 / placebo 0.4360 on ONE seed, and
# `MIN_SEEDS = 2`, so every registered check returned `pass: false` with an empty
# `scored` list. That refusal is the stage's rule working, and a second and third
# seed are the only thing that answers it. Nothing registered changes here: same
# term, same lambda=1, same margin 1.0, same eval stream, same bars.
#
# A SEED IS A DIFFERENT BODY. `PAIR_SEED` selects which of stage Y's three
# checkpoints to warm from, so this varies the basin as the stage intends and not
# merely the data order. Seed 0's code path is untouched.
#
# SEED 2 IS THE INTERESTING ONE AND THE PREDICTION IS SPLIT. Stage Y's own
# epoch-mean train nll is 1.5969 / 1.5933 / 1.6577, so seed 2's BODY already sits
# above the 1.630 convergence threshold before any margin term exists. Predicting
# seed 1 replicates (corrupt >= 0.65, placebo in band) and seed 2 replicates the
# AUROCs while failing `converged` for a reason that has nothing to do with
# L_pair.
#
# TWO RUNS, ONE JOB, because each is ~13 min and Rule 10 says aggregate. Each
# writes its own directory and its own checkpoints; `set -e` is off so seed 2
# still runs if seed 1 dies.
#
# TIME. Per seed: 30,000 instances (~4 min), two arms x 3,000 steps (~8), eval
# (~2). 1:30 against that ~28.
#
#SBATCH --job-name=stagepairSeeds12
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
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
  echo "################ PAIR_SEED=$s  $(date '+%T') ################"
  PAIR_SEED=$s python3 scripts/stagepair_pairprobe.py
  echo "seed $s exit=$?"
done

echo
echo "end $(date '+%F %T')"
