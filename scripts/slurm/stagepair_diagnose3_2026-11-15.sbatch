#!/bin/bash
#
# Three-arm instrumented diagnostic for finding 79 -- why stage PAIR's array
# (52423780) sat at uniform output for 45,000 steps. NOT A STAGE AND NOT A
# RESULT: no bar is read, nothing here goes in a deck, and nothing it finds
# changes the registered term. The three candidate design changes stay hshunt's
# call either way; this job is for the mechanism, which is currently unknown.
#
# WHAT IT ADDS OVER 52430713, which was the first version of this script:
#
#  1. A THIRD ARM at pair_weight = 1e-6. It still runs the second forward pass,
#     still consumes that pass's dropout RNG, still clones the batch -- and
#     exerts no force. So it differs from the control in everything but the
#     force and from the registered arm in nothing but the force, which is the
#     only clean way to separate "the margin pushes the body somewhere bad" from
#     "looking twice under dropout 0.1 is itself the problem".
#
#  2. THE PER-TERM BODY GRADIENT, measured on the CURRENT parameters every 250
#     steps by backward()ing each term alone. Smoke assertion 8 gives this ratio
#     at a fresh init with a live channel and it is 0.393 -- the margin does not
#     out-pull the main loss there. Nobody has measured it during training,
#     which is where the hypothesis says it crosses 1.0. In the training step
#     the terms are summed before backward, so the total the optimizer sees
#     cannot be decomposed after the fact; it has to be measured separately.
#
#  3. 7,500 STEPS, which is the control's real epoch 0, so `main` is directly
#     comparable to its 2.6375 rather than to nothing. 900 steps was 12% of an
#     epoch and could not date an onset -- and 52430713's auto-verdict line
#     tested `< 2.5`, a threshold no 900-step arm could reach, so it printed
#     "the term is NOT the cause" about a budget that could not have shown
#     either way. That line is deleted rather than re-thresholded: this script
#     now prints its tables and no conclusion.
#
# TIME. Arm A about 76 ms/step, arms B and C about 151 ms (two forwards), so
# 7,500 steps is roughly 10 + 19 + 19 = 48 minutes, plus about 6 for generating
# and DD-solving 60,000 instances. 1:40 is requested against that ~55.
#
#SBATCH --job-name=stagepairDiag3
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:40:00
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

# No `set -e`: if one arm dies the two that already printed are still the point.
python3 scripts/stagepair_diagnose.py
echo "diagnose exit=$?"

echo
echo "end $(date '+%F %T')"
