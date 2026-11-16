#!/bin/bash
#
# DIRECTION, NOT MAGNITUDE: the one open hypothesis finding 88 left behind.
#
# 88 destroyed three bodies by forcing the determinacy feedback open at a
# measured 10% of the residual stream -- train nll 9.18 / 6.45 / 6.73 nats at
# step 0, before any gradient step, against log 58 = 4.060. The published
# `/spin-gnn/loop/` deck offers one explanation and labels it UNMEASURED: the
# injected vector is an average of node-token embedding rows added at every one
# of the ~391 positions, so it may be a common-mode translation of the whole
# stream rather than a position-specific correction. 88 calibrated the SIZE of
# the injection and never looked at its DIRECTION.
#
# `scripts/stageloop_direction.py` looks. No training, no checkpoint written, no
# GPU: three forward passes per body through the same three stage Y bodies 87 and
# 88 warm-started from, with a forward hook on the shared block capturing the
# hidden state at each of the 16 applications, and `delta_t` recomputed from it
# by the model's own `determinacy_feedback_delta`. Two scale-free statistics --
# the common-mode fraction and the mean pairwise cosine across positions -- for
# the injection, for the block's own per-application update (the control that
# makes "common-mode" a property of the injection rather than of this cell) and
# for the residual stream itself (the control that makes it a property of the
# injection rather than of h). Predictions are registered in the script's
# docstring and committed before this is submitted, as 84-88's were.
#
# WHY A CPU JOB. One forward pass per mode: a GPU would spend the whole
# allocation idle. Rule 1 (nothing sustained on a login node) is why it is a job
# at all rather than an interactive python.
#
# NOTHING REGISTERED CHANGES AND NOTHING IS TRAINED. No bar in `stagepair_score`
# is read as a threshold, no artifact of 87 or 88 is touched, and the output goes
# to a NEW directory, `runs/stageloop_direction_2026-11-15`. NO VERDICT.
#
# TIME. Three bodies x three forwards of a 614k-parameter model on 32 sequences
# of ~391 tokens, plus the instance builds: minutes. 30 is margin.
#
#SBATCH --job-name=loopDirection
#SBATCH --partition=sganguli
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%j.out

set -uo pipefail
cd /home/users/hshunt/spin_gnn

echo "host=$(hostname)  job=${SLURM_JOB_ID}"
echo "start $(date '+%F %T')"
git rev-parse --short HEAD

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

echo
echo "################ SMOKE (shapes only)  $(date '+%T') ################"
if ! DIR_SMOKE=1 python3 scripts/stageloop_direction.py; then
  echo "SMOKE FAILED -- the real run is NOT started. Fix the code path first."
  exit 1
fi

echo
echo "################ THE RUN  $(date '+%T') ################"
python3 scripts/stageloop_direction.py
echo "exit=$?"

echo
echo "end $(date '+%F %T')"
