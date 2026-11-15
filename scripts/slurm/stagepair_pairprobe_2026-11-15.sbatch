#!/bin/bash
#
# Stage PAIR's registered risk P2: did the satisfied margin learn the predicate
# or an edit detector? The placebo is the discriminator and it is free.
#
# WHY NOW. STATUS next-step 2 called this "the highest-value unwritten script",
# and finding 80b unblocked it by producing a model whose margin is satisfied.
# 52441962 deliberately kept no checkpoint, so the two warm arms are retrained
# here -- and this time BOTH are written to disk, so the next probe is eval-only.
#
# THREE MODELS, ONE CASE LIST, at the registered PAIR_EVAL_SEED_START on 1,000
# cases: stage Y untouched (finding 78's 0.49, re-measured because 78's number
# came from a different stream), warm lambda=0 (controls for the 30,000 extra
# instances), warm lambda=1 (the satisfied margin). Without the middle arm a
# movement could be the data rather than the margin.
#
# NOT A STAGE: one seed, MIN_SEEDS is 2, and `check_p1` refuses on its own --
# the script prints the registered checks' own output and no verdict of mine.
#
# TIME. Data 30,000 (~3 min), two arms of 3,000 steps (~4 + ~8), the eval case
# list (~1), three probes of 1,000 cases x 3 forwards plus 600-instance
# stratified AUROC (~6). 1:15 against that ~22.
#
#SBATCH --job-name=stagepairProbe
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:15:00
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
python3 scripts/stagepair_pairprobe.py
echo "pairprobe exit=$?"

echo
echo "end $(date '+%F %T')"
