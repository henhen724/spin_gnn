#!/bin/bash
#SBATCH -J spin_gnn_test
#SBATCH -p sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o /scratch/users/hshunt/spin_gnn/%x-%j.out

# Runs the whole test suite. Cheap and CPU-only -- run it after any change to
# the model or the analysis modules, before trusting a result that came out of
# them.
#
# This script used to `source $SCRATCH/spin_gnn_venv/bin/activate` and then set
# PYTHONPATH to that venv's site-packages. Both are wrong now and the second
# always was: the venv was purged off $SCRATCH, and an inline PYTHONPATH
# shadows the py-* modules' own numpy/scipy/torch. Modules only, and the repo
# reached through its own directory rather than through PYTHONPATH.
#
# It also ran `python -m pytest`, which does not exist for py312 here, so the
# whole thing exited before collecting anything.

set -euo pipefail
module purge
# pandas/scipy are here because the eval modules the tests import reach for
# them; without those two a converted test file fails on an import rather than
# on anything it is checking.
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

echo "host=$(hostname)  job=${SLURM_JOB_ID}"
echo "start $(date '+%F %T')"
# `|| rc=$?` and not a bare call: with set -e a failing suite would exit here
# and the end line -- the thing that says the job reached the end at all --
# would never print.
rc=0
python3 scripts/run_tests.py "$@" || rc=$?
echo "end $(date '+%F %T')  rc=$rc"
exit "$rc"
