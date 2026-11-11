#!/bin/bash
#SBATCH --job-name=stagetWebFix
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagetWebFix_%j.out
#
# ONE BULLET, found by reading the first render (51957571) back.
#
#   "T* = 16, within 0.01 of T=16's mean solve"
#
# T* is DEFINED as the smallest T at or below the trained count whose mean solve
# is within NEAR of the trained count's, so that sentence states only that 16 is
# within 0.01 of itself. It now says what the test was: "T* = 16: the smallest T
# within 0.01 of the trained count".
#
# Still no publish. deck.md is read back again after this render and
# claude-slides-publish.sh runs from staget_publish_2026-11-11.sbatch.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists."
    exit 1
fi

python3 -m py_compile notebooks/make_staget_web.py || exit 1

echo
echo "=== commit the fix BEFORE rendering ==================================="
git add notebooks/make_staget_web.py scripts/slurm/staget_webfix_2026-11-11.sbatch || exit 1
staged=$(git diff --cached --name-only)
printf '%s\n' "$staged" | sed 's/^/  /'
if [ -n "$staged" ]; then
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -F - <<'MSG'
Stage T web deck: say what the T* test was, not that 16 equals itself

Read back off the first render (51957571). The page "All sixteen applications are
used: none is padding" opened with

    T* = 16, within 0.01 of T=16's mean solve

and T* is defined as the smallest T at or below the trained count whose mean
solve is within NEAR of the trained count's -- so the bullet asserted only that
16 is within 0.01 of itself. The PDF's longer wording ("the smallest T within
0.01 of T=16's mean solve") carried the test and the shortening dropped it.

Now: "T* = 16: the smallest T within 0.01 of the trained count". The claim the
page makes is unchanged and still rests on the second bullet, which is the one
with the number in it: T=12 costs 0.0186 of mean solve.
MSG
    echo "commit rc=$?"
else
    echo "  (nothing to commit)"
fi
SHA=$(git rev-parse --short HEAD)
echo "builder commit: $SHA"

echo
echo "=== re-render ========================================================="
python3 -u -m notebooks.make_staget_web --commit "$SHA"
rc=$?
echo "render rc=$rc"
echo
grep -n -A4 'All sixteen applications' /home/users/hshunt/slides/decks/spin-gnn/deck.md
echo
echo "end $(date '+%F %T')"
exit "$rc"
