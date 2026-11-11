#!/bin/bash
#SBATCH --job-name=stagetDraft
#SBATCH --partition=sganguli
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagetDraft_%j.out
#
# THE STAGE T WRITE-UP, DRAFT RENDER -- nothing is committed by this job.
#
# Stage S's deck rendered clean four times and shipped ten wrong pages, two of
# them wrong in their CLAIMS rather than their layout, and each fix meant another
# builder commit. So this render goes to $SCRATCH under a draft sha, the PNGs are
# read back from there, and only then are the builders committed and the real
# deck rendered (staget_writeup_2026-11-11.sbatch). A deck's provenance line has
# to carry the commit of the code that drew it, which is why the final render
# still comes after a commit -- but a KNOWN-WRONG builder should not be the thing
# that commit contains.
#
# Also runs the full suite here rather than in its own job: the session that
# fixed rung_label/sort_key (commit 9ce4bde) was killed afterwards, and the tree
# should be confirmed green before anything is written up on top of it.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
# viz + py-matplotlib are not optional: a deck job without the plotting module
# is a commit job with extra steps (stage S's 51905177 died 19 s in on
# `import matplotlib`).
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

OUT="${SCRATCH}/spin_gnn/staget_draft_${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"
echo "out=$OUT  (draft -- notebooks/staget is untouched until the builders commit)"

echo
echo "=== compile ==========================================================="
python3 -m py_compile notebooks/make_staget_deck.py notebooks/build_staget_nb.py || exit 1
echo "ok"

echo
echo "=== the whole suite, after the kill ==================================="
python3 -u scripts/run_tests.py 2>&1 | tail -15
echo "suite rc=${PIPESTATUS[0]}"

echo
echo "=== draft render ======================================================"
# --commit draft, not a sha: this PDF is never shipped, and stamping it with a
# real sha would make a draft indistinguishable from the artifact.
python3 -u -m notebooks.make_staget_deck --commit draft --out "$OUT"
echo "deck rc=$?"

echo
ls -la "$OUT" | head -25
echo
echo "end $(date '+%F %T')"
