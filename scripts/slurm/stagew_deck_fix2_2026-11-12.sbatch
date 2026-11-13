#!/bin/bash
#SBATCH --job-name=stagewDeck3
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagewDeck3_%j.out
#
# Third render of decks/spin-gnn/budget, from the first PNG read-back of all
# seven figures. Four defects, all of them things only a rendered image shows:
#
#  * fit.png printed "depth ladder +- its own LOO bar" straight through "seed 2
#    -0.0030" -- both anchored near the knee. The grey annotation moves to the
#    empty upper-left in axes coordinates, and LOO is expanded, since the
#    explanation page spells the bar out and the figure should not abbreviate
#    what the deck just defined.
#  * solve.png had blue = the 50-epoch control while curves.png two pages
#    earlier had blue = the 100-epoch arm. Now MUTED is the control and
#    SERIES[0] the arm under test on both, and orange stays what it is
#    everywhere else in this deck: the median annotation, never an arm.
#  * level.png's note said "per seed, not pooled", which says where the numbers
#    came from but not WHICH estimator drew them -- and the CENSOR page requires
#    that of every figure. It now names the bound-aware median.
#  * The reproducibility page's note spent its words re-deriving what a seed
#    fixes, which the RHO50 explanation page now defines. It spends them on the
#    specific instead: the control's seed-2 step at epoch 22, which is the whole
#    0.0602 gap and which is the unexplained cliff a reader otherwise hits on
#    the curve figure.
#
# No numbers change. Same 28 pages, same commit discipline, no publish: the
# second read-back of every page comes after this.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
            py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
            py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR=/scratch/users/hshunt/numba_cache

echo
echo "=== 1. compile and lint =========================================="
python3 -m py_compile notebooks/make_stagew_web.py notebooks/deck_explain.py \
                      notebooks/stagew_figs.py scripts/step_epoch_census.py || {
    echo "py_compile FAILED"; exit 1; }
python3 scripts/deck_lint.py notebooks/make_stagew_web.py || {
    echo "deck_lint FAILED"; exit 1; }
echo "lint: PASS"

echo
echo "=== 2. explanation pages ========================================="
python3 -u -m notebooks.deck_explain || { echo "deck_explain FAILED"; exit 1; }

echo
echo "=== 3. commit ===================================================="
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged or committed."
    exit 1
fi
git add notebooks/make_stagew_web.py notebooks/deck_explain.py \
        notebooks/stagew_figs.py scripts/step_epoch_census.py \
        scripts/slurm/stagew_deck_fix2_2026-11-12.sbatch
git diff --cached --name-only | sed 's/^/  staged /'
if ! git diff --cached --quiet; then
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -F - <<'MSG'
Stage W's deck: four defects from the first read-back of all seven figures

None of these are visible in the source and all four are visible in the image,
which is the whole argument for reading a deck back as PNGs.

fit.png printed the grey "depth ladder +- its own LOO bar" annotation through
the "seed 2 -0.0030" label; both were anchored near the knee of the isotonic
fit, where the only free space on that axes is not. It moves to the empty
upper-left in axes coordinates, and the abbreviation is expanded to
"leave-one-out", which is what the explanation page calls it.

solve.png used blue for the 50-epoch control, two pages after curves.png used
blue for the 100-epoch arm. A reader who learned the colour on one page read it
backwards on the other. The control is MUTED and the arm under test is SERIES[0]
on both now; orange keeps the one job it has everywhere else here, the median
annotation.

level.png's note said "per seed, not pooled" -- which answers where the numbers
came from, not which of the project's two censored-data estimators produced
them, and the CENSOR explanation page requires exactly that of every figure. It
now names the bound-aware median.

The reproducibility page's note was re-deriving what a seed does and does not
fix, now that the RHO50 page defines it. Those words go to the specific instead:
the control's seed 2 stepped at epoch 22 and its 100-epoch run never did, which
is the entire 0.0602 train NLL gap on that seed and which is the cliff a reader
hits, otherwise unexplained, on the curve figure two pages later.

No result changes. 28 pages, 14 explanation, 7 figures.
MSG
    echo "commit rc=$?"
fi
SHA=$(git rev-parse --short HEAD)
echo "builder commit: $SHA"

echo
echo "=== 4. render ===================================================="
python3 -u -m notebooks.make_stagew_web --commit "$SHA"
RC=$?
echo "render rc=$RC"

echo
echo "=== 5. push ======================================================"
module load system git/2.45.1
git push origin sudoku-dd
echo "push rc=$?"

echo "end $(date '+%F %T')"
exit $RC
