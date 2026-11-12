#!/bin/bash
#SBATCH --job-name=stagevDeck
#SBATCH --partition=sganguli
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagevDeck_%j.out
#
# Render stage V's results deck into decks/spin-gnn, REPLACING stage U's pages
# in the project's one gated deck (re-publishing keeps the password; stage U's
# pages stay in the slides repo's history).
#
# Step 2 re-runs the registered scorer. Stage V was scored interactively and its
# output was never written to disk -- grep for "STAGE V --" across the project's
# .out files finds only the REGISTRATION job, 51962711. So the scorer runs here,
# in the same job as the render, and this job's .out becomes the stage's scoring
# record. If the scorer and the builder ever disagree about a number, they
# disagree inside one file, at one timestamp, on one checkout.
#
# In a job because the builder imports torch through deck_tokenization (for the
# real instance the tokenization and trace pages show) and because $HOME I/O
# from sh02-ln02 wedges. The commit runs BEFORE the render so the provenance
# line on every page names a sha that actually contains the builder that drew it
# -- 52027787 published a deck crediting the wrong sha and the correction cost
# its own job.
#
# Rendering only. Publishing is deliberately NOT here: every page gets read back
# as a PNG, twice, before it goes out. Across stages S, T and U those read-backs
# caught thirteen defects, five of them wrong claims on decks that rendered
# clean.

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
echo "=== 1. the builder compiles and lints ============================"
python3 -m py_compile notebooks/make_stagev_web.py || {
    echo "py_compile FAILED"; exit 1; }
# Cheaper than finding out inside slides.py on page 17 of 19.
python3 scripts/deck_lint.py notebooks/make_stagev_web.py || {
    echo "deck_lint FAILED"; exit 1; }
echo "lint: PASS"

echo
echo "=== 2. the registered scorer, whose output was never saved ======="
python3 -u -m scripts.stagev_score
echo "scorer rc=$?"

echo
echo "=== 3. commit the builder, so --commit names real code =========="
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged or committed."
    exit 1
fi
git add notebooks/make_stagev_web.py scripts/slurm/stagev_deck_2026-11-12.sbatch
git diff --cached --name-only | sed 's/^/  staged /'
if ! git diff --cached --quiet; then
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -m "Stage V's deck: a longer trained schedule does not help, and destabilizes

V-a is falsified 0/3 against 0.4841, the bar stagev_score.py derived from S4's
steepness before any cell landed. The shape of the failure is the finding: two
seeds land at 0.4833 and 0.4834 against the control's 0.4830 -- unchanged, not
worse -- while the third never trained (0.4612, train NLL 1.6334, easy end 0.065,
rho_50 censored). The median rho_50 does move the wrong way, 0.757 -> 0.769, but
one broken seed moved it, and the seed spread went from 0.0010 to 0.0222.

The first render of this deck asserted its way out of two wrong titles: 'every
T=24 seed is worse than its control' is false (seed 2 is 0.757 against 0.765),
and the frozen-at-24 comparison tripped a guard because stage T reports a level
as the median over CROSSING seeds (0.819) where summarize() is bound-aware
(0.875). The pair page now draws one estimator everywhere and footnotes the
other; the claim holds under both.

The builder re-derives nothing. Rungs, levels and accuracies come from the
scorer's own load_pooled / rung / summarize / acc_of; per-bin curves are
load_all rows filtered by rung_key, so a point on a slide belongs to the cell
the scorer scored; stage T's frozen-at-24 arm is recomputed through cell_metrics
and checked against stage T's published 0.819 before it is drawn; the
sufficiency residuals come from acc_vs_solve's own collect / pava / predict /
rms. Every comparative page title is computed from the loaded data and asserted
in check_claims() before a page is written, so a re-scored grid stops the build
instead of republishing a stale sentence to a live URL.

V-d (T=32) was cancelled by V-f, not skipped: the adjudicator failed on the
statistic that cannot censor, and V-f named the training budget -- stage W, 100
epochs at constant 1e-3, k=1, T=16 -- as what runs next. The last page leaves
the standing minimum at J2's 0.580."
    echo "commit rc=$?"
fi
SHA=$(git rev-parse --short HEAD)
echo "builder commit: $SHA"

echo
echo "=== 4. render ===================================================="
python3 -u -m notebooks.make_stagev_web --commit "$SHA"
RC=$?
echo "render rc=$RC"

echo
echo "=== 5. push ======================================================"
module load system git/2.45.1
git push origin sudoku-dd
echo "push rc=$?"

echo "end $(date '+%F %T')"
exit $RC
