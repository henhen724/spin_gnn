#!/bin/bash
#SBATCH --job-name=stagewDeck2
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagewDeck2_%j.out
#
# Re-render decks/spin-gnn/budget after the FIGURE read-back. The first render
# (job 52148320) was clean and its text read-back fixed six defects; reading
# curves.png back then found a seventh and worst kind -- a page title the figure
# refutes. The page said "reached the basin gradually, with no step anywhere"
# over a plot showing an abrupt fall at epoch 58.
#
# What changed, and why each is not cosmetic:
#  * step_epoch_census.onset() -- reporting-only sibling of step_epoch(), which
#    returns the biggest one-epoch fall out of a flat run-in WITH its
#    persistence, instead of hiding every fall that misses only that test.
#    step_epoch() itself is untouched: the registered verdict does not move.
#    Seed 0: epoch 58, fall 0.0102 (bar 0.005), flat run-in 0.0020, persistence
#    0.0145 against the 0.020 bar -- so not a step by the registered definition,
#    and plainly a fall. It is also the epoch that crossed the 1.561 basin line,
#    which is asserted, because that is the verb in the new title.
#  * The control's seed 2 DOES step, at epoch 22, which the census hand-read
#    before stage W ran. "No step fires in either arm" was false twice over.
#  * paired() + two table pages: at a matched seed the two arrays diverge from
#    epoch 0 (train NLL 2.5167 vs 2.3656 on seed 0, up to 0.2413 on seed 2) and
#    are still apart at the control's last epoch (+0.0602 train NLL, -0.0246 val
#    accuracy on seed 2). Field-by-field both config.json files are identical
#    except the epoch count, the stored majority-class baseline matches to 16
#    digits, so the data and init are identical; nothing in the training script
#    pins GPU kernel nondeterminism. They are independent runs of one
#    configuration. Seed 2's 100-epoch run was already 0.0246 below its control
#    in accuracy BEFORE the extra budget existed, which is most of why W-a reads
#    1/3. The within-run form of the same quantity is +0.0016/+0.0011/+0.0007 --
#    reported as a measurement, not as a replacement arm, which the registration
#    does not allow and which the gate would not support anyway.
#  * LEVELS now states the step's three tests with their numbers, because a page
#    that says "missed only on persistence" needs the reader to have seen the
#    0.020. RHO50 now says what a seed does and does not fix.
#
# Two pages over the 10-25 guideline for the second time, and recorded as such:
# the guideline is a guideline, "report negative results honestly" is not, and a
# deck whose cross-run pages do not say they are cross-run is the thing being
# fixed.
#
# Still NO publish step here: the re-rendered figure and the new pages have to
# be read back first, twice, like everything else.

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
python3 -m py_compile notebooks/make_stagew_web.py notebooks/deck_explain.py \
                      notebooks/stagew_figs.py scripts/step_epoch_census.py || {
    echo "py_compile FAILED"; exit 1; }
python3 scripts/deck_lint.py notebooks/make_stagew_web.py || {
    echo "deck_lint FAILED"; exit 1; }
echo "lint: PASS"

echo
echo "=== 2. the step census still reads the same verdicts ============="
# onset() must not have disturbed step_epoch(): same census, same table.
python3 -u -m scripts.step_epoch_census | tail -n 12

echo
echo "=== 3. the shared explanation pages check against the scorer ====="
python3 -u -m notebooks.deck_explain || { echo "deck_explain FAILED"; exit 1; }

echo
echo "=== 4. commit, so --commit names the code that drew the pages ===="
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged or committed."
    exit 1
fi
git add notebooks/make_stagew_web.py notebooks/deck_explain.py \
        notebooks/stagew_figs.py scripts/step_epoch_census.py \
        scripts/slurm/stagew_deck_fix_2026-11-12.sbatch
git diff --cached --name-only | sed 's/^/  staged /'
if ! git diff --cached --quiet; then
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -F - <<'MSG'
Stage W's deck, after the figure read-back: the fall at epoch 58, and two runs that are not one

The curves page claimed "reached the basin gradually, with no step anywhere"
over a figure that shows seed 0 falling abruptly at epoch 58. The claim came
from step_epoch(), which answers the registered question (did a STEP happen) and
returns nothing for a fall that misses any one of its three tests. This one
misses exactly one: fall 0.0102 at epoch 58 against the 0.005 bar, flat run-in
0.0020, persistence 0.0145 against the 0.020 bar. So "no step" is right and "no
step anywhere" was a sentence a reader could see was wrong.

step_epoch_census.onset() now reports that candidate with its persistence, next
to an untouched step_epoch(); the page names the epoch, the fall and the missed
bar, the figure marks the same epoch from the same function, and check_claims()
asserts all of it -- including that epoch 58 is the epoch that crossed the 1.561
basin line, which is the verb in the title. It also asserts what the old note
got wrong in the other direction: the CONTROL's seed 2 does step, at epoch 22,
which the census hand-read before stage W ran.

The bigger finding is in the same figure. At a matched seed the two arrays
diverge from epoch 0 -- train NLL 2.5167 vs 2.3656 on seed 0, 2.5915 vs 2.3502
on seed 2 -- and are still apart at the control's last epoch: +0.0602 train NLL
and -0.0246 val accuracy on seed 2. Field by field, both cells' config.json
agree on everything but the epoch count, and the stored majority-class baseline
agrees to 16 digits, so the instances and the initialisation are identical;
nothing in the training script pins GPU kernel nondeterminism. The 100-epoch arm
is an INDEPENDENT RUN of stage S's configuration, not stage S's run continued.

That qualifies every cross-run page of this deck, and in both directions. Seed
2's 100-epoch run was already 0.0246 below its own control in accuracy before
the extra fifty epochs existed, which is most of why W-a reads 1/3 and why the
0.0263 spread is what it is. Measured within run, where the comparison is sound,
the extra budget moved accuracy by +0.0016 / +0.0011 / +0.0007. That is reported
as a measurement and not as a rescued verdict: W-a was registered across runs,
it failed, and rho_50 moved the wrong way regardless. W-c is untouched by any of
this -- it was registered within-run for this exact reason.

Two new table pages carry it, which puts the deck two pages over the 10-25
guideline for the second time. Recorded deliberately: the guideline is a
guideline, reporting negative results honestly is not, and a deck whose
cross-run pages do not say they are cross-run is the defect being fixed. LEVELS
now states the step's three tests with their numbers (a page that says "missed
only on persistence" has to show the 0.020), and RHO50 now says what a seed
fixes and what it does not.

No new arm. W-f stays registered and unsubmitted.
MSG
    echo "commit rc=$?"
fi
SHA=$(git rev-parse --short HEAD)
echo "builder commit: $SHA"

echo
echo "=== 5. render ===================================================="
python3 -u -m notebooks.make_stagew_web --commit "$SHA"
RC=$?
echo "render rc=$RC"

echo
echo "=== 6. push ======================================================"
module load system git/2.45.1
git push origin sudoku-dd
echo "push rc=$?"

echo "end $(date '+%F %T')"
exit $RC
