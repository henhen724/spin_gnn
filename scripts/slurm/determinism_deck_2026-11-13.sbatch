#!/bin/bash
#SBATCH --job-name=detDeck
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/detDeck_%j.out
#
# Render decks/spin-gnn/determinism from job 52181659's four cells. Submitted
# WITH --dependency=afterok on that job, so it starts the moment the cells land
# instead of being polled for.
#
# It RENDERS and does not PUBLISH. The figures have to be read back as PNGs
# twice before a URL starts serving them -- across stages S to W that read-back
# caught twenty-four defects, nine of them wrong CLAIMS on decks that rendered
# clean -- and that read-back is the one step a job cannot do.
#
# tools/build.sh is scoped to spin-gnn/determinism on purpose: ~/slides is
# shared with other sessions and one of them has an uncommitted deck of its own
# in that repo.

set -uo pipefail
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
date +'start %F %T'

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
            py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
            py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR=/scratch/users/hshunt/numba_cache
export PYTHONUNBUFFERED=1

echo
echo "=== 1. compile and lint =========================================="
python3 -m py_compile notebooks/make_determinism_web.py scripts/noise_floor.py || {
    echo "py_compile FAILED"; exit 1; }
python3 scripts/deck_lint.py notebooks/make_determinism_web.py || {
    echo "deck_lint FAILED"; exit 1; }
echo "lint: PASS"

echo
echo "=== 2. the noise floor, from stage W's own logs ==================="
python3 -u scripts/noise_floor.py || { echo "noise_floor FAILED"; exit 1; }

echo
echo "=== 3. explanation pages =========================================="
python3 -u -m notebooks.deck_explain || { echo "deck_explain FAILED"; exit 1; }

echo
echo "=== 4. commit the builder ========================================="
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged or committed."
    exit 1
fi
git add notebooks/make_determinism_web.py \
        scripts/noise_floor.py \
        scripts/deck_lint.py \
        scripts/slurm/determinism_deck_2026-11-13.sbatch
git diff --cached --name-only | sed 's/^/  staged /'
if ! git diff --cached --quiet; then
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -F - <<'MSG'
The determinism check as a deck: spin-gnn/determinism

Reports whether a rerun of one cell reproduces that cell. Not a stage: no
scientific knob moves and no published number can change, which the last page
says in those words.

Nothing is re-derived. The three stage-W pairs, the paired standard deviation
and the replicate count come from scripts/noise_floor.py; the 0.0011 effect
size and the S1 constants come through scripts/stagew_score.py from the stage
that measured them; the four cells' numbers are read from their history.json
and never from the job log. The one exception is wall clock, which is a
property of a run and has nowhere else to live, so it is parsed from the job's
own 'cell X rc=0 wall=..' lines and only for cells that exited 0.

check_claims() asserts, per cell, that config.json agrees with the arm it was
filed under (deterministic true/false), that it is S1's 613,818 trainable, seed
0, 3 epochs and 240,000 instances, and that noise_floor's effect size is the
scorer's to twelve digits -- two modules naming one number is how a deck comes
to contradict itself. The headline is chosen from the verdict rather than
written: there is a branch for H1 holding with H2 failing, which is the outcome
that would mean kernel nondeterminism was never the cause.

The shared explanation pages come from deck_explain minus CENSOR and LEAK,
which this deck never uses; five more are specific to the question -- what a
seed fixes and what it does not, what a replicate is, what the flag pins, the
replicate-count equation with every symbol defined, and the hypotheses as they
were written before the job was submitted.

Second attempt. The first (23a4b15, job 52183401) refused to render, and the
refusal was correct: check_claims compares noise_floor's effect size against
the scorer's to twelve digits and they disagreed by 4%, because noise_floor had
0.0011 typed into it while stage V DERIVES 0.0010565 and 0.0011 is only the
rounding this project publishes. noise_floor now imports NEED from the module
that defines it, which costs that file its stdlib-only property -- the import
chain reaches numpy, so it no longer runs on the login node's python 3.6.8.
Every figure still prints 0.0011 at %.4f; the replicate counts grow about 8%
because they divide by NEED squared: 1294 -> 1404 and 66 -> 71 replicates per
arm. Both were already out of reach and still are.

H3's page title also now branches on the measured sign instead of asserting a
cost in advance: the pinned cells came in at 34m37s and 34m39s against 32m14s
and 32m3s unpinned, so the cost is real here, but the title had been written
before any of those four numbers existed.

Third attempt. The second (48bade3, job 52189773) died in slides.py on a
points_page carrying 5 bullets against a cap of 3; deck_lint checked word
counts per literal and not how many literals a page had, so three over-full
pages passed the lint and the first of them cost a job. deck_lint now counts
bullets per points_page, and the three pages are split rather than compressed.
The existing stage builders still lint clean, so the new check does not
retroactively condemn a published deck.

Then the deck was rendered and read back twice on the login node before this
submission, which found five more defects that a render cannot see:

  - the spread figure's title, "Every run of this one cell lands somewhere
    different after epoch 0", was refuted by its own figure -- the pinned pair
    plots as one point, which IS the result. Now "Only the pinned pair lands
    twice in the same place";
  - that figure's x axis said det a / free a, this job's directory names,
    defined nowhere in the deck, while every other page says pinned/unpinned;
  - the LEAK explanation page had been dropped as unused, but the shared
    tokenization figure asserts "THE ORDER WITHIN A ROW IS THE LEAK" in bold.
    A term a FIGURE introduces is a term the deck has to define, so LEAK is
    back and only CENSOR is dropped;
  - the wall-clock table rounded to whole minutes, so a reader computing
    (35 - 32) / 3 got 1.0 min per epoch while the note said 0.8. One decimal;
  - "the count is out of reach: 562 GPU-hours at best" asserted that the
    optimistic reading is unreachable. 562 GPU-hours is about three weeks of
    one GPU -- expensive, not impossible. It now states both counts.
MSG
    echo "commit rc=$?"
fi
SHA=$(git rev-parse --short HEAD)
echo "builder commit: $SHA"

echo
echo "=== 5. render ====================================================="
python3 -u -m notebooks.make_determinism_web --commit "$SHA" --job 52181659 \
    --log /scratch/users/hshunt/spin_gnn/sud2N48Det_52181659.out
RC=$?
echo "render rc=$RC"

echo
echo "=== 6. KaTeX and delimiter warnings ==============================="
bash /home/users/hshunt/slides/tools/build.sh spin-gnn/determinism
echo "build rc=$?"

echo
echo "=== 7. push the builder ==========================================="
module load system git/2.45.1
git push origin sudoku-dd
echo "push rc=$?"

echo
echo "NOT PUBLISHED -- the figures still have to be read back twice."
date +'end %F %T'
exit $RC
