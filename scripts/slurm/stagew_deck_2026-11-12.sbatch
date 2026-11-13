#!/bin/bash
#SBATCH --job-name=stagewDeck
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagewDeck_%j.out
#
# Render stage W's deck into decks/spin-gnn/budget -- a NESTED deck. ~/CLAUDE.md
# changed on 2026-11-12: decks live at decks/<project>/<deck>/ and /<project>/
# becomes a contents page. Stage V's deck moved down to decks/spin-gnn/schedule/
# in slides commit 96324f2, so publishing the bare project name is now refused;
# the publish argument is `spin-gnn/budget`. One password covers both decks, so
# the link already shared keeps working.
#
# No scorer step here, unlike stage V's deck job: stage W's registered scorer
# already ran as the array's dependency (job 52062829) and its output is on disk
# at stagewScore_52062829.out. Re-running it would burn a second 27 seconds to
# print the same 175 lines, and Rule 10 asks for no duplicate submissions.
#
# In a job because the builder imports torch through deck_tokenization (for the
# real instance the tokenization and trace pages show) and because $HOME I/O
# from the login node wedges. The commit runs BEFORE the render so the
# provenance line on every page names a sha that actually contains the builder
# that drew it.
#
# Rendering only. Publishing is deliberately NOT here: every page gets read back
# as a PNG, twice, before it goes out. Across stages S-V those read-backs caught
# seventeen defects, five of them wrong claims on decks that rendered clean.

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
                      notebooks/stagew_figs.py || { echo "py_compile FAILED"; exit 1; }
python3 scripts/deck_lint.py notebooks/make_stagew_web.py || {
    echo "deck_lint FAILED"; exit 1; }
echo "lint: PASS"

echo
echo "=== 2. the shared explanation pages check against the scorer ====="
python3 -u -m notebooks.deck_explain || { echo "deck_explain FAILED"; exit 1; }

echo
echo "=== 3. commit the builder, so --commit names real code =========="
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged or committed."
    exit 1
fi
git add notebooks/make_stagew_web.py notebooks/deck_explain.py \
        scripts/slurm/stagew_deck_2026-11-12.sbatch
git diff --cached --name-only | sed 's/^/  staged /'
if ! git diff --cached --quiet; then
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -m "Stage W's deck: twice the budget bought training loss, and solving did not follow

One knob, zero parameters: stage S's tied d320 cell trained for 100 epochs
instead of 50, 613,818 trainable either way. Of the five registered predictions
one survives. W-c HOLDS -- train NLL fell within every run, by 0.0244 / 0.0059 /
0.0089, median 0.0089 against the 0.0065 bar stagew_score derived from stage S's
measured tail slope before any cell landed. Everything downstream of that loss
is flat or worse: W-a is falsified at 1/3 against 0.4841 (0.4856 / 0.4814 /
0.4594, and the 0.0263 seed spread is 24x the 0.0011 gain the bar asks for, so
one clearing seed is not evidence), W-b misses the gate with rho_50 0.757 ->
0.766 and 2/3 crossing where the control crossed 3/3, and W-e is falsified in
the direction that matters: the training loss fell and rho_50 did not.

W-d holds in one seed and is the stage's most interesting number. Seed 0 is the
first k=1 cell on disk to end below 1.561 AND clear the accuracy bar -- the
basin stage U only reached with extra blocks -- and it got there gradually: the
registered step detector fires in neither arm, so the basin is reachable by
descent and not only by a step. Its rho_50 is 0.760, inside the control's own
three-seed range of 0.754-0.765. All three seeds sit on the accuracy-to-solve
curve (-0.0158 / +0.0005 / -0.0030 against the 0.0423 leave-one-out bar), so the
budget moved cells ALONG the existing relation rather than changing it.

The deck is the first to carry the explanation pages ~/CLAUDE.md now requires:
thirteen stage-independent pages from notebooks/deck_explain.py, which define
the task, the hint density, direct deduction, the architecture, solve rate,
rho_50, censoring, the gate, token accuracy, the three NLL levels, the leak
control and the two baselines -- each computed quantity with an equation, every
symbol introduced, asserted against the scorer that owns the number. LEVELS is
amended here: 'the two basin cells are the best rho_50 seeds' is no longer true
of every basin cell, and stage W is why.

Nothing is re-derived. Rungs, levels and accuracies come through the scorer's
own load_pooled / rung / summarize / acc_of; per-bin curves are load_all rows
filtered by rung_key; the within-run W-c pair is the scorer's own nll_at() at
epoch 50; the step verdict and the training-curve figure come from
step_epoch_census and stagew_figs. check_claims() asserts every page title
against the loaded data -- including that the basin seed, the accuracy-clearing
seed and the no-better-rho_50 seed are the SAME seed -- so a re-scored grid
stops the build instead of republishing a stale sentence to a live URL.

No new arm is invented. W-f was registered by the scorer as what asks the next
question (100 epochs at k=2: do roles and budget compound, or is there one basin
either route finds?) and is not submitted here. The last page leaves the
standing minimum at J2's 0.580."
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
