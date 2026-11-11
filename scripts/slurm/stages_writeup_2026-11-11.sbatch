#!/bin/bash
#SBATCH --job-name=stagesWrite
#SBATCH --partition=sganguli
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagesWrite_%j.out
#
# THE STAGE S WRITE-UP: builders committed FIRST, then rendered.
#
# Stage N's two-commit pattern. The deck prints its own provenance line, and the
# commit in that line has to be the commit of the CODE THAT DREW IT -- so the
# builders are committed before anything renders, that sha is passed to the
# renderer, and the artifacts are committed afterwards by a second job. A deck
# stamped with the commit that contains the deck is stamped with a sha that did
# not exist when it was drawn.
#
# WAITS ON --dependency=afterany:51853969. The d448 arm of stage S was still
# training when this was submitted. Rendering before it lands would produce a
# deck whose width trend stops at d320 and whose accuracy page cannot make the
# saturation claim -- the builder detects that and softens both, but the honest
# deck is the one drawn after the whole array has reported, however it reported.
# afterany, not afterok: two d448 seeds already diverged, and "the arm failed"
# is a result this deck is meant to show, not a reason not to build it.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
# viz + py-matplotlib are NOT optional here: run 51905177 committed the builders,
# then died 19 seconds later on `import matplotlib` and rendered nothing. A deck
# job without the plotting module is a commit job with extra steps.
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"
sacct -j 51853969 -X -o JobID%16,State,ExitCode,Elapsed 2>/dev/null

if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists."
    exit 1
fi

echo
echo "=== compile ==========================================================="
python3 -m py_compile notebooks/make_stages_deck.py notebooks/build_stages_nb.py || exit 1
echo "ok"

echo
echo "=== commit the builders BEFORE rendering =============================="
PATHS="notebooks/make_stages_deck.py
notebooks/build_stages_nb.py
scripts/slurm/stages_writeup_2026-11-11.sbatch"
git add $PATHS || exit 1
staged=$(git diff --cached --name-only)
printf '%s\n' "$staged" | sed 's/^/  /'
if [ -n "$staged" ]; then
    while IFS= read -r p; do
        [ -z "$p" ] && continue
        printf '%s\n' "$staged" | grep -qx -- "$p" && continue
        # Not staged has TWO causes and only one of them is a fault. The fault
        # this guard exists for is an IGNORED path, where `git add` succeeds and
        # stages nothing; an UNCHANGED path stages nothing either, and that is
        # ordinary. Job 51919478 committed two of these three files before it
        # died on a render, so job 51922718 reached this line with
        # build_stages_nb.py already identical to HEAD and failed closed on it.
        # `git ls-files --error-unmatch` separates the two: tracked and unstaged
        # means identical to HEAD, which needs no commit.
        if git ls-files --error-unmatch -- "$p" >/dev/null 2>&1; then
            echo "  ($p unchanged since HEAD -- already committed)"
            continue
        fi
        echo "FAILING CLOSED: $p is not tracked (gitignored?) and did not stage."
        exit 1
    done <<< "$PATHS"
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -F - <<'MSG'
Stage S write-up: eight defects the PNG read-back found

The deck rendered clean (19 pages, 185 KB, exit 0) and eight of its pages
were wrong anyway. None of this is visible to a lint that counts words, and
none of it raised: a deck is checked by LOOKING at every page.

  p1   the subtitle ran off the right edge at "d192 -> d320 -> d448, ap".
       _claim_text wraps a title; title_page's subtitle is one fig.text that
       does not. Broken by hand into two lines, nothing cut.
  p4   "... 1 further rounds", in notebooks/deck_tokenization.py, on every
       instance that elides exactly one round.
  p5   "untied d192" struck through "tied d320": the two curves end 0.04 of
       solve rate apart and label_line writes at the curve's own end. New
       declutter() converts one line of 11pt type into this axes' data units
       and pushes the lower label down; draw_nll uses it too.
  p8   "1.6646 / 1.6721 / 1.6681 0/3" -- table_page gives every column the
       same 0.84/ncol, and at four columns that string overruns its share.
       The plateau column is dropped: it was 1.64 in every row and is in the
       note underneath.
  p9   the J2 line struck its own annotation through. The offset was 0.0015
       of accuracy, under two pixels at that y range; it is 5 points now.
  p13  "tied d320" sat under its 11pt marker. dx 6 -> 12.
  p14  the note quoted "the 30 discriminating cells in this fit" from S4's
       registered run. This fit has 46 of 81, because the rho_curve_*.csv glob
       fix made stage T's 18 grids visible and every off-schedule T entered the
       held-out set as its own cell -- so the worst miss is 0.0729, not 0.0222.
       The three tied d320 residuals are unchanged to the digit (+0.0165 /
       -0.0034 / +0.0065), so the claim stands; the note is computed now, and
       the renderer prints which cells miss, by rung tuple.
  p18  "What stage S does not show" is a LABEL -- it could head any stage's
       limitations page. Now "Stage S speaks only for N=48 at train depth
       16", which could not.

Both builders read every number through the registered scorer's own functions
(scripts/stages_score.py, committed by register job 51846608 before the first
cell trained) and share their figures: the notebook imports the deck's
module-level draw_* functions rather than redrawing, so the two cannot
disagree about what a curve looks like -- the five plot fixes above land in
the notebook without the notebook being touched.
MSG
    echo "commit rc=$?"
else
    echo "  (builders already committed)"
fi
SHA=$(git rev-parse --short HEAD)
echo "builder commit: $SHA"

echo
echo "=== render the deck ==================================================="
python3 -u -m notebooks.make_stages_deck --commit "$SHA"
deck_rc=$?
echo "deck rc=$deck_rc"

echo
echo "=== build and execute the results notebook ============================"
python3 -u notebooks/build_stages_nb.py || exit 1
python3 -u -m notebooks.exec_nb \
    notebooks/sudoku_n48_stages_2026-11-11_results.ipynb
echo "notebook rc=$?"

echo
ls -la notebooks/stages/ 2>/dev/null | head -25
echo
echo "end $(date '+%F %T')"
exit "$deck_rc"
