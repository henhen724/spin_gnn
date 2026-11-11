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
Stage S write-up: an unchanged path is not an ignored one

Whatever of the three paths below has changed since HEAD is in this commit;
the builders themselves went in as 41e49e4 and 33a01bd, from the two runs of
this job that committed and then died in the renderer (51905177 on a missing
matplotlib, 51919478 on a points page with four bullets where slides.py caps
at three).

That second death is what this fixes. The staging guard required every listed
path to appear in `git diff --cached --name-only`, which is true of a path
that CHANGED and false of a path that is merely identical to HEAD -- so run
51922718 failed closed in 6 seconds on notebooks/build_stages_nb.py, a file
that was already committed and correct. The guard's real target is an IGNORED
path, where `git add` succeeds and stages nothing, and `git ls-files
--error-unmatch` is what tells the two apart.

Both builders read every number through the registered scorer's own functions
(scripts/stages_score.py, committed by register job 51846608 before the first
cell trained) and share their figures: the notebook imports the deck's
module-level draw_* functions rather than redrawing, so the two cannot
disagree about what a curve looks like.
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
