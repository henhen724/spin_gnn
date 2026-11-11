#!/bin/bash
#SBATCH --job-name=stagetWrite
#SBATCH --partition=sganguli
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagetWrite_%j.out
#
# THE STAGE T WRITE-UP: builders committed FIRST, then rendered.
#
# Stage N's two-commit pattern, same as stage S. The deck prints its own
# provenance line and the commit in that line has to be the commit of the CODE
# THAT DREW IT, so the builders are committed before anything renders, that sha
# is passed to the renderer, and the artifacts are committed afterwards by a
# second job.
#
# No --dependency here. Stage T trained nothing: its 18 grids have been on disk
# since array 51904636 (+51905401) and were scored by 51919538. The stage U
# array 51922710 is still running and is deliberately NOT waited on -- stage U
# is the next cut, not part of this reading, and its cells have 6-9 h left.
#
# The builders go in already corrected. Draft renders 51939930 and 51941245 went
# to $SCRATCH first (staget_draft_2026-11-11.sbatch) and the PNG read-back of
# 51939930 found two pages that were wrong in their CLAIMS:
#   p5  "The easy end never reaches the registered 0.90 bar, at any T" -- its own
#       figure refutes it: seed 2 sits exactly ON 0.900 at the trained count.
#       T1's statistic is the WORST seed, so what fails is unanimity.
#   p9  "All three seeds peak at 16" -- seed 2 peaks at T=20 (0.3152 against
#       0.3148). The MEDIAN peaks at 16; the seeds do not, unanimously.
# Both now read their sentences off the data, and check_claims asserts the
# exact-equality case and the no-seed-improves-past-20 case so neither sentence
# can come back. Page 1's provenance line also ran off the right edge (~170
# chars) and is held to ~125.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
# viz + py-matplotlib are NOT optional: stage S's 51905177 committed the
# builders and then died 19 s later on `import matplotlib`, rendering nothing.
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists."
    exit 1
fi

echo
echo "=== compile ==========================================================="
python3 -m py_compile notebooks/make_staget_deck.py notebooks/build_staget_nb.py || exit 1
echo "ok"

echo
echo "=== commit the builders BEFORE rendering =============================="
PATHS="notebooks/make_staget_deck.py
notebooks/build_staget_nb.py
scripts/slurm/staget_draft_2026-11-11.sbatch
scripts/slurm/staget_writeup_2026-11-11.sbatch
notebooks/make_stages_deck.py"
git add $PATHS || exit 1
staged=$(git diff --cached --name-only)
printf '%s\n' "$staged" | sed 's/^/  /'
if [ -n "$staged" ]; then
    while IFS= read -r p; do
        [ -z "$p" ] && continue
        printf '%s\n' "$staged" | grep -qx -- "$p" && continue
        # Not staged has two causes and only one is a fault: an IGNORED path
        # stages nothing and is a fault, an UNCHANGED path stages nothing and is
        # ordinary. ls-files --error-unmatch separates them (stage S's 51922718
        # failed closed on an unchanged file before this guard was split).
        if git ls-files --error-unmatch -- "$p" >/dev/null 2>&1; then
            echo "  ($p unchanged since HEAD -- already committed)"
            continue
        fi
        echo "FAILING CLOSED: $p is not tracked (gitignored?) and did not stage."
        exit 1
    done <<< "$PATHS"
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -F - <<'MSG'
Stage T write-up: the shared block is a 16-step schedule, not an operator

Stage T is the question only a tied cell can ask. T -- how many times the one
shared block is applied -- is a dial at EVAL time on frozen weights, so stage S's
d320 checkpoints were re-scored at T = 4, 8, 12, 16, 20, 24 with nothing
retrained. If the block were a deduction step, more applications would be
tolerated. They are not.

  T0 FALSIFIED. rho_50 bottoms at the trained count (0.757, matching stage S to
     the digit) and rises to 0.819 at T=24: |T24 - T16| = 0.062 against a
     registered 0.01 bar. Below the trained count it censors outright (T=4 and
     T=8 cross 0.5 in 0/3 seeds).
  T1 FALSIFIED, and not only at T=24. Registered as solve_rate >= 0.90 at
     realized rho >= 0.85, it is missed by at least one seed at EVERY T >= 12,
     the trained count included, where the seeds are 0.895 / 0.887 / 0.900.
  T2  T* = 16. No T at or below the trained count is within 0.01 of its mean
     solve; T=12 already costs 0.0186. None of the sixteen rounds is padding.
  T3  the registered deep-end gain is +-0.005 in 1 of 3 seeds: nothing. The cost
     of over-iterating lands at the EASY end, where depth should be free.

So "one application = one round" is a reading of a trained network, not a
property of the architecture. That is a clean negative, registered in advance as
equally publishable, and it is what sends the next cut at k distinct blocks
(stage U, array 51922710) rather than at more depth.

Two pages the draft read-back caught, both wrong in their CLAIMS and not their
layout, on a deck that rendered clean

  p5  "The easy end never reaches the registered 0.90 bar, at any T." Its own
      figure refutes it -- seed 2 sits exactly ON 0.900 at the trained count.
      T1's statistic is the worst seed, so what fails is unanimity: no T clears
      the bar in all three seeds. check_claims now asserts the exact-equality
      case too, so the overstated sentence cannot return.
  p9  "All three seeds peak at 16 and fall away on both sides." Seed 2 peaks at
      T=20, by 0.0004 (0.3152 against 0.3148) -- smaller than this eval resolves,
      and still not what the page said. The median over seeds peaks at 16; the
      page now says "no seed improves past T=20; 2 of 3 peak at 16" and reads
      both halves off mean_peaks() rather than asserting unanimity by hand.
  p1  the provenance line ran ~170 chars and clipped "NO TRAINING" off the right
      edge. Held to ~125 now; what it drops (N=48, --shuffle-adj --fall) is on
      the tokenization and trace pages.

The same correction is in the docstring and in the notebook's "What T1 says, in
full", which is the section that exists because the stage S deck and STATUS.md
both reported T1 as a collapse at T=24 and stopped there.

Every number is read through the registered scorer's own functions
(scripts/staget_score.py, committed by register job 51904633 before the first
grid, scored by 51919538). The notebook imports the deck's module-level draw_*
functions rather than redrawing, so the two cannot disagree about a curve.

Also here: a stale comment in make_stages_deck.py justifying rung tuples over
labels by a rung_label bug that 9ce4bde fixed. The tuple is still what prints,
for the reason that outlives the bug -- the scorers' frozen literals are written
as tuples, so a miss can be grepped for directly.
MSG
    echo "commit rc=$?"
else
    echo "  (builders already committed)"
fi
SHA=$(git rev-parse --short HEAD)
echo "builder commit: $SHA"

echo
echo "=== render the deck ==================================================="
python3 -u -m notebooks.make_staget_deck --commit "$SHA"
deck_rc=$?
echo "deck rc=$deck_rc"

echo
echo "=== build and execute the results notebook ============================"
python3 -u notebooks/build_staget_nb.py || exit 1
python3 -u -m notebooks.exec_nb \
    notebooks/sudoku_n48_staget_2026-11-11_results.ipynb
echo "notebook rc=$?"

echo
ls -la notebooks/staget/ 2>/dev/null | head -25
echo
echo "end $(date '+%F %T')"
exit "$deck_rc"
