#!/bin/bash
#SBATCH --job-name=statusCommitB
#SBATCH --partition=sganguli
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/statusCommitB_%j.out
#
# STATUS.md rewritten for the end of session 7, plus two LOG lines and the S4
# refit check script. In a job because $HOME I/O on sh02-ln02 wedges
# intermittently; the lock is never removed by hand.
#
# The LOG append is GUARDED BY ITS OWN SESSION HEADER, so this is safe to
# resubmit: appending the same lines twice is how a log stops being readable.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists."
    exit 1
fi

if grep -q '^## 2026-11-11 (session 7' LOG.md; then
    echo "LOG.md already has the session 7 section -- not appending again."
else
    cat >> LOG.md <<'LOGMSG'

## 2026-11-11 (session 7, stage S shipped while stage U trains)

- **The stage S deck shipped, and the second PNG read-back is what made it fit to ship.** Four renders: the first exited 0 with 19 pages, 186 KB and a clean word-count lint, and TEN of its pages were wrong. Eight were layout (a subtitle clipped at "d192 -> d320 -> ap" because `title_page` wraps a title but not a subtitle, "... 1 further rounds", two right-hand labels struck through each other, a 4-column table cell overrunning `table_page`'s equal 0.84/ncol share, an annotation struck through by its own line at a 0.0015 offset, a label under its 11pt marker, a note quoting "30 discriminating cells" from a superseded fit, and a title that was a LABEL). The two found on the SECOND read-back were not layout at all: p16 asserted "the object to improve is the optimization, not the architecture", which stage T contradicts, and p17 posed stage T's test as though it were still open. A page can be well-typeset and say the wrong thing. Deck `673301e` (builders `d9b177f`), notebook re-executed, uploaded to OneDrive; **no improvement claimed — rho_50 0.757 against the standing 0.580, simpler and worse.**
- **`rung_label` named the wrong axis, and fixing it surfaced a TypeError that would have landed mid-stage-U** (`51932519`, verified by `51933077`). Its 7th positional parameter was `epochs`, so stage T's T=24 cell printed "ep24" and stage U's k=2 cell would have printed "epk2" in every table and deck built from a key. The tail is now read by TYPE, which required `rung_key` to write eval-T as the string "T24" — and that raised `TypeError: '<' not supported between instances of 'int' and 'str'` in `acc_vs_solve.collect`'s `sorted(cells.items())`, because a cell key is a rung plus its seed, so a string tail meets the SEED of the same rung's shorter key: `('T24', 0)` against `(0,)`. **Stage U's "k2" tail has the identical shape**, so this was going to raise tonight, on the first load of stage U's rows beside stage S's, after six 8-to-11-hour cells. `minarch_analysis.sort_key` now gives a total order over mixed-shape keys and the test asserts the raw sort still raises. Re-rendering the deck's own S4 fit reproduces 81 held-out / 46 discriminating / worst miss +0.0729 at tied d192 seed 0 and the three d320 residuals to the digit: a relabelling, not a regrouping.
LOGMSG
    echo "appended 2 lines to LOG.md"
fi

echo
echo "=== commit ============================================================"
PATHS="STATUS.md
LOG.md
scripts/slurm/s4_refit_check_2026-11-11.sbatch
scripts/slurm/status_commit_2026-11-11b.sbatch"

git add $PATHS || exit 1
staged=$(git diff --cached --name-only)
printf '%s\n' "$staged" | sed 's/^/  /'
if [ -z "$staged" ]; then
    echo "nothing staged, exiting clean."
    exit 0
fi

git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
    commit -q -F - <<'MSG'
STATUS/LOG for session 7: stage S shipped, key-sort bug fixed before stage U

Stage S's write-up is delivered (deck 673301e, builders d9b177f, uploaded) and
STATUS.md now carries what the second PNG read-back cost and found, the
rung_label/sort_key fix with both job ids, and the one shape that can still
collide in a key sort -- a tied d320 cell at a non-default epoch count, which
is exactly the k=1 optimization-budget cut that next step 3 points at.

Also commits s4_refit_check_2026-11-11.sbatch, which exists because the
previous commit claimed a verification its job had not actually performed:
51932519 ran acc_vs_solve's CLI, which reports well-determined RUNGS over its
own dir list, not the three per-CELL numbers the deck shipped. 51933077 renders
the deck's own S4 fit to $SCRATCH -- leaving the committed, uploaded PDF
untouched -- and prints them: 81 held-out, 46 discriminating, worst +0.0729 at
(16, 4, tied_no_mlp, 192, 1024, 0) seed 0, d320 residuals +0.0165 / -0.0034 /
+0.0065. Unchanged, as a relabelling should be.

Stage U (array 51922710, six cells, 0 restarts) is still training; rho_50 after
cannot be read before ~16:00 today.
MSG
rc=$?
echo "commit rc=$rc"
[ "$rc" -ne 0 ] && exit "$rc"
git --no-pager log --oneline -1

echo
module load system git/2.45.1
git push origin sudoku-dd
echo "push rc=$?"
echo "end $(date '+%F %T')"
