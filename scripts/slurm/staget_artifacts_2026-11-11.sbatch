#!/bin/bash
#SBATCH --job-name=stagetArt
#SBATCH --partition=sganguli
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagetArt_%j.out
#
# The stage T ARTIFACTS: deck PDF + executed notebook, plus STATUS.md rewritten
# for the end of session 8 and two LOG lines. In a job because $HOME I/O on
# sh02-ln02 wedges intermittently; the lock is never removed by hand.
#
# Only the PDF is added from notebooks/staget/ -- the per-page PNGs are
# gitignored (.gitignore: notebooks/**/page*.png) and are read-back scratch.
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

if grep -q '^## 2026-11-11 (session 8' LOG.md; then
    echo "LOG.md already has the session 8 section -- not appending again."
else
    cat >> LOG.md <<'LOGMSG'

## 2026-11-11 (session 8, stage T shipped after a kill; nothing finished training)

- **The process was killed mid-session; nothing was lost and nothing new finished training.** The suite is green after the kill (29 files, 267 cases, 0 failures, 2 skipped — `torch_geometric` is absent and nothing on the sudoku line imports it), the tree was clean and fully pushed at `a6a5541`, and array `51922710` (stage U, six cells) has been running since 07:50 with 0 restarts, 2h31 in at the time of the check against 8–11.5 h per cell. **No model finished training in this window**, so the only finished run owed a results notebook was stage T — which trained nothing at all: it re-scored stage S's frozen d320 checkpoints at six values of T.
- **Stage T shipped, and reading the registered scorer instead of the earlier write-ups changed what T1 says.** The stage S deck and STATUS.md both reported T1 as a collapse at T=24. The scorer's own output (job `51919538`) shows the 0.90 easy-bin bar is missed by at least one seed at **every `T ≥ 12`**, the trained count included, where the three seeds are 0.895 / 0.887 / 0.900 — so one seed sits exactly ON the bar at T=16 and T1 fails there on unanimity, not on a collapse. Deck `aec838b`, 16 pages, 222 KB; notebook executed clean. **No improvement claimed: no T beats 0.757, and the standing minimum is still J2's 0.580.**
- **Two draft pages were wrong in their CLAIMS, and only the PNG read-back could have caught either.** p5's title ("the easy end never reaches the registered 0.90 bar, at any T") was refuted by its own figure, and p9's note ("all three seeds peak at 16") by a 0.0004 margin at seed 2, which peaks at T=20. Both now read their sentences off the data through `mean_peaks()` and a widened `check_claims`, which asserts the exact-equality case and that no seed improves past T=20. The draft render goes to `$SCRATCH` under `--commit draft` so a known-wrong builder is never what a commit contains.
LOGMSG
    echo "appended 3 lines to LOG.md"
fi

echo
echo "=== commit ============================================================"
PATHS="STATUS.md
LOG.md
notebooks/staget/staget_slides.pdf
notebooks/sudoku_n48_staget_2026-11-11_results.ipynb
scripts/slurm/staget_artifacts_2026-11-11.sbatch"

git add $PATHS || exit 1
staged=$(git diff --cached --name-only)
printf '%s\n' "$staged" | sed 's/^/  /'
if [ -z "$staged" ]; then
    echo "nothing staged, exiting clean."
    exit 0
fi

git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
    commit -q -F - <<'MSG'
Stage T artifacts: deck (16 pages) and executed notebook, STATUS/LOG

The deck and notebook that the builders at aec838b produced. No improvement is
claimed and none was available: stage T trains nothing, so the best it could do
was find a T that reads better than the trained count, and rho_50 bottoms at the
trained count itself (0.757, matching stage S to the digit) and rises to 0.819
at T=24. The standing minimum is still J2's 0.580 at 16 separate blocks.

STATUS.md is rewritten for the end of session 8 and now states T1 correctly:
the 0.90 easy-bin bar is missed by at least one seed at every T >= 12, the
trained count included (0.895 / 0.887 / 0.900), so one seed sits exactly on the
bar at T=16 and T1 fails there on unanimity rather than on a collapse. What is
specific to over-iterating is seed 0's 0.887 -> 0.000 between T=16 and T=24.
The previous wording -- "the representation collapses past the trained count"
-- was true of T=24 and not of the bar.

Stage U (array 51922710, six cells, 0 restarts) is still training; rho_50 after
cannot be read before ~16:00 today. No model finished training in this window.
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
