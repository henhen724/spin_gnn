#!/bin/bash
#SBATCH --job-name=stagesArt
#SBATCH --partition=sganguli
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagesArt_%j.out
#
# THE SECOND COMMIT of stage N's two-commit pattern: the ARTIFACTS, after the
# builders that drew them were committed (d9b177f, job 51928045) and after every
# page was read back as a PNG. The deck's own provenance line names d9b177f, so
# the deck cannot be committed in the same commit as the code that drew it
# without stamping a sha that did not exist at render time.
#
# In a job because $HOME I/O on sh02-ln02 wedges intermittently and a wedged
# `git commit` has held a 0-byte .git/index.lock for 33 minutes; the lock is
# never removed by hand. Fails closed on the lock rather than waiting on it.
#
# STATUS.md is deliberately NOT here: it is rewritten once more at the end of
# the session and committed then.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists."
    exit 1
fi

# The PNGs are gitignored (notebooks/**/page*.png) on purpose -- they are the
# read-back medium, not an artifact; the PDF is what ships. Assert the PDF this
# commit is about is the one the last render wrote.
if [ ! -s notebooks/stages/stages_slides.pdf ]; then
    echo "FAILING CLOSED: no deck PDF on disk."
    exit 1
fi
ls -l notebooks/stages/stages_slides.pdf

echo
echo "=== commit ============================================================"
PATHS="notebooks/stages/stages_slides.pdf
notebooks/sudoku_n48_stages_2026-11-11_results.ipynb
notebooks/deck_tokenization.py
scripts/slurm/sudoku2_n48_tiedT_T_2026-11-11.sbatch
LOG.md
scripts/slurm/stages_artifacts_2026-11-11.sbatch"

git add $PATHS || exit 1
staged=$(git diff --cached --name-only)
printf '%s\n' "$staged" | sed 's/^/  /'
if [ -z "$staged" ]; then
    echo "nothing staged, exiting clean."
    exit 0
fi
while IFS= read -r p; do
    [ -z "$p" ] && continue
    printf '%s\n' "$staged" | grep -qx -- "$p" && continue
    # Tracked-and-unstaged means identical to HEAD, which needs no commit; the
    # fault this guard is for is an IGNORED path, where `git add` succeeds and
    # stages nothing.
    if git ls-files --error-unmatch -- "$p" >/dev/null 2>&1; then
        echo "  ($p unchanged since HEAD -- already committed)"
        continue
    fi
    echo "FAILING CLOSED: $p is not tracked (gitignored?) and did not stage."
    exit 1
done <<< "$PATHS"

git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
    commit -q -F - <<'MSG'
Stage S deck and results notebook (19 pages, builders d9b177f)

The artifacts, after four renders and two PNG read-backs. The deck exited 0 on
the first render with 19 pages and a clean word-count lint, and ten of those
pages were wrong: eight layout defects on pages 1-9 (previous commits) and two
pages that stated something the project no longer believes (p16's "the object
to improve is the optimization, not the architecture", which stage T
contradicts, and p17, which posed stage T's test as though it were still open).
None of the ten raised. A deck is checked by LOOKING at every page.

What it says: tied `no_mlp T=16 d320`, 613,818 trainable, one shared block
applied 16 times, rho_50 0.757 (0.754/0.757/0.765), 3/3 seeds crossing,
mean_solve 0.3148, val acc 0.4830, plateau escapes 3/3. 0.017 past the free line
0.740, so the gate is NOT cleared, and the final page says so against the
standing J2 untied d192 (0.580) on the same 14-point grid, same 400 instances
per bin, same --shuffle-adj --fall. No improvement is claimed: the tied cell is
simpler and worse.

Also in this commit, both uncommitted since the work that needed them:

  notebooks/deck_tokenization.py -- "... 1 further rounds", on every traced
  instance that elides exactly one round.

  sudoku2_n48_tiedT_T_2026-11-11.sbatch -- the run tag is DISCOVERED under the
  cell directory instead of spelled out. train_sudoku_transformer appends _sNN
  for non-zero seeds and nothing for seed 0, so the tag that was right for seed
  0 failed closed on 51904636_1 and _2. Exactly one checkpoint dir per cell or
  it stops, rather than guessing.

LOG.md: stage T's T1 reading was written up as "solve_rate 0.0000 at T=24 in
2/3 seeds". The registered scorer printed per seed 0.000 / 0.550 / 0.882 -- all
three miss the 0.90 easy-bin bar, ONE collapses to zero. Verdict unchanged, the
sentence was wrong.
MSG
rc=$?
echo "commit rc=$rc"
[ "$rc" -ne 0 ] && exit "$rc"
git --no-pager log --oneline -2

echo
echo "=== push =============================================================="
module load system git/2.45.1
git push origin sudoku-dd
echo "push rc=$?"
echo "end $(date '+%F %T')"
