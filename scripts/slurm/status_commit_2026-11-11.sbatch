#!/bin/bash
#SBATCH --job-name=statusCommit
#SBATCH --partition=sganguli
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/statusCommit_%j.out
#
# STATUS.md + LOG.md for the stage S result and the stage T launch, plus the
# run_tests skip list. In a job because $HOME I/O on sh02-ln02 wedges
# intermittently and a wedged `git commit` has held a 0-byte .git/index.lock for
# 33 minutes; the lock is never removed by hand.
#
# The LOG.md append is GUARDED BY ITS OWN DATE HEADER: this job is safe to
# resubmit, and appending the same three lines twice is how a log stops being
# readable.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists."
    exit 1
fi

if grep -q '^## 2026-11-11' LOG.md; then
    echo "LOG.md already has a 2026-11-11 section -- not appending again."
else
    cat >> LOG.md <<'LOGMSG'

## 2026-11-11 (session 6, check-in)

- **STAGE S ANSWERED STAGE P'S CENSOR: the plateau was a CAPACITY FLOOR, not an obstruction created by weight sharing.** Tied `no_mlp T=16 d320` (613,818 trainable) trained 3 of 3, escaped the plateau in every seed (1.5851/1.5915/1.5864 against d192's 1.6646/1.6721/1.6681, 0/3), and scored `rho_50` **0.757** (0.754/0.757/0.765), mean_solve 0.3148, val acc 0.4830 — 0.017 past the free line 0.740, so the gate is not cleared. **S4 is the result that matters: all three seeds sit ON the N=48 accuracy-to-solve curve** (+0.0165 / −0.0034 / +0.0065 against the 0.0423 LOO bar, and DISCRIMINATING, so the fit could have rejected them), so tying moves the cell along the curve rather than bending the relation, and the whole 0.177 gap against untied J2 is what J2's extra 0.011 of token accuracy buys. Accuracy saturates in width (d448's one surviving seed is 0.4837 against d320's 0.4830), and the epoch confound is excluded — J2, P1 and S1 are all 240k × 50 epochs.
- **STAGE T registered (`51904633`) and launched (array `51904636`): is the shared block an ITERABLE OPERATOR or a 16-step SCHEDULE?** No training — the trained d320 block applied `T ∈ {4,8,12,16,20,24}` on fixed weights, 3 seeds, paired instances, T=16 re-run through the same code path rather than copied from stage S. `rho_transition_eval.py` gained `--n-iters` (which REFUSES on an untied checkpoint instead of silently scoring the trained depth) plus `n_iters_eval`/`n_iters_trained` columns, and `minarch_analysis.rung_key` was made T-aware — without that fix stage T's six curves per seed would have shared one cell key and been silently averaged. **The S5 warmup array stays unsubmitted on purpose:** its motivation was the censoring-floor confound for a *censored* `rho_50`, and d320 produced a number.
- **The test suite's exit code carries signal again.** 27 files, 237 cases, 0 failures, but rc was permanently 1 from the two files that import `spin_gnn/model.py` → `torch_geometric` (absent on Sherlock), which meant a NEW import failure would not have shown up in it. `scripts/run_tests.py` now has an explicit `ABSENT` list, reports those files as SKIPPED **by name with the reason**, excludes them from the exit code, and fails the suite if a listed package ever becomes importable and the entry is left behind.
LOGMSG
    echo "appended 3 lines to LOG.md"
fi

echo
echo "=== commit ============================================================"
PATHS="STATUS.md
LOG.md
scripts/run_tests.py
scripts/slurm/status_commit_2026-11-11.sbatch"

git add $PATHS || exit 1
staged=$(git diff --cached --name-only)
printf '%s\n' "$staged" | sed 's/^/  /'
if [ -z "$staged" ]; then
    echo "nothing staged, exiting clean."
    exit 0
fi
while IFS= read -r p; do
    [ -z "$p" ] && continue
    if ! printf '%s\n' "$staged" | grep -qx -- "$p"; then
        echo "FAILING CLOSED: $p did not stage."
        exit 1
    fi
done <<< "$PATHS"

git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
    commit -q -F - <<'MSG'
Stage S result in STATUS/LOG; run_tests reports absent packages as skips

Stage S: tied no_mlp T=16 d320 trained 3 of 3 and scored rho_50 0.757
(0.754/0.757/0.765), mean_solve 0.3148, val acc 0.4830, plateau escapes 3/3.
Stage P's d192 censor was a CAPACITY FLOOR -- one block at d192 cannot serve 16
roles -- not an obstruction created by sharing. The cell is 0.017 past the free
line 0.740 and does not clear the gate.

S4 is the headline over the level: all three d320 seeds sit on the N=48
accuracy-to-solve curve (+0.0165 / -0.0034 / +0.0065 against stage J's 0.0423
leave-one-out bar) and they are DISCRIMINATING, so the fit could have rejected
them. Tying moves the cell along the curve; it does not bend the relation. The
0.177 rho_50 gap against untied J2 is therefore what J2's extra 0.011 of token
accuracy buys at this steepness, and accuracy has saturated in width.

run_tests.py: the suite's exit code was permanently 1 because two files import
spin_gnn/model.py -> torch_geometric, which `ml spider` does not find here. A
permanently-1 exit code carries no signal, so a NEW import failure would have
been invisible in it. An explicit ABSENT list now names those packages with the
check that established each, reports the blocked files as SKIPPED by name,
excludes them from the exit code, and FAILS the suite if a listed package ever
becomes importable with its entry left behind. 27 files, 237 cases, 0 failures,
2 skipped.
MSG
rc=$?
echo "commit rc=$rc"
[ "$rc" -ne 0 ] && exit "$rc"
git --no-pager log --oneline -1

echo
echo "=== push =============================================================="
module load system git/2.45.1
git push origin sudoku-dd
echo "push rc=$?"
echo "end $(date '+%F %T')"
