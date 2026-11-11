#!/bin/bash
#SBATCH --job-name=stagetReg
#SBATCH --partition=sganguli
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagetReg_%j.out
#
# REGISTER STAGE T BEFORE ITS CELLS RUN. The eval array waits on this job with
# --dependency=afterok, so if the registration does not compile, does not commit
# and does not push, no T curve ever gets written and there is nothing to read
# selectively after the fact.
#
# What is being registered: scripts/staget_score.py, whose docstring fixes the
# readings T0-T5 and the six guards, against the competing hypothesis (depth 16
# is a trained SCHEDULE, so quality peaks at exactly 16 and falls on either side)
# rather than against nothing.
#
# Also committed here, because stage T cannot run without them and they are the
# changes the readings depend on:
#   scripts/rho_transition_eval.py  --n-iters, refusing on untied cells, plus the
#                                   n_iters_eval / n_iters_trained columns.
#   scripts/minarch_analysis.py     rung_key made T-aware -- without it stage T's
#                                   six curves per seed collapse to ONE cell key
#                                   and get averaged with nothing in the output
#                                   saying so.
# The second is a silent-pooling fix found by reading rung_key before trusting
# it, and is the reason this commit exists as its own job rather than riding
# along with the results notebook later.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

# Fail closed rather than racing another session's commit, and never remove the
# lock (org policy: a wedged index is reported, not cleared).
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists -- another git process holds the index."
    exit 1
fi

echo
echo "=== compile ==========================================================="
python3 -m py_compile \
    scripts/staget_score.py \
    scripts/rho_transition_eval.py \
    scripts/minarch_analysis.py || exit 1
echo "py_compile ok"

echo
echo "=== the registration, printed so it is in the job log too =============="
python3 - <<'PY'
import scripts.staget_score as s
print(s.__doc__)
PY

echo
echo "=== dry run (no cells have landed yet; 'no rows' is the expected result)"
python3 -u -m scripts.staget_score || echo "scorer rc=$? -- expected before any curve exists"

echo
echo "=== commit ============================================================"
PATHS="scripts/staget_score.py
scripts/rho_transition_eval.py
scripts/minarch_analysis.py
scripts/slurm/staget_register_2026-11-11.sbatch
scripts/slurm/sudoku2_n48_tiedT_T_2026-11-11.sbatch
scripts/slurm/stages_s4_curve_2026-11-11.sbatch
scripts/slurm/stages_score_2026-11-11.sbatch"

git add $PATHS || exit 1
staged=$(git diff --cached --name-only)
echo "staged:"
printf '%s\n' "$staged" | sed 's/^/  /'

if [ -z "$staged" ]; then
    echo "nothing staged -- already committed, exiting clean."
    exit 0
fi

# A partial set means one path silently failed to stage, and a registration that
# is only half committed is not a registration.
while IFS= read -r p; do
    [ -z "$p" ] && continue
    if ! printf '%s\n' "$staged" | grep -qx -- "$p"; then
        echo "FAILING CLOSED: $p did not stage."
        exit 1
    fi
done <<< "$PATHS"

git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
    commit -q -F - <<'MSG'
Register stage T: is the tied block an iterable operator or a 16-step schedule?

Stage S answered the question stage P left censored. Tied no_mlp at d320
trained 3 of 3 seeds (train NLL 1.5851/1.5915/1.5864), crossed 0.5 in all
three, and scored rho_50 median 0.757 (0.754/0.757/0.765), mean_solve 0.3148,
val token accuracy 0.4830. So P's plateau was a CAPACITY FLOOR, not an
obstruction created by weight sharing -- d192 censored, d320 measures. The
tied cell lands 0.017 past the free line 0.740 rather than clearing it.

S4 then placed those three cells ON the N=48 accuracy-to-solve curve:
residuals +0.0165 / -0.0034 / +0.0065, inside a run where 26/26 well-determined
held-out rungs sit on the curve and the worst miss among the 30 discriminating
cells is 0.0222 against the 0.0423 LOO bar. Tying therefore moves the cell
ALONG the curve; it does not bend the relation. The whole rho_50 gap against
untied J2 is what J2's extra 0.011 of token accuracy buys at this steepness.

That makes the next question about the block rather than about the level: a
tied stack has one dial no untied stack has, the number of APPLICATIONS at
eval. If the shared block is an iterable deduction operator, quality should
hold when applied past the 16 it was trained with; if depth 16 is a trained
schedule, quality should peak at exactly 16 and fall on both sides. Stage T
reads that on fixed weights -- T in {4,8,12,16,20,24} x 3 seeds, 18 grids, no
training -- and T1 (does the easy end still solve at all) is read first and
never censors.

Carried with it, because T cannot run without them:
- rho_transition_eval.py: --n-iters, which REFUSES on an untied checkpoint
  rather than silently scoring the trained depth, plus n_iters_eval and
  n_iters_trained columns so a T=24 row is distinguishable from the trained row
  of the same checkpoint.
- minarch_analysis.py: rung_key appends the eval T only when it differs from
  num_layers. Without this, stage T's six curves per seed share one key and
  pool into a single cell with nothing in the output saying so; old CSVs have
  no n_iters_eval column and key exactly as before.

No improvement is claimed here. d320 is 0.757 against the standing minimum
J2's 0.580; the tied cell is smaller (613,818 trainable vs 2,504,890) and
simpler, and it is worse on the eval.
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

echo
echo "end $(date '+%F %T')"
