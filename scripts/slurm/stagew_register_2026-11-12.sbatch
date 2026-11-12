#!/bin/bash
#SBATCH --job-name=stagewRegister
#SBATCH --partition=sganguli
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagewRegister_%j.out
#
# REGISTER STAGE W BEFORE ITS FIRST CELL CAN START.
#
# The array scripts/slurm/sudoku2_n48_budget_W_2026-11-12.sbatch is submitted
# --dependency=afterok on THIS job, so every step below fails closed:
#
#   1. the budget dry check must pass. The ENTIRE tractability case for this
#      stage is that the epoch count costs no parameters, so the cell must count
#      613,818 EXACTLY -- the control's count -- build ONE distinct block with
#      n_iters 16, and the 100-epoch rung key must separate from stage S's
#      50-epoch key, from stage U's "k2" tail and from stage T's "T24" tail. The
#      three tails are distinguishable by TYPE, which is the only reason epochs
#      kept its bare-integer form, and that is asserted here rather than assumed.
#   2. the registered scorer must run and exit 0. Its exit code is guard 4 --
#      stage S's 50-epoch cell is re-read from disk and checked against the
#      frozen literals (613,818 trainable, rho_50 0.757, escapes 3/3). If it has
#      drifted, this job fails, the array's afterok never fires, and no GPU is
#      spent on a difference whose baseline is unverifiable.
#   3. the commit must happen. The docstring of scripts/stagew_score.py IS the
#      registration -- the hypothesis, the bar W-a must clear, the WITHIN-RUN
#      pair W-c rests on, W-d's confound stated before the result exists, and
#      what runs next in each case -- so it is in git before any number exists to
#      tempt it.
#
# STAGE U's STEP 2 IS DELIBERATELY ABSENT, for the same reason it was absent
# from stage V's registration: stage W changes NO code at all. --epochs has been
# a flag since the first sweep and epochs has been in rung_key since stage G.
# Running the full suite here would test nothing this stage touches. Said
# explicitly so the missing step reads as a decision.
#
# From a compute node, like every git step on this project: $HOME I/O on
# sh02-ln02 wedges intermittently in cl_sync_io_wait. The commit refuses to run
# if a .git/index.lock exists and never deletes it.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

echo "=== STEP 1/3  the budget dry check ====================================="
python3 -u - << 'PY'
import sys

from scripts.minarch_analysis import EP_DEFAULT, rung_key
from spin_gnn.sat_transformer2 import SATReasoningTransformer

D, K, T, CONTROL = 320, 1, 16, 613818
S1_KEY = (16, 4, "tied_no_mlp", 320, 1024, 0)

ok = True
m = SATReasoningTransformer(vocab_size=58, d_model=D, num_layers=T, nhead=4,
                            max_len=512, dim_feedforward=1024, norm_first=True,
                            arch_mode="tied_no_mlp", n_blocks=K)
got = m.trainable_parameter_count()
nb = len(m.encoder.layers)
print("T=%d  trainable %s  blocks %d  n_iters %d" % (T, format(got, ","), nb,
                                                     m.n_iters))
if got != CONTROL:
    print("  FAIL: the cell is not the control's count, so the budget is not the")
    print("  only thing that differs.")
    ok = False
if nb != 1 or m.n_iters != T:
    print("  FAIL: not one block applied 16 times.", nb, m.n_iters)
    ok = False

# The keys. A W cell differs from the control ONLY in the epochs tail, so if
# that tail does not appear the whole stage silently pools into stage S's rung
# and reports its own control as part of itself.
base = {"num_layers": T, "nhead": 4, "arch_mode": "tied_no_mlp", "d_model": D,
        "dim_feedforward": 1024, "warmup_epochs": 0, "n_blocks": K,
        "n_iters_eval": T}
k50 = rung_key(dict(base, epochs=EP_DEFAULT))
k100 = rung_key(dict(base, epochs=100))
kk2 = rung_key(dict(base, epochs=EP_DEFAULT, n_blocks=2))
kt24 = rung_key(dict(base, epochs=EP_DEFAULT, n_iters_eval=24))
print("keys:  50ep %s" % (k50,))
print("      100ep %s" % (k100,))
print("       k=2  %s" % (kk2,))
print("      T=24  %s" % (kt24,))
if k50 != S1_KEY:
    print("  FAIL: the control's key moved.", k50)
    ok = False
if k100 == k50:
    print("  FAIL: the 100-epoch cell would pool into stage S's rung.")
    ok = False
if k100 != S1_KEY + (100,):
    print("  FAIL: the epochs tail is not what the registration describes.", k100)
    ok = False
if k100 in (kk2, kt24):
    print("  FAIL: the budget rung collides with stage U's or stage T's tail.")
    ok = False
# Type, not value: an int tail and a string tail are the reason three different
# knobs can share one key format at all.
if not isinstance(k100[-1], int) or isinstance(kk2[-1], int) \
        or isinstance(kt24[-1], int):
    print("  FAIL: the tails are no longer distinguishable by type -- epochs must")
    print("  stay a bare int and k/T must stay strings.", k100[-1], kk2[-1],
          kt24[-1])
    ok = False
if EP_DEFAULT != 50:
    print("  FAIL: EP_DEFAULT moved, so the 50-epoch control now carries a tail of")
    print("  its own and every earlier rung key in this programme changed.",
          EP_DEFAULT)
    ok = False

print("dry check:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
DRY=$?
echo "dry check rc=$DRY"
if [ "$DRY" -ne 0 ]; then
    echo "FAILING CLOSED: the budget rung is not what the registration says."
    echo "No cell may start. Nothing committed."
    exit 1
fi

echo
echo "=== STEP 2/3  run the registered scorer DRY (no new cells on disk) ====="
echo "Everything below should read PENDING for W1, and the CONTROL block should"
echo "find stage S's 50-epoch d320 cell on disk and matching its literals."
python3 -u -m scripts.stagew_score
SCORER=$?
echo "scorer rc=$SCORER"
if [ "$SCORER" -ne 0 ]; then
    echo "FAILING CLOSED: stage S's 50-epoch control is missing or has drifted, so"
    echo "the budget has nothing to be a difference against. The array's afterok"
    echo "will not fire."
    exit 1
fi

echo
echo "=== STEP 3/3  commit the registration =================================="
if [ -e .git/index.lock ]; then
    echo "FAILING CLOSED: .git/index.lock exists. Nothing staged, nothing"
    echo "committed, and the lock is left alone."
    ls -l .git/index.lock
    exit 1
fi

git add scripts/stagew_score.py \
    scripts/slurm/sudoku2_n48_budget_W_2026-11-12.sbatch \
    scripts/slurm/stagew_register_2026-11-12.sbatch
echo "add rc=$?"
echo "--- staged ---"
git diff --cached --name-only

python3 -u - << 'PY'
import subprocess
import sys

staged = subprocess.run(["git", "diff", "--cached", "--name-only"],
                        capture_output=True, text=True).stdout.split()
tracked = subprocess.run(["git", "ls-files"], capture_output=True,
                         text=True).stdout.split()
need = ["scripts/stagew_score.py",
        "scripts/slurm/sudoku2_n48_budget_W_2026-11-12.sbatch",
        "scripts/slurm/stagew_register_2026-11-12.sbatch"]
for p in need:
    # Staged OR already committed -- a re-run of this job must not fail just
    # because there is nothing left to add. An IGNORED path appears in neither,
    # which is the case this check actually exists to catch.
    if p not in staged and p not in tracked:
        print("NEITHER STAGED NOR TRACKED -- is it gitignored? a `git add` on an "
              "ignored path says nothing: %s" % p)
        sys.exit(1)
print("the registration and the array are staged or already committed")
PY
if [ "$?" -ne 0 ]; then
    echo "FAILING CLOSED: the registration did not stage. Nothing committed."
    exit 1
fi

if git diff --cached --quiet; then
    echo "nothing staged -- already committed; no commit made"
else
    git -c user.name='Claude (hshunt)' -c user.email='hshunt@stanford.edu' \
        commit -q -F - << 'MSG'
Register stage W: twice the training budget for the shared block

This is the arm stage V's V-f registered before any V cell landed: "V-a misses
in 2/3 -> the schedule-length explanation is DEAD ... the registered arm is 100
epochs at constant 1e-3, k=1, T=16, one knob. Explicitly NOT more width
(saturated), NOT cosine (finding 47), NOT larger k if stage U is null." V-a came
back 0/3 at T=24 (0.4833/0.4612/0.4834 against the 0.4841 bar) and V1's rho_50
median was 0.769 with one seed censored, worse than T=16's 0.757. So this runs
that arm, at the k that branch names, rather than the k=2 variant the stage U
basin seeds would also motivate -- two knobs at once is how a stage stops being
able to answer.

Hypothesis W: the tied cell is not short of capacity (width saturated), roles
(stage U) or serial applications (stage V). It is short of STEPS. Finding 45
called this family step-bound rather than converged and finding 47 is the
evidence -- cosine decay HURT, which only happens to a cell with descent left to
do -- and stage S's tail slope at epoch 50 was still about -0.00013 nll/epoch.
Fifty more epochs at the same constant 1e-3 is the knob that reasoning names.

The count does not move: (630d + 58) + (4d^2 + 8d)k has no epoch term, so W1
holds 613,818 trainable parameters, byte for byte the control's and byte for byte
stage V's at every T. The object does not change at all -- one block, one map,
h_{t+1} = F(h_t), sixteen applications -- only the number of gradient steps. The
budget is the second free knob on this board after the application count.

The learning rate stays constant for a second reason beyond finding 47: it is
what makes epoch 50 of a 100-epoch run the same point in optimization as epoch 50
of a 50-epoch run, which is what W-c reads.

W-a is read first and is the adjudicator: val token accuracy >= 0.4841 in at
least 2 of 3 seeds, the bar IMPORTED from stagev_score rather than retyped so the
two stages are judged against one number. W-b is the gate. W-c is the only
WITHIN-RUN paired number the stage has -- train NLL at epoch 100 at least 0.0065
below epoch 50, stage S's measured tail slope over the added epochs -- and it
outranks both for a NULL: if the budget bought no loss, the cell was converged at
50 and a knob that did not turn cannot be reported as a knob that did not work.
W-d carries its confound in the registration: at twice the budget a low train NLL
is not by itself stage U's basin, since more steps lower it mechanically, so the
basin claim needs NLL < 1.561 AND accuracy >= 0.4841 in the same seed. The 1.561
line is placed between the two populations stage U separated (worst basin 1.5371,
best plateau 1.5845), not chosen.

W-e states the asymmetry rather than discovering it later: the 100-epoch run
never evaluates rho_50 at epoch 50, so rho_50 and accuracy are cross-run
comparisons and only W-c is paired.

Guard 7 is stated rather than omitted, as in stage V: at 613,818 this rung sits
BELOW H1's 879,418 censoring floor, so that argument does not protect it. What
protects it is the identical count -- a censor at a parameter count that crosses
3 of 3 seeds at half the budget cannot be a size effect.

Registered in advance for the null: if W-c holds and W-a misses, the budget
bought training loss without validation accuracy and the 0.017 is a
generalization gap pointing at data, which must then be run as a matched 240k/480k
pair inside one stage because n_train_instances is not in rung_key. If everything
is null, the honest statement is that the tied family does not clear the free line
at N=48 under this recipe -- a boundary published as one, with the standing
minimum left at J2's attention-only L=16 d192 at 0.580.

No model code changes, which is why this registration runs three steps and not
stage U's four: --epochs has been a flag since the first sweep and epochs has been
in rung_key since stage G, so the full suite would test nothing this stage
touches.
MSG
    echo "commit rc=$?"
fi

module load system git/2.45.1 2>/dev/null
git push origin sudoku-dd
echo "push rc=$? (non-fatal: the commit is the record)"
git log --oneline -3

echo
echo "=== registered ==="
date +'%F %T'
echo "the array may now start; score with: python3 -u -m scripts.stagew_score"
