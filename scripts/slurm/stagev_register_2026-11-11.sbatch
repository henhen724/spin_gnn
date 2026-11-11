#!/bin/bash
#SBATCH --job-name=stagevRegister
#SBATCH --partition=sganguli
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagevRegister_%j.out
#
# REGISTER STAGE V BEFORE ITS FIRST CELL CAN START.
#
# The array scripts/slurm/sudoku2_n48_sched_V_2026-11-11.sbatch is submitted
# --dependency=afterok on THIS job, so every step below fails closed:
#
#   1. the schedule dry check must pass. The ENTIRE tractability case for this
#      stage is that the application count costs no parameters, so T=16, T=24
#      and T=32 must all count 613,818 EXACTLY -- the control's count -- each
#      must build one distinct block with n_iters equal to its own T, and the
#      rung keys must separate on num_layers so a V cell cannot pool into stage
#      S's rung or collide with stage T's frozen "T24"-tailed re-reads.
#   2. the registered scorer must run and exit 0. Its exit code is guard 4 --
#      stage S's T=16 cell is re-read from disk and checked against the frozen
#      literals (613,818 trainable, rho_50 0.757, escapes 3/3). If it has
#      drifted, this job fails, the array's afterok never fires, and no GPU is
#      spent on a ladder whose bottom rung is unverifiable.
#   3. the commit must happen. The docstring of scripts/stagev_score.py IS the
#      registration -- the hypothesis, what stage T could not test, which reading
#      outranks which, the bar V-a has to clear and what runs next in each case --
#      so it is in git before any number exists to tempt it.
#
# STAGE U's STEP 2 IS DELIBERATELY ABSENT. Stage U changed the model forward
# itself (sat_transformer2.py, config, save/load, rung_key, the eval row) and so
# had to run the full suite before spending a GPU-hour. Stage V changes NO code
# at all: num_layers is the application count for a tied cell and has been since
# stage S, every guard it needs already exists, and the only new files are a
# scorer and two job scripts. Running the suite here would test nothing this
# stage touches. Said explicitly so the missing step reads as a decision.
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

echo "=== STEP 1/3  the schedule dry check ==================================="
python3 -u - << 'PY'
import sys

from scripts.minarch_analysis import rung_key
from spin_gnn.sat_transformer2 import SATReasoningTransformer

D, K, CONTROL = 320, 1, 613818
S1_KEY = (16, 4, "tied_no_mlp", 320, 1024, 0)

ok = True
for T in (16, 24, 32):
    m = SATReasoningTransformer(vocab_size=58, d_model=D, num_layers=T, nhead=4,
                                max_len=512, dim_feedforward=1024,
                                norm_first=True, arch_mode="tied_no_mlp",
                                n_blocks=K)
    got = m.trainable_parameter_count()
    nb = len(m.encoder.layers)
    row = {"num_layers": T, "nhead": 4, "arch_mode": "tied_no_mlp",
           "d_model": D, "dim_feedforward": 1024, "warmup_epochs": 0,
           "epochs": 50, "n_blocks": K, "n_iters_eval": T}
    key = rung_key(row)
    print("T=%2d  trainable %s  blocks %d  n_iters %d  key %s"
          % (T, format(got, ","), nb, m.n_iters, key))
    if got != CONTROL:
        print("  FAIL: the application count moved the parameter count. The whole")
        print("  point of this stage is that it does not.")
        ok = False
    if nb != 1 or m.n_iters != T:
        print("  FAIL: not one block applied T times.", nb, m.n_iters)
        ok = False
    if T != 16 and key == S1_KEY:
        print("  FAIL: this cell would pool into stage S's rung.")
        ok = False
    if T != 16 and key == S1_KEY + ("T%d" % T,):
        print("  FAIL: keys like stage T's FROZEN re-read at the same T. Those are")
        print("  16-trained weights run longer; this is weights TRAINED at T.")
        ok = False

# The pair V-e rests on: frozen-at-24 and trained-at-24 must be distinguishable
# on disk, or the comparison silently averages the two models it compares.
frozen24 = S1_KEY + ("T24",)
trained24 = (24, 4, "tied_no_mlp", 320, 1024, 0)
print("V-e's pair: frozen %s  vs  trained %s" % (frozen24, trained24))
if frozen24 == trained24:
    print("  FAIL: V-e would compare a cell with itself.")
    ok = False

print("dry check:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
DRY=$?
echo "dry check rc=$DRY"
if [ "$DRY" -ne 0 ]; then
    echo "FAILING CLOSED: the schedule ladder is not what the registration says."
    echo "No cell may start. Nothing committed."
    exit 1
fi

echo
echo "=== STEP 2/3  run the registered scorer DRY (no new cells on disk) ====="
echo "Everything below should read PENDING for V1 and V2, and the CONTROL block"
echo "should find stage S's T=16 d320 cell on disk and matching its literals."
python3 -u -m scripts.stagev_score
SCORER=$?
echo "scorer rc=$SCORER"
if [ "$SCORER" -ne 0 ]; then
    echo "FAILING CLOSED: stage S's T=16 control is missing or has drifted, so the"
    echo "ladder has nothing to be a difference against. The array's afterok will"
    echo "not fire."
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

git add scripts/stagev_score.py \
    scripts/slurm/sudoku2_n48_sched_V_2026-11-11.sbatch \
    scripts/slurm/stagev_register_2026-11-11.sbatch \
    scripts/slurm/staget_publish_2026-11-11.sbatch
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
need = ["scripts/stagev_score.py",
        "scripts/slurm/sudoku2_n48_sched_V_2026-11-11.sbatch",
        "scripts/slurm/stagev_register_2026-11-11.sbatch"]
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
Register stage V: train the shared block at a LONGER schedule

Stage T applied stage S's frozen tied d320 weights T = 4..24 times and found
rho_50 bottoms exactly at the trained count (0.757) and rises to 0.819 at T=24,
with the easy end collapsing in 2 of 3 seeds. The registered reading was "one
weight set tuned to a 16-step schedule, not an iterable operator". But stage T
could only ever move T at EVAL -- its own page 14 says what it does not speak
for: no tied cell was ever TRAINED at another T.

Stage V trains at T=24 and T=32, three seeds each, everything else held at stage
S's d320 cell. The knob costs nothing: (630d + 58) + (4d^2 + 8d)k has no T in it,
so all three rungs hold 613,818 trainable parameters -- byte for byte the
control's count. If a longer schedule clears the 0.740 gate, the gate is cleared
by the most tractable cell in the whole ladder rather than a bigger one, which is
the opposite of stage U, where k buys accuracy at 1.7x and 3.0x the weights.

The motivating number is in the data, not the architecture: the generator's own
draw puts reasoning depth at median 5 rounds with range 1-19, and sixteen
applications cannot serially propagate a 19-round chain. The deepest chains are
the sparse instances, which is exactly where rho_50 is read.

V-a is read first AND is the adjudicator: val token accuracy >= 0.4841 (the
control's 0.4830 plus the 0.0011 the gate needs, derived from S4's steepness, one
full control seed spread above it) in at least 2 of 3 seeds. Stage U had to
register in advance that accuracy could not adjudicate because the required gain
sat inside the seed spread; here the bar and the spread are printed against each
other. V-b is the gate, V-c the easy end stage T failed at every T, V-d the trend
across trained lengths, and V-e closes the pair stage T set up but could not
reach: frozen-at-24 (0.819, 2/3, worst easy bin 0.0000) against trained-at-24, same
architecture, same count, same 24 applications, same grid, differing only in
whether the weights were trained for the schedule they run at.

Ruled out before choosing this knob, each on existing evidence rather than taste:
cosine decay (finding 47 falsified it -- both cells got worse, since finding 45's
cells are step-bound rather than converged), more data (n_train_instances is not
in rung_key, so a 480k cell would pool into stage S's rung and destroy its own
control), more width (accuracy saturated 0.4567/0.4830/0.4837) and best-val
checkpoint selection (the cells save only the final model.pt).

Guard 7 is stated rather than omitted: at 613,818 these rungs sit BELOW H1's
879,418 censoring floor, so that argument does not protect this stage. What
protects it is the identical count -- a censor at a parameter count that crosses
3 of 3 seeds when trained at 16 cannot be a size effect.

No model code changes, which is why this registration runs three steps and not
stage U's four: num_layers has been the application count for a tied cell since
stage S, and the full suite would test nothing this stage touches.
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
echo "the array may now start; score with: python3 -u -m scripts.stagev_score"
