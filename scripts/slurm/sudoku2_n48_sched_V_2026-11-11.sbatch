#!/bin/bash
# =============================================================================
# STAGE V -- TRAIN the shared block at a LONGER SCHEDULE. Is the remaining 0.017
# a schedule-LENGTH limit rather than a capacity, sharing or budget limit?
#
# WRITTEN 2026-11-11, NOT YET SUBMITTED: scripts/stagev_score.py does not exist
# and the standing non-negotiable is that the scorer is registered BEFORE the
# first cell lands, with the array --dependency=afterok on it. Submitting this
# without it would be the exact corner the discipline exists to cut off.
#
# WHERE THIS STARTS
#
#     S1  tied_no_mlp T=16 d320 k=1   613,818   rho_50 0.757 (0.754/0.757/0.765)
#         3/3 crossing   mean_solve 0.3148   val token acc 0.4830 (spread 0.0010)
#
# 0.017 past the 0.740 gate, and S4 put all three seeds ON the N=48
# accuracy-to-solve curve, so the 0.017 is worth about 0.0011 of token accuracy.
# What has been ruled out for supplying it:
#
#   WIDTH      saturates: 0.4567 -> 0.4830 -> 0.4837 for d192 / d320 / d448.
#   LR DECAY   finding 47 FALSIFIED it on this task -- cosine made both cells
#              worse (rho_50 0.679 -> 0.830 and 0.564 -> 0.726), because finding
#              45's cells are step-bound rather than converged and decaying
#              through the bulk descent starves them. The optimizer stays at
#              constant 1e-3, warmup 0.
#   MORE DATA  not keyable: rung_key carries (L, nhead, arch, d_model, ff,
#              warmup) + epochs + kK, and NOT n_train_instances, so a 480k cell
#              would pool into S1's rung and contaminate its own control.
#   BEST-VAL   finding 14's +0.0280 solve_rate is not free to collect: the cells
#              save only the final model.pt, so it needs a re-train either way.
#
# THE KNOB THIS STAGE MOVES, AND WHY IT IS THE RIGHT ONE
#
# Stage T (array 51904636, scorer 51919538) applied S1's FROZEN weights
# T = 4..24 and found rho_50 bottoms exactly at the trained count and rises to
# 0.819 at T=24: ONE WEIGHT SET TUNED TO A 16-STEP SCHEDULE, not an iterable
# operator. Stage T could only ever move T at EVAL. The cell it explicitly does
# not speak for is its own page 14: "no tied cell trained at another T".
#
# And the task has chains longer than sixteen. The instance generator's own draw
# (deck_tokenization, 40 draws from this exact config) reports reasoning depth
# median 5 rounds with range 1-19. A network applied 16 times cannot serially
# propagate a 19-round deduction chain at all, and the chains that run deepest
# are the SPARSE instances -- which is precisely where rho_50 is read.
#
#     V1  tied_no_mlp T=24 d320 k=1   613,818   this array, tasks 0-2
#     V2  tied_no_mlp T=32 d320 k=1   613,818   this array, tasks 3-5
#
# THE PARAMETER COUNT DOES NOT MOVE. (630d + 58) + (4d^2 + 8d)k has no T in it:
# a tied cell at T=32 is 613,818 trainable, byte for byte the control's count.
# Under the 11-10 brief this is the cheapest knob on the board -- the object
# stays ONE block and ONE map iterated, h_{t+1} = F(h_t), and only the number of
# iterations changes. If a longer schedule clears the gate, the gate is cleared
# by the MOST tractable cell in the whole ladder, not by a bigger one.
#
# REGISTERED PREDICTIONS, IN READING ORDER (accuracy first: it cannot censor)
#
#   V-a  FIRST, and it is the adjudicator. Final val token accuracy at T=24 is
#        >= 0.4841 (the control's 0.4830 plus the 0.0011 the gate needs) in at
#        least 2 of 3 seeds. The control's own seed spread is 0.0010, so the bar
#        sits one full spread above it.
#   V-b  rho_50 at T=24 < 0.740 in ALL THREE seeds -- the gate itself. Lower is
#        better. CENSORING is impossible to rule out in advance but would be
#        informative here: it cannot be a size effect, since the count is
#        identical to a cell that crosses 3/3.
#   V-c  the easy-end bar (solve_rate >= 0.90 at realized rho >= 0.85) cleared in
#        3/3 seeds. NO T cleared it unanimously on frozen weights, the trained
#        count included (0.895 / 0.887 / 0.900). If training at the longer
#        schedule does not fix the easy end, nothing about this is a schedule.
#   V-d  TREND: rho_50(T=32) <= rho_50(T=24) <= 0.757. The two rungs exist so the
#        reading is a trend and not a point -- stage T's own lesson is that one
#        T cannot separate "schedule" from "operator". If T=32 comes back WORSE
#        than T=24, the honest reading is a tuned schedule again, at a longer
#        length, and the knob is exhausted.
#
# FALSIFICATION, STATED BEFORE THE CELLS RUN. If V-a misses in 2 of 3 seeds, the
# schedule-length explanation is dead and the remaining candidate for the 0.017
# is the single shared weight set itself -- which is what stage U (array
# 51922710, k in {2,4}) is already testing. A null here AND a null there would
# say the tied family cannot clear the free line at N=48, which is a boundary
# worth publishing rather than a failure to report.
#
# WALL TIME. The knob multiplies per-step FLOPs: T=24 is 1.5x and T=32 is 2.0x
# stage S's 16 applications, against 8h05-11h29 per d320 cell. 30:00:00 is a
# ceiling with margin on the 2.0x arm (sganguli's limit is 7 days), and the
# registered 50 epochs is asserted in the read-back, so a cell cut short fails
# loudly instead of being compared against a full one.
#
# CONTENTION. Submit tasks 0-2 (T=24) alone if the array would otherwise run
# beside stage U's six GPU cells on sganguli's three nodes. The trend in V-d
# needs both rungs, but it does not need them concurrently.
#
# NON-NEGOTIABLES CARRIED FORWARD: `--shuffle-adj` on every cell (16/17/23),
# 3 seeds with the spread quoted (41/42/47/51/56), validation token accuracy read
# BEFORE rho_50 (4+6), one knob at a time, and the scorer registered before the
# first cell lands.
#
# N=48 ONLY. Finding 55: the N=48 minimum does not transfer to N=96.
# =============================================================================

#SBATCH --job-name=sud2N48SchedV
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=30:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%A_%a.out

set -euo pipefail

# applications:seed -- two schedule lengths, three seeds each. d_model, nhead,
# ff, warmup, k and the 240k x 50ep budget are fixed at stage S's d320 cell for
# every task; the number of APPLICATIONS is the one thing that moves.
SPECS=(
  "24:0"  "24:1"  "24:2"
  "32:0"  "32:1"  "32:2"
)

TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
IFS=: read -r LAYERS SEED <<< "${SPECS[$TASK]}"
LAYERS=$((10#$LAYERS)); SEED=$((10#$SEED))

DMODEL=320
KBLOCKS=1
FF=1024
NHEAD=4
WARMUP=0
ARCH=tied_no_mlp

LL=$(printf '%02d' "$LAYERS")
HH=$(printf '%02d' "$NHEAD")
TT=$(printf '%02d' "$TASK")
DD=$(printf '%04d' "$DMODEL")
FFS=$(printf '%05d' "$FF")

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}"

echo "=== smoke check ==="
python3 - "$DMODEL" "$NHEAD" "$KBLOCKS" "$LAYERS" <<'PY'
import sys
import torch, numpy, numba
import spin_gnn.sudoku_hparam_sweep  # noqa: F401
import scripts.rho_transition_eval   # noqa: F401
from spin_gnn.sat_transformer2 import ARCH_MODES, SATReasoningTransformer
d, nhead, k, L = (int(a) for a in sys.argv[1:5])
print("torch", torch.__version__, "numpy", numpy.__version__, "numba", numba.__version__)
assert "tied_no_mlp" in ARCH_MODES, ("the tied mode is not in this checkout",
                                     ARCH_MODES)
assert d == 320 and nhead == 4 and k == 1, (
    "stage V is stage S's d320 k=1 cell with ONE knob moved", d, nhead, k)
assert L in (24, 32), ("stage V is the two registered schedule lengths only", L)
m = SATReasoningTransformer(vocab_size=58, d_model=d, num_layers=L, nhead=nhead,
                            max_len=512, dim_feedforward=1024, norm_first=True,
                            arch_mode="tied_no_mlp", n_blocks=k)
# THE HEADLINE GUARD OF THIS STAGE: the count has no T in it, so a longer
# schedule costs ZERO parameters and the cell stays as tractable as the control.
want = (630 * d + 58) + (4 * d * d + 8 * d) * k
got = m.trainable_parameter_count()
assert got == want == 613818, ("the count moved with the schedule length -- this "
                               "is not a pure schedule knob", got, want)
assert len(m.encoder.layers) == 1 and m.n_iters == L, (len(m.encoder.layers),
                                                       m.n_iters)
print("T=%d  one distinct block applied %d times, %s trainable -- IDENTICAL to "
      "the T=16 control's count" % (L, L, format(got, ",")))
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n48_sched_V_2026-11-11"
cell="${OUT}/t${TT}_d${DD}_ff${FFS}_h${HH}_${ARCH}_L${LL}_s${SEED}"
mkdir -p "$cell"

echo "task=${TASK}  applications=${LAYERS} d_model=${DMODEL} ff=${FF} nhead=${NHEAD} arch=${ARCH} k=${KBLOCKS} seed=${SEED}"
echo "cell -> ${cell}"
date +'cell start %F %T'

python3 -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 48 --degree 5 --n-colors 4 --shuffle-adj --fall \
    --d-model "$DMODEL" --nhead "$NHEAD" --dim-feedforward "$FF" \
    --num-layers "$LAYERS" --arch-mode "$ARCH" --n-blocks "$KBLOCKS" \
    --seed "$SEED" --warmup-epochs "$WARMUP" \
    --n-train-instances 240000 --epochs 50 \
    --n-val-instances 300 --batch-size 32 --n-per-bin 100 \
    --out-dir "$cell"

date +'cell end   %F %T'

TAG="d${DD}_h${HH}_L${LL}_ff${FFS}_lr1e-03_pre_shuf_fall_noff_tied"
if [ "$WARMUP" -ne 0 ]; then
    TAG="${TAG}_w$(printf '%02d' "$WARMUP")"
fi
if [ "$SEED" -ne 0 ]; then
    TAG="${TAG}_s$(printf '%02d' "$SEED")"
fi
CKPT="${cell}/${TAG}"
[ -f "${CKPT}/model.pt" ] || { echo "expected checkpoint missing: ${CKPT}"; ls -1 "$cell"; exit 1; }

echo
echo "=== W: what the checkpoint says it is (read back, not trusted) ==="
python3 - "${CKPT}/config.json" "$ARCH" "$LAYERS" "$NHEAD" "$DMODEL" "$FF" "$WARMUP" "$SEED" "$KBLOCKS" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
want_arch, want_L, want_h = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
want_d, want_ff, want_w = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
want_seed, want_k = int(sys.argv[8]), int(sys.argv[9])
assert c["shuffle_adj"] is True, "LEAK: this cell is not shuffle_adj"
assert c["fall"] is True, "wrong ensemble: fall is not set"
assert c["arch_mode"] == want_arch, (c["arch_mode"], want_arch)
assert c["nhead"] == want_h and c["d_model"] == want_d
assert c["dim_feedforward"] == want_ff, (c["dim_feedforward"], want_ff)
assert int(c.get("warmup_epochs", 0) or 0) == want_w, (c.get("warmup_epochs"), want_w)
assert c["n_nodes"] == 48, ("stage V is N=48 only (finding 55)", c["n_nodes"])
assert int(c["seed"]) == want_seed, ("wrong seed; the triple would not pool",
                                     c["seed"], want_seed)
assert int(c.get("n_blocks", 1) or 1) == want_k, ("k moved; stage V holds k=1",
                                                  c.get("n_blocks"))
# The knob. num_layers IS the application count for a tied cell, and it is in
# rung_key -- which is why this stage keys distinctly from S1 and the data route
# (n_train_instances, not in the key) was rejected.
assert c["num_layers"] == want_L, ("the schedule length did not round-trip; this "
                                   "cell would pool into S1's rung",
                                   c["num_layers"], want_L)
assert c["max_len_margin"] == 2 * c["n_nodes"] + 2, c["max_len_margin"]
d, L = c["d_model"], c["num_layers"]
BLOCK = 4 * d * d + 8 * d
pred = (630 * d + 58) + BLOCK            # k=1, and T does not appear
got = c.get("n_trainable_params", -1)
print("arch_mode=%s applications=%d (one block, %d times) nhead=%d d_model=%d "
      "ff=%d warmup=%d seed=%d  trainable %s params"
      % (c["arch_mode"], L, L, c["nhead"], d, c["dim_feedforward"],
         int(c.get("warmup_epochs", 0) or 0), int(c["seed"]), format(got, ",")))
assert got == pred == 613818, (
    "this is not the registered cell: a longer schedule must cost ZERO "
    "parameters, and the whole tractability case rests on that", got, pred)
print("613,818 trainable at T=%d -- IDENTICAL to the T=16 control. The object is "
      "still h_{t+1} = F(h_t), iterated %d times instead of 16." % (L, L))
PY

echo
echo "=== the transition curve for this cell (V-b) ==="
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.33 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== V-a: token accuracy, read BEFORE any level ==="
python3 - "${CKPT}/history.json" <<'PY'
import json, sys
PLATEAU = 1.64          # finding 44, an N=48 number
S1_NLL = (1.5851, 1.5915, 1.5864)       # stage S's T=16 d320 cell, the control
S1_ACC = 0.4830
BAR = 0.4841            # V-a: the control plus the 0.0011 the gate needs
h = json.load(open(sys.argv[1]))
best = min(h, key=lambda r: r["nll"])
final_train = h[-1]["train_nll"]
tail = [r["train_nll"] for r in h[-5:]]
slope = (tail[-1] - tail[0]) / max(len(tail) - 1, 1)
acc = h[-1]["accuracy"]
print("epochs %d  final train_nll %.4f  final val_nll %.4f  best val_nll %.4f at epoch %d"
      % (len(h), final_train, h[-1]["nll"], best["nll"], best["epoch"]))
print("final val TOKEN ACCURACY %.4f  %+.4f against the T=16 control's %.4f  "
      "(V-a bar %.4f: %s)"
      % (acc, acc - S1_ACC, S1_ACC, BAR, "CLEARS" if acc >= BAR else "MISSES"))
print("tail slope over the last 5 epochs: %+.5f nll/epoch" % slope)
print("the control finished at %s -- this cell is %+.4f against their median"
      % ("/".join("%.4f" % v for v in S1_NLL), final_train - sorted(S1_NLL)[1]))
first = next((r["epoch"] for r in h if r["train_nll"] < PLATEAU), None)
print("plateau %s" % ("escaped at epoch %d" % first if first is not None
                      else "NOT ESCAPED (min %.4f)" % min(r["train_nll"] for r in h)))
assert len(h) == 50, ("the run did not reach the registered 50 epochs; a "
                      "truncated cell does not compare with its matched rung",
                      len(h))
PY

echo
echo "=== done ==="
date +'%F %T'
echo "score with: python3 -u -m scripts.stagev_score   (NOT YET WRITTEN)"
