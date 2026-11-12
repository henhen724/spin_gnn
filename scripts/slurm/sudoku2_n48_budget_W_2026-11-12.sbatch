#!/bin/bash
# =============================================================================
# STAGE W -- give the shared block TWICE THE TRAINING BUDGET. Is the remaining
# 0.017 a limit on STEPS OF OPTIMIZATION rather than on capacity, sharing or
# serial depth?
#
# THIS IS THE ARM STAGE V REGISTERED, NOT ONE CHOSEN AFTERWARDS. Stage V's V-f,
# written before any V cell landed, said: "V-a misses in 2/3 -> the
# schedule-length explanation is DEAD ... the remaining candidate for the 0.017
# is the training budget, and the registered arm is 100 epochs at constant 1e-3,
# k=1, T=16, one knob. Explicitly NOT more width (saturated), NOT cosine
# (finding 47), NOT larger k if stage U is null." V-a came back 0/3 at T=24
# (0.4833 / 0.4612 / 0.4834 against the 0.4841 bar), so this is that arm, at the
# k that branch names.
#
# WHERE THIS STARTS
#
#     S1  tied_no_mlp T=16 d320 k=1  50 ep  613,818  rho_50 0.757
#         (0.754/0.757/0.765)  3/3 crossing  mean_solve 0.3148  acc 0.4830
#
# 0.017 past the 0.740 gate, and S4 put all three seeds ON the N=48
# accuracy-to-solve curve, so the 0.017 is worth about 0.0011 of token accuracy.
# What has been ruled out for supplying it, each on evidence:
#
#   WIDTH      saturates: 0.4567 -> 0.4830 -> 0.4837 for d192 / d320 / d448.
#   LR DECAY   finding 47 FALSIFIED cosine on this task (rho_50 0.679 -> 0.830,
#              0.564 -> 0.726): finding 45's cells are step-bound rather than
#              converged, and decaying through the bulk descent starves them.
#   BEST-VAL   the cells save only the final model.pt; collecting finding 14's
#              +0.0280 needs a re-train either way.
#   ROLE COUNT stage U: k in {2,4} gave 0.752 / 0.755 against k=1's 0.757, all
#              within 0.01. U0 FALSIFIED.
#   SCHEDULE   stage V: trained at T=24, val accuracy 0/3 over the bar and
#              rho_50 median 0.769 with one seed censored -- WORSE than T=16.
#   MORE DATA  not keyable against stage S: rung_key carries (L, nhead, arch,
#              d_model, ff, warmup) + epochs + kK + TN, and NOT
#              n_train_instances, so a 480k cell pools into S1's rung. If it is
#              ever run it must be a MATCHED PAIR inside one stage.
#
# THE KNOB THIS STAGE MOVES, AND WHY FINDING 47 POINTS AT IT
#
# Finding 45 called this family step-bound rather than converged, and finding 47
# is the evidence: cosine decay HURT, which only happens to a cell that still
# had descent left to do. Stage S's tail slope at epoch 50 was about -0.00013
# nll/epoch and it was still falling. Fifty more epochs at the SAME constant
# 1e-3 is the knob that reasoning names.
#
#     W1  tied_no_mlp T=16 d320 k=1  100 ep  613,818   this array, tasks 0-2
#
# THE PARAMETER COUNT DOES NOT MOVE. (630d + 58) + (4d^2 + 8d)k has no epoch
# count in it either: W1 is 613,818 trainable, byte for byte S1's count and byte
# for byte stage V's at every T. The object does not change AT ALL -- one block,
# one map, h_{t+1} = F(h_t), sixteen applications -- only the number of gradient
# steps. Under the 11-10 brief that makes the budget the second free knob on the
# board, after the application count: if the gate opens here it is opened by the
# most tractable object in the ladder, and the price is GPU hours rather than
# structure.
#
# THE LEARNING RATE STAYS CONSTANT FOR A SECOND REASON. Beyond finding 47: a
# constant schedule is what makes epoch 50 of a 100-epoch run the same point in
# optimization as epoch 50 of a 50-epoch run. Under any epoch-indexed decay the
# two runs would differ from step one and W-c below could not be read at all.
#
# REGISTERED PREDICTIONS, IN READING ORDER (accuracy first: it cannot censor)
#
#   W-a  FIRST, and the adjudicator. Final val token accuracy >= 0.4841 -- the
#        control's 0.4830 plus the 0.0011 the gate needs -- in at least 2 of 3
#        seeds. The bar is IMPORTED from stagev_score, not retyped, so the two
#        stages are judged against one number. The control's spread is 0.0010.
#   W-b  rho_50 < 0.740 in ALL THREE seeds -- the gate. Lower is better. A
#        censor cannot be a size effect: identical count to a cell that crosses
#        3/3 at half the budget.
#   W-c  WITHIN-RUN, and it decides whether this stage tested anything: train
#        NLL at epoch 100 is at least 0.0065 below epoch 50 (stage S's measured
#        tail slope over the added 50 epochs) in >= 2 of 3 seeds. If this FAILS
#        the cell was already converged at 50, the budget hypothesis never had a
#        chance, and W is a null about CONVERGENCE, not about budget.
#   W-d  the basin, with its confound registered: stage U's two basin seeds hit
#        NLL 1.5371/1.5306 with accuracy 0.4929/0.4948 while k=1 sat at
#        1.586/0.4830. At twice the budget a low train NLL is NOT by itself a
#        basin -- more steps lower it mechanically -- so the basin claim needs
#        NLL < 1.561 AND accuracy >= 0.4841 in the SAME seed.
#
# FALSIFICATION, STATED BEFORE THE CELLS RUN. If W-c holds and W-a misses, the
# budget bought training loss without validation accuracy and the 0.017 is a
# GENERALIZATION gap, which points at data rather than at optimization. If W-c
# fails, the cell was converged and this stage says nothing about budget. If
# both W and the closed arms above are null, the honest statement is that the
# tied family does not clear the free line at N=48 under this recipe -- a
# boundary worth publishing as one, with the standing minimum left where it is
# (J2's attention-only L=16 d192 at 0.580).
#
# WALL TIME. The knob multiplies steps, not per-step FLOPs: 100 epochs at T=16
# is 2.0x stage S's budget, against 8h05-11h29 per d320 cell at 50 epochs, plus
# about an hour for the 14-point grid at 400 per bin. 40:00:00 is a ceiling with
# margin (sganguli's limit is 7 days), and the registered 100 epochs is asserted
# in the read-back, so a cell cut short fails loudly instead of being compared
# against a full one.
#
# NON-NEGOTIABLES CARRIED FORWARD: `--shuffle-adj` on every cell (16/17/23),
# 3 seeds with the spread quoted (41/42/47/51/56), validation token accuracy read
# BEFORE rho_50 (4+6), one knob at a time, and the scorer registered before the
# first cell lands -- scripts/stagew_score.py, committed by
# scripts/slurm/stagew_register_2026-11-12.sbatch, which this array depends on
# with --dependency=afterok.
#
# N=48 ONLY. Finding 55: the N=48 minimum does not transfer to N=96.
# =============================================================================

#SBATCH --job-name=sud2N48BudgetW
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=40:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%A_%a.out

set -euo pipefail

# epochs:seed -- ONE budget, three seeds. d_model, nhead, ff, warmup, k, the
# application count and the 240k instance pool are fixed at stage S's d320 cell
# for every task; the number of EPOCHS is the one thing that moves.
SPECS=(
  "100:0"  "100:1"  "100:2"
)

TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
IFS=: read -r EPOCHS SEED <<< "${SPECS[$TASK]}"
EPOCHS=$((10#$EPOCHS)); SEED=$((10#$SEED))

LAYERS=16
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
python3 - "$DMODEL" "$NHEAD" "$KBLOCKS" "$LAYERS" "$EPOCHS" <<'PY'
import sys
import torch, numpy, numba
import spin_gnn.sudoku_hparam_sweep  # noqa: F401
import scripts.rho_transition_eval   # noqa: F401
from spin_gnn.sat_transformer2 import ARCH_MODES, SATReasoningTransformer
d, nhead, k, L, ep = (int(a) for a in sys.argv[1:6])
print("torch", torch.__version__, "numpy", numpy.__version__, "numba", numba.__version__)
assert "tied_no_mlp" in ARCH_MODES, ("the tied mode is not in this checkout",
                                     ARCH_MODES)
assert d == 320 and nhead == 4 and k == 1, (
    "stage W is stage S's d320 k=1 cell with ONE knob moved", d, nhead, k)
assert L == 16, ("the application count is stage V's knob, not this stage's; W "
                 "holds it at the trained 16", L)
assert ep == 100, ("stage W is the 100-epoch budget", ep)
m = SATReasoningTransformer(vocab_size=58, d_model=d, num_layers=L, nhead=nhead,
                            max_len=512, dim_feedforward=1024, norm_first=True,
                            arch_mode="tied_no_mlp", n_blocks=k)
# THE HEADLINE GUARD OF THIS STAGE: the count has no epoch term in it, so twice
# the budget costs ZERO parameters and the cell stays as tractable as the
# control. Same guard stage V made about T, one axis over.
want = (630 * d + 58) + (4 * d * d + 8 * d) * k
got = m.trainable_parameter_count()
assert got == want == 613818, ("the count moved with the budget -- that is not "
                               "possible, so something else moved", got, want)
assert len(m.encoder.layers) == 1 and m.n_iters == L, (len(m.encoder.layers),
                                                       m.n_iters)
print("%d epochs  one distinct block applied %d times, %s trainable -- IDENTICAL "
      "to the 50-epoch control's count" % (ep, L, format(got, ",")))
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n48_budget_W_2026-11-12"
cell="${OUT}/t${TT}_d${DD}_ff${FFS}_h${HH}_${ARCH}_L${LL}_ep${EPOCHS}_s${SEED}"
mkdir -p "$cell"

echo "task=${TASK}  epochs=${EPOCHS} applications=${LAYERS} d_model=${DMODEL} ff=${FF} nhead=${NHEAD} arch=${ARCH} k=${KBLOCKS} seed=${SEED}"
echo "cell -> ${cell}"
date +'cell start %F %T'

python3 -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 48 --degree 5 --n-colors 4 --shuffle-adj --fall \
    --d-model "$DMODEL" --nhead "$NHEAD" --dim-feedforward "$FF" \
    --num-layers "$LAYERS" --arch-mode "$ARCH" --n-blocks "$KBLOCKS" \
    --seed "$SEED" --warmup-epochs "$WARMUP" \
    --n-train-instances 240000 --epochs "$EPOCHS" \
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
echo "=== what the checkpoint says it is (read back, not trusted) ==="
python3 - "${CKPT}/config.json" "$ARCH" "$LAYERS" "$NHEAD" "$DMODEL" "$FF" "$WARMUP" "$SEED" "$KBLOCKS" "$EPOCHS" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
want_arch, want_L, want_h = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
want_d, want_ff, want_w = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
want_seed, want_k, want_ep = int(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10])
assert c["shuffle_adj"] is True, "LEAK: this cell is not shuffle_adj"
assert c["fall"] is True, "wrong ensemble: fall is not set"
assert c["arch_mode"] == want_arch, (c["arch_mode"], want_arch)
assert c["nhead"] == want_h and c["d_model"] == want_d
assert c["dim_feedforward"] == want_ff, (c["dim_feedforward"], want_ff)
assert int(c.get("warmup_epochs", 0) or 0) == want_w, (c.get("warmup_epochs"), want_w)
assert c["n_nodes"] == 48, ("stage W is N=48 only (finding 55)", c["n_nodes"])
assert int(c["seed"]) == want_seed, ("wrong seed; the triple would not pool",
                                     c["seed"], want_seed)
assert int(c.get("n_blocks", 1) or 1) == want_k, ("k moved; stage W holds k=1",
                                                  c.get("n_blocks"))
assert c["num_layers"] == want_L, ("the application count moved; that is stage "
                                   "V's knob and this stage holds it at 16",
                                   c["num_layers"], want_L)
assert int(c.get("n_train_instances", 240000)) == 240000, (
    "the data size moved, and n_train_instances is NOT in rung_key -- this cell "
    "would pool into stage S's rung and destroy its own control",
    c.get("n_train_instances"))
# THE KNOB. epochs IS in rung_key (as a bare int, distinguishable by type from
# stage U's "k2" and stage T's "T24" tails), which is why this stage keys
# distinctly from S1 and the data route above was rejected.
assert int(c["epochs"]) == want_ep, ("the budget did not round-trip; this cell "
                                     "would pool into S1's rung",
                                     c["epochs"], want_ep)
assert c["max_len_margin"] == 2 * c["n_nodes"] + 2, c["max_len_margin"]
d, L = c["d_model"], c["num_layers"]
BLOCK = 4 * d * d + 8 * d
pred = (630 * d + 58) + BLOCK            # k=1, and the budget does not appear
got = c.get("n_trainable_params", -1)
print("arch_mode=%s epochs=%d applications=%d (one block, %d times) nhead=%d "
      "d_model=%d ff=%d warmup=%d seed=%d  trainable %s params"
      % (c["arch_mode"], int(c["epochs"]), L, L, c["nhead"], d,
         c["dim_feedforward"], int(c.get("warmup_epochs", 0) or 0),
         int(c["seed"]), format(got, ",")))
assert got == pred == 613818, (
    "this is not the registered cell: twice the budget must cost ZERO "
    "parameters, and the whole tractability case rests on that", got, pred)
print("613,818 trainable at %d epochs -- IDENTICAL to the 50-epoch control. The "
      "object is still h_{t+1} = F(h_t), iterated 16 times." % int(c["epochs"]))
PY

echo
echo "=== the transition curve for this cell (W-b) ==="
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.33 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== W-a: token accuracy, read BEFORE any level; then W-c, within-run ==="
python3 - "${CKPT}/history.json" "$EPOCHS" <<'PY'
import json, sys
PLATEAU = 1.64          # finding 44, an N=48 number
BASIN = 1.561           # W-d's line, between stage U's two populations
S1_NLL = (1.5851, 1.5915, 1.5864)       # stage S's 50-epoch cell, the control
S1_ACC = 0.4830
BAR = 0.4841            # W-a: the control plus the 0.0011 the gate needs
NLL_BAR = 0.0065        # W-c: stage S's tail slope over the added 50 epochs
h = json.load(open(sys.argv[1]))
want_ep = int(sys.argv[2])
best = min(h, key=lambda r: r["nll"])
final_train = h[-1]["train_nll"]
tail = [r["train_nll"] for r in h[-5:]]
slope = (tail[-1] - tail[0]) / max(len(tail) - 1, 1)
acc = h[-1]["accuracy"]
print("epochs %d  final train_nll %.4f  final val_nll %.4f  best val_nll %.4f at epoch %d"
      % (len(h), final_train, h[-1]["nll"], best["nll"], best["epoch"]))
print("final val TOKEN ACCURACY %.4f  %+.4f against the 50-epoch control's %.4f  "
      "(W-a bar %.4f: %s)"
      % (acc, acc - S1_ACC, S1_ACC, BAR, "CLEARS" if acc >= BAR else "MISSES"))
print("tail slope over the last 5 epochs: %+.5f nll/epoch" % slope)
print("the control finished at %s -- this cell is %+.4f against their median"
      % ("/".join("%.4f" % v for v in S1_NLL), final_train - sorted(S1_NLL)[1]))
# W-c, the only PAIRED number this stage has: same seed, same data order, same
# weights up to epoch 50, and the constant lr is what makes the two comparable.
if len(h) >= 50:
    at50 = h[49]["train_nll"]
    print("W-c  epoch 50 train_nll %.4f -> epoch %d %.4f   fell %+.4f  (bar %.4f: %s)"
          % (at50, len(h), final_train, at50 - final_train, NLL_BAR,
             "CLEARS" if at50 - final_train >= NLL_BAR else "BELOW BAR -- this "
             "seed was converged at 50"))
    print("     val accuracy %.4f at epoch 50 -> %.4f at epoch %d"
          % (h[49]["accuracy"], acc, len(h)))
first = next((r["epoch"] for r in h if r["train_nll"] < PLATEAU), None)
print("plateau %s" % ("escaped at epoch %d" % first if first is not None
                      else "NOT ESCAPED (min %.4f)" % min(r["train_nll"] for r in h)))
low = next((r["epoch"] for r in h if r["train_nll"] < BASIN), None)
print("W-d  loss line %.3f %s -- and a low NLL at twice the budget is NOT a basin "
      "on its own; it needs the accuracy above too"
      % (BASIN, "crossed at epoch %d" % low if low is not None else "not reached"))
assert len(h) == want_ep, ("the run did not reach the registered %d epochs; a "
                           "truncated cell does not compare with its matched "
                           "rung" % want_ep, len(h))
PY

echo
echo "=== done ==="
date +'%F %T'
echo "score with: python3 -u -m scripts.stagew_score"
