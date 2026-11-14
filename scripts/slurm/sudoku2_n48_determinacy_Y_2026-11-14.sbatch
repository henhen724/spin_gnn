#!/bin/bash
# =============================================================================
# STAGE Y -- GIVE DETERMINACY ITS OWN CHANNEL, SO IT CAN RE-RANK NODES INSTEAD
# OF SHIFTING THE WHOLE BLOCK. 320 new parameters, +0.05%.
#
# WHERE THIS STARTS, and it starts from a failure
#
#     S1       tied_no_mlp T=16 d320 k=1  613,818  rho_50 0.754/0.757/0.765
#              val acc 0.4830/0.4830/0.4833          free line 0.740
#     stage X  the same cell with a determinacy BCE on the READOUT's node
#              logits, lambda=1, zero new parameters. Verdict from the
#              registered scorer: representation_without_use.
#                AUROC        0.7610/0.7635/0.7774 -> 0.9462/0.9467/0.9469
#                guard rate   1.0000/1.0000/1.0000 -> 1.0000/1.0000/0.9568
#                val acc      0.4830/0.4830/0.4833 -> 0.4780/0.4767/0.4778
#                rho_50       0.754/0.757/0.765    -> 0.866/0.864/0.855
#
# X2 is the decisive one. Determinacy made linearly readable at AUROC 0.947
# INSIDE the logits that choose the emission bought ZERO counterfactual
# responsiveness: destroy node v's determinacy and the stage-X model still
# names v, at the control's rate, in every seed. So there is no over-eager
# guard to tune down -- there is no conditional guard at all. What the BCE
# moved is the ABSOLUTE height of the node block against the terminal tokens,
# a density-independent bias toward stopping, which the 14-point grid shows
# directly: the SOLVED/UNSOLVED call is flat near 1.00 for the control at
# every density and traces a U for stage X, 0.93 at rho 0.94, 0.42-0.45 at
# rho 0.69-0.75, 0.94 at rho 0.33.
#
# THE LESSON STAGE Y IS BUILT ON: a per-node loss on absolute logit height
# does not produce per-node conditional behaviour. It shifts the block.
#
# AND THE SHAPE IT SHOULD HAVE HAD (finding 74, from finding 68's stored CSVs,
# measured before this was designed): when the UNTIED model's guard fires it
# emits a node token 96.6% of the time, that node is still forced 90.3% of the
# time, and it names the broken node again in 0 of 350 cases. The real guard
# does not stop. It RE-RANKS. A BCE on the height of z_{p,v} says "v is or is
# not deducible" one node at a time and can never say "u before v".
#
# THE ONE THING THIS CHANGES
#
#     z_{p,v}                      the existing readout, untouched
#     g_{p,v} = <h[p] * s, E_v>    the determinacy channel, s in R^d learned
#     emission over node tokens:   z'_{p,v} = z_{p,v} + log sigmoid(g_{p,v})
#     everything else:             unchanged
#
# E is token_embed and W is readout, SEPARATE matrices in this model and never
# tied, so g is a different function of h than z and the sum reorders the node
# block. Two structural properties, and both are the point:
#
#   1  It can re-rank. The cheap monotone alternative cannot: z -> z + log
#      sigmoid(a(z - tau)) is strictly increasing in z, preserves the order
#      WITHIN the node block exactly, and can only suppress the block against
#      the non-node tokens -- stage X's failure mode at two scalars. I was
#      going to build that until finding 74.
#   2  CE and the BCE stop competing for the same scalars. In stage X both
#      wrote to z: CE wants one node far above the rest, the BCE wants ~35% of
#      the open nodes high at once. -0.0055 of token accuracy is what that
#      conflict cost.
#
# PRICE: one vector, d_model = 320 parameters. 613,818 -> 614,138, +0.052%.
# The arm ratio against J2's 2,504,890 stays 4.08x. The brief is the simplest
# analysable object that learns direct deduction, not the fewest parameters,
# and a diagonal rescale of h read through the embedding the model already has
# is about the smallest structural change that can reorder anything.
#
# HELD FIXED, so this is ONE knob: the same tied cell (tied_no_mlp T=16 d320
# ff1024 h4 k=1), the same 240k instances, the same 50 epochs, the same seeds
# 0/1/2, the same lambda = 1.0 NOT swept, the same deducibility target at the
# same pre_node positions over still-uncoloured nodes only (positive rate
# 0.3544, scored fraction 0.2466), the same --deterministic kernels as stage X,
# the same 14-point rho grid at 400 per point.
#
# REGISTERED PREDICTIONS -- scripts/stagey_score.py, whose docstring IS the
# registration, committed at a05847a BEFORE this launcher was written. The
# bars are IMPORTED below, never retyped.
#
#   Y4  SANITY, FIRST, and it can veto everything. val token accuracy >= 0.4800
#       in all 3 seeds. Set against the MEASURED control (0.4830/0.4830/0.4833)
#       and not a historical floor: stage X passed an X4 of this shape at
#       0.4776 because its floor was 0.4330, ten times the damage. Stage X
#       would FAIL this bar.
#   Y1  THE POINT. max(rho_50) < 0.754, complete separation below the control.
#       Whether it also clears the 0.740 free line is reported SEPARATELY.
#   Y2  THE MECHANISM. Finding 68's guard probe unchanged: conditional guard
#       rate below 0.90 in >= 2 of 3 seeds.
#   Y3  THE SHAPE, new in stage Y. Among fired cases, node_token >= 0.90 AND
#       still_forced >= 0.60 in >= 2 of 3 seeds -- reproducing the untied
#       model's 0.966 and 0.903. Y2 WITHOUT Y3 is a stop-shaped guard again
#       and the scorer calls it "design refuted", not a partial success.
#
# WHAT A NULL WOULD MEAN. Y4 pass with Y1 and Y2 refuted says the coupling is
# not the missing piece and the determinacy signal, however it is wired into
# this emission, does not reach the schedule -- which would point at the tied
# cell's depth rather than its objective (finding 71: the untied guard's
# provenance-specificity is partly depth). Y2 with Y3 refuted says the channel
# built stage X's guard again at 320 parameters' extra cost.
#
# WHAT IT CANNOT SHOW: g is read from the same h[p] the readout sees, so a
# negative rules out this coupling and not every coupling; and a positive
# cannot separate re-ranking capacity from the end of the CE/BCE conflict,
# which needs a third arm with g trained but DETACHED from the emission. That
# is the natural stage Z and is not submitted here.
#
# SMOKE-TESTED BEFORE SUBMISSION: scripts/stagey_smoke.py, six assertions,
# all pass -- 613,818 -> 614,138; channel off is bit-identical to the old
# forward; z'_node == z_node + log sigmoid(g) exactly and no other column
# moves; d(BCE)/d(readout) is exactly zero; the emission does reorder the
# node block with only the 320-vector moving; and the channel is INERT AT
# INITIALIZATION.
#
# THE FIRST SUBMISSION OF THIS ARRAY FAILED, AND THE SIXTH ASSERTION IS WHY
# THIS ONE IS DIFFERENT. 52336800 initialized s to ones, on the untested
# reasoning -- written in the code comment as if it were a result -- that
# g = <h, E_v> would be "a small common offset". Measured, it is neither:
# h is LayerNorm'd and E_v ~ N(0, I), so g had std 21 and log sigmoid(g) std
# 10.3, reaching -59, against node logits whose own spread is 0.61. A random
# PER-NODE penalty seventeen times the signal it perturbs, scrambling the
# ranking before step 1. Seeds 0 and 2 trained through it (epoch 12: val acc
# 0.4600/0.4582, AUROC 0.940); seed 1 never left the initial plateau
# (train_nll 3.67 -> 3.96 in epoch 0, val acc 0.032, AUROC 0.59, flat for 13
# epochs). Y4 needs all three seeds, so the array was cancelled at 2h29
# rather than spend 21 more GPU-h on a verdict the scorer would have had to
# return as "Y4 failed, Y1-Y3 not interpreted" -- and on a model whose
# channel began as noise injection rather than as the registered design.
#
# s = 0 makes the channel inert: g is identically 0 and log sigmoid(g) =
# -log 2 for every node at every position, so the ORDER within the node block
# starts exactly the control's, which is what the original comment claimed
# and did not deliver. Not a saddle (dL/ds_i = sum_{p,v} (dL/dg) h_pi E_vi is
# O(1)) and not a parameter (still +320 exactly). THE ONLY CHANGE FROM
# 52336800 is this initialization: the design, the bars and the registration
# are untouched.
#
# WALL TIME. Stage X's cells ran 9h15-9h31 with the same data, depth and
# pinned kernels. The channel adds one (48 x 320) matmul per position per
# forward, which is under a percent of the readout's own (58 x 320). 20:00:00
# has margin; sganguli's limit is 7 days.
#
# NON-NEGOTIABLES CARRIED FORWARD: `--shuffle-adj` on every cell (16/17/23), 3
# seeds with the spread quoted (41/42/47/51/56), validation token accuracy read
# BEFORE rho_50 (4+6), ONE knob at a time, N=48 only (finding 55).
# =============================================================================

#SBATCH --job-name=sud2N48DetermY
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%A_%a.out

set -euo pipefail

# determinacy_weight:seed -- ONE weight, three seeds. Everything else is stage
# X's cell, which is stage S's d320 cell, unchanged.
SPECS=(
  "1.0:0"  "1.0:1"  "1.0:2"
)

TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
IFS=: read -r DETW SEED <<< "${SPECS[$TASK]}"
SEED=$((10#$SEED))

LAYERS=16
DMODEL=320
KBLOCKS=1
FF=1024
NHEAD=4
WARMUP=0
EPOCHS=50
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
# Read ONCE at cuBLAS init, so it must be exported before python starts.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${SLURM_SUBMIT_DIR}"

echo "=== smoke check ==="
python3 - "$DMODEL" "$NHEAD" "$KBLOCKS" "$LAYERS" "$EPOCHS" "$DETW" <<'PY'
import sys
import torch, numpy, numba
import spin_gnn.sudoku_hparam_sweep  # noqa: F401
import scripts.rho_transition_eval   # noqa: F401
import scripts.stagey_score as sy
from spin_gnn.sat_transformer2 import ARCH_MODES, SATReasoningTransformer
from spin_gnn.sudoku_determinacy import (determinacy_loss_from_channel,  # noqa: F401
                                         determinacy_targets)
d, nhead, k, L, ep = (int(a) for a in sys.argv[1:6])
detw = float(sys.argv[6])
print("torch", torch.__version__, "numpy", numpy.__version__, "numba", numba.__version__)
assert "tied_no_mlp" in ARCH_MODES, ("the tied mode is not in this checkout", ARCH_MODES)
assert d == 320 and nhead == 4 and k == 1, (
    "stage Y is stage X's cell with the determinacy term MOVED to its own "
    "channel and nothing else", d, nhead, k)
assert L == 16 and ep == 50, ("stage Y holds the application count and the "
                              "budget at stage S's", L, ep)
assert detw == sy.LAMBDA == 1.0, ("this is not the registered lambda", detw, sy.LAMBDA)

base = SATReasoningTransformer(vocab_size=58, d_model=d, num_layers=L, nhead=nhead,
                               max_len=512, dim_feedforward=1024, norm_first=True,
                               arch_mode="tied_no_mlp", n_blocks=k)
m = SATReasoningTransformer(vocab_size=58, d_model=d, num_layers=L, nhead=nhead,
                            max_len=512, dim_feedforward=1024, norm_first=True,
                            arch_mode="tied_no_mlp", n_blocks=k,
                            determinacy_channel=True, node_offset=2, n_nodes=48)
# THE PRICE OF THIS STAGE, asserted rather than claimed. (630d + 58) +
# (4d^2 + 8d)k is stage S's count and the channel adds exactly d.
want = (630 * d + 58) + (4 * d * d + 8 * d) * k
got_base, got = base.trainable_parameter_count(), m.trainable_parameter_count()
assert got_base == want == 613818, (got_base, want)
assert got == want + d == 614138, ("the channel is not one vector", got, want + d)
assert got - got_base == sy.NEW_PARAMS == 320, (got - got_base, sy.NEW_PARAMS)
assert len(m.encoder.layers) == 1 and m.n_iters == L
print("one distinct block applied %d times, %s -> %s trainable (+%d, +%.3f%%)"
      % (L, format(got_base, ","), format(got, ","), got - got_base,
         100 * (got - got_base) / got_base))
print("registered bars: acc>=%.4f  rho<%.3f  guard<%.2f  shape node>=%.2f "
      "forced>=%.2f  in >=%d of 3  (free line %.3f reported separately)"
      % (sy.ACC_BAR, sy.RHO_BAR, sy.GUARD_BAR, sy.SHAPE_NODE_BAR,
         sy.SHAPE_FORCED_BAR, sy.MIN_SEEDS, sy.FREE_LINE))
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))
PY

echo
echo "=== the six structural assertions of the channel, in the job ==="
# These are what distinguish stage Y from stage X at all, and four of the six
# can only fail SILENTLY: an off-by-default flag that does not take, an
# emission that drops log sigma(g), a BCE that still reaches the readout, or a
# channel that is not inert at initialization would each produce a plausible
# run of the WRONG stage -- and the last of those is not hypothetical, it is
# what 52336800 did. Checked here rather than trusted from a login node.
python3 scripts/stagey_smoke.py

echo
echo "=== the determinacy target is pinned against DD itself, not assumed ==="
python3 - <<'PY'
import importlib
m = importlib.import_module("tests.test_sudoku_determinacy")
names = sorted(n for n in dir(m) if n.startswith("test_"))
for n in names:
    getattr(m, n)()
    print("PASS", n)
print("%d determinacy tests pass" % len(names))
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n48_determinacy_Y_2026-11-14b"
cell="${OUT}/t${TT}_d${DD}_ff${FFS}_h${HH}_${ARCH}_L${LL}_dw${DETW}_chan_s${SEED}"
mkdir -p "$cell"

echo
echo "task=${TASK}  determinacy_weight=${DETW} CHANNEL=on applications=${LAYERS} d_model=${DMODEL} ff=${FF} nhead=${NHEAD} arch=${ARCH} k=${KBLOCKS} epochs=${EPOCHS} seed=${SEED}"
echo "cell -> ${cell}"
date +'cell start %F %T'

python3 -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 48 --degree 5 --n-colors 4 --shuffle-adj --fall \
    --d-model "$DMODEL" --nhead "$NHEAD" --dim-feedforward "$FF" \
    --num-layers "$LAYERS" --arch-mode "$ARCH" --n-blocks "$KBLOCKS" \
    --seed "$SEED" --warmup-epochs "$WARMUP" --deterministic \
    --determinacy-weight "$DETW" --determinacy-channel \
    --n-train-instances 240000 --epochs "$EPOCHS" \
    --n-val-instances 300 --batch-size 32 --n-per-bin 100 \
    --out-dir "$cell"

date +'cell end   %F %T'

TAG="d${DD}_h${HH}_L${LL}_ff${FFS}_lr1e-03_pre_shuf_fall_noff_tied_det_dw1_chan"
if [ "$SEED" -ne 0 ]; then
    TAG="${TAG}_s$(printf '%02d' "$SEED")"
fi
CKPT="${cell}/${TAG}"
[ -f "${CKPT}/model.pt" ] || { echo "expected checkpoint missing: ${CKPT}"; ls -1 "$cell"; exit 1; }

echo
echo "=== what the checkpoint says it is (read back, not trusted) ==="
python3 - "${CKPT}/config.json" "$ARCH" "$LAYERS" "$NHEAD" "$DMODEL" "$FF" "$WARMUP" "$SEED" "$KBLOCKS" "$EPOCHS" "$DETW" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
want_arch, want_L, want_h = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
want_d, want_ff, want_w = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
want_seed, want_k, want_ep = int(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10])
want_dw = float(sys.argv[11])
assert c["shuffle_adj"] is True, "LEAK: this cell is not shuffle_adj"
assert c["fall"] is True, "wrong ensemble: fall is not set"
assert c["arch_mode"] == want_arch, (c["arch_mode"], want_arch)
assert c["nhead"] == want_h and c["d_model"] == want_d
assert c["dim_feedforward"] == want_ff, (c["dim_feedforward"], want_ff)
assert int(c.get("warmup_epochs", 0) or 0) == want_w
assert c["n_nodes"] == 48, ("stage Y is N=48 only (finding 55)", c["n_nodes"])
assert int(c["seed"]) == want_seed, ("wrong seed; the triple would not pool",
                                     c["seed"], want_seed)
assert int(c.get("n_blocks", 1) or 1) == want_k, ("k moved; stage Y holds k=1",
                                                  c.get("n_blocks"))
assert c["num_layers"] == want_L, ("the application count moved; that is stage "
                                   "V's knob and Y holds it at 16", c["num_layers"])
assert int(c["epochs"]) == want_ep, ("the budget moved; that is stage W's knob",
                                     c["epochs"])
assert int(c.get("n_train_instances", 240000)) == 240000, (
    "the data size moved, and n_train_instances is NOT in rung_key -- this cell "
    "would pool into stage S's rung and destroy its own control",
    c.get("n_train_instances"))
assert c.get("deterministic") is True, "the pinned-kernel flag did not round-trip"
assert float(c.get("determinacy_weight", 0.0)) == want_dw, (
    "determinacy_weight did not round-trip", c.get("determinacy_weight"), want_dw)
# THE KNOB OF THIS STAGE. Unlike every earlier flag this one changes the state
# dict and the forward map, so a loader that misses it gets a shape error --
# but a TRAINER that misses it would have silently rerun stage X under a
# stage-Y directory name, which is what this line exists to prevent.
assert c.get("determinacy_channel") is True, (
    "determinacy_channel did not round-trip -- this cell is stage X again, "
    "not stage Y", c.get("determinacy_channel"))
assert c["max_len_margin"] == 2 * c["n_nodes"] + 2, c["max_len_margin"]
# The positional embedding is TRAINABLE, so max_len is a term in the count: one
# position is d_model = 320 parameters, and 511 would give 613,498 + 320. A
# cell that lands anywhere but 512 is not the matched control.
assert c["max_len"] == 512, (
    "max_len is %s, not stage S's 512 -- this cell has a different positional "
    "table and its count will not match the control" % c["max_len"])
d, L = c["d_model"], c["num_layers"]
pred = (630 * d + 58) + (4 * d * d + 8 * d) + d   # k=1, plus the channel's s
got = c.get("n_trainable_params", -1)
print("arch_mode=%s lambda=%g channel=%s applications=%d (one block, %d times) "
      "nhead=%d d_model=%d ff=%d seed=%d  trainable %s params"
      % (c["arch_mode"], float(c["determinacy_weight"]), c["determinacy_channel"],
         L, L, c["nhead"], d, c["dim_feedforward"], int(c["seed"]), format(got, ",")))
assert got == pred == 614138, (
    "this is not the registered cell: stage Y costs EXACTLY one vector of "
    "d_model, 320 parameters on top of the control's 613,818", got, pred)
print("614,138 trainable = the control's 613,818 + one 320-vector, +0.052%. "
      "The object is still h_{t+1} = F(h_t), iterated 16 times.")
PY

echo
echo "=== Y1: the transition curve for this cell ==="
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.33 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== Y4 first (it can veto the rest), then the within-run AUROC ==="
python3 - "${CKPT}/history.json" "$EPOCHS" <<'PY'
import json, sys
sys.path.insert(0, ".")
import scripts.stagey_score as sy
h = json.load(open(sys.argv[1]))
want_ep = int(sys.argv[2])
acc = h[-1]["accuracy"]
ctrl = sy.CONTROL_ACC[1]
print("epochs %d  final train_nll %.4f  final val_nll %.4f"
      % (len(h), h[-1]["train_nll"], h[-1]["nll"]))
print("Y4  final val TOKEN ACCURACY %.4f  %+.4f against the control's %.4f  "
      "(bar %.4f: %s)"
      % (acc, acc - ctrl, ctrl, sy.ACC_BAR,
         "OK" if acc >= sy.ACC_BAR else "THE CHANNEL DID NOT BUY BACK THE "
         "INTERFERENCE -- Y1-Y3 are not interpreted for this seed"))
print("    stage X scored 0.4780/0.4767/0.4778 on this bar and would fail it")
print("    determinacy AUROC is now measured on the CHANNEL g, not on the "
      "readout: %.4f (train det_bce %.4f)"
      % (h[-1]["determinacy_auroc"], h[-1]["train_determinacy_bce"]))
print("    first epoch %.4f -> last %.4f;  det_bce %.4f -> %.4f"
      % (h[0]["determinacy_auroc"], h[-1]["determinacy_auroc"],
         h[0]["train_determinacy_bce"], h[-1]["train_determinacy_bce"]))
print("    NOT comparable to stage X's 0.947: that scored z, this scores g. "
      "A high AUROC here is necessary and, per finding 75, nowhere near "
      "sufficient -- Y2 and Y3 are the bars that matter.")
first = next((r["epoch"] for r in h if r["train_nll"] < 1.64), None)
print("plateau %s" % ("escaped at epoch %d" % first if first is not None
                      else "NOT ESCAPED (min %.4f)" % min(r["train_nll"] for r in h)))
low = next((r["epoch"] for r in h if r["train_nll"] < 1.561), None)
print("basin line 1.561 %s" % ("crossed at epoch %d" % low
                               if low is not None else "not reached"))
assert len(h) == want_ep, ("the run did not reach the registered %d epochs; a "
                           "truncated cell does not compare with its matched "
                           "rung" % want_ep, len(h))
PY

echo
echo "=== done ==="
date +'%F %T'
echo "Y2 and Y3 need the guard probe: scripts/slurm/stagey_guard_2026-11-14.sbatch"
echo "chained with --dependency=afterok on this array, so nothing is polled"
