#!/bin/bash
# =============================================================================
# STAGE X -- TRAIN THE READOUT TO SAY WHICH NODES ARE DEDUCIBLE. Does the guard
# appear, and does any of the standing 0.017 close?
#
# THIS IS NOT A HYPERPARAMETER. Stages T, U, V and W closed every knob that
# could have supplied the gap -- role count k, schedule length T, training
# budget, width, decay, data -- and each came back inside the 0.0423 LOO bar or
# outright worse. The board has no knob left, and stage X does not look for
# one: it changes the OBJECTIVE and holds the object fixed.
#
# WHERE THIS STARTS
#
#     S1  tied_no_mlp T=16 d320 k=1  50 ep  613,818  rho_50 0.757
#         (0.754/0.757/0.765)  acc 0.4830 (spread 0.0010)   free line 0.740
#
# WHY CROSS-ENTROPY CANNOT ALREADY HAVE TAUGHT THE GUARD -- the gauge argument
#
# The main loss sees the readout's logit vector only through a softmax, and a
# softmax is invariant under adding a constant to every logit. So CE constrains
# the node logits ONLY UP TO THAT CONSTANT. It can say "node v outranks the
# others here"; it cannot say "node v is not emittable at all", because
# emittability is a claim about ABSOLUTE height and absolute height is the one
# direction CE integrates out. A guard lives exactly there, and nothing in
# stages S-W ever put a gradient on it.
#
# That is a mechanism for what findings 68 and 69 measured, not a restatement
# of them:
#
#   68  S1 applies DD's elimination RULE flawlessly (rule-flip 1.0000) and does
#       not run DD's SCHEDULER. Destroy a node's determinacy with one token and
#       S1 names it anyway: guard rate 1.0000/1.0000/1.0000, spread 0.0000,
#       against the untied arm's 0.0000/0.0000/0.8444.
#   69  The determinacy is COMPUTED REGARDLESS. Under the same broken edit the
#       tied model's p_freed reads 0.5253/0.5228/0.5354 -- correctly uncertain,
#       the ambiguity is represented -- while the node readout does not move to
#       four decimals. Present and unused.
#
# THE KNOB, AND ITS PRICE IN PARAMETERS: ZERO
#
#     L = CE_next_token  +  lambda * BCE_determinacy
#
# BCE_determinacy is a multi-label binary cross-entropy on the EXISTING
# readout's node logits, at every position about to emit a node, over the nodes
# still uncoloured there, against DD's own deducibility predicate. No head, no
# projection, no parameter. A stage-X cell is 613,818 trainable weights and
# S1's state dict exactly, so every probe written for findings 62-72 reads one
# untouched, and S1's three seeds are the genuine lambda=0 control.
#
# A SEPARATE AUXILIARY HEAD WAS CONSIDERED AND REJECTED. Finding 69 already
# established the determinacy representation EXISTS; a head that reads it more
# sharply is predicted-null by that finding. The question is whether the
# EMISSION can be made to use it, so the term has to land on the logits that
# choose the emission.
#
# LAMBDA = 1.0, REGISTERED, NOT SWEPT. Both terms are means of per-item losses;
# at N=48 the determinacy term starts at 0.733 against a main term of 3.95, so
# it enters at roughly 0.19x. One value chosen a priori. A sweep would be a
# second knob and is not part of this stage.
#
# WHAT THE TERM SCORES, ON THE DISTRIBUTION THESE CELLS ACTUALLY TRAIN ON:
# scored_fraction 0.2486, positive_rate 0.3543 (157,470 positive / 286,992
# negative / 1,343,058 masked). An earlier reading of 0.156 / 0.654 was taken
# at a single high hint density and does NOT describe the training mix -- at
# high rho few nodes are open and most of those are determined, at low rho the
# reverse. The mix is the better-conditioned of the two for a BCE.
# Already-coloured nodes are MASKED, not labelled negative: they are trivially
# undeducible and they are most of every row, so labelling them would make the
# task mostly trivial and the AUROC unreadable.
#
# PINNED KERNELS, AND THE ONE WAY THIS DIFFERS FROM ITS CONTROL. These cells run
# --deterministic and stage S did not. Finding 67 measured that pinning removes
# run-to-run noise WITHOUT shifting the distribution (the pinned pair landed at
# 2.4249, inside the unpinned 2.3656-2.5950), and with 3 seeds the variance that
# matters is the basin lottery, which pinning does not touch. So this buys
# reproducibility rather than moving a second knob -- but it IS a difference
# from the control and a reader is entitled to discount it. Stated here rather
# than discovered later.
#
# REGISTERED PREDICTIONS -- scripts/stagex_score.py, whose docstring IS the
# registration, committed at 1b190a0 BEFORE this array was written. The bars are
# IMPORTED from it below, never retyped, so one number judges the stage.
#
#   X4  FIRST, and it can veto everything. Final val token accuracy >= 0.4330
#       (the control's 0.4830 less 0.05) in ALL THREE seeds. If this fails the
#       term broke training and X1-X3 are uninterpretable.
#   X1  PRIMARY. rho_50 of every stage-X seed strictly below rho_50 of every
#       stage-S seed -- complete separation of 3 against 3, which is the
#       one-sided rank test at exactly p = 1/20. Reported separately and not
#       conflated: whether all three clear the 0.740 free line, which is the
#       standing gate.
#   X2  MECHANISM. Finding 68's guard probe, unchanged, on its byte-identical
#       350 round-0 cases: conditional guard rate below 0.90 in >= 2 of 3
#       seeds. Lower is better. The control measured 1.0000 three times with
#       spread 0.0000, so 0.90 is ten times its entire spread.
#   X3  DID THE TERM TAKE. Final val determinacy AUROC above every stage-S
#       cell's, the control measured with the same function on the same
#       instances by scripts/determinacy_eval.py. Above chance is NOT the bar:
#       the next node emitted is always deducible, so CE alone already lifts it
#       and the stage-S cells are expected to beat 0.5.
#
# FALSIFICATION, STATED BEFORE THE CELLS RUN, AND THE NULL IS THE INTERESTING
# ONE. X3 confirmed with X2 refuted -- the determinacy made linearly explicit in
# the very logits that choose the emission, and the emission still ignoring it
# -- is the sharpest negative this project can produce, and it is what finding
# 69 predicts. It would say the schedule has to be STRUCTURALLY coupled to the
# determinacy (a logit mask at the readout) rather than merely trained
# alongside it, and that is a different and smaller follow-up than another
# training stage. X3 refuted means the term never took and nothing downstream
# is readable.
#
# WALL TIME. The term adds one BCE over 15.6% of a (32, ~13, 48) slice per step
# and a 300-instance AUROC per epoch: both negligible. The cost is stage S's
# 8h05-11h29 per d320 cell plus finding 67's measured +7.7% for pinned kernels,
# plus about an hour for the 14-point grid at 400 per bin. 20:00:00 has margin
# (sganguli's limit is 7 days), and the registered 50 epochs is asserted in the
# read-back so a truncated cell fails loudly instead of being compared against
# a full one.
#
# NON-NEGOTIABLES CARRIED FORWARD: `--shuffle-adj` on every cell (16/17/23), 3
# seeds with the spread quoted (41/42/47/51/56), validation token accuracy read
# BEFORE rho_50 (4+6), ONE knob at a time, N=48 only (finding 55: the N=48
# minimum does not transfer to N=96).
# =============================================================================

#SBATCH --job-name=sud2N48DetermX
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%A_%a.out

set -euo pipefail

# determinacy_weight:seed -- ONE weight, three seeds. Everything else is stage
# S's d320 cell, unchanged, down to the 240k instance pool and the 50 epochs.
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
# Read ONCE at cuBLAS init, so it must be exported before python starts. The
# trainer asserts on it rather than running non-deterministically while the
# directory name claims otherwise.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "${SLURM_SUBMIT_DIR}"

echo "=== smoke check ==="
python3 - "$DMODEL" "$NHEAD" "$KBLOCKS" "$LAYERS" "$EPOCHS" "$DETW" <<'PY'
import sys
import torch, numpy, numba
import spin_gnn.sudoku_hparam_sweep  # noqa: F401
import scripts.rho_transition_eval   # noqa: F401
import scripts.stagex_score as sx
from spin_gnn.sat_transformer2 import ARCH_MODES, SATReasoningTransformer
from spin_gnn.sudoku_determinacy import determinacy_loss, determinacy_targets  # noqa: F401
d, nhead, k, L, ep = (int(a) for a in sys.argv[1:6])
detw = float(sys.argv[6])
print("torch", torch.__version__, "numpy", numpy.__version__, "numba", numba.__version__)
assert "tied_no_mlp" in ARCH_MODES, ("the tied mode is not in this checkout", ARCH_MODES)
assert d == 320 and nhead == 4 and k == 1, (
    "stage X is stage S's d320 k=1 cell with the OBJECTIVE moved and nothing "
    "else", d, nhead, k)
assert L == 16 and ep == 50, ("stage X holds the application count and the "
                              "budget at stage S's", L, ep)
# The registered lambda, read from the registration rather than from this file,
# so the two cannot drift apart.
assert detw == sx.LAMBDA == 1.0, ("this is not the registered lambda", detw, sx.LAMBDA)
m = SATReasoningTransformer(vocab_size=58, d_model=d, num_layers=L, nhead=nhead,
                            max_len=512, dim_feedforward=1024, norm_first=True,
                            arch_mode="tied_no_mlp", n_blocks=k)
# THE HEADLINE GUARD OF THIS STAGE, and the reason the loss attaches to the
# existing readout rather than to a head: the objective moves and the OBJECT
# does not. (630d + 58) + (4d^2 + 8d)k has no term for a loss.
want = (630 * d + 58) + (4 * d * d + 8 * d) * k
got = m.trainable_parameter_count()
assert got == want == 613818, ("the count moved with the objective -- that is "
                               "only possible if a head crept in", got, want)
assert len(m.encoder.layers) == 1 and m.n_iters == L
print("one distinct block applied %d times, %s trainable -- IDENTICAL to the "
      "lambda=0 control's count" % (L, format(got, ",")))
print("registered bars: free_line=%.3f harm_drop=%.2f guard_bar=%.2f need=%d/3"
      % (sx.FREE_LINE, sx.HARM_DROP, sx.GUARD_BAR, sx.GUARD_MIN_SEEDS))
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))
PY

echo
echo "=== the determinacy target is pinned against DD itself, not assumed ==="
# tests/test_sudoku_determinacy.py compares the target cell-for-cell against
# sudoku_activations.dd_state_labels. If the predicate is wrong the whole stage
# trains on the wrong supervision, so it is checked IN the job rather than
# trusted from a laptop. (pytest is py39-only on this stack; this is the
# inline runner the project uses.)
python3 - <<'PY'
import importlib
m = importlib.import_module("tests.test_sudoku_determinacy")
names = sorted(n for n in dir(m) if n.startswith("test_"))
for n in names:
    getattr(m, n)()
    print("PASS", n)
print("%d determinacy tests pass" % len(names))
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n48_determinacy_X_2026-11-13"
cell="${OUT}/t${TT}_d${DD}_ff${FFS}_h${HH}_${ARCH}_L${LL}_dw${DETW}_s${SEED}"
mkdir -p "$cell"

echo
echo "task=${TASK}  determinacy_weight=${DETW} applications=${LAYERS} d_model=${DMODEL} ff=${FF} nhead=${NHEAD} arch=${ARCH} k=${KBLOCKS} epochs=${EPOCHS} seed=${SEED}"
echo "cell -> ${cell}"
date +'cell start %F %T'

python3 -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 48 --degree 5 --n-colors 4 --shuffle-adj --fall \
    --d-model "$DMODEL" --nhead "$NHEAD" --dim-feedforward "$FF" \
    --num-layers "$LAYERS" --arch-mode "$ARCH" --n-blocks "$KBLOCKS" \
    --seed "$SEED" --warmup-epochs "$WARMUP" --deterministic \
    --determinacy-weight "$DETW" \
    --n-train-instances 240000 --epochs "$EPOCHS" \
    --n-val-instances 300 --batch-size 32 --n-per-bin 100 \
    --out-dir "$cell"

date +'cell end   %F %T'

TAG="d${DD}_h${HH}_L${LL}_ff${FFS}_lr1e-03_pre_shuf_fall_noff_tied_det_dw1"
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
assert c["n_nodes"] == 48, ("stage X is N=48 only (finding 55)", c["n_nodes"])
assert int(c["seed"]) == want_seed, ("wrong seed; the triple would not pool",
                                     c["seed"], want_seed)
assert int(c.get("n_blocks", 1) or 1) == want_k, ("k moved; stage X holds k=1",
                                                  c.get("n_blocks"))
assert c["num_layers"] == want_L, ("the application count moved; that is stage "
                                   "V's knob and X holds it at 16", c["num_layers"])
assert int(c["epochs"]) == want_ep, ("the budget moved; that is stage W's knob",
                                     c["epochs"])
assert int(c.get("n_train_instances", 240000)) == 240000, (
    "the data size moved, and n_train_instances is NOT in rung_key -- this cell "
    "would pool into stage S's rung and destroy its own control",
    c.get("n_train_instances"))
assert c.get("deterministic") is True, "the pinned-kernel flag did not round-trip"
# THE KNOB. The only thing in this checkpoint that distinguishes a stage-X cell
# from its stage-S control -- the architecture, the count and the state dict
# are identical, so if this does not round-trip the two are indistinguishable
# on disk and the stage has no record of what it ran.
assert float(c.get("determinacy_weight", 0.0)) == want_dw, (
    "determinacy_weight did not round-trip", c.get("determinacy_weight"), want_dw)
assert c["max_len_margin"] == 2 * c["n_nodes"] + 2, c["max_len_margin"]
# The positional embedding is TRAINABLE, so max_len is a term in the parameter
# count: one position is d_model = 320 parameters. The 2000-instance smoke
# resolved max_len to 511 and came out at 613,498 for exactly that reason. A
# cell that lands anywhere but 512 holds a different number of positions than
# stage S and is not the matched control, so this is checked BEFORE the count
# and says why.
assert c["max_len"] == 512, (
    "max_len is %s, not stage S's 512 -- the longest training sequence came "
    "out different, so this cell has a different positional table and its "
    "parameter count will not match the control" % c["max_len"])
d, L = c["d_model"], c["num_layers"]
pred = (630 * d + 58) + (4 * d * d + 8 * d)      # k=1, and lambda does not appear
got = c.get("n_trainable_params", -1)
print("arch_mode=%s lambda=%g applications=%d (one block, %d times) nhead=%d "
      "d_model=%d ff=%d seed=%d  trainable %s params"
      % (c["arch_mode"], float(c["determinacy_weight"]), L, L, c["nhead"], d,
         c["dim_feedforward"], int(c["seed"]), format(got, ",")))
assert got == pred == 613818, (
    "this is not the registered cell: the determinacy term must cost ZERO "
    "parameters, and the whole tractability case rests on that", got, pred)
print("613,818 trainable with the term ON -- byte for byte the lambda=0 "
      "control's count. The object is still h_{t+1} = F(h_t), iterated 16 times.")
PY

echo
echo "=== X1: the transition curve for this cell ==="
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.33 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== X4 first (it can veto the rest), then X3 within-run ==="
python3 - "${CKPT}/history.json" "$EPOCHS" <<'PY'
import json, sys
sys.path.insert(0, ".")
import scripts.stagex_score as sx
h = json.load(open(sys.argv[1]))
want_ep = int(sys.argv[2])
acc = h[-1]["accuracy"]
floor = sx.S1_VAL_ACC - sx.HARM_DROP
print("epochs %d  final train_nll %.4f  final val_nll %.4f"
      % (len(h), h[-1]["train_nll"], h[-1]["nll"]))
print("X4  final val TOKEN ACCURACY %.4f  %+.4f against the lambda=0 control's "
      "%.4f  (floor %.4f: %s)"
      % (acc, acc - sx.S1_VAL_ACC, sx.S1_VAL_ACC, floor,
         "OK" if acc >= floor else "THE TERM BROKE TRAINING -- X1-X3 are "
         "uninterpretable for this seed"))
print("X3  final val determinacy AUROC %.4f  (train det_bce %.4f)"
      % (h[-1]["determinacy_auroc"], h[-1]["train_determinacy_bce"]))
print("    first epoch %.4f -> last %.4f   the stage-S control for this number "
      "comes from scripts/determinacy_eval.py, NOT from 0.5"
      % (h[0]["determinacy_auroc"], h[-1]["determinacy_auroc"]))
print("    det_bce %.4f -> %.4f" % (h[0]["train_determinacy_bce"],
                                    h[-1]["train_determinacy_bce"]))
first = next((r["epoch"] for r in h if r["train_nll"] < sx.PLATEAU), None)
print("plateau %s" % ("escaped at epoch %d" % first if first is not None
                      else "NOT ESCAPED (min %.4f)" % min(r["train_nll"] for r in h)))
low = next((r["epoch"] for r in h if r["train_nll"] < sx.BASIN), None)
print("basin line %.3f %s" % (sx.BASIN, "crossed at epoch %d" % low
                              if low is not None else "not reached"))
assert len(h) == want_ep, ("the run did not reach the registered %d epochs; a "
                           "truncated cell does not compare with its matched "
                           "rung" % want_ep, len(h))
PY

echo
echo "=== done ==="
date +'%F %T'
echo "X2 still needs the guard probe on ${CKPT}"
echo "score with: python3 scripts/stagex_score.py <summary.json>"
