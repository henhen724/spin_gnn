#!/bin/bash
# =============================================================================
# STAGE U -- k DISTINCT blocks cycled T/k times. Is the remaining 0.017 a
# ROLE-COUNT limit, or is it the recipe?
#
# WHERE THIS STARTS (stage S, array 51853969, scorer 51904146)
#
#     S1  tied_no_mlp T=16 d320 k=1   613,818   rho_50 0.757 (0.754/0.757/0.765)
#         3/3 crossing   mean_solve 0.3148   val token acc 0.4830
#         plateau escapes 3/3 (1.5851/1.5915/1.5864)
#
# One shared block does direct deduction, and it lands 0.017 PAST the 0.740
# gate. S4 put all three seeds ON the N=48 accuracy-to-solve curve (residuals
# +0.0165/-0.0034/+0.0065 against the 0.0423 LOO bar), so the 0.177 rho_50 gap
# to the untied twin J2 is what J2's extra 0.011 of token accuracy buys at that
# steepness -- and the 0.017 still needed is worth about 0.001 of accuracy.
# Width will not supply it: 0.4567 -> 0.4830 -> 0.4837 across d192/d320/d448.
#
# WHAT STAGE T CONTRIBUTED (array 51904636, scorer 51919538), IN ITS ORDER
#
#   T1 FALSIFIED  the easy end stops solving past the trained count: worst-bin
#                 solve_rate 0.0000/0.0525/0.7525/0.8875/0.7300/0.0000 for
#                 T=4..24. The representation COLLAPSES in 2 of 3 seeds at T=24.
#   level table   rho_50 CENSORED/CENSORED/0.769/0.757/0.763/0.819, crossings
#                 0/0/3/3/3/2, mean_solve ..0.2963/0.3148/0.3020/0.1873
#   T0 FALSIFIED  |T=24 - T=16| = 0.062 and mean_solve PEAKS at the trained
#                 count. Not an iterable operator -- stable for about as many
#                 applications as it was trained for.
#   T2            T* = 16: no smaller T comes within 0.01. All 16 applications
#                 are used; there is no smaller object hiding inside.
#
# T1 outranks T0 by registration, so the honest reading is ONE WEIGHT SET TUNED
# TO A 16-STEP SCHEDULE. That is exactly what k > 1 relieves: k blocks each fit
# to 16/k roles instead of one block fit to 16.
#
# WHY THIS DOES NOT ABANDON THE 11-10 TRACTABILITY BRIEF
#
# k distinct blocks cycled is STILL one map iterated:
#     G = F_{k-1} o ... o F_0        h_{t+1} = G(h_t), applied 16/k times.
# Every rung is a fixed-point iteration. What grows is how much there is to read
# inside G, which is why the figure of merit is the SMALLEST k that clears the
# gate and NOT the parameter count (which rises with k and is not the score).
#
# WHAT CHANGES: EXACTLY ONE KNOB, k
#
#     S1  tied_no_mlp T=16 d320 k=1     613,818   (stage S, on disk, NOT re-run)
#     U1  tied_no_mlp T=16 d320 k=2   1,025,978   ABOVE H1's floor   this array
#     U2  tied_no_mlp T=16 d320 k=4   1,850,298   ABOVE H1's floor   this array
#
# Everything else is S1's: d320, nhead 4, ff 1024 (the feedforward writes
# identically zero in EVERY block), warmup 0, lr 1e-3 constant, 240k x 50ep,
# `--shuffle-adj --fall`, the 14-point grid to 0.33 at 400 per bin, seeds 0/1/2,
# 16 APPLICATIONS throughout. k=1 is re-read from disk by the scorer as the
# control -- never re-run, never retyped.
#
# WARMUP STAYS 0, deliberately. Stage S's d448 arm diverged at warmup 0 in 2 of
# 3 seeds and diagnostic 51860479 traced that to warmup rather than to the cell.
# But k holds width FIXED at 320, where all three seeds trained cleanly and all
# three escaped the plateau. Changing warmup here would be a second knob and
# would make a null result unreadable.
#
# BOTH NEW RUNGS ARE ABOVE H1's 879,418 CENSORING FLOOR, so unlike stage P's
# d192 cell a censor at either is about the architecture and not about size.
# This is said before the cells run, as stage S said it.
#
# VERIFIED BEFORE SUBMISSION (scripts/_cycled_dry_check.py, run by the
# registration job this array depends on with --dependency=afterok): k=1 still
# counts 613,818 so the control has not moved; k=2 and k=4 count 1,025,978 and
# 1,850,298 and match (630d+58)+(4d^2+8d)k; encoder.layers holds exactly k
# blocks while n_iters stays 16; the application order is the cycle 0,1,..,k-1
# repeated and not k applications; all k blocks are identical at init so k does
# not confound capacity with the draw; the feedforward writes zero and is frozen
# in EVERY block; iterating the cycle 16/k times differs from one turn of it;
# and both counts straddle H1's floor on the right side. The full test suite
# runs in the same job, because this is the first stage that changes the model's
# forward pass.
#
# WALL TIME. Stage S's d320 cells took 8h05-11h29 at this width, depth and
# token budget. k changes the weight COUNT, not the per-step FLOPs (still 16
# attention applications at d320), so 16:00:00 is a ceiling with margin rather
# than an estimate; sganguli's limit is 7 days. The registered 50 epochs is
# asserted in the read-back below, so a cell cut short by the limit fails loudly
# instead of being compared against a full one.
#
# NON-NEGOTIABLES CARRIED FORWARD: `--shuffle-adj` on every cell (16/17/23),
# 3 seeds with the spread quoted (41/42/47/51/56), validation token accuracy
# read BEFORE rho_50 (4+6), and the scorer registered before the first cell
# lands -- this array is submitted `--dependency=afterok` on the registration.
#
# N=48 ONLY. Finding 55: the N=48 minimum does not transfer to N=96, so every
# claim from this stage is an N=48 claim and is labelled as one.
# =============================================================================

#SBATCH --job-name=sud2N48CycledU
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%A_%a.out

set -euo pipefail

# n_blocks:seed -- two k, three seeds each. d_model, nhead, ff, warmup and the
# 16 APPLICATIONS are fixed at stage S's d320 cell for every task; k is the one
# thing that moves.
SPECS=(
  "2:0"  "2:1"  "2:2"
  "4:0"  "4:1"  "4:2"
)

TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
IFS=: read -r KBLOCKS SEED <<< "${SPECS[$TASK]}"
KBLOCKS=$((10#$KBLOCKS)); SEED=$((10#$SEED))

DMODEL=320
LAYERS=16
FF=1024
NHEAD=4
WARMUP=0
ARCH=tied_no_mlp

LL=$(printf '%02d' "$LAYERS")
HH=$(printf '%02d' "$NHEAD")
TT=$(printf '%02d' "$TASK")
DD=$(printf '%04d' "$DMODEL")
FFS=$(printf '%05d' "$FF")
KK=$(printf '%02d' "$KBLOCKS")

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
assert "tied_no_mlp" in ARCH_MODES, ("the tied mode is not in this checkout -- "
                                     "the array outran the registration commit", ARCH_MODES)
assert d % nhead == 0, ("d_model must divide by nhead", d, nhead)
assert d == 320, ("stage U is stage S's d320 cell with ONE knob moved", d)
assert k in (2, 4), ("stage U is the two registered k only", k)
assert L % k == 0, ("k must divide the applications", k, L)
# The cell this array describes, built here and counted here, before 8+ hours of
# GPU time: the registration's arithmetic must be true of THIS checkout.
m = SATReasoningTransformer(vocab_size=58, d_model=d, num_layers=L, nhead=nhead,
                            max_len=512, dim_feedforward=1024, norm_first=True,
                            arch_mode="tied_no_mlp", n_blocks=k)
want = (630 * d + 58) + (4 * d * d + 8 * d) * k
got = m.trainable_parameter_count()
assert got == want, ("trainable count is not the registered one", got, want)
assert len(m.encoder.layers) == k and m.n_iters == L, (len(m.encoder.layers), m.n_iters)
assert got != (630 * d + 58) + (4 * d * d + 8 * d), (
    "the count equals the k=1 control's; the extra blocks were not built")
print("k=%d  %d distinct block(s), %d applications each, %s trainable"
      % (k, k, L // k, format(got, ",")))
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n48_cycled_U_2026-11-11"
cell="${OUT}/t${TT}_d${DD}_ff${FFS}_h${HH}_${ARCH}_L${LL}_k${KK}_s${SEED}"
mkdir -p "$cell"

echo "task=${TASK}  k=${KBLOCKS} applications=${LAYERS} d_model=${DMODEL} ff=${FF} nhead=${NHEAD} arch=${ARCH} seed=${SEED}"
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

TAG="d${DD}_h${HH}_L${LL}_ff${FFS}_lr1e-03_pre_shuf_fall_noff_tied_k${KK}"
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
assert c["num_layers"] == want_L and c["nhead"] == want_h
assert c["d_model"] == want_d, (c["d_model"], want_d)
assert c["d_model"] % c["nhead"] == 0, ("d_model must divide by nhead",
                                        c["d_model"], c["nhead"])
assert c["dim_feedforward"] == want_ff, (c["dim_feedforward"], want_ff)
assert int(c.get("warmup_epochs", 0) or 0) == want_w, (c.get("warmup_epochs"), want_w)
assert c["max_len_margin"] == 2 * c["n_nodes"] + 2, c["max_len_margin"]
assert c["n_nodes"] == 48, ("stage U is N=48 only (finding 55)", c["n_nodes"])
assert int(c["seed"]) == want_seed, ("wrong seed; the triple would not pool",
                                     c["seed"], want_seed)
assert want_seed in (0, 1, 2), ("stage U reuses stage S's seeds", want_seed)
# k is what this stage varies, and it is the one field that did not exist in any
# checkpoint before 2026-11-11. If it is absent or wrong the cell pools into
# stage S's rung and the whole comparison disappears silently (rung_key only
# appends the "kK" tail when k != 1).
got_k = int(c.get("n_blocks", 1) or 1)
assert got_k == want_k, ("n_blocks did not round-trip into the checkpoint; this "
                         "cell would pool into stage S's k=1 rung", got_k, want_k)
assert want_L % got_k == 0, ("k must divide the applications", got_k, want_L)
d, L, k = c["d_model"], c["num_layers"], got_k
# A cycled cell pays the per-block term once PER BLOCK, not per application.
BLOCK = 4 * d * d + 8 * d
pred_k = (630 * d + 58) + BLOCK * k
pred_shared = (630 * d + 58) + BLOCK          # the k=1 control
pred_untied = (630 * d + 58) + BLOCK * L
got = c.get("n_trainable_params", -1)
print("arch_mode=%s k=%d applications=%d (%d each) nhead=%d d_model=%d ff=%d "
      "warmup=%d seed=%d  trainable %s params  max_len=%d (margin %d)"
      % (c["arch_mode"], k, L, L // k, c["nhead"], d, c["dim_feedforward"],
         int(c.get("warmup_epochs", 0) or 0), int(c["seed"]), format(got, ","),
         c["max_len"], c["max_len_margin"]))
assert got == pred_k, ("this is not the registered k cell: the count matches "
                       "neither the prediction nor anything else",
                       got, pred_k, pred_shared, pred_untied)
assert got != pred_shared, ("the count equals the k=1 control's -- the extra "
                            "blocks were not built and the ladder is void")
assert got != pred_untied, "cycled and untied counts coincide; the comparison is void"
RUNGS = {1025978: "U1  tied_no_mlp T=16 d320 k=2",
         1850298: "U2  tied_no_mlp T=16 d320 k=4"}
assert got in RUNGS, ("not a registered stage U rung", got, sorted(RUNGS))
FLOOR = 879418      # H1, no_mlp L=12 d128, CENSORED at 1/3 seeds
BASE = 613818       # S1, the shared-block d320 cell this ladder starts from
print("%s -- %.2fx the parameters of stage S's k=1 cell (%s vs %s) at the SAME "
      "width and the SAME %d applications" % (RUNGS[got], got / BASE,
                                              format(got, ","), format(BASE, ","), L))
assert got > FLOOR, ("guard 6: every stage U rung is above H1's censoring floor, "
                     "so a censor here is about the architecture", got, FLOOR)
print("ABOVE H1's censoring floor %s: a censor HERE is about the ARCHITECTURE "
      "and not about size." % format(FLOOR, ","))
print("the object: G = F_%d o ... o F_0 applied %d times -- still ONE map "
      "iterated, which is the whole reason k is an allowed knob."
      % (k - 1, L // k))
PY

echo
echo "=== the transition curve for this cell (U0) ==="
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.33 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== W: plateau escape (U1, read before any level) ==="
python3 - "${CKPT}/history.json" <<'PY'
import json, sys
PLATEAU = 1.64          # finding 44, an N=48 number
S1_NLL = (1.5851, 1.5915, 1.5864)       # stage S's k=1 d320 cell, the control
S1_ACC = 0.4830
h = json.load(open(sys.argv[1]))
best = min(h, key=lambda r: r["nll"])
final_train = h[-1]["train_nll"]
tail = [r["train_nll"] for r in h[-5:]]
slope = (tail[-1] - tail[0]) / max(len(tail) - 1, 1)
first = next((r["epoch"] for r in h if r["train_nll"] < PLATEAU), None)
print("epochs %d  final train_nll %.4f  final val_nll %.4f  best val_nll %.4f at epoch %d"
      % (len(h), final_train, h[-1]["nll"], best["nll"], best["epoch"]))
# history.json's key is "accuracy"; the TRAINING LOG prints it as val_accuracy=.
print("final val TOKEN ACCURACY %.4f  (U2 reads this BEFORE rho_50; it never "
      "censors)  %+.4f against the k=1 control's %.4f"
      % (h[-1]["accuracy"], h[-1]["accuracy"] - S1_ACC, S1_ACC))
print("tail slope over the last 5 epochs: %+.5f nll/epoch" % slope)
print("stage S's k=1 cell finished at %s -- this cell is %+.4f against their median"
      % ("/".join("%.4f" % v for v in S1_NLL), final_train - sorted(S1_NLL)[1]))
if first is None:
    print("PLATEAU NOT ESCAPED: train_nll never went below %.2f (min %.4f). The "
          "k=1 control escaped 3/3, so a cell that does NOT escape here is a "
          "fact about this k, not a missing number -- it still contributes its "
          "level to U1's trend." % (PLATEAU, min(r["train_nll"] for r in h)))
else:
    print("plateau escaped at epoch %d (train_nll < %.2f). U1 asks whether the "
          "FLOOR falls with k, so the level above matters more than the escape: "
          "the control already escaped." % (first, PLATEAU))
print("steps bound: %d epochs completed of the registered 50" % len(h))
assert len(h) == 50, ("the run did not reach the registered 50 epochs; a "
                      "truncated cell does not compare with its matched rung",
                      len(h))
PY

echo
echo "=== done ==="
date +'%F %T'
echo "score with: python3 -u -m scripts.stageu_score"
