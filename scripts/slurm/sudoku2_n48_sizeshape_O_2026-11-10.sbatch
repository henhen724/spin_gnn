#!/bin/bash
# =============================================================================
# STAGE O -- is the free/not-free transition set by SIZE or by SHAPE?
#
# Two rungs of MATCHED TRAINABLE COUNT and DIFFERENT SHAPE, placed inside the
# gap that every previous rung has only bracketed, N=48, attention-only.
#
# WHAT THE LADDER LOOKS LIKE WHEN IT IS SORTED BY SIZE INSTEAD OF BY SHAPE
#
#     rung                 trainable   rho_50   cross   escapes   call
#     H1  L=12 d128          879,418   CENSORED  1/3      1/3      censors
#     KB  L=16 d128        1,145,658   0.761     6/6      5/6      past 0.740
#     I1  L=12 d160        1,345,018   0.773     3/3      3/3      past 0.740
#     KA  L=16 d160        1,759,738   0.757     3/3      3/3      past 0.740
#     H2  L=12 d192        1,908,922   0.754     3/3      3/3      past 0.740
#     ------------------ the transition is in here -------------------
#     J2  L=16 d192        2,504,890   0.580     3/3      3/3      FREE (standing)
#     J1  L=12 d224        2,571,130   0.579     3/3      3/3      free
#
# Two DIFFERENT shapes reach ~2.5M and return the same rho_50 to 0.001. Two
# different shapes sit at ~1.8-1.9M and return the same rho_50 to 0.003. Inside
# this family rho_50 reads like a function of TRAINABLE COUNT rather than of how
# the parameters are arranged. That has never been tested: no two cells of
# matched size and different shape exist anywhere in the gap, so the pattern
# could equally be a coincidence of a ladder that only moved one knob at a time.
#
# WHAT CHANGES: TWO RUNGS, ONE COMPARISON
#
#     OA  no_mlp L=16 d176   2,115,930 trainable   the width route into the gap
#     OB  no_mlp L=20 d160   2,174,458 trainable   the depth route into the gap
#
# Matched to 2.8% in trainable count, 4 layers and 16 channels apart in shape,
# both strictly between H2 (past the line) and J2 (free). Every other flag is
# stage K's and stage N's, copied rather than retyped: nhead 4, ff 1024, warmup
# 0, 240k x 50ep, `--shuffle-adj --fall`, the 14-point grid to 0.33, 400 per bin,
# seeds 0/1/2. The tasks interleave the two rungs so that a partial array still
# has seeds of both.
#
# WHY THIS AND NOT A RUNG BELOW d128. Finding 56 (stage N, six seeds) settled
# the resolution question that was blocking the width ladder: conditioned on the
# pre-registered plateau criterion, the five seeds that trained span 0.018 while
# a width step is 0.181, so width steps ARE resolvable at three seeds. But every
# rung below d128 is further into a region already measured flat and past the
# line -- 0.761, 0.773, 0.757, 0.754 across a 2.2x range of size -- and the only
# place the level has ever moved is the gap above. The breaking point is the
# result, and the breaking point is in the gap.
#
# COST, FROM MEASURED WALL CLOCK, NOT FROM AN ESTIMATE. KA (L=16 d160) ran
# 04:32-04:37 per cell. OA is the same depth at 1.21x the per-layer width cost;
# OB is the same width at 1.25x the depth. ~5-6 h per cell, 6 cells at once on
# sganguli, ~6 h wall. --time stays at stage K's 10:00:00: the bound that worked
# is not a knob this stage is turning.
#
# -----------------------------------------------------------------------------
# REGISTERED BEFORE THE FIRST CELL LANDS -- scripts/stageo_score.py, committed by
# stageo_register_2026-11-10.sbatch, which this array depends on with
# --dependency=afterok. No cell can start ahead of its own registration.
#
# THE BARS ARE N=48 BARS, REUSED AND NOT REFROZEN: s* = 0.1484 (rho_50), 0.2918
# (mean_solve), free line 0.740, plateau 1.64, LOO bar 0.0423. Stage M's N=96
# bars (0.117 / 0.1586 / 0.0079) are a DIFFERENT SIZE and appear nowhere here --
# different grid floors (0.33 vs 0.32) and different controls. Finding 55 is the
# standing reason the two sizes are never pooled or differenced.
#
# O0  THE HEADLINE, AND IT IS DISCRIMINATING. OA and OB land on the SAME side of
#     0.740 and agree within one s*. That is what "rho_50 is a function of
#     trainable count" predicts of a 2.8% size match. THE COMPETING HYPOTHESIS
#     PREDICTS THE OPPOSITE: the only shape step ever measured as moving the
#     level is depth at fixed width (H2 -> J2, -0.174), so "depth buys
#     deduction" says OB -- four layers deeper than any cell in the programme --
#     comes in free while OA does not. One run separates them.
#
# O1  THE LEVEL. PREDICTION: both rungs PAST 0.740 and the standing minimum
#     UNCHANGED at 2,504,890 / 0.580; interpolating in log-trainable between the
#     bracketing pairs puts the step near 2.2M and both cells sit just under it.
#     REGISTERED UPSIDE, stated before the fact: a rung that is free under the
#     standing rule (3 of 3 crossing AND median <= 0.740) REPLACES the minimum --
#     OA 1.18x smaller, OB 1.15x -- and a tie goes to the smaller count.
#
# O2  THE CROSSING. All 6 cells return a MEASURED rho_50 on a grid whose floor is
#     0.33. Both rungs are more than twice the only rung that has ever censored
#     (H1, 879,418), so a censor here is this stage's result: it would mean the
#     transition is not monotone in size and the table above is not a ladder.
#
# O3  THE PLATEAU. At least 5 of 6 cells escape train NLL 1.64. Every rung at or
#     above 1.75M escaped 3 of 3; only the 1.15M KB has had a stuck seed (5 of 6,
#     finding 56). Fewer than 5 says these cells are optimization-limited rather
#     than capacity-limited, which outranks O0.
#
# O4  ACCURACY FIRST, AS EVERY STAGE. Validation token accuracy is read before
#     rho_50 because it never censors. All 6 cells sit on acc_vs_solve.py's N=48
#     fitted curve within the 0.0423 LOO bar; a cell OFF it is the headline over
#     O0. That curve is an N=48 object and applies only because this is N=48.
#
# WATCHED, NOT PREDICTED
#   * trainable params must read back EXACTLY 2,115,930 (OA) or 2,174,458 (OB) --
#     (630*d + 58) + (4*d^2 + 8*d)*L, the formula verified against KA and KB. A
#     different count means a different cell and O0 is not the comparison made.
#   * d_model must stay divisible by nhead (176/4 = 44, 160/4 = 40).
#   * shuffle_adj True, fall True, n_nodes 48 on every cell, asserted from the
#     written config.json rather than from these flags.
#   * OA and OB are never pooled with each other: matched SIZE is the design,
#     matched CELL is not. Only seeds of one cell pool.
#   * the checkpoint is the final epoch, as everywhere in this programme.
# =============================================================================

#SBATCH --job-name=sud2N48SizeShapeO
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%A_%a.out

set -euo pipefail

# d_model:num_layers:seed -- the two rungs interleaved, so a partial array
# still holds seeds of both and O0 stays a comparison rather than a report.
SPECS=(
  "176:16:0"  "160:20:0"  "176:16:1"
  "160:20:1"  "176:16:2"  "160:20:2"
)

TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
IFS=: read -r DMODEL LAYERS SEED <<< "${SPECS[$TASK]}"
DMODEL=$((10#$DMODEL)); LAYERS=$((10#$LAYERS)); SEED=$((10#$SEED))

FF=1024
NHEAD=4
WARMUP=0
ARCH=no_mlp

LL=$(printf '%02d' "$LAYERS")
HH=$(printf '%02d' "$NHEAD")
TT=$(printf '%02d' "$TASK")
DD=$(printf '%04d' "$DMODEL")
FFS=$(printf '%05d' "$FF")

case "$ARCH" in
  full)   ATAG="" ;;
  linear) ATAG="_lin" ;;
  rf_mlp) ATAG="_rfm" ;;
  rf_all) ATAG="_rfa" ;;
  no_mlp) ATAG="_noff" ;;
  *) echo "unknown arch_mode: $ARCH"; exit 1 ;;
esac

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}"

echo "=== smoke check ==="
python3 - <<'PY'
import torch, numpy, numba
import spin_gnn.sudoku_hparam_sweep  # noqa: F401
import scripts.rho_transition_eval   # noqa: F401
from spin_gnn.sat_transformer2 import ARCH_MODES
print("torch", torch.__version__, "numpy", numpy.__version__, "numba", numba.__version__)
print("arch modes", ARCH_MODES)
assert "no_mlp" in ARCH_MODES, ARCH_MODES
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n48_sizeshape_O_2026-11-10"
cell="${OUT}/t${TT}_d${DD}_ff${FFS}_h${HH}_${ARCH}_L${LL}_s${SEED}"
mkdir -p "$cell"

echo "task=${TASK}  layers=${LAYERS} d_model=${DMODEL} ff=${FF} nhead=${NHEAD} arch=${ARCH} seed=${SEED}"
echo "cell -> ${cell}"
date +'cell start %F %T'

python3 -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 48 --degree 5 --n-colors 4 --shuffle-adj --fall \
    --d-model "$DMODEL" --nhead "$NHEAD" --dim-feedforward "$FF" \
    --num-layers "$LAYERS" --arch-mode "$ARCH" --seed "$SEED" \
    --warmup-epochs "$WARMUP" \
    --n-train-instances 240000 --epochs 50 \
    --n-val-instances 300 --batch-size 32 --n-per-bin 100 \
    --out-dir "$cell"

date +'cell end   %F %T'

TAG="d${DD}_h${HH}_L${LL}_ff${FFS}_lr1e-03_pre_shuf_fall${ATAG}"
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
python3 - "${CKPT}/config.json" "$ARCH" "$LAYERS" "$NHEAD" "$DMODEL" "$FF" "$WARMUP" "$SEED" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
want_arch, want_L, want_h = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
want_d, want_ff, want_w = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
want_seed = int(sys.argv[8])
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
assert c["n_nodes"] == 48, ("stage O is N=48 only -- the gap it probes is an "
                            "N=48 gap (finding 55)", c["n_nodes"])
assert int(c["seed"]) == want_seed, ("wrong seed; the triple would not pool",
                                     c["seed"], want_seed)
assert want_seed in (0, 1, 2), ("stage O is the first triple of a new rung",
                                want_seed)
d, L = c["d_model"], c["num_layers"]
# 630*d + 58 is the N=48 fixed term; asserted above that this IS N=48.
pred = (630 * d + 58) + (4 * d * d + 8 * d) * L
got = c.get("n_trainable_params", -1)
print("arch_mode=%s L=%d nhead=%d d_model=%d ff=%d warmup=%d seed=%d  trainable %s "
      "params (header predicted %s)  max_len=%d (margin %d)"
      % (c["arch_mode"], L, c["nhead"], d, c["dim_feedforward"],
         int(c.get("warmup_epochs", 0) or 0), int(c["seed"]), format(got, ","),
         format(pred, ","), c["max_len"], c["max_len_margin"]))
assert got == pred, ("trainable count is not what the registration predicted", got, pred)
RUNGS = {2115930: "OA  no_mlp L=16 d176", 2174458: "OB  no_mlp L=20 d160"}
assert got in RUNGS, ("this is neither registered rung; O0's matched-size design "
                      "is the whole comparison", got, sorted(RUNGS))
GAP_LO, GAP_HI = 1908922, 2504890   # H2 (past the line), J2 (free)
assert GAP_LO < got < GAP_HI, ("the cell is not inside the measured gap", got)
STANDING = 2504890                  # J2, no_mlp L=16 d192, the standing minimum at 0.580
print("%s -- inside the gap (%s, %s), %.2fx smaller than the standing minimum %s"
      % (RUNGS[got], format(GAP_LO, ","), format(GAP_HI, ","), STANDING / got,
         format(STANDING, ",")))
OTHER = [k for k in RUNGS if k != got][0]
print("against the other rung of this stage: %s vs %s -> %.1f%% apart in size, "
      "which is what makes O0 a shape comparison"
      % (format(got, ","), format(OTHER, ","), 100 * abs(got - OTHER) / got))
PY

echo
echo "=== the transition curve for this cell (O0, O1, O2) ==="
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.33 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== W: plateau escape (O3) ==="
python3 - "${CKPT}/history.json" <<'PY'
import json, sys
PLATEAU = 1.64          # finding 44, an N=48 number
h = json.load(open(sys.argv[1]))
best = min(h, key=lambda r: r["nll"])
final_train = h[-1]["train_nll"]
tail = [r["train_nll"] for r in h[-5:]]
slope = (tail[-1] - tail[0]) / max(len(tail) - 1, 1)
first = next((r["epoch"] for r in h if r["train_nll"] < PLATEAU), None)
print("epochs %d  final train_nll %.4f  final val_nll %.4f  best val_nll %.4f at epoch %d"
      % (len(h), final_train, h[-1]["nll"], best["nll"], best["epoch"]))
# history.json's key is "accuracy"; the TRAINING LOG prints it as val_accuracy=.
print("final val TOKEN ACCURACY %.4f  (O4 reads this BEFORE rho_50; it never censors)"
      % h[-1]["accuracy"])
print("tail slope over the last 5 epochs: %+.5f nll/epoch" % slope)
if first is None:
    print("PLATEAU NOT ESCAPED: train_nll never went below %.2f (min %.4f). "
          "Every rung at or above 1.75M has escaped 3 of 3; this is what O3 is "
          "about." % (PLATEAU, min(r["train_nll"] for r in h)))
else:
    print("plateau escaped at epoch %d (train_nll < %.2f)" % (first, PLATEAU))
print("steps bound: %d epochs completed of the registered 50" % len(h))
assert len(h) == 50, ("the run did not reach the registered 50 epochs; a "
                      "truncated cell does not compare with its matched rung",
                      len(h))
PY

echo
echo "=== done ==="
date +'%F %T'
echo "score with: python3 -u -m scripts.stageo_score"
