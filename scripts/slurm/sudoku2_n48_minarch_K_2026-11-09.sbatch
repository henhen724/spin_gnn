#!/bin/bash
# =====================================================================
# STAGE K (2026-11-09) -- DOES DEPTH BUY BACK WIDTH BELOW THE MINIMUM?
#
# THE QUESTION. Stage J moved the standing minimum to attention-only
# L=16 d192, 2,504,890 trainable at rho_50 0.580, and STATUS currently
# reads the frontier as "a FLOOR near 2M parameters, and everything
# above it is flat". That floor has never been tested at L=16. It was
# read off the L=12 width ladder alone:
#
#     no_mlp L=12  d128   CENSORED (1/3 cross)       879,418
#     no_mlp L=12  d160   0.773                    1,345,018
#     no_mlp L=12  d192   0.754                    1,908,922
#     no_mlp L=12  d224   0.579                    2,571,130   FREE
#     no_mlp L=12  d256   0.605                    3,331,642   FREE
#     no_mlp L=16  d192   0.580                    2,504,890   FREE  <- standing minimum
#
# The load-bearing number is the LAST ROW AGAINST THE THIRD. At d192,
# holding width fixed and going L=12 -> L=16 moved rho_50 0.754 -> 0.580,
# a gain of 0.174, and carried the rung from "not free" to "free". Every
# rung below 2M in this table is an L=12 rung. Nobody has asked whether
# the same depth step carries d160 across the 0.740 line too.
#
# Stage K is that question and nothing else: the SAME width ladder, one
# depth up. Two rungs, three seeds each.
#
#     KA   no_mlp L=16 d160   predicted 1,759,738 trainable  (1.42x below the minimum)
#     KB   no_mlp L=16 d128   predicted 1,145,658 trainable  (2.19x below the minimum)
#
# The rungs are KA and KB; the registered hypotheses below are K0-K4.
# They are deliberately lettered apart so that no verdict can be read
# as the name of a rung.
#
# ONE KNOB FROM THE STANDING MINIMUM. KA and KB differ from J2 in
# d_model alone -- same arch_mode, same depth, same nhead, same ff, same
# ensemble, same 240000 x 50ep budget, same 14 x 400 rho grid. Nothing
# else in this file differs from the stage J sbatch it was copied from.
#
# -------------------------- PRE-REGISTERED ---------------------------
#
# K0 -- THE HEADLINE: does the d192 depth gain TRANSPORT to d160?
#   Transporting the one measured depth step onto d160 gives
#   0.773 - 0.174 = 0.599. REGISTERED: KA's median rho_50 lands in
#   [0.55, 0.70], and the binary call -- which is the robust one and the
#   one the standing-minimum rule actually uses -- is that KA IS FREE,
#   median rho_50 <= 0.740 on 3 of 3 seeds crossing.
#   FALSIFIER, NAMED IN BOTH DIRECTIONS BEFORE ANY CELL LANDS:
#     HIGH  rho_50 > 0.740, or any seed censoring: the depth gain does
#           NOT transport across width, (L, d) is not separable on this
#           grid, and the ~2M floor in STATUS stands as written. This is
#           a real outcome, not a formality -- d160 at L=12 is 0.773,
#           which is 0.033 the WRONG side of the line, and the whole
#           prediction rests on a single measured depth step at ONE width.
#     LOW   rho_50 < 0.55: the gain is LARGER at narrow width than at
#           d192. That falsifies the transport model in our favour, so
#           it earns no victory lap -- it would mean the ladder is not
#           separable in (L, d) and stage J's convexity reading needs
#           rewriting before any of it is quoted.
#
# K1 -- THE BREAKING POINT, which is the result if it fires. d128 at
#   L=12 CENSORED with 1 of 3 seeds crossing and a train-NLL that never
#   left finding 44's 1.64 plateau -- a TRAINABILITY failure, not a
#   capacity one. REGISTERED: KB CENSORS AGAIN (<= 2 of 3 seeds cross),
#   because a per-layer width bottleneck is not something more layers
#   can pay for.
#   FALSIFIER: 3 of 3 crossing at d128 means the d128 wall at L=12 was
#   an artifact of the DEPTH budget rather than of width, and the floor
#   moves to ~1.15M. That would be the largest single result in this
#   programme since stage E, so it gets held to the same 3/3 rule as
#   everything else and is re-run before it is believed.
#
# K2 -- THE STANDING-MINIMUM RULE (verdict, not a rung), unchanged from H3 = I3 = J3
#   (finding 51): a rung replaces the standing minimum ONLY IF 3 of 3
#   seeds cross AND median rho_50 <= 0.740. TWO OF THREE IS NEVER
#   ENOUGH. If both K rungs qualify, the smaller one wins on the
#   pre-named tie-break (parameter count), as in J3.
#
# K3 -- SEPARABILITY, so that "smaller" is never silently read as
#   "better". s* = 0.1484 (rho_50), frozen, NOT recomputed. Any rho_50
#   move between the new minimum and J2's 0.580 that is under s* is
#   reported as "no separable cost", never as an improvement, and the
#   claim is "N times smaller at no separable cost".
#
# K4 -- OUT-OF-SAMPLE SUFFICIENCY, free, and reported ahead of
#   everything if it fails: BOTH new rungs land on the depth ladder's
#   accuracy-to-solve curve within the 0.0423 LOO bar. A rung falling
#   OFF that curve is where a width-specific failure mode would show up,
#   and it would be the headline over K0. Refit over the NINE dirs
#   (SUFFICIENCY_DIRS in scripts/stagej_score.py), which is 22/22 over
#   47 cells today.
#
# ------------------------------ TRAPS --------------------------------
#  * --shuffle-adj on every cell (findings 16/17/23). The smaller the
#    model the more tempting the decoding shortcut, and these are the
#    smallest cells ever run on this task.
#  * 3 seeds per rung, spread quoted (findings 41/42/47, finding 51).
#  * These rungs carry a NON-DEFAULT d_model, so they key as 6-tuples
#    and cannot enter the pooled bar by accident. SCAN_K stays OUT of
#    SCAN_DIRS, as E/G/H/I/J all do.
#  * N=48 numbers only. The N=96 yardstick s*_96 = 0.119 and stage F's
#    rungs are a different problem size and are never pooled with these
#    (N=48 rho grid ends at 0.33, N=96's at 0.32).
#  * The 630*d + 58 fixed term is an N=48 expression. It is correct here
#    and would not be at N=96.
#
# COST: 6 cells. Stage J's L=16 d192 cells ran 4:35-5:37 at this budget;
# these are narrower, so <= that. Budgeted 10:00:00 with margin, %6 so
# all six run as one wave (sganguli had 11 idle GPUs at submission).
# =====================================================================
#SBATCH --job-name=sud2MinArchK
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-5%6
#SBATCH --output=%x_%A_%a.out

set -euo pipefail

# d_model:num_layers:seed -- widths interleaved so that a short array
# still returns seeds of BOTH rungs rather than all of one.
SPECS=(
  "160:16:0"  "128:16:0"  "160:16:1"
  "128:16:1"  "160:16:2"  "128:16:2"
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

OUT="${SCRATCH}/spin_gnn/sudoku_n48_minarch_K_2026-11-09"
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
python3 - "${CKPT}/config.json" "$ARCH" "$LAYERS" "$NHEAD" "$DMODEL" "$FF" "$WARMUP" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
want_arch, want_L, want_h = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
want_d, want_ff, want_w = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
assert c["shuffle_adj"] is True, "LEAK: this cell is not shuffle_adj"
assert c["fall"] is True, "wrong ensemble: fall is not set"
assert c["arch_mode"] == want_arch, (c["arch_mode"], want_arch)
assert c["num_layers"] == want_L and c["nhead"] == want_h
assert c["d_model"] == want_d, (c["d_model"], want_d)
assert c["dim_feedforward"] == want_ff, (c["dim_feedforward"], want_ff)
assert int(c.get("warmup_epochs", 0) or 0) == want_w, (c.get("warmup_epochs"), want_w)
assert c["max_len_margin"] == 2 * c["n_nodes"] + 2, c["max_len_margin"]
assert c["n_nodes"] == 48, ("stage K is N=48 only", c["n_nodes"])
d, L = c["d_model"], c["num_layers"]
# 630*d + 58 is the N=48 fixed term; asserted above that this IS N=48.
pred = (630 * d + 58) + (4 * d * d + 8 * d) * L
got = c.get("n_trainable_params", -1)
print("arch_mode=%s L=%d nhead=%d d_model=%d ff=%d warmup=%d  trainable %s params "
      "(header predicted %s)  max_len=%d (margin %d)"
      % (c["arch_mode"], L, c["nhead"], d, c["dim_feedforward"],
         int(c.get("warmup_epochs", 0) or 0), format(got, ","), format(pred, ","),
         c["max_len"], c["max_len_margin"]))
assert got == pred, ("trainable count is not what the registration predicted", got, pred)
STANDING = 2504890          # J2, no_mlp L=16 d192, the standing minimum at rho_50 0.580
print("vs the standing minimum: %s against %s -> %.2fx SMALLER"
      % (format(got, ","), format(STANDING, ","), STANDING / got))
PY

echo
echo "=== the transition curve for this cell (K0, K1) ==="
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.33 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== W: plateau escape ==="
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
print("overfit gap (final - best val) = %+.4f" % (h[-1]["nll"] - best["nll"]))
print("last-5-epoch train_nll slope = %+.5f/epoch (negative = still descending)" % slope)
print("final val token accuracy = %.4f" % h[-1]["accuracy"])
print("first epoch under the plateau = %s" % (first if first is not None else "never"))
print("PLATEAU_ESCAPE=%s  (final train_nll %.4f vs finding 44's plateau %.2f)"
      % ("yes" if final_train < PLATEAU else "NO", final_train, PLATEAU))
PY

date +'task end  %F %T'
