#!/bin/bash
# =============================================================================
# STAGE S5 -- the registered schedule arm, PREPARED AND NOT SUBMITTED.
#
# DO NOT SUBMIT YET, for two reasons:
#   1. ~1.6 GPU-days. Stage S's own registration says the screen is hshunt's
#      call on the GPUs, and this script exists so that the call is one
#      command rather than an evening of setup.
#   2. The scorer for these arms is NOT WRITTEN. The standing discipline is
#      scorer registered first, array submitted `--dependency=afterok` on it
#      (16/17/23). Write it, then submit; the readings below are what it has
#      to implement, fixed in advance here so they cannot be chosen later.
#
# WHY THIS EXISTS: STAGE S's DISCRIMINATING ARM DOES NOT TRAIN AT WARMUP 0
#
# Array 51853969, read at epoch 12-17 of 50 on 11-10:
#
#     d320 s0/s1/s2   train_nll 1.5947 / 1.6087 / 1.6474    healthy
#     d448 s0         train_nll 1.6510                      healthy
#     d448 s1         train_nll 3.9655  val_accuracy 0.0318  DIVERGED
#     d448 s2         train_nll 3.9389  val_accuracy 0.0328  DIVERGED
#
# Both failures were already at ~3.89 by the END OF EPOCH 0 and never left:
# train NLL near 3.9 with token accuracy near 0.03 is the uniform predictor
# over the whole vocabulary. d448 is the ONLY arm above H1's 879,418 censoring
# floor, so it is the only arm that can separate "tying costs the deduction"
# from "270k is too few" (S3). With 2 of its 3 seeds dead, stage S cannot
# answer its own question: a median over {trained, dead, dead} censors for an
# optimization reason, which is stage P's confound a second time.
#
# DIAGNOSTIC 51860479 ALREADY SETTLED THE MECHANISM (8 epochs, one knob,
# --warmup-epochs 0 -> 5, d448 SEED 1 -- one of the two that died):
#
#     warmup 0   epoch 3  train_nll 3.9563  val_accuracy 0.0313
#     warmup 5   epoch 7  train_nll 1.6228  val_accuracy 0.4804
#
# It crossed 2.0 at epoch 2 and kept falling through epochs 5-7, which run at
# the FULL 1e-3 -- so the ramp does not merely postpone the blow-up. Same
# width, same seed, same everything else.
#
# WHAT CHANGES HERE: EXACTLY ONE KNOB, --warmup-epochs 0 -> 5.
#
# WHY BOTH WIDTHS AND NOT JUST d448. A schedule change at one width only would
# leave d448 differing from P1 in two things at once, which is the discipline
# this programme is built on. d192 at warmup 5 is the CONTROL: P1's trajectory
# showed no instability anywhere (2.3154 -> 1.8909 -> ... -> 1.6662, monotone,
# decelerating, best val at epoch 44), so a ramp should land it in the same
# place. If it does, the ramp is inert where there was nothing to stabilize and
# the d448 comparison is clean; if it does not, the schedule is not a free
# variable and NOTHING here compares with stage P.
#
# WHY d320 IS DROPPED: it is below H1's floor, S3 already says nothing is
# called on it alone, and its only role was a trend point that stage S's
# healthy seeds have now supplied. Dropping it halves the cost.
#
# READINGS, FIXED BEFORE THE RUN
#   S5-1 (read FIRST, before any level). d192 w5 finishes within 0.005 of
#        P1's median final train NLL 1.6681. If not, the ramp is not inert and
#        no claim here compares with stage P -- report that and stop.
#   S5-2 All six seeds leave the uniform predictor (final train NLL < 2.0).
#        A dead seed here means warmup 5 is not enough and the next knob is
#        the lr (3e-4 cosine), NOT a longer ramp.
#   S5-3 If >= 2/3 d448 seeds escape train NLL 1.64, the tied floor is a
#        CAPACITY floor; rho_50 at d448 then answers whether a tied cell above
#        H1's floor learns DD, and that is an architecture claim.
#   S5-4 If d448 trains and does NOT escape 1.64, the floor does not move with
#        width at a schedule that trains: sharing is the obstruction, and that
#        is the result stage P could not deliver. This outcome is as
#        publishable as the other and is not a failed stage.
#   S5-5 Validation token accuracy is read BEFORE rho_50, always (4+6), and
#        every cell must reach the registered 50 epochs (asserted below).
#
# N=48 ONLY (finding 55). Nothing here is pooled with N=96.
# =============================================================================

#SBATCH --job-name=sud2N48TiedWidthS5warm
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%A_%a.out

set -euo pipefail

# d_model:num_layers:seed -- the discriminating width and its control, three
# seeds each. num_layers is how many times the SHARED block is applied.
SPECS=(
  "192:16:0"  "192:16:1"  "192:16:2"
  "448:16:0"  "448:16:1"  "448:16:2"
)

TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
IFS=: read -r DMODEL LAYERS SEED <<< "${SPECS[$TASK]}"
DMODEL=$((10#$DMODEL)); LAYERS=$((10#$LAYERS)); SEED=$((10#$SEED))

FF=1024
NHEAD=4
WARMUP=5          # THE ONE KNOB. Stage S and stage P both run these cells at 0.
ARCH=tied_no_mlp

LL=$(printf '%02d' "$LAYERS")
HH=$(printf '%02d' "$NHEAD")
TT=$(printf '%02d' "$TASK")
DD=$(printf '%04d' "$DMODEL")
FFS=$(printf '%05d' "$FF")
ATAG="_noff_tied"

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}"

echo "=== smoke check ==="
python3 - "$DMODEL" "$NHEAD" "$WARMUP" <<'PY'
import sys
import torch, numpy, numba
import spin_gnn.sudoku_hparam_sweep  # noqa: F401
import scripts.rho_transition_eval   # noqa: F401
from spin_gnn.sat_transformer2 import ARCH_MODES
from spin_gnn.sudoku_hparam_sweep import ARCH_TAG
d, nhead, warmup = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
print("torch", torch.__version__, "numpy", numpy.__version__, "numba", numba.__version__)
assert "tied_no_mlp" in ARCH_MODES, ARCH_MODES
assert ARCH_TAG["tied_no_mlp"] == "_noff_tied", ARCH_TAG
assert d % nhead == 0, ("d_model must divide by nhead", d, nhead)
assert d in (192, 448), ("S5 is the discriminating width and its control only", d)
assert warmup == 5, ("the one knob is warmup 5; at 0 this duplicates stage S", warmup)
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n48_tiedwidth_S5warm_2026-11-11"
cell="${OUT}/t${TT}_d${DD}_ff${FFS}_h${HH}_${ARCH}_L${LL}_s${SEED}_w$(printf '%02d' "$WARMUP")"
mkdir -p "$cell"

echo "task=${TASK}  layers=${LAYERS} d_model=${DMODEL} warmup=${WARMUP} seed=${SEED}"
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

TAG="d${DD}_h${HH}_L${LL}_ff${FFS}_lr1e-03_pre_shuf_fall${ATAG}_w$(printf '%02d' "$WARMUP")"
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
assert c["dim_feedforward"] == want_ff, (c["dim_feedforward"], want_ff)
assert int(c.get("warmup_epochs", 0) or 0) == want_w, (
    "the one knob did not reach the run", c.get("warmup_epochs"), want_w)
assert c["n_nodes"] == 48, ("S5 is N=48 only (finding 55)", c["n_nodes"])
assert int(c["seed"]) == want_seed, (c["seed"], want_seed)
d, L = c["d_model"], c["num_layers"]
BLOCK = 4 * d * d + 8 * d
pred_tied = (630 * d + 58) + BLOCK
got = c.get("n_trainable_params", -1)
print("arch_mode=%s T=%d d_model=%d warmup=%d seed=%d  trainable %s params"
      % (c["arch_mode"], L, d, int(c.get("warmup_epochs", 0) or 0), int(c["seed"]),
         format(got, ",")))
assert got == pred_tied, ("not a tied cell", got, pred_tied)
RUNGS = {270010: "P1 control  tied_no_mlp T=16 d192 (below H1's floor)",
         1088698: "S2 discriminating  tied_no_mlp T=16 d448 (ABOVE H1's floor)"}
assert got in RUNGS, ("not a registered S5 rung", got, sorted(RUNGS))
print(RUNGS[got])
PY

echo
echo "=== the transition curve for this cell ==="
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.33 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== W: did the ramp do what the diagnostic said it would (S5-1, S5-2) ==="
python3 - "${CKPT}/history.json" "$DMODEL" <<'PY'
import json, sys
PLATEAU = 1.64          # finding 44, an N=48 number
P1_MEDIAN = 1.6681      # stage P's tied d192 median final train NLL, warmup 0
DEAD = 2.0              # above this the cell never left the uniform predictor
h = json.load(open(sys.argv[1]))
d = int(sys.argv[2])
final_train = h[-1]["train_nll"]
print("epochs %d  final train_nll %.4f  final val_nll %.4f  final val TOKEN "
      "ACCURACY %.4f" % (len(h), final_train, h[-1]["nll"], h[-1]["accuracy"]))
if final_train >= DEAD:
    print("S5-2 FAILS ON THIS SEED: still at the uniform predictor at warmup 5. "
          "The next knob is the lr (3e-4 cosine), not a longer ramp.")
elif d == 192:
    delta = final_train - P1_MEDIAN
    print("S5-1 control: %+.4f against stage P's warmup-0 median %.4f" % (delta, P1_MEDIAN))
    print("the ramp is INERT here (|delta| <= 0.005), so the d448 arm compares "
          "with stage P" if abs(delta) <= 0.005 else
          "THE RAMP IS NOT INERT: the schedule is not a free variable and no "
          "claim here compares with stage P. Report this and stop.")
else:
    print("S5-3/S5-4: escaped %.2f" % PLATEAU if final_train < PLATEAU else
          "S5-3/S5-4: did NOT escape %.2f -- the floor did not move with width at "
          "a schedule that trains, i.e. sharing is the obstruction" % PLATEAU)
assert len(h) == 50, ("the run did not reach the registered 50 epochs", len(h))
PY

echo
echo "=== done ==="
date +'%F %T'
