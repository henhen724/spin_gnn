#!/bin/bash
# =============================================================================
# STAGE S -- buy the capacity back at FIXED sharing. Is the tied plateau a
# CAPACITY floor, or an obstruction that tying itself creates?
#
# WHAT STAGE P RETURNED (array 51815311, scorer 51815318, both COMPLETED 11-10)
#
#     P1  tied_no_mlp T=16 d192   270,010   rho_50 CENSORED (1/3 crossing)
#         plateau escapes 0/3     train NLL 1.6646 / 1.6721 / 1.6681
#         val token acc median 0.4567   vs the untied twin J2's 0.4940
#
# P2 was FALSIFIED -- 0 of 3 escaped train NLL 1.64 -- and by its own
# registration P2 falsified OUTRANKS P0. So stage P returned no architecture
# result about tying at all: an optimization-limited cell's rho_50 is a fact
# about the recipe. P1's confound stands exactly as named in advance, because
# 270,010 is BELOW H1's 879,418 censoring floor and "tying costs the deduction"
# and "270k is too few" both predict a censor there.
#
# WHY WIDTH, THOUGH P2's REGISTERED NEXT MOVE WAS "warmup or LR"
#
# The trajectory does not support the mechanism a warmup ramp fixes. Seed 0 runs
# 2.3154 -> 1.8909 -> 1.8252 -> 1.8013 -> 1.7882 -> ... -> 1.6662, monotone and
# decelerating, no spike anywhere, best val NLL at epoch 44 of 50, tail slope
# -0.00041 nll/epoch. All three seeds have that same shape and finish within
# 0.0075 of each other. That is a curve ASYMPTOTING to a floor near 1.66, not
# one destabilized at init, and warmup and a smaller LR both treat early
# instability. There is none here to treat.
#
# The knob that discriminates is the one P1 registered, and it tests P2's
# diagnosis at the same time: if the floor near 1.66 is CAPACITY, train NLL
# falls with width and a wide enough tied cell escapes 1.64. If sharing is the
# obstruction, the floor sits in the same place at every width, because no width
# changes the fact that one block must serve 16 roles. One knob, both questions.
# The schedule arm is registered as the follow-up (S5) if the floor does not
# move, and is NOT abandoned -- it is sequenced behind the test that can tell
# the two stories apart.
#
# WHAT CHANGES: EXACTLY ONE KNOB, d_model
#
#     P1  tied_no_mlp T=16 d192     270,010   below H1's floor   (stage P, disk)
#     S1  tied_no_mlp T=16 d320     613,818   below H1's floor   this array
#     S2  tied_no_mlp T=16 d448   1,088,698   ABOVE H1's floor   this array
#
# Everything else is P1's: tied_no_mlp, T=16 applications of ONE shared block,
# nhead 4, ff 1024 (the feedforward writes identically zero), warmup 0, lr 1e-3
# constant, 240k x 50ep, `--shuffle-adj --fall`, the 14-point grid to 0.33 at
# 400 per bin, seeds 0/1/2. Stage P's three d192 seeds are the third point of
# the width trend and are RE-READ FROM DISK by the scorer, never retyped.
#
# WHICH ARM CARRIES WHICH QUESTION, SAID IN ADVANCE (S3). Only d448 is above
# H1's floor, so only a censor THERE is about tying rather than size. d320 is a
# TREND point for S1 and nothing is called on it alone. Stage O was called on
# one arm after its other arm was cancelled; this stage states up front that it
# will not be.
#
# VERIFIED BEFORE SUBMISSION (scripts/_tiedwidth_dry_check.py): d192 still counts
# 270,010 so the trend's baseline has not moved; d320 and d448 count 613,818 and
# 1,088,698 exactly and match (630d+58)+(4d^2+8d); both widths divide by nhead 4;
# at each width encoder.layers has ONE entry while n_iters is 16; the tied
# forward differs from one application of the same weights; the feedforward
# writes zero; and H1's floor straddles the two widths, which is the one fact the
# stage's discriminating power rests on.
#
# WALL TIME. Stage P's d192 cell took 5:40. At T=393 tokens, d448 is about 3.4x
# the per-step FLOPs of d192 (T^2 d grows 2.3x, T d^2 grows 5.4x), so 20:00:00 is
# a ceiling with margin rather than an estimate; sganguli's limit is 7 days. The
# registered 50 epochs is asserted in the read-back below, so a cell cut short by
# the limit fails loudly instead of being compared against a full one.
#
# NON-NEGOTIABLES CARRIED FORWARD: `--shuffle-adj` on every cell (16/17/23),
# 3 seeds with the spread quoted raw AND conditioned on the plateau criterion
# (41/42/47/51/56), validation token accuracy read BEFORE rho_50 (4+6), and the
# scorer registered before the first cell lands -- this array is submitted
# `--dependency=afterok` on the registration job.
#
# N=48 ONLY. Finding 55: the N=48 minimum does not transfer to N=96, so every
# claim from this stage is an N=48 claim and is labelled as one.
# =============================================================================

#SBATCH --job-name=sud2N48TiedWidthS
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%A_%a.out

set -euo pipefail

# d_model:num_layers:seed -- two widths, three seeds each. num_layers is the
# number of TIMES THE SHARED BLOCK IS APPLIED, which is what joins these cells to
# the depth ladder; it costs no parameters.
SPECS=(
  "320:16:0"  "320:16:1"  "320:16:2"
  "448:16:0"  "448:16:1"  "448:16:2"
)

TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
IFS=: read -r DMODEL LAYERS SEED <<< "${SPECS[$TASK]}"
DMODEL=$((10#$DMODEL)); LAYERS=$((10#$LAYERS)); SEED=$((10#$SEED))

FF=1024
NHEAD=4
WARMUP=0
ARCH=tied_no_mlp

LL=$(printf '%02d' "$LAYERS")
HH=$(printf '%02d' "$NHEAD")
TT=$(printf '%02d' "$TASK")
DD=$(printf '%04d' "$DMODEL")
FFS=$(printf '%05d' "$FF")

case "$ARCH" in
  full)        ATAG="" ;;
  linear)      ATAG="_lin" ;;
  rf_mlp)      ATAG="_rfm" ;;
  rf_all)      ATAG="_rfa" ;;
  no_mlp)      ATAG="_noff" ;;
  tied_no_mlp) ATAG="_noff_tied" ;;
  *) echo "unknown arch_mode: $ARCH"; exit 1 ;;
esac

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR}"

echo "=== smoke check ==="
python3 - "$DMODEL" "$NHEAD" <<'PY'
import sys
import torch, numpy, numba
import spin_gnn.sudoku_hparam_sweep  # noqa: F401
import scripts.rho_transition_eval   # noqa: F401
from spin_gnn.sat_transformer2 import ARCH_MODES
from spin_gnn.sudoku_hparam_sweep import ARCH_TAG
d, nhead = int(sys.argv[1]), int(sys.argv[2])
print("torch", torch.__version__, "numpy", numpy.__version__, "numba", numba.__version__)
assert "tied_no_mlp" in ARCH_MODES, ("the tied mode is not in this checkout -- "
                                     "the array outran the registration commit", ARCH_MODES)
assert ARCH_TAG["tied_no_mlp"] == "_noff_tied", ARCH_TAG
assert d % nhead == 0, ("d_model must divide by nhead", d, nhead)
assert d in (320, 448), ("stage S is the two registered widths only", d)
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n48_tiedwidth_S_2026-11-10"
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
assert c["n_nodes"] == 48, ("stage S is N=48 only (finding 55)", c["n_nodes"])
assert int(c["seed"]) == want_seed, ("wrong seed; the triple would not pool",
                                     c["seed"], want_seed)
assert want_seed in (0, 1, 2), ("stage S reuses stage P's seeds", want_seed)
d, L = c["d_model"], c["num_layers"]
# A TIED cell pays the per-block term ONCE however many times the block is
# applied. That identity is the whole stage, so it is asserted, not described.
BLOCK = 4 * d * d + 8 * d
pred_tied = (630 * d + 58) + BLOCK
pred_untied = (630 * d + 58) + BLOCK * L
got = c.get("n_trainable_params", -1)
print("arch_mode=%s T=%d nhead=%d d_model=%d ff=%d warmup=%d seed=%d  trainable %s "
      "params  max_len=%d (margin %d)"
      % (c["arch_mode"], L, c["nhead"], d, c["dim_feedforward"],
         int(c.get("warmup_epochs", 0) or 0), int(c["seed"]), format(got, ","),
         c["max_len"], c["max_len_margin"]))
assert got == pred_tied, ("this is not a TIED cell: the count matches neither the "
                          "registered tied prediction nor anything else",
                          got, pred_tied, pred_untied)
assert got != pred_untied, "tied and untied counts coincide; the comparison is void"
RUNGS = {613818: "S1  tied_no_mlp T=16 d320", 1088698: "S2  tied_no_mlp T=16 d448"}
assert got in RUNGS, ("not a registered stage S rung", got, sorted(RUNGS))
FLOOR = 879418      # H1, no_mlp L=12 d128, CENSORED at 1/3 seeds
BASE = 270010       # P1, the tied d192 cell this trend starts from
print("%s -- %.2fx the parameters of stage P's tied d192 (%s vs %s) at the SAME "
      "sharing and the SAME depth %d" % (RUNGS[got], got / BASE, format(got, ","),
                                         format(BASE, ","), L))
if got > FLOOR:
    print("ABOVE H1's censoring floor %s: a censor HERE is about TYING and not "
          "about size. This is stage S's discriminating cell (S0, S3)."
          % format(FLOOR, ","))
else:
    print("BELOW H1's censoring floor %s: a censor HERE cannot separate tying "
          "from size. This cell is a TREND point for S1 and nothing is called "
          "on it alone (S3)." % format(FLOOR, ","))
PY

echo
echo "=== the transition curve for this cell (S0, S3) ==="
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.33 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== W: plateau escape (S1, read before any level) ==="
python3 - "${CKPT}/history.json" <<'PY'
import json, sys
PLATEAU = 1.64          # finding 44, an N=48 number
P1_NLL = (1.6646, 1.6721, 1.6681)       # stage P's tied d192, the trend's start
h = json.load(open(sys.argv[1]))
best = min(h, key=lambda r: r["nll"])
final_train = h[-1]["train_nll"]
tail = [r["train_nll"] for r in h[-5:]]
slope = (tail[-1] - tail[0]) / max(len(tail) - 1, 1)
first = next((r["epoch"] for r in h if r["train_nll"] < PLATEAU), None)
print("epochs %d  final train_nll %.4f  final val_nll %.4f  best val_nll %.4f at epoch %d"
      % (len(h), final_train, h[-1]["nll"], best["nll"], best["epoch"]))
# history.json's key is "accuracy"; the TRAINING LOG prints it as val_accuracy=.
print("final val TOKEN ACCURACY %.4f  (S2 reads this BEFORE rho_50; it never censors)"
      % h[-1]["accuracy"])
print("tail slope over the last 5 epochs: %+.5f nll/epoch" % slope)
print("stage P's tied d192 finished at %s -- this cell is %+.4f against their median"
      % ("/".join("%.4f" % v for v in P1_NLL),
         final_train - sorted(P1_NLL)[1]))
if first is None:
    print("PLATEAU NOT ESCAPED: train_nll never went below %.2f (min %.4f). S1 asks "
          "whether the floor MOVES with width, so a cell that did not escape still "
          "contributes its level to the trend -- it is not a failed cell."
          % (PLATEAU, min(r["train_nll"] for r in h)))
else:
    print("plateau escaped at epoch %d (train_nll < %.2f). If this is the d448 cell, "
          "S1 reads CAPACITY FLOOR and S0 becomes readable as an architecture claim."
          % (first, PLATEAU))
print("steps bound: %d epochs completed of the registered 50" % len(h))
assert len(h) == 50, ("the run did not reach the registered 50 epochs; a "
                      "truncated cell does not compare with its matched rung",
                      len(h))
PY

echo
echo "=== done ==="
date +'%F %T'
echo "score with: python3 -u -m scripts.stages_score"
