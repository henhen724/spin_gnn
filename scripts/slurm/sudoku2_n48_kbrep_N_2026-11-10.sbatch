#!/bin/bash
# =============================================================================
# STAGE N -- does KB replicate? `no_mlp L=16 d128` on three FRESH seeds, N=48.
#
# Next step 4 of the 11-09 handoff: the one loose thread stage K left.
#
# WHAT STAGE K MEASURED (finding 54, array 51668331, scored 51699439)
#
#     rung                 rho_50   trainable   cross   escapes   call
#     KA no_mlp L=16 d160   0.757    1,759,738    3/3     3/3      past the free line
#     KB no_mlp L=16 d128   0.766    1,145,658    3/3     2/3      past the free line
#
# Neither took the standing minimum (`no_mlp L=16 d192`, 2,504,890 at 0.580),
# because the replacement rule H3 = I3 = J3 = K2 needs BOTH 3-of-3 crossing and
# a median at or below the free line 0.740, and KB's 0.766 is +0.026 = +0.18 s*
# past it. That is a NARROW miss on a 14-point grid, measured on three seeds.
#
# WHY THIS RUNG AND NOT ANOTHER. Two things about KB are unresolved:
#
#   1. d128 CENSORS at L=12 and then crosses 3 of 3 at L=16. That is the single
#      surprise of stage K (K1 was falsified in the interesting direction), and
#      K1's own registration says it is not quoted outside the scorer until
#      fresh seeds confirm it.
#   2. KB crossed with only 2 of 3 plateau escapes. A rung whose seeds do not
#      all get below train NLL 1.64 is one whose level may be an optimization
#      artifact rather than a capacity one.
#
# WHAT CHANGES: THE SEED TRIPLE, AND NOTHING ELSE. Seeds 3, 4, 5 of the exact
# KB cell -- same arch, same shape, same ff, same nhead, same warmup, same
# 240k x 50ep budget, same `--shuffle-adj --fall` ensemble, same 14-point grid
# ending at 0.33, same 400 instances per bin. Every flag below is stage K's,
# copied rather than retyped, so the two triples pool.
#
# COST, FROM STAGE K's MEASURED WALL CLOCK, NOT FROM AN ESTIMATE. K's three
# d128 cells ran 04:32:16, 04:36:54 and 04:33:04. Three cells, ~14 GPU-h, ~5 h
# wall if all three start together. --time stays at K's 10:00:00 unchanged:
# the bound that worked is not a knob this stage is turning.
#
# -----------------------------------------------------------------------------
# REGISTERED BEFORE THE FIRST CELL LANDS -- scripts/stagen_score.py, committed
# by stagen_register_2026-11-10.sbatch, which this array depends on with
# --dependency=afterok. No cell can start ahead of its own registration.
#
# THE BARS ARE N=48 BARS AND ARE REUSED, NOT REFROZEN. This rung is a replicate
# of an N=48 cell against N=48 neighbours, so it is scored against the yardstick
# those neighbours were scored against: s* = 0.1484 (rho_50), 0.2918
# (mean_solve), free line 0.740, plateau 1.64, LOO bar 0.0423. Stage M's N=96
# bars (0.117 / 0.1586 / 0.0079) are a DIFFERENT SIZE and never appear here --
# different grid floors (0.33 vs 0.32) and different controls. Finding 55 is the
# standing reason the two sizes are never pooled.
#
# N0  REPLICATION OF THE LEVEL. The fresh triple's median rho_50 lands within
#     one s* of KB's 0.766, i.e. in [0.618, 0.914]. Either falsification says
#     something: BELOW means 0.766 was pessimistic and the pooled six-seed
#     median may cross the free line; ABOVE means 0.766 was optimistic and the
#     bottom of the ladder is worse than finding 54 reports.
#
# N1  REPLICATION OF THE CROSSING. All 3 fresh seeds return a MEASURED rho_50
#     (no censor) on a grid whose floor is 0.33. A censor in the fresh triple
#     is this stage's result: it would mean KB's 3-of-3 crossing at L=16 -- the
#     thing that made d128's L=12 censor surprising -- does not replicate, and
#     K1's surprise is a three-seed accident.
#
# N2  THE POOLED CALL, WITH THE POOLING RULE FIXED NOW. Pool stage K's seeds
#     0/1/2 with this stage's 3/4/5 and apply the replacement rule to all six:
#     KB takes the standing minimum only if 6 of 6 cross AND the SIX-SEED median
#     rho_50 is at or below 0.740. Choosing the pooling rule after seeing the
#     fresh numbers is how a narrow miss becomes a hit, so it is chosen here.
#     PREDICTION: KB stays past the free line and the standing minimum is
#     UNCHANGED at 2,504,890 trainable / 0.580. THE REGISTERED UPSIDE, stated
#     before the fact rather than discovered after it: if the six-seed median
#     does come in at or below 0.740, KB at 1,145,658 REPLACES the standing
#     minimum, a 2.19x reduction in trainable parameters.
#
# N3  THE PLATEAU. Pooled escapes below train NLL 1.64 are at least 4 of 6.
#     Fewer would mean d128 at L=16 is optimization-limited rather than
#     capacity-limited, which reinterprets the bottom of the whole ladder --
#     the ~2M floor would be a floor on what this recipe TRAINS, not on what
#     the task needs.
#
# N4  ACCURACY FIRST, AS EVERY STAGE. Validation token accuracy is read before
#     rho_50 because it never censors. All six KB cells sit on
#     scripts/acc_vs_solve.py's N=48 fitted curve within the 0.0423 LOO bar.
#     A cell that falls OFF it is the interesting one. That curve is an N=48
#     object and is applied only because this stage is N=48.
#
# WATCHED, NOT PREDICTED
#   * trainable params must read back EXACTLY 1,145,658 on all three cells --
#     (630*128 + 58) + (4*128^2 + 8*128)*16. A different count means a different
#     cell and the triples do not pool.
#   * shuffle_adj True, fall True, n_nodes 48 on every cell, asserted from the
#     written config.json rather than from these flags.
#   * seeds 3/4/5 are fresh, so they cannot collide with K's 0/1/2 in the
#     pooled cell key -- the pooling is sound mechanically as well as
#     statistically.
#   * the checkpoint is the final epoch, as everywhere in this programme.
# =============================================================================

#SBATCH --job-name=sud2N48KbrepN
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%A_%a.out

set -euo pipefail

# d_model:num_layers:seed -- KB's cell on the three fresh seeds.
SPECS=(
  "128:16:3"  "128:16:4"  "128:16:5"
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

OUT="${SCRATCH}/spin_gnn/sudoku_n48_kbrep_N_2026-11-10"
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
assert c["dim_feedforward"] == want_ff, (c["dim_feedforward"], want_ff)
assert int(c.get("warmup_epochs", 0) or 0) == want_w, (c.get("warmup_epochs"), want_w)
assert c["max_len_margin"] == 2 * c["n_nodes"] + 2, c["max_len_margin"]
assert c["n_nodes"] == 48, ("stage N is N=48 only -- it replicates an N=48 rung", c["n_nodes"])
assert int(c["seed"]) == want_seed, ("wrong seed; the triples would not pool",
                                     c["seed"], want_seed)
assert want_seed in (3, 4, 5), ("stage N is the FRESH triple; 0/1/2 are stage K's",
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
KB_TRAINABLE = 1145658      # stage K's KB, the cell this stage replicates
assert got == KB_TRAINABLE, ("this is not KB's cell; the triples do not pool",
                             got, KB_TRAINABLE)
STANDING = 2504890          # J2, no_mlp L=16 d192, the standing minimum at rho_50 0.580
print("vs the standing minimum: %s against %s -> %.2fx SMALLER"
      % (format(got, ","), format(STANDING, ","), STANDING / got))
print("(that factor is what N2 buys IF the six-seed median comes in free; "
      "stage K's three seeds said 0.766, which is past 0.740.)")
PY

echo
echo "=== the transition curve for this cell (N0, N1) ==="
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.33 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== W: plateau escape (N3) ==="
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
# Reading h[-1]["val_accuracy"] here raises KeyError, and minarch_analysis's
# history()[2] is the val NLL, not this -- both were mistakes in this file.
print("final val TOKEN ACCURACY %.4f  (N4 reads this BEFORE rho_50; it never censors)"
      % h[-1]["accuracy"])
print("tail slope over the last 5 epochs: %+.5f nll/epoch" % slope)
if first is None:
    print("PLATEAU NOT ESCAPED: train_nll never went below %.2f (min %.4f). "
          "KB had 2 of 3 escapes; this is the measurement N3 is about."
          % (PLATEAU, min(r["train_nll"] for r in h)))
else:
    print("plateau escaped at epoch %d (train_nll < %.2f)" % (first, PLATEAU))
print("steps bound: %d epochs completed of the registered 50" % len(h))
assert len(h) == 50, ("the run did not reach the registered 50 epochs; "
                      "a truncated cell does not pool with stage K's", len(h))
PY

echo
echo "=== done ==="
date +'%F %T'
echo "score with: python3 -u -m scripts.stagen_score"
