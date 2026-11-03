#!/bin/bash
#SBATCH --job-name=sud2N96Cos
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --array=0-1
#SBATCH --output=%x_%A_%a.out
#
# DOES AN LR SCHEDULE SHRINK THE SEED TERM? (and does it push rho_50 lower?)
#
#   sbatch scripts/slurm/sudoku2_n96_cosine_2026-11-03.sbatch
#
# WHY. 50860571 set out to measure depth (H2) and data (H3) at N=96 and
# instead measured its own noise floor. H4 -- the pre-registered seed control,
# t0 vs t4, the SAME config at seeds 0 and 1 -- came back at 0.115 in rho_50
# against a bar of 0.05, with per-rho |delta solve_rate| peaking at 0.435. The
# sbatch header pre-committed the reading: "IF IT IS LARGER THAN 0.05 HERE,
# this whole grid is one draw and only the ordering survives". It is, so:
#   H2's L=8 -> L=12 gap is +0.012, a tenth of the seed term -- unreadable.
#   H3's falsifier fires nominally (120k beats 240k by 0.069) -- unreadable.
# Averaging the seed away is not an option. At sigma_hat = 0.102 per cell,
# resolving H2's own 0.05 bar needs ~66 seeds per rung, ~1450 GPU-h. The seed
# term has to be made SMALLER, not sampled more.
#
# THE HYPOTHESIS, and it is a mechanism and not a hunch. Across the four
# landed cells val_nll rank-orders rho_50 exactly (Spearman +1.000, n=4):
#   t4 val 1.7283 -> rho_50 0.564    t3 val 1.7332 -> 0.610
#   t0 val 1.7455 -> 0.679           t1 val 1.7495 -> 0.691
# A val_nll range of 0.021 maps to a rho_50 range of 0.128. And every cell is
# STILL DESCENDING when training stops (last-5-epoch val drop negative in all
# four). So the seed is not changing what the network can represent -- it is
# changing how far down each run happened to get, and where the iterate
# happened to be sitting when the budget ran out. This grid trains at
# CONSTANT lr 1e-3 with no decay, which is exactly the setting that makes the
# end-of-training point a lottery.
#
# PREDICTION, written before submission.
#   P1 (the one that matters) rho_50 spread between these two seeds is BELOW
#      0.115, the constant-lr spread measured at this identical config. If it
#      is not, the mechanism hypothesis is wrong, the seed term is intrinsic
#      to the task at N=96, and no depth scan at this N is worth running --
#      say that rather than retrying a third schedule.
#   P2 both cells reach val_nll below 1.7283, the best constant-lr value, since
#      cosine converts the remaining descent into a lower final point.
#   P3 best rho_50 beats 0.564. Weakest of the three and NOT the test: 0.564
#      is itself a lucky draw, so beating it once proves nothing. P1 is what
#      makes it reproducible, and P1 is what this run is for.
#   Falsifier for the whole idea: spread >= 0.115 AND val_nll not improved.
#
# ONE KNOB. Every argument below is byte-identical to task 0/4 of 50860571
# except --lr-schedule cosine. lr_min_factor stays at its 0.01 default, so lr
# runs 1e-3 -> 1e-5 over the 50 epochs; warmup_epochs stays 0, so this is a
# decay test and not a decay+warmup test. Warmup is the obvious second knob
# and it is deliberately NOT turned here -- if cosine alone moves P1 we know
# which knob did it.
#
# SEEDS 0 AND 1, the same two as t0/t4, so the before and after spreads are
# computed from the same pair of draws and not from a different pair.
#
# EVAL. Identical to the scan: scripts.rho_transition_eval, the same 14-point
# rho grid, 400 instances per bin, default --seed-start -- so these curves are
# paired instance-for-instance with all six scan cells and rho_50 means the
# same thing it means there.
#
# OUTPUT. A NEW dated directory, sudoku_n96_cosine_2026-11-03, so nothing
# lands in 50860571's scan dir and the scan reader's cell table stays exactly
# the six cells it pre-registered.
#
# COST. t0 (this config, constant lr) COMPLETED in 11:42:11 including
# generation and the 14-point eval. A schedule changes no FLOPs, so 11-12 h is
# the expectation and --time 16:00:00 is ~37% headroom. sh_part at 10:55 shows
# 1 of sganguli's 16 GPUs idle with 8 queued, so the second task will wait;
# that is fine, nothing downstream depends on both finishing together.

set -euo pipefail

declare -A SPECS=( [0]="12:240000:50:0" [1]="12:240000:50:1" )   # L:n_train:epochs:seed
TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
[ -n "${SPECS[$TASK]+x}" ] || { echo "no spec for task ${TASK}; this script defines 0 and 1"; exit 1; }
IFS=: read -r LAYERS NTRAIN EPOCHS SEED <<< "${SPECS[$TASK]}"
FF=1024
LL=$(printf '%02d' "$LAYERS")
FFS=$(printf '%05d' "$FF")
TT=$(printf '%02d' "$TASK")

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
from spin_gnn.config import SudokuReasoningConfig, SudokuReasoningTrainConfig
from spin_gnn.train_sudoku_transformer import resolve_max_len_margin
print("torch", torch.__version__, "numpy", numpy.__version__, "numba", numba.__version__)
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))
m = resolve_max_len_margin(SudokuReasoningTrainConfig(),
                           SudokuReasoningConfig(n_nodes=96, degree=5, n_colors=4))
assert m == 194, f"expected a 2*96+2 generation margin, got {m}"
print("max_len margin", m)
# The knob under test must actually exist and must not be the default, or this
# run silently repeats t0/t4 and the comparison is vacuous.
c = SudokuReasoningTrainConfig()
assert c.lr_schedule == "constant", f"baseline default moved: {c.lr_schedule}"
assert c.warmup_epochs == 0, f"baseline warmup moved: {c.warmup_epochs}"
print("baseline defaults confirmed: lr_schedule=constant warmup_epochs=0")
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n96_cosine_2026-11-03"
cell="${OUT}/t${TT}_L${LL}_ff${FFS}_n${NTRAIN}_s${SEED}_cos"
mkdir -p "$cell"

echo "task=${TASK}  L=${LAYERS} ff=${FF} n_train=${NTRAIN} epochs=${EPOCHS} seed=${SEED} lr_schedule=cosine"
echo "cell -> ${cell}"
date +'cell start %F %T'

python3 -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 96 --degree 5 --n-colors 4 --shuffle-adj --fall \
    --d-model 256 --nhead 4 --dim-feedforward "$FF" \
    --num-layers "$LAYERS" --seed "$SEED" \
    --lr-schedule cosine \
    --n-train-instances "$NTRAIN" --epochs "$EPOCHS" \
    --n-val-instances 300 --batch-size 32 --n-per-bin 100 \
    --out-dir "$cell"

date +'cell end   %F %T'

TAG="d0256_h04_L${LL}_ff${FFS}_lr1e-03_pre_shuf_fall_cos"
if [ "$SEED" -ne 0 ]; then
    TAG="${TAG}_s$(printf '%02d' "$SEED")"
fi
CKPT="${cell}/${TAG}"
[ -f "${CKPT}/model.pt" ] || { echo "expected checkpoint missing: ${CKPT}"; ls -1 "$cell"; exit 1; }

echo
echo "=== the transition curve, same grid and same instances as 50860571 ==="
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.32 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== P2: did the schedule actually buy a lower final loss? ==="
python3 - "${CKPT}/history.json" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
best = min(h, key=lambda r: r["nll"])
print("epochs %d  final train_nll %.4f  final val_nll %.4f  best val_nll %.4f at epoch %d"
      % (len(h), h[-1]["train_nll"], h[-1]["nll"], best["nll"], best["epoch"]))
print("overfit gap (final - best) = %+.4f" % (h[-1]["nll"] - best["nll"]))
tail = [r["nll"] for r in h[-5:]]
print("last 5 val_nll:", " ".join("%.4f" % x for x in tail),
      "  drop over them %+.4f" % (tail[-1] - tail[0]))
# The constant-lr reference at this identical config, from 50860571.
print("reference: t0 (seed 0) final val 1.7455 ; t4 (seed 1) final val 1.7283")
print("P2 asks for both cosine cells below 1.7283.")
PY
date +'task end  %F %T'
