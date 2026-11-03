#!/bin/bash
#SBATCH --job-name=sud2N96FallS1L8
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=14:00:00
#SBATCH --array=5-5
#SBATCH --output=%x_%A_%a.out
#
# A SECOND SEED AT L=8, SO H2'S HEADLINE GAP HAS A REPLICATE AT BOTH ENDS.
#
#   sbatch scripts/slurm/sudoku2_n96_fall_seed1_L8_2026-11-03.sbatch
#
# WHY. 50860571 measures the seed yardstick (H4) at L=12 ONLY -- tasks 0 and 4
# are the same config at seeds 0 and 1. But H2's load-bearing comparison is
# the L=8 -> L=12 gap, and with a replicate at one end only, the honest
# reading of that gap is "one draw at L=8 against two at L=12". Finding 41 is
# the precedent and it is not hypothetical: at N=48 the seed was worth 0.1627
# of solve_rate at L=8 -- larger than the depth gap finding 38's headline
# rested on -- and it REVERSED the sign of a published inversion. Finding 44
# then made "seed replicates at every rung" a standing design constraint, and
# this grid does not yet meet it.
#
# This cell is that replicate: task 1 of 50860571 (L=8/240k/50ep) at seed 1.
# It is a REPLICATE, not a new condition -- every other knob is byte-identical
# to the running script, so the only difference from t1 is --seed.
#
# WHAT IT BUYS, concretely. With t1+t5 and t0+t4 the H2 gap is read as a
# difference of two 2-seed means with a measured spread at each end, instead
# of a difference of two single draws against a yardstick borrowed from the
# other rung. If s(L=8) turns out comparable to s(L=12) ~ the H4 number, the
# depth claim stands or falls on its own error bar. If s(L=8) is much larger
# -- which is what N=48 saw at the SHALLOW end specifically -- then H2 is
# "ordering only" and this cell is what reveals that rather than hiding it.
#
# WHY L=8 AND NOT L=16. The pre-registered bar in H2 is the 8->12 gap (>=
# 0.05); L=16 only has to beat L=12 by the yardstick to confirm, and loses by
# it to falsify, so a replicate there sharpens a weaker clause. Shallow is
# also where N=48 actually measured the big seed term.
#
# OUTPUT. Writes into 50860571's own scan directory with task tag t05, so the
# reader picks it up with no change: cell key (L, ff, n_train, seed) =
# (8, 1024, 240000, 1) is distinct from t1's seed 0, and the CSV filename
# carries this job's OWN array job id, so nothing overwrites. --seed-start is
# left at the default, so this cell sees the SAME 400 instances per rho as all
# five of 50860571's and is paired with them point for point.
#
# COST, now MEASURED rather than modelled. The pre-flight model for 50860571
# (55 min/layer + 2.2 h serial generation) was wrong in both terms: t1
# COMPLETED in 7:54:59 wall for the whole cell including generation and the
# 14-point eval, and generation turned out to cost minutes, not hours. This
# cell is the same shape, so 8 h is the expectation and --time 14:00:00 is 77%
# headroom. One GPU; sh_part at 04:20 on 11-03 shows 10 of sganguli's 16 idle,
# and 50860571's four remaining tasks are already allocated, so this does not
# compete with them.

set -euo pipefail

# Keyed by array task id on purpose: this script owns task 5 of the N=96 fall
# scan's numbering and refuses to run as any other, so it cannot be submitted
# with a bare --array and silently retrain a cell 50860571 already has.
declare -A SPECS=( [5]="8:240000:50:1" )   # L:n_train:epochs:seed
TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
[ -n "${SPECS[$TASK]+x}" ] || { echo "no spec for task ${TASK}; this script only defines 5"; exit 1; }
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
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n96_fall_scan_2026-11-02"
cell="${OUT}/t${TT}_L${LL}_ff${FFS}_n${NTRAIN}_s${SEED}"
mkdir -p "$cell"

echo "task=${TASK}  L=${LAYERS} ff=${FF} n_train=${NTRAIN} epochs=${EPOCHS} seed=${SEED}"
echo "cell -> ${cell}"
date +'cell start %F %T'

python3 -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 96 --degree 5 --n-colors 4 --shuffle-adj --fall \
    --d-model 256 --nhead 4 --dim-feedforward "$FF" \
    --num-layers "$LAYERS" --seed "$SEED" \
    --n-train-instances "$NTRAIN" --epochs "$EPOCHS" \
    --n-val-instances 300 --batch-size 32 --n-per-bin 100 \
    --out-dir "$cell"

date +'cell end   %F %T'

TAG="d0256_h04_L${LL}_ff${FFS}_lr1e-03_pre_shuf_fall"
if [ "$SEED" -ne 0 ]; then
    TAG="${TAG}_s$(printf '%02d' "$SEED")"
fi
CKPT="${cell}/${TAG}"
[ -f "${CKPT}/model.pt" ] || { echo "expected checkpoint missing: ${CKPT}"; ls -1 "$cell"; exit 1; }

echo
echo "=== the transition curve for this cell (the L=8 seed term) ==="
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.32 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== watch items: steps-bound? trainability plateau? ==="
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
PY
date +'task end  %F %T'
