#!/bin/bash
#SBATCH --job-name=sud2FallScan
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=09:00:00
#SBATCH --array=0-12%5
#SBATCH --output=%x_%A_%a.out
#
# HOW SOLVE RATE SCALES ACROSS THE TRANSITION, WITH TRAINING SET SIZE, AND
# WITH MODEL SIZE -- the three axes hshunt asked for, on the fall ensemble.
#
#   sbatch scripts/slurm/sudoku2_n48_fall_scan_2026-11-02.sbatch
#
# WHY NOW, AND WHY NONE OF THIS WAS POSSIBLE YESTERDAY. Commit 5c17d3c
# explains the mechanism: under the unconstrained stub split, 0.3827 of nodes
# at q=4,d=5 are missing a colour in their neighbourhood, are therefore
# undeducible by DD forever, and pin greedy removal at rho_min = 0.469 at
# N=48 -- ABOVE the slides' rho_c ~ 0.46. Every N=48 result in this project is
# consequently a fast-phase result, and "across the transition" was not a
# measurable thing to ask for. A fall colouring (every node adjacent to every
# other colour) drops that floor to 0.332 +- 0.024 and lifts T_end from 7.9 to
# 13.8 at N=48, so the grid below now actually crosses into the slow side.
#
# =====================================================================
# THE GRID. 13 cells, all N=48 d=5 q=4, --fall --shuffle-adj, d_model 256,
# nhead 4, pre-norm, lr 1e-3 constant, batch 32, n_val 300.
#
#   task  L   ff    n_train  epochs  seed   steps   est h   role
#    0    4  1024     30000    400    0    375,200   2.2    data axis, L=4
#    1    4  1024    120000    100    0    375,000   2.2
#    2    4  1024    240000     50    0    375,000   2.2
#    3    8  1024     30000    400    0    375,200   3.9    data axis, L=8
#    4    8  1024    120000    100    0    375,000   3.9
#    5    8  1024    240000     50    0    375,000   3.9
#    6   12  1024     30000    400    0    375,200   5.5    data axis, L=12
#    7   12  1024    120000    100    0    375,000   5.5
#    8   12  1024    240000     50    0    375,000   5.5
#    9    8  1024     30000    400    1    375,200   3.9    SEED REPLICATE
#   10    8  1024    120000    100    1    375,000   3.9    of tasks 3-5
#   11    8  1024    240000     50    1    375,000   3.9
#   12    4  4096    120000    100    0    375,000   5.5    WIDTH CONTROL
#
# EQUAL GRADIENT STEPS, NOT EQUAL EPOCHS. Every cell takes 375k optimizer
# steps, so the n_train axis varies the number of DISTINCT examples with the
# amount of optimization held fixed. The alternative -- 100 epochs everywhere,
# which is what all eight existing N=48 cells used -- confounds data quantity
# with step count and would make "more data helps" unreadable. The cost is
# that these cells are not step-comparable to those eight; they are not
# instance-comparable to them either (different graph ensemble), so nothing is
# lost that was not already lost. lr_schedule is constant and warmup is 0, so
# equal steps really is equal optimization and there is no schedule to rescale.
#
# WIDTH CONTROL (task 12) IS THE POINT OF THE MODEL-SIZE AXIS. L=4 with
# ff=4096 has 9,598,010 parameters against L=12/ff=1024's 9,612,346 (the
# latter read from sweep.csv, the former from the same per-layer formula that
# reproduces it) -- a 0.15% match -- at one third the depth. hshunt's standing question is whether required
# reasoning depth tracks MLP depth; the L ladder alone cannot answer it,
# because depth and parameter count move together along it. This one cell
# separates them.
#
# SEED REPLICATE (tasks 9-11) IS NOT OPTIONAL HERE. Finding 41 measured a
# same-architecture seed spread of 0.1627 at L=8 on the shared n=1500 pool --
# LARGER than the 0.1120 one-rung depth gap the ladder is built from. A
# single-seed grid therefore cannot support any level comparison at all. These
# three cells put an error bar on the entire data axis at one depth, which is
# the cheapest placement that makes H3 falsifiable rather than decorative.
#
# =====================================================================
# PRE-REGISTERED, written before submission.
#
# Definitions. rho_50 = the realized density at which a cell's solve rate
# crosses 0.5, by linear interpolation between adjacent grid points. s_L8(rho)
# = |solve(seed 1) - solve(seed 0)| at L=8, from tasks 3-5 against 9-11; this
# is THE yardstick, and no effect smaller than it is claimed as an effect.
#
#  H1 -- THE TRANSITION IS A TRANSITION. High confidence.
#      Prediction: for every cell reaching solve >= 0.5 at rho=0.95,
#      solve(0.95) - solve(0.35) >= 0.4, monotone non-increasing in rho to
#      within sampling noise, and rho_50 > 0.50 -- i.e. models break down well
#      ABOVE the ensemble's structural floor of 0.332, so what the curve
#      measures is the model and not the instance generator running out of
#      room. dd_mean_T is recorded per bin so the x-axis can be re-read as
#      required reasoning depth rather than density.
#      FALSIFIER: a curve flat to within 0.1 across the whole grid (the
#      transition is invisible at N=48 and the scan must move to larger N), or
#      rho_50 < 0.40 for every cell (the models track the generator's floor
#      and the measurement is about the ensemble, not about learning).
#
#  H2 -- DEPTH BUYS THE TAIL AND NOTHING ELSE. Moderate confidence.
#      Prediction: rho_50 is monotone decreasing in L over {4, 8, 12} at each
#      n_train, with rho_50(L=4) - rho_50(L=12) >= 0.10; and the cells agree
#      to within 0.03 at rho=0.95, separating only below rho ~ 0.7. Mechanism:
#      low rho means large DD depth T, a fixed-depth transformer can chain
#      only so many serial deduction rounds, so layers should pay off exactly
#      where T is large and nowhere else.
#      FALSIFIER: a non-monotone L ordering whose size exceeds s_L8 at the
#      same rho, or depth separating the cells at rho >= 0.9 as much as at
#      rho <= 0.5 (which would make it capacity, not serial computation).
#
#  H3 -- DATA HELPS, AND HELPS MOST IN THE TAIL. Moderate confidence.
#      Prediction: at matched steps, solve(240k) - solve(30k) > s_L8 at rho <=
#      0.5, and smaller than that gap at rho >= 0.8. Rationale: deep instances
#      are rare in a rho-uniform training mix, so the tail is where a small
#      pool runs out of distinct examples first.
#      FALSIFIER: 30k matching or beating 240k anywhere by more than s_L8 --
#      30k x 400 epochs would then be sufficient, the task would not be
#      data-limited at this scale, and the memorization question (30k seen 400
#      times, still generalizing) becomes the interesting one instead.
#      Watch: val_nll in history.json will show if the 30k cells overfit. If
#      they do, that is part of the answer, not a reason to discard the cell.
#
#  H4 -- IT IS DEPTH, NOT PARAMETERS. The discriminator, low-moderate
#      confidence, and the single most informative cell in the grid.
#      Prediction: task 12 (L=4, ff=4096, 9.60M) has rho_50 within 0.03 of
#      task 1 (L=4, ff=1024, 3.29M) and at least 0.08 above task 7 (L=12,
#      ff=1024, 9.61M) -- i.e. it behaves like its depth, not like its
#      parameter count.
#      FALSIFIER: it tracks L=12 instead. That would say the L ladder has been
#      measuring capacity all along, and every depth reading in findings
#      11/14/22/33/38 would need restating as a capacity reading.
#
#  C -- CONTROLS, low confidence anything moves.
#      truncated_rate <= 0.01 and outcome_accuracy >= 0.99 at every rho for
#      every cell with solve > 0.1, so a gap is deduction and not grammar
#      (finding 25's failure mode). realized_rho identical across all 13 cells
#      at each requested rho, since --seed-start is fixed and every cell
#      shares an instance config -- this is what makes the curves paired point
#      by point. The floor note must fire for at most the last grid point;
#      if it fires higher, the fall ensemble's rho_min at N=48 is above 0.35
#      and the low end of the grid has to be rebuilt.
#
#  SCORING RULE, fixed now so it cannot be chosen afterwards: any level
#  comparison across cells is read against s_L8 at the same rho. Where the
#  effect is smaller, the statement is "not separable at this budget" -- not a
#  result with a p-value. Curve SHAPE within one checkpoint is paired and does
#  not need this caveat; curve LEVEL across checkpoints always does.
#
# =====================================================================
# COST. Measured law from the three seed-0 reference cells (50527634 L=8
# 07:12:09, 50607549 L=10 08:58:51, 50527633 L=12 10:31:58, all at 750k
# steps): 50.0 min per layer plus ~32 min fixed. At 375k steps that is 25.0
# min per layer plus 32 min, giving the est h column above; task 12's layers
# cost 3.0x an ff=1024 layer by FLOPs (4d^2 + 2*d*ff = 2.36M vs 0.79M params
# per layer), so 4 of them cost what 12 cheap ones do. Instance generation is
# 2.1 ms each, measured 11-02, so even 240k is 0.14 h and is not in the
# critical path. Sum 51.9 GPU-h over 13 cells; --array throttled to 5 at a
# time leaves 11 of sganguli's 16 GPUs for the rest of the group and puts wall
# clock near 11 h. --time 09:00:00 is 58% headroom over the longest cell
# (5.5 h + ~5 min eval); sganguli allows 7 d and 32 GB/core (sh_part, 11-02).
#
# ENVIRONMENT. ${SCRATCH}/spin_gnn_venv is dead (90-day purge), so this is the
# module stack only and calls python3 explicitly -- bare `python` is the
# system 2.7. The smoke check turns an environment failure into a 30-second
# job instead of a 5-hour one that dies in the eval.

set -euo pipefail

# task: L:ff:n_train:epochs:seed
SPECS=(
  "4:1024:30000:400:0"
  "4:1024:120000:100:0"
  "4:1024:240000:50:0"
  "8:1024:30000:400:0"
  "8:1024:120000:100:0"
  "8:1024:240000:50:0"
  "12:1024:30000:400:0"
  "12:1024:120000:100:0"
  "12:1024:240000:50:0"
  "8:1024:30000:400:1"
  "8:1024:120000:100:1"
  "8:1024:240000:50:1"
  "4:4096:120000:100:0"
)
TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
IFS=: read -r LAYERS FF NTRAIN EPOCHS SEED <<< "${SPECS[$TASK]}"
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
print("torch", torch.__version__, "numpy", numpy.__version__, "numba", numba.__version__)
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n48_fall_scan_2026-11-02"
# One directory PER TASK, not per L: 13 cells run concurrently and
# sudoku_hparam_sweep appends to <out-dir>/sweep.csv, so a shared directory
# would interleave rows from several writers into one file.
cell="${OUT}/t${TT}_L${LL}_ff${FFS}_n${NTRAIN}_s${SEED}"
mkdir -p "$cell"

echo "task=${TASK}  L=${LAYERS} ff=${FF} n_train=${NTRAIN} epochs=${EPOCHS} seed=${SEED}"
echo "cell -> ${cell}"
date +'cell start %F %T'

python3 -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 48 --degree 5 --n-colors 4 --shuffle-adj --fall \
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
echo "=== the transition curve for this cell (H1-H4) ==="
# Grid bottoms out at 0.33 because the fall ensemble's rho_min at N=48 is
# 0.332 +- 0.024 (200 seeds, measured 11-02); 0.33 is there so the floor note
# fires and labels where the ensemble stops, rather than leaving a flat tail
# to be read as a plateau. --seed-start is left at the default 6e10, shared
# with the "before" job, so all 13 cells see the SAME 400 instances per rho.
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.33 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== H3 watch: val_nll trajectory (overfitting at 30k x 400 epochs?) ==="
python3 - "${CKPT}/history.json" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
best = min(h, key=lambda r: r["nll"])
print("epochs %d  final val_nll %.4f  best val_nll %.4f at epoch %d  final token_acc %.4f"
      % (len(h), h[-1]["nll"], best["nll"], best["epoch"], h[-1]["accuracy"]))
print("overfit gap (final - best) = %+.4f" % (h[-1]["nll"] - best["nll"]))
PY
date +'task end  %F %T'
