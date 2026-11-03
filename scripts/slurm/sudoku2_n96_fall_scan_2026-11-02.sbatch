#!/bin/bash
#SBATCH --job-name=sud2N96FallScan
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=26:00:00
#SBATCH --array=0-4%5
#SBATCH --output=%x_%A_%a.out
#
# THE SAME THREE AXES AS FINDING 44, AT N=96 -- the size where the model
# should stop outrunning the generator.
#
#   sbatch scripts/slurm/sudoku2_n96_fall_scan_2026-11-02.sbatch
#
# WHY N=96 AND WHY NOW. Finding 44 answered "how does solve rate scale across
# the transition, with training-set size, and with model size" at N=48, and
# the answer came with one defect that is not fixable at N=48: H1's
# `rho_50 > 0.50` clause FAILED for both L=12 cells because the best model
# never dropped below 0.5 anywhere on the grid -- it still solved 0.755 at the
# ensemble floor (rho=0.333, dd_mean_T=13.79), where the generator stops. The
# model outran the problem. `rho_50` for that cell is a bound, not a
# measurement, and no amount of re-running N=48 changes it.
#
# At N=96 the fall ensemble floors at rho ~ 0.32 with dd_mean_T 21.9 (finding
# 43, 200 seeds), and a 30-instance probe on 11-02 confirmed the generator
# end to end at this size: rho_min 0.312, T up to 28, mean 8.4, encoded
# sequence mean 780 / max 800, 33.4 ms per instance, vocab 106. So the grid
# below reaches roughly twice the reasoning depth that N=48 could offer.
#
# BLOCKER CLEARED FIRST. `max_len` used to be the longest TRAINING sequence
# exactly, and generate() hard-stops at pos_embed.shape[0], so finding 44's
# two L=12 cells were truncated in up to 0.0275 of their deepest bins. At
# N=96, with traces 2x longer, that would have bound everywhere. Fixed and
# verified in 41baaf5 / 42086ac (job 50853243, ALL CHECKS PASSED): max_len is
# now the longest training sequence + 2*n_nodes+2, so ~800 + 194 ~ 1000 here
# against a longest observed trace of 800. truncated_rate is a watch item
# below precisely because this is the first run that depends on that fix.
#
# =====================================================================
# THE GRID. 5 cells, all N=96 d=5 q=4, --fall --shuffle-adj, d_model 256,
# nhead 4, ff 1024, pre-norm, lr 1e-3 constant, batch 32, n_val 300, NO
# warmup. EQUAL 375k GRADIENT STEPS EVERYWHERE, exactly as at N=48, so the
# data axis varies distinct examples with optimization held fixed and the two
# scans are read on the same protocol. The ONLY protocol change from
# 50760297 is n_nodes: one thing at a time.
#
#   task  L   n_train  epochs  seed   role                         est h
#     0   12   240000     50     0    the headline cell              13.6
#     1    8   240000     50     0    depth rung below                9.9
#     2   16   240000     50     0    depth rung above               17.2
#     3   12   120000    100     0    the data axis at fixed steps   12.5
#     4   12   240000     50     1    the SEED YARDSTICK             13.6
#
# 30k is deliberately absent. Finding 44 measured that at 30k the L ordering
# INVERTS by 0.157 against a seed spread of 0.154, i.e. that row measured the
# loss plateau and the seed, not the architecture, and all three 30k cells
# were the only ones that overfit -- more so the deeper they were. A grid
# that spends GPU-hours there buys a row that cannot be read.
#
# =====================================================================
# PRE-REGISTRATION. Written before any cell runs; scored in the notebook.
#
# H1  THE CURVE DOES NOT SATURATE AT N=96. Every cell has a `rho_50` strictly
#     inside the grid -- it crosses solve_rate 0.5 somewhere above the floor
#     -- so N=96 measures the transition rather than bounding it.
#     FALSIFIER: the best cell again fails to cross 0.5 at rho=0.32. Then
#     N=96 is still too small and the next move is N=192, NOT another N=96
#     grid. Opposite failure, equally informative: if even L=16/240k is below
#     0.5 at rho=0.95, the budget is wrong rather than the size.
#
# H2  DEPTH STILL BUYS TRANSITION DEPTH, AND THE KNEE HAS MOVED PAST 12.
#     rho_50(L=8) > rho_50(L=12) > rho_50(L=16) at 240k, with the 8->12 gap
#     at least 0.05. Leak-free L* at N=48 is (8, 12] (finding 38) and the
#     leaked ladder went 3/4/8 at N=24/48/96, so if the knee scales with N at
#     all, N=96 leak-free should still want more than 12 layers.
#     FALSIFIER: L=16 ties or loses to L=12 by more than the H4 yardstick.
#     That would be the first time depth stopped paying in the leak-free task
#     at a size where the model is demonstrably not saturated, and it would
#     put the knee at or below 12.
#
# H3  COVERAGE STILL MATTERS AT FIXED OPTIMIZATION, AND LESS THAN DEPTH.
#     120k -> 240k at L=12 and equal 375k steps moves rho_50 by at least 0.02
#     toward 240k, and by less than the L=8 -> L=12 gap.
#     FALSIFIER: 120k beats 240k by more than the yardstick. Then optimization
#     and not coverage is binding at N=96, and the next move is more steps,
#     not more instances -- the mirror of finding 23, where at N=24 the extra
#     epochs did nothing and only instance coverage moved the number.
#
# H4  THE SEED YARDSTICK, and the reason task 4 exists. |rho_50(s0) -
#     rho_50(s1)| at L=12/240k is <= 0.05. EVERY level comparison in H1-H3 is
#     read against it; where an effect is smaller, the statement is "not
#     separable at this budget", not a result with a p-value. Finding 41 is
#     the precedent and it is not a formality: at N=48 the seed was worth
#     0.1627 of solve_rate at L=8, more than the L=12-minus-L=8 gap that
#     finding 38's headline rested on, and it REVERSED the sign of a
#     published inversion. At 240k the N=48 rho_50 spread was 0.023.
#     IF IT IS LARGER THAN 0.05 HERE, this whole grid is one draw and only
#     the ordering survives -- say so rather than quoting the levels.
#
# WATCH ITEMS (not hypotheses; they decide what the numbers MEAN).
#   * GRAMMAR. outcome_accuracy and truncated_rate per bin. With the margin
#     in place truncated_rate should be ~0 everywhere; finding 44's 0.0275
#     came from a flush table. If L=16 collapses on well-formedness the way
#     finding 25's deep-thin N=96 cell did (0.853), the follow-up is warmup
#     (finding 32 recovered +0.0800 that way), NOT a re-run.
#   * STEPS-BOUND? Each cell prints its val_nll trajectory. If the final
#     val_nll is still the minimum and still falling, N=96 is steps-bound at
#     375k and this grid measures that rather than capacity. Finding 15 is
#     the warning: "no overfitting gap, so more data will not help" was wrong
#     there, and 4x data then gained 2.1x what the lr schedule did.
#   * TRAINABILITY PLATEAU. Finding 44's three L=4 cells never left a train
#     NLL plateau near 1.64, so that row measured trainability, not capacity.
#     Any N=96 cell that plateaus is read the same way and its rho_50 is not
#     a depth result.
#
# NOT MEASURED HERE, ON PURPOSE. There is no default-ensemble "before" job
# this time: every existing N=96 checkpoint is LEAKED-ORDER (findings
# 15/20/22/25/31/32), so none of them is comparable to these cells, and a
# paired before/after of the kind 50760252 provided at N=48 does not exist at
# this size. The cross-size comparison to N=48 is curve shape against
# dd_mean_T only, never a McNemar: different N is a different ensemble.
#
# =====================================================================
# COST, from the measured N=48 law and the 11-02 N=96 probe. The three seed-0
# N=48 reference cells gave 50.0 min per layer at 750k steps, i.e. 4.0
# ms/step/layer at batch 32 x seq 394 = 12,608 tokens/step, so 0.317 us per
# token per layer on an H100. At N=96 the probe measures seq 780, so 24,960
# tokens/step, and attention's share rises from ~13% to ~25% of per-token
# cost: 4.0 * (24960/12608) * 1.11 = 8.8 ms/step/layer, i.e. 55 min per layer
# at 375k steps. Training is therefore 7.3 / 11.0 / 14.7 h at L=8 / 12 / 16.
#
# INSTANCE GENERATION IS NO LONGER FREE AND THAT IS THE ONE SURPRISE. At
# N=48 it was 2.1 ms each, 8 min for 240k, noise. At N=96 the probe measures
# 33.4 ms each -- 16x, because greedy hint removal runs DD to fixation after
# every deletion and both the node count and the depth have grown. 240k
# instances is 2.2 h of a GPU sitting idle on a CPU loop, 1.1 h at 120k. It
# is budgeted below rather than optimized away: parallelizing the generator
# is a code change this run should not also be testing.
#
# Eval is ~20 min: 14 bins x 400 = 5,600 free generations of ~800 tokens,
# unbatched, measured at 0.13 ms per forward on these GPUs from 50760252
# (35,200 generations in 31 min at N=48), plus 187 s to build the instances.
#
#   cell          train   gen    eval   total
#   L=12 240k     11.0 h  2.2 h  0.3 h  13.6 h
#   L=8  240k      7.3 h  2.2 h  0.3 h   9.9 h
#   L=16 240k     14.7 h  2.2 h  0.3 h  17.2 h
#   L=12 120k     11.0 h  1.1 h  0.3 h  12.5 h
#   L=12 240k s1  11.0 h  2.2 h  0.3 h  13.6 h
#                                       66.8 GPU-h
#
# --time 26:00:00 is 51% headroom over the longest cell, which is the right
# margin when the 2.2x step-cost factor is an estimate rather than a
# measurement. --array throttled to 5 runs all five at once and still leaves
# 11 of sganguli's 16 GPUs for the rest of the group; wall clock ~17.5 h.
# sganguli allows 7 d and 32 GB/core (sh_part, 11-02); its GPUs are H100
# SXM5 80GB (sh_node_feat), so the ~20 GB of L=16 attention maps at seq 800 x
# batch 32 is not close to a constraint.
#
# ENVIRONMENT. ${SCRATCH}/spin_gnn_venv is dead (90-day purge), so this is the
# module stack only and calls python3 explicitly -- bare `python` is the
# system 2.7. The smoke check turns an environment failure into a 30-second
# job instead of a 14-hour one that dies in the eval.

set -euo pipefail

# task: L:n_train:epochs:seed
SPECS=(
  "12:240000:50:0"
  "8:240000:50:0"
  "16:240000:50:0"
  "12:120000:100:0"
  "12:240000:50:1"
)
TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
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
# The max_len fix is load-bearing at this size (traces ~800 tokens), so assert
# it is present rather than discovering a flush table in the truncated_rate.
m = resolve_max_len_margin(SudokuReasoningTrainConfig(),
                           SudokuReasoningConfig(n_nodes=96, degree=5, n_colors=4))
assert m == 194, f"expected a 2*96+2 generation margin, got {m}"
print("max_len margin", m)
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n96_fall_scan_2026-11-02"
# One directory PER TASK, not per L: the cells run concurrently and
# sudoku_hparam_sweep appends to <out-dir>/sweep.csv, so a shared directory
# would interleave rows from several writers into one file. Tasks 0 and 3
# would also collide on the checkpoint TAG, which carries L but not n_train.
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
echo "=== the transition curve for this cell (H1-H4) ==="
# Grid bottoms out at 0.32 because the fall ensemble's rho_min at N=96 is
# 0.324 (finding 43, 200 seeds) and the 11-02 probe saw 0.312 in 30 draws;
# 0.32 is there so the floor note fires and labels where the ensemble stops,
# rather than leaving a flat tail to be read as a plateau. --seed-start is
# left at the default 6e10, so all five cells see the SAME 400 instances per
# rho and are paired point for point.
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
# Still falling at the end means steps-bound, not capacity-bound.
tail = [r["nll"] for r in h[-5:]]
print("last 5 val_nll:", " ".join("%.4f" % x for x in tail),
      "  drop over them %+.4f" % (tail[-1] - tail[0]))
PY
date +'task end  %F %T'
