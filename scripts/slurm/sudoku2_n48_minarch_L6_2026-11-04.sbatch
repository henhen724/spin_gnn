#!/bin/bash
#SBATCH --job-name=sud2MinArchL6
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --array=0-14%6
#SBATCH --output=%x_%A_%a.out
#
# STAGE B: STRIP THE MACHINERY AT THE DEPTH WHERE DEDUCTION STILL WORKS.
#
#   sbatch scripts/slurm/sudoku2_n48_minarch_L6_2026-11-04.sbatch
#
# WHY THIS JOB EXISTS, I.E. WHAT STAGE A (51032450) GOT WRONG BY DESIGN.
# Stage A pinned the architecture axis at L=4 because finding 44 said L=4
# solves the task. Its own depth ladder then measured, at three seeds each:
#
#     L    rho_50 per seed          median   spread   mean_solve [spread]
#    12    0.385 / 0.370 / 0.485    0.385    0.115    0.8163 [0.1530]
#     8    0.432 / 0.581 / 0.471    0.471    0.148    0.7111 [0.2205]
#     6    0.593 / 0.582 / 0.592    0.592    0.012    0.5389 [0.0141]
#     4    0.760 / >0.95  / 0.763   CENSORED   --     0.3084 [0.2784]
#
# So B = 4: L=4 is the BREAKING depth, the rung where one of three seeds stops
# crossing 0.5 at all. Pinning the arch axis there was pinning it to the edge
# of the cliff. The consequence is arithmetic, not opinion: at L=4 the hub's
# own across-seed spread in mean_solve (0.2784) is as large as its own score
# (0.3084), so NO arch contrast measured at L=4 can be resolved against the
# seed term -- and indeed Stage A's `linear` cell came back at 0.3089 against
# the hub's 0.3084, a "tie" between two cells that are both sitting on the
# floor and would also tie with a model that had learned nothing.
#
# L=6 is the SHALLOWEST rung where all three seeds still solve the task, and
# its seed term there is small (0.012 in rho_50, 0.014 in mean_solve). That is
# where the question "what machinery can be removed" has room to be answered.
# This job therefore re-runs the whole arch axis and the head axis at L=6.
#
# THIS IS NOT A FREE CHOICE OF RUNG, AND THE SCORING RULE BELOW IS WHAT STOPS
# IT FROM BEING ONE. s* is POOLED over Stage A and Stage B -- the largest
# across-seed spread seen on ANY complete rung in EITHER job -- so moving to a
# quiet rung does not buy a smaller yardstick. At submission s* = 0.1484 in
# rho_50 and 0.2784 in mean_solve, both inherited from Stage A's L=8 and L=4
# rungs, and it can only grow as Stage A's last cells land.
#
# =====================================================================
# THE GRID. 15 cells, all N=48 d=5 q=4, --fall --shuffle-adj, d_model 256,
# ff 1024, L=6, pre-norm, lr 1e-3 CONSTANT, batch 32, 240k instances x 50
# epochs = 375k gradient steps, n_val 300. Three seeds everywhere.
#
#   tasks   L  nhead  arch_mode  seeds    est h ea   axis
#    0-2    6    4    linear     0,1,2      2.8     nonlinearity removed (N1)
#    3-5    6    4    rf_mlp     0,1,2      2.8     random features, literal (N2)
#    6-8    6    1    full       0,1,2      2.8     heads 4 -> 1 (N5)
#    9-11   6    4    no_mlp     0,1,2      2.8     attention only (N4)
#   12-14   6    4    rf_all     0,1,2      2.3     random features, convex (N3)
#
# THE CONTROL IS NOT IN THIS JOB. L=6 / nhead=4 / full is Stage A tasks 6-8,
# already run to completion at the same budget, same 14-point grid and the same
# default --seed-start, so it is PAIRED with every cell here point by point.
# Re-running it would have cost 8.4 GPU-h to re-measure a number I already have
# to three seeds. Every cell below differs from those three in exactly one
# respect.
#
# ORDERED BY SCIENTIFIC PRIORITY, not by cost, because all five arch modes cost
# about the same at fixed L: `linear` first because N1 is the headline if it
# ties, `rf_all` last because M5 already came back a clean null at L=4 (0.0000
# mean_solve, 3/3 seeds, 0/3 plateau escapes) and its value here is to confirm
# that the null is about the architecture and not about the breaking depth.
# Throttled to 6 concurrent: sganguli had 8 of 16 GPUs idle and only 3 of 384
# cores free at submission, with 13 of brennerj's jobs pending, so this job
# queues politely behind the group rather than claiming the rest of the GPUs.
#
# WHAT EACH arch_mode TAKES AWAY is unchanged from Stage A; see
# scripts/slurm/sudoku2_n48_minarch_2026-11-04.sbatch and
# SudokuReasoningModelConfig.arch_mode. Every mode starts from the SAME
# initialization as full at a given seed, so each contrast is about training
# the weights, not about drawing them.
#
# =====================================================================
# PRE-REGISTERED, written before submission. HYPOTHESIS IN ONE SENTENCE:
# at the shallowest depth that still deduces, the MLP's nonlinearity and three
# of its four heads are removable without raising rho_50, but training the
# attention is not.
#
# METRIC OF RECORD: rho_50, the realized hint density at which solve rate
# crosses 0.5 on the 14-point x 400-instance grid below, by linear
# interpolation. LOWER IS BETTER. Secondary metrics that never censor, carried
# wherever a cell fails to cross 0.5: mean_solve_rate (unweighted mean over the
# 14 bins) and solve_at_floor (solve rate at rho = 0.3333). A censored rho_50
# is reported as "> 0.95" and never as a number.
#
# BEFORE, the number every cell below is scored against -- Stage A tasks 6-8,
# L=6 nhead=4 full, seeds 0/1/2:
#     rho_50       0.593 / 0.582 / 0.592   median 0.592   spread 0.012
#     mean_solve   median 0.5389           spread 0.0141
#     solve_at_floor median 0.030
#     plateau escapes 3/3 (worst final train_nll 1.617 vs the 1.64 plateau)
#
#  N1 -- REMOVING THE ELEMENTWISE NONLINEARITY COSTS NOTHING. Moderate
#      confidence, UPGRADED from Stage A's M3 (which predicted the opposite)
#      because Stage A's L=4 linear cell did not separate from its hub -- a
#      result that was uninformative there for the floor reason above, but
#      which moved the prior.
#      Prediction: rho_50(linear, L=6) - 0.592 < s*, i.e. median at or below
#      ~0.74, with all three seeds crossing 0.5.
#      IF IT TIES IT IS THE HEADLINE OF THE PROJECT. With an identity MLP
#      activation there is no elementwise nonlinearity left anywhere in the
#      stack. What remains is the attention softmax and the 2L+1 = 13 pre-norm
#      LayerNorms, and both act as input-dependent SCALARS on otherwise linear
#      maps -- softmax sets mixing coefficients, LayerNorm subtracts a mean
#      (linear) and divides by an RMS (one scalar per token). The network is
#      then an input-dependent linear operator on the embedded sequence: a far
#      more tractable object than a ReLU stack, though NOT yet closed form.
#      That distinction is load-bearing and must survive into the write-up.
#      FALSIFIER: linear resolvably worse than full (gap > s*), or any seed
#      censored while all three full seeds solved. Then the nonlinearity is
#      doing real work at the depth where deduction lives, and the honest
#      reading of the L=4 tie is that it was a floor artifact.
#
#  N2 -- RANDOM FEATURES ARE ENOUGH, THE LITERAL READING. Moderate confidence.
#      Prediction: rho_50(rf_mlp, L=6) ties full within s*. Mechanism: at
#      d_model=256 a frozen 1024-wide random ReLU bank is over-complete, so a
#      trained linear2 can synthesize essentially any smooth per-block MLP the
#      full cell would have learned. Attention still trains here, so this cell
#      is NOT convex and NOT yet the analyzable object.
#      FALSIFIER: rf_mlp resolvably worse. Then linear1 is learning
#      task-specific features rather than supplying a basis, and WHAT those
#      features are becomes the next question.
#
#  N3 -- THE CONVEX CELL STILL DOES NOT REACH THE GRID AT L=6. High confidence
#      it fails; this cell is a CONTROL on Stage A's null, not a new bet.
#      Prediction: rf_all's rho_50 censored at all three seeds and mean_solve <
#      0.10, matching Stage A's L=4 result (0.0000 at 3/3).
#      FALSIFIER: mean_solve >= 0.10. Then two more blocks of frozen random
#      features changed a flat zero into partial deduction, a convex problem
#      does part of DD, and the entire analytic programme runs through this
#      cell -- it becomes the main object of study at once.
#      Reported either way: mean_solve, solve_at_floor, well_formed_rate and
#      token accuracy, since how much of the GRAMMAR a convex readout learns is
#      interesting even at zero solve rate.
#
#  N4 -- ATTENTION ONLY: NO REGISTERED PREDICTION. no_mlp brackets N1 and N2
#      from below rather than testing a belief. If it solves the task, nothing
#      in the MLP is needed and the minimal architecture is an attention stack;
#      if it fails while rf_mlp succeeds, the gap between them is exactly what
#      the MLP contributes, measured rather than argued.
#
#  N5 -- ONE HEAD IS ENOUGH. Moderate confidence.
#      Prediction: nhead 4 -> 1 at L=6 ties within s*. Evidence: finding 27
#      found no single critical head (worst single-head ablation left 84% of
#      performance), so the heads look redundant rather than specialized.
#      FALSIFIER: nhead=1 resolvably worse. Then the four heads carry
#      complementary work that test-time ablation cannot see because the others
#      compensate, and the head axis needs its own ladder.
#
#  W -- WATCH ITEMS, not hypotheses, but they decide how a failed cell is
#      WRITTEN UP, so they are fixed in advance.
#      TRAINABILITY vs CAPACITY. Each cell prints a plateau-escape flag (final
#      train_nll against finding 44's 1.64, plus the last-5-epoch slope). A
#      cell where 0 of 3 seeds escape is a TRAINABILITY result at this budget
#      -- "this architecture did not train here" -- and must NOT be written as
#      "this architecture cannot represent DD". Stage A's rf_all is exactly
#      that case and is reported that way.
#      GRAMMAR vs DEDUCTION. truncated_rate <= 0.01 and well_formed_rate >=
#      0.99 at every rho for every cell with solve > 0.1, so a gap is deduction
#      and not finding 25's grammar failure. Stage A ALREADY BREACHES this at
#      the margin -- max truncated 0.0200, min well_formed 0.9500 over its live
#      bins, concentrated at low density on the L=8 rung -- far too small to
#      carry any gap reported here, but it is recorded rather than rounded away
#      and the same check runs on these cells.
#      LEAK. Every cell trains and evaluates with --shuffle-adj, and the
#      assertion block below re-reads shuffle_adj, fall, arch_mode, num_layers,
#      nhead and max_len_margin out of each written config.json rather than
#      trusting the command line. Findings 16/17/23: a minimal architecture
#      that scores well on the leaky task has told us nothing, and the smaller
#      the model the more tempting the decoding shortcut.
#
#  SCORING RULE, fixed now so it cannot be chosen afterwards: every comparison
#  is median-vs-median against the POOLED s* defined above, computed over both
#  jobs by scripts/minarch_analysis.py and nothing else. Smaller than s* is
#  "not separable at this budget", never a result.
#
# =====================================================================
# COST. Measured law at this budget: 0.40 h per layer + 0.40 h fixed, confirmed
# live by Stage A (L=12 5.2 h, L=8 3:37, L=6 2:48, L=4 2:00). 15 cells x ~2.8 h
# = ~42 GPU-h, ~7 h wall at 6 concurrent. --time 05:00:00 is 79% headroom over
# the longest expected cell; sganguli allows 7 d. Instance generation is 2.1 ms
# each at N=48 fall, so 240k is 0.14 h per cell and is not in the critical path.
#
# ENVIRONMENT. Module stack only -- ${SCRATCH}/spin_gnn_venv is long dead to
# the 90-day purge -- and python3 explicitly, since bare `python` is the system
# 2.7. The smoke check turns an environment failure into a 30-second job
# instead of a three-hour one that dies in the eval.

set -euo pipefail

# task: L:nhead:arch_mode:seed    (priority order, see the grid table above)
SPECS=(
  "6:4:linear:0"   "6:4:linear:1"   "6:4:linear:2"
  "6:4:rf_mlp:0"   "6:4:rf_mlp:1"   "6:4:rf_mlp:2"
  "6:1:full:0"     "6:1:full:1"     "6:1:full:2"
  "6:4:no_mlp:0"   "6:4:no_mlp:1"   "6:4:no_mlp:2"
  "6:4:rf_all:0"   "6:4:rf_all:1"   "6:4:rf_all:2"
)
TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
IFS=: read -r LAYERS NHEAD ARCH SEED <<< "${SPECS[$TASK]}"
LL=$(printf '%02d' "$LAYERS")
HH=$(printf '%02d' "$NHEAD")
TT=$(printf '%02d' "$TASK")
FF=1024
FFS=$(printf '%05d' "$FF")

# Mirrors ARCH_TAG in spin_gnn/sudoku_hparam_sweep.py, which is the authority;
# the assertion block below re-derives the checkpoint path from the written
# config.json, so a mismatch here fails loudly instead of scoring nothing.
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
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n48_minarch_L6_2026-11-04"
# One directory PER TASK: sudoku_hparam_sweep appends to <out-dir>/sweep.csv and
# six cells run concurrently, so a shared directory would interleave rows from
# several writers into one file.
cell="${OUT}/t${TT}_L${LL}_h${HH}_${ARCH}_s${SEED}"
mkdir -p "$cell"

echo "task=${TASK}  L=${LAYERS} nhead=${NHEAD} arch=${ARCH} seed=${SEED}"
echo "cell -> ${cell}"
date +'cell start %F %T'

python3 -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 48 --degree 5 --n-colors 4 --shuffle-adj --fall \
    --d-model 256 --nhead "$NHEAD" --dim-feedforward "$FF" \
    --num-layers "$LAYERS" --arch-mode "$ARCH" --seed "$SEED" \
    --n-train-instances 240000 --epochs 50 \
    --n-val-instances 300 --batch-size 32 --n-per-bin 100 \
    --out-dir "$cell"

date +'cell end   %F %T'

TAG="d0256_h${HH}_L${LL}_ff${FFS}_lr1e-03_pre_shuf_fall${ATAG}"
if [ "$SEED" -ne 0 ]; then
    TAG="${TAG}_s$(printf '%02d' "$SEED")"
fi
CKPT="${cell}/${TAG}"
[ -f "${CKPT}/model.pt" ] || { echo "expected checkpoint missing: ${CKPT}"; ls -1 "$cell"; exit 1; }

echo
echo "=== W: what the checkpoint says it is (read back, not trusted) ==="
# arch_mode is load-bearing at EVAL time the way fall is (finding 43): if it
# were missing from config.json, the eval below would rebuild a linear cell
# with a ReLU and print a plausible number with no error at all.
python3 - "${CKPT}/config.json" "$ARCH" "$LAYERS" "$NHEAD" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
want_arch, want_L, want_h = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
assert c["shuffle_adj"] is True, "LEAK: this cell is not shuffle_adj"
assert c["fall"] is True, "wrong ensemble: fall is not set"
assert c["arch_mode"] == want_arch, (c["arch_mode"], want_arch)
assert c["num_layers"] == want_L and c["nhead"] == want_h
assert c["max_len_margin"] == 2 * c["n_nodes"] + 2, c["max_len_margin"]
print("arch_mode=%s L=%d nhead=%d ff=%d  trainable %s params  max_len=%d (margin %d)"
      % (c["arch_mode"], c["num_layers"], c["nhead"], c["dim_feedforward"],
         format(c.get("n_trainable_params", -1), ","), c["max_len"], c["max_len_margin"]))
PY

echo
echo "=== the transition curve for this cell (N1-N5) ==="
# Same 14-point grid and the same default --seed-start as Stage A and the 11-02
# fall scan, so these curves are paired with Stage A's L=6 control point by
# point. Bottoms out at 0.33 because the fall ensemble's rho_min at N=48 is
# 0.332 +- 0.024.
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.33 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== W: plateau escape (trainability, NOT capacity -- see watch items) ==="
python3 - "${CKPT}/history.json" <<'PY'
import json, sys
# 1.64 is where finding 44's L=4/ff=1024/240k cell sat for the whole run at
# this exact budget. A cell that does not get below it has not shown that its
# architecture cannot represent DD; it has shown it did not train in 375k steps.
PLATEAU = 1.64
h = json.load(open(sys.argv[1]))
best = min(h, key=lambda r: r["nll"])
final_train = h[-1]["train_nll"]
tail = [r["train_nll"] for r in h[-5:]]
slope = (tail[-1] - tail[0]) / max(len(tail) - 1, 1)
print("epochs %d  final train_nll %.4f  final val_nll %.4f  best val_nll %.4f at epoch %d"
      % (len(h), final_train, h[-1]["nll"], best["nll"], best["epoch"]))
print("overfit gap (final - best val) = %+.4f" % (h[-1]["nll"] - best["nll"]))
print("last-5-epoch train_nll slope = %+.5f/epoch (negative = still descending)" % slope)
print("PLATEAU_ESCAPE=%s  (final train_nll %.4f vs finding 44's plateau %.2f)"
      % ("yes" if final_train < PLATEAU else "NO", final_train, PLATEAU))
PY
date +'task end  %F %T'
