#!/bin/bash
#SBATCH --job-name=sud2MinArch
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-35%8
#SBATCH --output=%x_%A_%a.out
#
# HOW LITTLE MACHINERY DOES LEARNING DIRECT DEDUCTION ACTUALLY REQUIRE --
# stage A: walk every axis DOWNWARDS from L=12/ff=1024 until it breaks.
#
#   sbatch scripts/slurm/sudoku2_n48_minarch_2026-11-04.sbatch
#
# THE OBJECTIVE CHANGED ON 11-04 AND THIS GRID IS THE FIRST MEASUREMENT UNDER
# IT. Success is no longer a lower rho_50. There is already a model that solves
# the leak-free task; the question now is how small, and how analyzable, the
# thing that does it can be. A small model that solves it beats a large one,
# and one that can be written down in closed form beats both. So the grid below
# removes machinery -- layers, heads, the MLP nonlinearity, the training of the
# MLP, the MLP itself -- one kind at a time, and the BREAKING POINT is the
# result, not a failure. Where deduction stops is the number being measured.
#
# WHY N=48 FALL AND NOT N=96 OR N=24. N=96 costs ~0.96 h per layer (finding 45
# elapsed times) and 36 cells of it would be ~250 GPU-h for a question that is
# about architecture, not scale. N=24 is cheap but its fall ensemble has never
# been calibrated, so a rho grid there would need its own calibration run
# first. N=48 fall has finding 44's calibrated reference grid, a measured
# structural floor (rho_min = 0.332 +- 0.024, 200 seeds), and it already
# BRACKETS the breaking point in depth: L=4 solves it and L=12 outruns the
# instance generator, so the interesting rungs are all inside L in [1, 4].
#
# =====================================================================
# THE GRID. 36 cells, all N=48 d=5 q=4, --fall --shuffle-adj, d_model 256,
# ff 1024, pre-norm, lr 1e-3 CONSTANT, batch 32, 240k instances x 50 epochs =
# 375k gradient steps, n_val 300. Three seeds everywhere, no exceptions.
#
#   tasks   L  nhead  arch_mode  seeds    est h ea   axis
#    0-2   12    4    full       0,1,2      5.2      depth ladder (the top rung
#    3-5    8    4    full       0,1,2      3.6       is finding 44's cell, so
#    6-8    6    4    full       0,1,2      2.8       the ladder is anchored to
#    9-11   4    4    full       0,1,2      2.0       an already-measured point)
#   12-14   4    4    linear     0,1,2      2.0      nonlinearity removed
#   15-17   4    4    rf_mlp     0,1,2      2.0      random features, literal
#   18-20   4    4    rf_all     0,1,2      2.0      random features, convex
#   21-23   4    4    no_mlp     0,1,2      2.0      attention only (the floor)
#   24-26   4    1    full       0,1,2      2.0      heads 4 -> 1
#   27-29   3    4    full       0,1,2      1.6      depth ladder, lower rungs
#   30-32   2    4    full       0,1,2      1.2
#   33-35   1    4    full       0,1,2      0.8
#
# TASKS 9-11 ARE THE HUB. L=4/ff=1024/full/nhead=4 is the control for the arch
# axis AND for the head axis AND a rung of the depth ladder, so it is measured
# once and compared against three ways. Everything else differs from it in
# exactly one respect -- one of hshunt's "change one thing at a time".
#
# ORDERED LONGEST-FIRST so that with --array throttled to 8 the five-hour L=12
# cells are already running while the eight-minute-per-layer rungs queue behind
# them, instead of the reverse. 8 of sganguli's 16 GPUs leaves half the
# partition for the rest of the group.
#
# WHAT EACH arch_mode TAKES AWAY (see SudokuReasoningModelConfig.arch_mode;
# every mode starts from the SAME initialization as full at a given seed, so
# each contrast is about training the weights, not about drawing them):
#   linear  identity MLP activation. Each block's MLP collapses to one linear
#           map and the attention softmax is the ONLY nonlinearity left.
#   rf_mlp  linear1 frozen at init in every block, linear2 trains. A 1024-wide
#           random ReLU bank read out by a trained linear map, per block.
#   rf_all  everything frozen but the readout. Softmax cross-entropy on fixed
#           features is CONVEX in the trained weights -- the one cell in this
#           grid that can be written down and analyzed rather than only
#           measured, which is why it is here even though it is expected to
#           fail outright.
#   no_mlp  the MLP write is identically zero. x = x + attn(norm1(x)) exactly.
#
# =====================================================================
# PRE-REGISTERED, written before submission.
#
# METRIC OF RECORD: rho_50, the realized hint density at which solve rate
# crosses 0.5 on the 14-point x 400-instance grid below, by linear
# interpolation between adjacent points. LOWER IS BETTER. --seed-start is left
# at its default, shared with finding 44's cells, so every curve in this job
# and every curve in that one sees the SAME 400 instances per rho and the
# comparisons are paired point by point.
#
# rho_50 CENSORS, so two secondary metrics that never do are reported
# alongside it and carry the comparison wherever a cell fails to cross 0.5:
#   mean_solve_rate  the unweighted mean of solve rate over all 14 bins
#   solve_at_floor   solve rate at rho = 0.33, next to the structural floor
# A censored rho_50 is reported as "> 0.95" and never as a number.
#
# THE YARDSTICK. Every rung is three seeds, reported as the MEDIAN with the
# spread (max - min) beside it. s* = the largest spread seen on any rung in
# this job, and no gap smaller than s* is called a difference anywhere below.
# This is not optional caution: finding 41/42 measured a same-architecture seed
# spread of 0.1627 at N=48 and 0.115-0.138 at N=96, LARGER than the one-rung
# depth gaps the earlier ladders were built from, and finding 47 then showed
# that term is not a schedule artifact and cannot be cheaply reduced (cosine
# decay made both cells worse, 0.679 -> 0.830 and 0.564 -> 0.726, because
# finding 45's cells are step-bound rather than converged). A single-seed rung
# is not a measurement of its architecture. The optimizer is therefore left
# EXACTLY as finding 44/45 had it -- constant lr 1e-3, no warmup -- because the
# one thing that has been tried instead made things worse.
#
#  M1 -- THERE IS A BREAKING DEPTH AND IT IS LOW. High confidence.
#      Definition: B = the largest L at which some seed fails to cross 0.5
#      anywhere in the grid. Prediction: B = 2, i.e. L=3 solves the task at
#      three seeds and L=2 does not at some seed. Mechanism: rho_50 ~ 0.33-0.68
#      corresponds to DD depth T_end up to 13.8 at N=48 fall, far more serial
#      rounds than there are blocks, so the model cannot be running one round
#      per layer and the depth it does need is an open empirical number.
#      FALSIFIER: L=1 solves it at all three seeds (no breaking depth inside
#      this ladder -- then one block suffices and the whole depth axis was
#      measuring something else), or B >= 6 (the task needs more depth than
#      finding 44's L=4 cell had, which would contradict finding 44 directly
#      and send me back to check for a leak before believing it).
#
#  M2 -- THE LADDER CAN RESOLVE A KNEE AT ALL. Moderate confidence, and the
#      precondition for reading M1 as anything but a yes/no.
#      Prediction: rho_50(L=2) - rho_50(L=4) >= 2 s*.
#      FALSIFIER: smaller than that. Then the depth gaps at this budget are
#      inside the seed term, the ladder locates no knee, and the honest report
#      is a breaking point bracketed to within a factor of two in L and nothing
#      finer. That is a real result about measurement cost, not a null.
#
#  M3 -- REMOVING THE NONLINEARITY COSTS SOMETHING. Moderate confidence.
#      Prediction: linear at L=4 is resolvably worse than full at L=4
#      (difference > s*), and most likely does not cross 0.5 at all.
#      FALSIFIER: linear ties full within s*. IF THAT HAPPENS IT IS THE
#      HEADLINE OF THE PROJECT, not a null result: a stack of linear maps with
#      attention softmaxes between them has no MLP nonlinearity anywhere, the
#      blocks compose into a single input-dependent linear operator, and the
#      model becomes analyzable in closed form -- which is exactly what hshunt
#      asked for. It would be followed up immediately rather than filed.
#
#  M4 -- RANDOM FEATURES ARE ENOUGH, THE LITERAL READING. Moderate confidence.
#      Prediction: rf_mlp ties full at L=4 within s*. Mechanism: at d_model=256
#      a frozen 1024-wide random ReLU bank is over-complete, so a trained
#      linear2 can synthesize essentially any smooth MLP the full cell would
#      have learned. Attention still trains here, so the training problem is
#      NOT convex and rf_mlp is not yet the analyzable object.
#      FALSIFIER: rf_mlp resolvably worse than full. Then the MLP's first layer
#      is learning task-specific features rather than supplying a basis, and
#      WHAT those features are becomes the next question.
#
#  M5 -- THE CONVEX CELL DOES NOT REACH THE GRID. High confidence that it
#      fails; the informative number is HOW FAR it gets.
#      Prediction: rf_all's rho_50 is censored at every seed and its
#      mean_solve_rate < 0.10. With attention frozen at init, nothing in the
#      network can learn to look at the right neighbours, and reading a
#      deduction off fixed features is the one thing a linear readout should
#      not be able to do.
#      FALSIFIER: mean_solve_rate >= 0.10. Then a convex problem on frozen
#      features does part of DD, and the entire analytic programme runs through
#      this cell -- it would become the main object of study at once.
#      Reported either way: mean_solve_rate, solve_at_floor, well_formed_rate
#      and token accuracy, since "how much of the GRAMMAR a convex readout
#      learns" is interesting even when the solve rate is zero.
#
#  M6 -- ATTENTION ONLY: NO REGISTERED PREDICTION. This cell is here to
#      bracket M3 and M4 from below rather than to test a belief. It is the
#      floor of the arch axis: if no_mlp solves the task, then nothing in the
#      MLP is needed at all and the minimal architecture is an attention stack;
#      if it fails while rf_mlp succeeds, the gap between them is exactly what
#      the MLP contributes, measured rather than argued.
#
#  M7 -- ONE HEAD IS ENOUGH. Moderate confidence.
#      Prediction: nhead 4 -> 1 at L=4 ties within s*. Evidence: finding 27
#      found no single critical head -- the worst single-head ablation left 84%
#      of performance -- so the heads look redundant rather than specialized.
#      FALSIFIER: nhead=1 resolvably worse. That would say the four heads carry
#      complementary work that ablation at test time cannot see because the
#      others compensate, and the head axis would need its own ladder.
#
#  W -- WATCH ITEMS, not hypotheses, but they decide how a failed rung is
#      WRITTEN UP, so they are fixed in advance too.
#      TRAINABILITY vs CAPACITY. Finding 44 saw L=4/ff=1024/240k plateau at
#      train NLL ~1.64 and never escape within 375k steps. Each cell therefore
#      prints an explicit plateau-escape flag (final train_nll against 1.64 and
#      the last-5-epoch slope). A rung where 0 of 3 seeds escape is a
#      TRAINABILITY result at this budget -- "this architecture did not train
#      here" -- and must NOT be reported as "this architecture cannot represent
#      DD". The two are distinguishable only by more steps or a better
#      optimizer, neither of which is in this job.
#      GRAMMAR vs DEDUCTION. truncated_rate <= 0.01 and well_formed_rate >=
#      0.99 at every rho for every cell with solve > 0.1, so a gap is deduction
#      and not finding 25's grammar failure. Small models are MORE likely to
#      fail on grammar, so this control matters more here than it did above.
#      LEAK. Every cell trains and evaluates with --shuffle-adj, and the
#      assertion block below re-reads shuffle_adj and fall out of each written
#      config.json rather than trusting the command line. Findings 16, 17 and
#      23 are why: a minimal architecture that scores well on the leaky task
#      has told us nothing, and the smaller the model the more tempting the
#      decoding shortcut becomes, so this is exactly the regime where that
#      failure mode bites hardest. max_len_margin is asserted at its auto value
#      2N+2 = 98 so no cell is quietly sized flush to its longest sequence.
#
#  SCORING RULE, fixed now so it cannot be chosen afterwards: every cross-cell
#  comparison is median-vs-median read against s*. Smaller than s* is "not
#  separable at this budget", never a result. Curve SHAPE within one checkpoint
#  is paired and needs no such caveat; curve LEVEL across checkpoints always
#  does. scripts/minarch_analysis.py scores M1-M7 by these rules and nothing
#  else.
#
# =====================================================================
# COST. Measured law from the N=48 fall scan at this exact budget (sacct
# 50760297: L=4 tasks 1:53-2:06, L=8 tasks 3:35-3:38, elapsed, eval included):
# 0.40 h per layer plus 0.40 h fixed. That gives the est h column; the arch
# cells are all L=4 and cost what the L=4 depth rung costs (rf_all and no_mlp
# are slightly cheaper in the backward pass, which is headroom, not a saving
# worth modelling). Sum ~82 GPU-h over 36 cells, ~11 h wall clock at 8
# concurrent. --time 08:00:00 is 54% headroom over the longest cell (5.2 h);
# sganguli allows 7 d. Instance generation is 2.1 ms each at N=48 fall, so
# 240k is 0.14 h per cell and is not in the critical path.
#
# ENVIRONMENT. Module stack only -- ${SCRATCH}/spin_gnn_venv is long dead to
# the 90-day purge -- and python3 explicitly, since bare `python` is the system
# 2.7. The smoke check turns an environment failure into a 30-second job
# instead of a five-hour one that dies in the eval.

set -euo pipefail

# task: L:nhead:arch_mode:seed    (longest-first, see the grid table above)
SPECS=(
  "12:4:full:0"    "12:4:full:1"    "12:4:full:2"
  "8:4:full:0"     "8:4:full:1"     "8:4:full:2"
  "6:4:full:0"     "6:4:full:1"     "6:4:full:2"
  "4:4:full:0"     "4:4:full:1"     "4:4:full:2"
  "4:4:linear:0"   "4:4:linear:1"   "4:4:linear:2"
  "4:4:rf_mlp:0"   "4:4:rf_mlp:1"   "4:4:rf_mlp:2"
  "4:4:rf_all:0"   "4:4:rf_all:1"   "4:4:rf_all:2"
  "4:4:no_mlp:0"   "4:4:no_mlp:1"   "4:4:no_mlp:2"
  "4:1:full:0"     "4:1:full:1"     "4:1:full:2"
  "3:4:full:0"     "3:4:full:1"     "3:4:full:2"
  "2:4:full:0"     "2:4:full:1"     "2:4:full:2"
  "1:4:full:0"     "1:4:full:1"     "1:4:full:2"
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

OUT="${SCRATCH}/spin_gnn/sudoku_n48_minarch_2026-11-04"
# One directory PER TASK: sudoku_hparam_sweep appends to <out-dir>/sweep.csv
# and 8 cells run concurrently, so a shared directory would interleave rows
# from several writers into one file.
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
echo "=== the transition curve for this cell (M1-M7) ==="
# Same 14-point grid and the same default --seed-start as the 11-02 fall scan,
# so these curves are paired with finding 44's point by point. Bottoms out at
# 0.33 because the fall ensemble's rho_min at N=48 is 0.332 +- 0.024.
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
# this exact budget. A rung that does not get below it has not shown that its
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
