#!/bin/bash
#SBATCH --job-name=sud2MinArchI
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=07:00:00
#SBATCH --array=0-2%3
#SBATCH --output=%x_%A_%a.out
#
# STAGE I: BISECT THE WIDTH TRAINABILITY FLOOR. ONE KNOB -- d160.
#
#   sbatch scripts/slurm/sudoku2_n48_minarch_I_2026-11-05.sbatch
#
# WHY THIS JOB EXISTS, AND WHY IT IS SUBMITTED BEFORE STAGE H IS SCORED.
# Stage H (51223007) is still running, but the half of it that decides what to
# run next has already landed, and it landed on the side the header did NOT
# bet on. EXACT STATE AT SUBMISSION, written down so this header cannot be
# re-read later as having known more than it did:
#
#   task  rung  cell                 state at submission
#   t00   H1    d128 L=12 s0         COMPLETE: CENSORED, mean_solve 0.0411,
#                                    best solve 0.422, final train_nll 1.6768,
#                                    NEVER below 1.64 in 50 epochs
#   t02   H1    d128 L=12 s1         COMPLETE: CENSORED, mean_solve 0.0316,
#                                    best solve 0.357, final train_nll 1.6703,
#                                    NEVER below 1.64 in 50 epochs
#   t01   H2    d192 L=12 s0         RUNNING, epoch 40/50, train_nll 1.5670,
#                                    val_acc 0.4842 -- ESCAPED, and descending
#                                    at about 0.0007/epoch over epochs 38-40
#   t03,t04,t05                      running or pending, nothing on disk
#
# So at L=12 and 375k steps, d128 does not train on 2 of 2 seeds and d192 does
# train on the one seed far enough along to say. THE TRAINABILITY FLOOR IS
# BRACKETED IN [128, 192], A FACTOR OF 1.5, AND NOTHING IN THE GRID SITS
# INSIDE IT. d160 bisects it. That is the whole job.
#
# THIS IS ON-BRIEF, NOT A DETOUR. The brief asks for the smallest architecture
# that still learns direct deduction and says to treat the breaking point as
# the result. The breaking point is the thing this job measures. It is also
# the next rung down the ladder the project has been walking: if d160 trains
# and crosses, the standing minimum falls again; if it does not, the bracket
# closes to [128, 160] and the breaking point is located to within 1.25x.
#
# WHAT WOULD MAKE THIS JOB WASTED, STATED HONESTLY. If H2 comes back censored
# or lands above 0.740, then d192 trains but does not solve well enough to be
# a standing minimum, and d160 cannot either -- the escape-bracket result
# survives but the ladder result does not. That is a real risk and it is taken
# knowingly: the width ladder at L=6 moved only 0.003 across a 3x parameter
# range (0.763 at d256, 0.760 at d384), so the prior that d192 at L=12 lands
# near E2's 0.605 rather than back in the band is strong but not measured.
#
# =====================================================================
# THE GRID. 3 cells, all N=48 d=5 q=4, --fall --shuffle-adj, NUM_LAYERS 12,
# ff 1024 (unused by no_mlp but written for key compatibility), nhead 4,
# pre-norm, no warmup, lr 1e-3, batch 32, 240k instances x 50 epochs = 375k
# gradient steps, n_val 300. Seeds 0, 1, 2.
#
#   rung   d_model  arch     seeds  trainable (est)  vs hub   asks
#   I1       160    no_mlp   0,1,2    1,345,018      3.64x    the bisector
#
# Trainable count from the same arithmetic stage H used and stage H's own
# config.json confirmed: 4*d^2 + 8*d per layer, 630*d + 58 fixed. At d160 that
# is 100,858 + 103,680*12 = 1,345,018. The assertion block reads the real
# number back out of config.json and the write-up quotes THAT; stage D's
# header estimated d384 no_mlp at "~3.6M" against a built 3,799,354.
#
# ONE KNOB MOVES. I1 differs from H2 in d_model alone, and from H1 in d_model
# alone, and from E2 (d256 L=12 no_mlp, rho_50 0.605, 3,331,642) in d_model
# alone. Depth, arch, heads, warmup, budget, grid and seed-start are all E2's,
# so no new control cell is needed: this job has three paired partners already
# on disk or in flight at 128, 192 and 256.
#
# =====================================================================
# PRE-REGISTERED, written and committed before any I cell landed and while
# 51223007 was still running. git log is the timestamp.
#
# METRIC OF RECORD: rho_50, the realized hint density where solve rate crosses
# 0.5 on the 14-point x 400-instance grid below, by linear interpolation.
# LOWER IS BETTER. Censored rho_50 is reported as "> 0.95", never as a number.
# Secondary metrics that never censor: mean_solve_rate (unweighted mean over
# the 14 bins) and solve_at_floor (solve rate at rho = 0.3333).
#
# SCORED AGAINST THE FROZEN BAR stages A-E were scored against: s* = 0.1484 in
# rho_50 and 0.2918 in mean_solve. NOT RECOMPUTED. d160 is a non-default
# d_model so this rung keys as the 6-tuple (12,4,'no_mlp',160,1024,0), which
# cannot collide with anything run: no_mlp has been run at d128 (L=6 and L=12),
# d192 (L=12), d256 and d384, and never at d160. SCAN_I stays OUT of SCAN_DIRS
# for the same reason SCAN_G and SCAN_H do.
#
#  I1 -- THE TRAINABILITY PREDICTION, AND IT IS A NUMBER, NOT A DIRECTION.
#      This is the primary and it is registered quantitatively because the
#      bracket's two endpoints are measured. Final train_nll at L=12, no_mlp,
#      375k steps, as a function of width:
#
#          d128   1.6736   (mean of 1.6768, 1.6703; 0 of 2 escapes)
#          d160   ???      <- this job
#          d192   ~1.560   (t01 extrapolated from 1.5670 at epoch 40 at its
#                           own 0.0007/epoch tail rate -- AN EXTRAPOLATION OF
#                           A RUNNING CELL, not a measurement)
#          d256   1.5330   (mean of 1.5372, 1.5243, 1.5374; 3 of 3 escapes)
#
#      PREDICTION: I1 ESCAPES finding 44's 1.64 plateau on at least 2 of 3
#      seeds, with final train_nll in [1.600, 1.620]. The interval is the two
#      honest interpolations of the bracketing segment and nothing fancier:
#      linear in 1/d gives 1.605, linear in d gives 1.617. Both clear 1.64 by
#      0.023 to 0.035, which is why this is registered as an escape rather
#      than as a coin flip, and the interval is quoted so that being right
#      about the direction and wrong about the number is still visible.
#      FALSIFIER, AND IT IS THE INTERESTING OUTCOME: 0 of 3 escapes, or a
#      final train_nll above 1.64. That says the 128->192 transition is a
#      CLIFF rather than a smooth curve in width -- the model either has
#      enough residual stream to find the algorithm in 375k steps or it does
#      not, with no graded middle -- and by the brief's own instruction that
#      breaking point is the result, not a failed cell. It would also mean
#      the two bracketing interpolations are both wrong, which is worth
#      knowing before any further width rung is designed.
#
#  I2 -- THE LADDER, conditional on I1 escaping and reported second.
#      PREDICTION: rho_50(I1) lies between rho_50(H2) and rho_50(H1), i.e. the
#      width ladder at L=12 is monotone, as the width ladder at L=6 is and as
#      the complete model's is (0.592 at d256, 0.752 at d192, CENSORED at
#      d128).
#      FALSIFIER OF MONOTONICITY: I1 outside [min, max] of its two measured
#      neighbours by more than s* = 0.1484, reported ahead of everything else.
#
#  I3 -- THE STANDING-MINIMUM RULE, unchanged from stage H's H3 and NOT
#      restated in softer terms because this job's rung is smaller. A rung
#      replaces the standing minimum ONLY IF it has 3 of 3 crossing seeds AND
#      a median rho_50 at or below 0.740. Two of three is not enough. If I1
#      clears both, the standing minimum falls from E2's 3,331,642 to
#      1,345,018 -- 3.64x below the hub and 2.48x below E2.
#      TIE-BREAK, stated in advance: a median anywhere in [0.72, 0.740] is
#      scored as "did not clear the hub". A boundary value is not evidence.
#
#  I4 -- OUT-OF-SAMPLE TEST OF THE SUFFICIENCY RESULT. Free, no extra cells.
#      scripts/acc_vs_solve.py fits the depth ladder alone and holds at 16/16
#      well-determined held-out rungs over 37 cells, worst discriminating miss
#      0.0099 against a leave-one-out bar of 0.0423. PREDICTION: I1 lands on
#      the curve within 0.0423.
#      FALSIFIER: it falls off, which would be the grid's first evidence of an
#      architectural change doing something SOLVE-SPECIFIC, and would be
#      reported ahead of I1.
#
#  W -- WATCH ITEMS.
#      FINDING 51 AND WHAT A 2-OF-3 MEANS AT 50 EPOCHS. Stage G (51194118)
#      doubled the budget on the attention-only rungs and found that the d256
#      L=6 rung's 2-of-3 was ONE UNTRAINED SEED, not three different answers:
#      at 100 epochs seed 1 went from censored (best solve 0.465, train_nll
#      1.688) to rho_50 0.761 and best solve 0.963, crossing seeds 2/3 -> 3/3
#      and the rung's mean_solve spread 0.2671 -> 0.0041, a 65x collapse.
#      So a 2-of-3 here must be written as "one seed is not finished at this
#      budget", never as a property of the architecture, and I3 deliberately
#      refuses to move the standing minimum on it.
#      AND THE SAME FINDING CUTS THE OTHER WAY AGAINST I1's FALSIFIER: stage G
#      ALSO ran d128 no_mlp at L=6 for 750k steps and got 0 of 3 escapes, so
#      the one place in this grid where a wider budget was spent on d128, it
#      bought nothing. A d160 that does not escape in 375k steps is therefore
#      NOT automatically "needs longer"; it is the second independent width at
#      which the budget argument has been tried and failed.
#      GRAMMAR vs DEDUCTION. truncated_rate <= 0.01 and well_formed_rate >=
#      0.99 at every rho for every cell with solve > 0.1.
#      LEAK. Every cell trains and evaluates with --shuffle-adj, and the
#      assertion block re-reads shuffle_adj, fall, arch_mode, num_layers,
#      nhead, d_model, dim_feedforward, warmup_epochs and max_len_margin out
#      of each written config.json rather than trusting the command line
#      (findings 16/17/23).
#
#  SCORING RULE, fixed now so it cannot be chosen afterwards: every comparison
#  is median-vs-median against the frozen s* above. Smaller than s* is "not
#  separable at this budget", never a result. Three seeds and the spread
#  quoted, always (findings 41/42/47).
#
# =====================================================================
# COST. Measured on this exact depth and arch: d128 L=12 ran 03:27:40 and
# 03:25:39 per cell on stage H tasks 0 and 2; d256 L=12 ran 4h50m-4h51m on
# stage E. d160 is between them, so ~3h50m is the expectation and --time
# 07:00:00 is stage E's own limit with 31% headroom on the slower end of the
# bracket. 3 cells x ~4 h = ~12 GPU-h, ~4 h wall at 3 concurrent.
#
# THROTTLED TO 3 CONCURRENT. Checked with ListAgents and squeue before
# submitting: four peer sessions exist, none is running a spin_gnn job, and
# the only sud2MinArch jobs in the queue are my own 51223007. sganguli's two
# GPU nodes showed 15 and 10 idle CPUs at 4 CPUs/job, so this fills idle group
# capacity alongside stage H rather than displacing it.
#
# ENVIRONMENT. Module stack only -- ${SCRATCH}/spin_gnn_venv is long dead to
# the 90-day purge -- and python3 explicitly, since bare `python` is the system
# 2.7. The smoke check turns an environment failure into a 30-second job
# instead of a four-hour one that dies in the eval.

set -euo pipefail

# task: seed. d_model is fixed at 160 on this job -- it is the only knob.
SPECS=( "160:0"  "160:1"  "160:2" )
TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
IFS=: read -r DMODEL SEED <<< "${SPECS[$TASK]}"
# 10# forces base 10, as in stages C, D, E and H.
DMODEL=$((10#$DMODEL)); SEED=$((10#$SEED))
LAYERS=12
FF=1024
NHEAD=4
WARMUP=0
ARCH=no_mlp
LL=$(printf '%02d' "$LAYERS")
HH=$(printf '%02d' "$NHEAD")
TT=$(printf '%02d' "$TASK")
DD=$(printf '%04d' "$DMODEL")
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
assert "no_mlp" in ARCH_MODES, ARCH_MODES
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n48_minarch_I_2026-11-05"
# One directory PER TASK: sudoku_hparam_sweep appends to <out-dir>/sweep.csv and
# three cells run concurrently, so a shared directory would interleave rows
# from several writers into one file.
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

# Tag order follows sudoku_hparam_sweep.py exactly: arch, then schedule, then
# warmup, then seed, each suffixed only when non-default.
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
# 4*d^2 + 8*d per layer, 630*d + 58 fixed -- the same arithmetic stage H used
# and its config.json confirmed at both d128 (879,418) and d192 (1,908,922).
d, L = c["d_model"], c["num_layers"]
print("arch_mode=%s L=%d nhead=%d d_model=%d ff=%d warmup=%d  trainable %s params "
      "(header predicted %s)  max_len=%d (margin %d)"
      % (c["arch_mode"], L, c["nhead"], d, c["dim_feedforward"],
         int(c.get("warmup_epochs", 0) or 0), format(c.get("n_trainable_params", -1), ","),
         format((630 * d + 58) + (4 * d * d + 8 * d) * L, ","),
         c["max_len"], c["max_len_margin"]))
PY

echo
echo "=== the transition curve for this cell (I1) ==="
# Same 14-point grid and the same default --seed-start as stages A-H, so these
# curves are paired with E2 (d256), H2 (d192) and H1 (d128) point by point.
# Bottoms out at 0.33 because the fall ensemble's rho_min at N=48 is
# 0.332 +- 0.024.
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.33 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== I1: plateau escape -- THE registered prediction on this job ==="
python3 - "${CKPT}/history.json" <<'PY'
import json, sys
# 1.64 is where finding 44's L=4/ff=1024/240k cell sat for the whole run at
# this exact budget. This job registered a NUMBER, not a direction: final
# train_nll in [1.600, 1.620], from the two interpolations of the measured
# bracket (d128 -> 1.6736 at 0/2 escapes, d192 -> ~1.560 escaping). Printing
# the prediction next to the measurement is the point.
PLATEAU, LO, HI = 1.64, 1.600, 1.620
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
# I4 is scored off this line together with history.json's accuracy field: the
# sufficiency curve is fitted on validation TOKEN ACCURACY, not on NLL.
print("final val token accuracy = %.4f" % h[-1]["accuracy"])
print("first epoch under the plateau = %s" % (first if first is not None else "never"))
print("PLATEAU_ESCAPE=%s  (final train_nll %.4f vs finding 44's plateau %.2f)"
      % ("yes" if final_train < PLATEAU else "NO", final_train, PLATEAU))
print("REGISTERED_INTERVAL=[%.3f, %.3f]  observed %.4f  -> %s"
      % (LO, HI, final_train,
         "inside" if LO <= final_train <= HI else
         "BELOW (better than predicted)" if final_train < LO else
         "ABOVE (worse than predicted)"))
PY
date +'task end  %F %T'
