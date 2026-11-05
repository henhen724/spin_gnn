#!/bin/bash
#SBATCH --job-name=sud2MinArchH
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=07:00:00
#SBATCH --array=0-5%3
#SBATCH --output=%x_%A_%a.out
#
# STAGE H: THE CHEAP END OF THE DEPTH RESULT. ONE KNOB FROM E2 -- WIDTH.
#
#   sbatch scripts/slurm/sudoku2_n48_minarch_H_2026-11-05.sbatch
#
# WHY THIS JOB EXISTS, AND IT IS A DIFFERENT JOB FROM THE ONE I WOULD HAVE
# WRITTEN YESTERDAY. Stage E closed this morning (51171824, 6/6) and E2
# FALSIFIED the barrier reading the whole grid had been organised around:
#
#   rung                       rho_50   mean_solve   trainable   esc   cross
#   L=12 full                  0.385      0.8163     9,638,458   3/3    3/3
#   L=8  full                  0.471      0.7111     6,479,418   3/3    3/3
#   L=6  full  (the hub)       0.592      0.5389     4,899,898   3/3    3/3
#   L=12 no_mlp  (E2)          0.605      0.5214     3,331,642   3/3    3/3
#   ------------------------------------------------- hub + s* = 0.740 -------
#   E1   L=8  no_mlp           0.749      0.3214     2,274,874   3/3    3/3
#   D1   L=6  d192 full        0.752      0.3217     3,381,562   3/3    3/3
#   L=3  full                  0.757      0.3102     2,530,618   3/3    3/3
#   D2   L=6  d384 no_mlp      0.760      0.3155     3,799,354   3/3    3/3
#   L=6  no_mlp (std. minimum) 0.763      0.3039     1,746,490   2/3    2/3
#   L=6  d128 no_mlp          CENSORED    0.0291       480,058   0/3    1/3
#
# AN ATTENTION-ONLY STACK WITH NO MLP ANYWHERE, GIVEN TWICE THE HUB'S DEPTH,
# LANDS AT 0.605 AGAINST THE HUB'S 0.592 -- A GAP OF 0.013, which is 0.09 s*,
# on 3 of 3 crossing seeds and 3 of 3 plateau escapes, at 1.47x FEWER
# parameters than the hub. The 0.76 pile-up is not a floor. Depth bought the
# MLP back where width (D2, 1.5x the hub's stream) did not.
#
# THAT MAKES THE CHEAP END THE INTERESTING END AGAIN, and it is the brief's
# actual question -- the smallest architecture that still learns direct
# deduction, not the best solver. E2 is a WIN on all-seed solving (2/3 -> 3/3
# crossing) and on rho_50 (0.763 -> 0.605) but a LOSS on size: it is 1.91x
# BIGGER than the standing minimum it replaces. Stage H asks whether the depth
# that bought the descent survives being made narrow, which is the only way
# this line of work produces a smaller model rather than a better one.
#
# =====================================================================
# THE GRID. 6 cells, all N=48 d=5 q=4, --fall --shuffle-adj, NUM_LAYERS 12,
# ff 1024 (unused by no_mlp but written for key compatibility), nhead 4,
# pre-norm, no warmup, lr 1e-3, batch 32, 240k instances x 50 epochs = 375k
# gradient steps, n_val 300. Three seeds (0,1,2) per rung.
#
#   rung   d_model  arch     seeds  trainable (est)  vs hub   asks
#   H1       128    no_mlp   0,1,2      879,418      5.6x     the cheap end
#   H2       192    no_mlp   0,1,2    1,908,922      2.6x     the bisector
#
# Trainable counts are arithmetic, not guesses, and the arithmetic is now
# pinned by three measured attention-only rungs rather than extrapolated:
# per layer is 4*d^2 + 8*d (263,168 + 1,024 = 264,192 at d256, confirmed
# against 1,746,490 at L=6 and 3,331,642 at L=12), and the fixed part is
# linear in d at 630*d + 58 (80,698 at d128, confirmed against the measured
# 480,058 for d128 L=6; 161,338 at d256). The assertion block reads the real
# number back out of config.json and the write-up quotes THAT -- stage D's
# header estimated d384 no_mlp at "~3.6M" against a built 3,799,354, and
# estimates in a header are not results.
#
# ONE KNOB MOVES. Both rungs differ from E2 in d_model ALONE. Everything else
# -- depth, arch, heads, warmup, budget, grid, seed-start -- is E2's, so E2 is
# the paired partner at full width and no new control is needed. The other
# paired partner already exists at the bottom: d128 no_mlp at L=6 (480,058,
# CENSORED, 0/3 escapes), which is this job's rung with the depth taken away.
#
# WHY d128 AND d192 AND NOT d64. d128 is the width at which the attention-only
# stack has already been measured at L=6, so H1 has a same-width partner as
# well as a same-depth one and sits at the intersection of two measured lines.
# d192 bisects 128 and 256 and is the width where the COMPLETE model is
# measured (D1), so if H1 fails there is a cell between it and E2 rather than
# a two-to-one jump. d64 is not in this job: the complete model at d64 scored
# mean_solve 0.0211 with 0/3 escapes, so a d64 attention-only cell is a
# trainability experiment, not an architecture one, and it would come back
# unscoreable.
#
# =====================================================================
# PRE-REGISTERED, written before submission and before stage G's tasks 3-5
# finished (51194118 tasks 3/4/5 were 3h16m, 3h13m and 1h52m into their runs
# at the time this file was committed; git log is the timestamp).
#
# METRIC OF RECORD: rho_50, the realized hint density where solve rate crosses
# 0.5 on the 14-point x 400-instance grid below, by linear interpolation.
# LOWER IS BETTER. Censored rho_50 is reported as "> 0.95", never as a number.
# Secondary metrics that never censor: mean_solve_rate (unweighted mean over
# the 14 bins) and solve_at_floor (solve rate at rho = 0.3333).
#
# SCORED AGAINST THE FROZEN BAR stages A-E were scored against: s* = 0.1484 in
# rho_50 and 0.2918 in mean_solve. NOT RECOMPUTED. These rungs carry a
# non-default d_model, so rung_key() gives them their own 6-tuple keys and they
# stay out of the pooled bar automatically, as stages C and D did and as stage
# E's defaulted rungs did not. SCAN_H is still kept out of SCAN_DIRS for the
# same reason the others are: a dir is added to the default list only once
# every verdict scored against the old bar has been re-read.
#
#  H1 -- ATTENTION-ONLY, L=12, d128. The cell this job is for.
#      Prediction under (D) DEPTH IS THE CURRENCY, which is the reading E2
#      just installed and which this header bets on at MODERATE confidence:
#      rho_50 median at or below 0.740 (hub + s*) on at least 2 of 3 crossing
#      seeds. That would put a model of ~879,418 parameters -- 5.6x below the
#      hub and 2.0x below the standing minimum -- below the 0.76 rung, and the
#      standing minimum moves there.
#      Prediction under (W) WIDTH IS A SEPARATE FLOOR: rho_50 median back in
#      [0.72, 0.80], or censored. That says depth only pays above some width,
#      the two resources are not interchangeable, and the cheap end is bounded
#      by the residual stream rather than by the MLP or by depth.
#      TIE-BREAK, stated in advance for the same reason stage E stated one:
#      a median of exactly 0.740, or anywhere in [0.72, 0.740], is scored as
#      (W) and written as "did not clear the hub". (D) is the claim that needs
#      the evidence here and a boundary value is not evidence.
#
#  H2 -- ATTENTION-ONLY, L=12, d192. The bisector, and a monotonicity check.
#      Prediction: rho_50(H2) lies between rho_50(H1) and rho_50(E2) = 0.605,
#      i.e. the width ladder at L=12 is monotone, as the width ladder at L=6
#      is (0.763 at d256, CENSORED at d128) and as the complete model's is
#      (0.592 at d256, 0.752 at d192, CENSORED at d128).
#      FALSIFIER OF MONOTONICITY: H2 outside [min, max] of its two neighbours
#      by more than s* = 0.1484. A non-monotone width ladder at L=12 would
#      mean width and depth interact in a way no ladder in this grid has
#      shown, and it would be reported ahead of H1.
#
#  H3 -- THE STANDING-MINIMUM RULE, fixed now so it cannot be chosen after the
#      numbers land. A rung replaces the standing minimum ONLY IF it has 3 of 3
#      crossing seeds AND a median rho_50 at or below 0.740. Two of three is
#      not enough: the rung it would replace (L=6 no_mlp, 0.763) is itself a
#      2-of-3 rung and that is precisely why it was never a satisfying answer.
#      By this rule E2 (3/3, 0.605, 3,331,642) is the standing minimum as of
#      this morning even though it is BIGGER than the rung it replaces, and
#      "smallest architecture that still learns direct deduction" is read as
#      smallest among rungs that actually learn it on every seed.
#
#  H4 -- OUT-OF-SAMPLE TEST OF THE SUFFICIENCY RESULT. Free, no extra cells.
#      scripts/acc_vs_solve.py fits the DEPTH LADDER ALONE and now holds at
#      16/16 well-determined held-out rungs, 34 cells, worst discriminating
#      miss 0.0099 against a leave-one-out bar of 0.0423 -- stage E's two rungs
#      went in this morning and landed at +0.0004 and -0.0029. Prediction: both
#      stage H rungs land on the curve within 0.0423.
#      FALSIFIER: either falls off, which would be the grid's first evidence of
#      an architectural change doing something SOLVE-SPECIFIC rather than
#      moving the model along one quality axis, and would be reported ahead of
#      H1 and H2.
#
#  W -- WATCH ITEMS.
#      TRAINABILITY IS AGAIN THE MOST LIKELY WAY THIS JOB PRODUCES NOTHING,
#      and at d128 it is worse than it was on stage E. Attention-only at d128
#      and L=6 escaped finding 44's 1.64 plateau on 0 of 3 seeds. What makes
#      the cell worth running anyway is that DEPTH HAS MONOTONICALLY IMPROVED
#      ESCAPES on this axis: no_mlp at d256 escapes 0/3 at L=4, 2/3 at L=6,
#      3/3 at L=8 and 3/3 at L=12. The registered expectation is therefore
#      >= 2 of 3 escapes at H1. If H1 comes back 0/3 it is written as "did not
#      train in 375k steps" and is NOT evidence for (W) -- that is the trap
#      stage C's P2 fell into. The budget is deliberately NOT raised to
#      compensate; stage G is the budget arm and it is a separate job.
#      AND THE INDEPENDENT REASON TO EXPECT 0/3 ANYWAY: scripts/escape_timing.py
#      Q3 found there is no knee -- all 14 late escapes in the grid crossed
#      from train NLL 1.640-1.652 and 13 of 14 were within five epochs of the
#      plateau at their own prior rate -- and it puts d128 no_mlp's three
#      seeds a median 87 epochs short at their own tail slope. Stage H does not
#      move the budget, so Q3 predicts H1 FAILS TO ESCAPE. That contradicts the
#      monotone-depth expectation above, both are on record, and whichever way
#      it lands one of them is wrong in print.
#      GRAMMAR vs DEDUCTION. truncated_rate <= 0.01 and well_formed_rate >=
#      0.99 at every rho for every cell with solve > 0.1.
#      LEAK. Every cell trains and evaluates with --shuffle-adj, and the
#      assertion block re-reads shuffle_adj, fall, arch_mode, num_layers,
#      nhead, d_model, dim_feedforward, warmup_epochs and max_len_margin out of
#      each written config.json rather than trusting the command line
#      (findings 16/17/23).
#      SILENT KNOBS. d_model is non-default on both rungs, so they key as
#      6-tuples (12,4,'no_mlp',128,1024,0) and (12,4,'no_mlp',192,1024,0).
#      Neither can collide: no_mlp has been run at d128 only at L=6, at d384
#      only at L=6, and at d192 never.
#
#  SCORING RULE, fixed now so it cannot be chosen afterwards: every comparison
#  is median-vs-median against the frozen s* above. Smaller than s* is "not
#  separable at this budget", never a result. Three seeds per rung and the
#  spread quoted, always (findings 41/42/47).
#
# =====================================================================
# COST. Measured: attention-only at d256 L=12 ran 4h50m-4h51m per cell on
# stage E (51171824 tasks 1, 3, 5). The per-layer matmul cost falls as d^2 but
# attention's sequence-length term falls only as d, and no attention-only cell
# has been timed below d256 at this depth, so the law is NOT extrapolated on a
# guess: --time 07:00:00 is stage E's own limit, which covered its L=12 cells
# with 31% headroom, and H1/H2 cannot be slower than the cells it was measured
# on. 6 cells x ~4 h = ~24 GPU-h, ~8 h wall at 3 concurrent.
#
# THROTTLED TO 3 CONCURRENT. sganguli showed 8 of 16 GPUs idle at submission
# with stage G (51194118) holding 3 of the busy ones, so this fills idle group
# capacity rather than competing with its own earlier stage. Checked with
# ListAgents and squeue first: no other session is running a spin_gnn job.
#
# ENVIRONMENT. Module stack only -- ${SCRATCH}/spin_gnn_venv is long dead to
# the 90-day purge -- and python3 explicitly, since bare `python` is the system
# 2.7. The smoke check turns an environment failure into a 30-second job
# instead of a five-hour one that dies in the eval.

set -euo pipefail

# task: d_model:seed  (num_layers is fixed at 12 on this job -- it is the knob
# stage E already moved, and the only knob moving here is the width)
# INTERLEAVED BY SEED, not grouped by rung: at %3 the first wave is tasks 0-2,
# so this order returns a first seed of BOTH rungs in the first wave instead of
# all of H1 and nothing of H2. The grid table above is the authority on what
# the rungs are; this is only the order they run in.
SPECS=(
  "128:0"  "192:0"  "128:1"
  "192:1"  "128:2"  "192:2"
)
TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
IFS=: read -r DMODEL SEED <<< "${SPECS[$TASK]}"
# 10# forces base 10, as in stages C, D and E. Neither field can be a leading-
# zero literal here, but the guard costs nothing and the one time it was left
# out it would have bitten on "08".
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

OUT="${SCRATCH}/spin_gnn/sudoku_n48_minarch_H_2026-11-05"
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
# The header's arithmetic, now pinned by three measured attention-only rungs:
# 4*d^2 + 8*d per layer, 630*d + 58 fixed. Printed beside the real number so a
# wrong estimate is visible in the log rather than surviving into the write-up.
d, L = c["d_model"], c["num_layers"]
print("arch_mode=%s L=%d nhead=%d d_model=%d ff=%d warmup=%d  trainable %s params "
      "(header predicted %s)  max_len=%d (margin %d)"
      % (c["arch_mode"], L, c["nhead"], d, c["dim_feedforward"],
         int(c.get("warmup_epochs", 0) or 0), format(c.get("n_trainable_params", -1), ","),
         format((630 * d + 58) + (4 * d * d + 8 * d) * L, ","),
         c["max_len"], c["max_len_margin"]))
PY

echo
echo "=== the transition curve for this cell (H1-H2) ==="
# Same 14-point grid and the same default --seed-start as stages A-E, so these
# curves are paired with E2 (d256 L=12 no_mlp) and with stage C's d128 no_mlp
# at L=6 point by point. Bottoms out at 0.33 because the fall ensemble's
# rho_min at N=48 is 0.332 +- 0.024.
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.33 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== W: plateau escape -- THE deciding watch item on this job ==="
python3 - "${CKPT}/history.json" <<'PY'
import json, sys
# 1.64 is where finding 44's L=4/ff=1024/240k cell sat for the whole run at
# this exact budget. A cell that does not get below it has not shown that its
# architecture cannot represent DD; it has shown it did not train in 375k
# steps. On THIS job two registered predictions disagree about which happens:
# depth has monotonically improved escapes on the attention-only axis
# (0/3 at L=4, 2/3 at L=6, 3/3 at L=8, 3/3 at L=12, all d256), while
# escape_timing.py Q3 puts d128 no_mlp a median 87 epochs short at its own
# tail slope and this job does not raise the budget.
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
# H4 is scored off this line together with history.json's accuracy field: the
# sufficiency curve is fitted on validation TOKEN ACCURACY, not on NLL.
print("final val token accuracy = %.4f" % h[-1]["accuracy"])
print("PLATEAU_ESCAPE=%s  (final train_nll %.4f vs finding 44's plateau %.2f)"
      % ("yes" if final_train < PLATEAU else "NO", final_train, PLATEAU))
PY
date +'task end  %F %T'
