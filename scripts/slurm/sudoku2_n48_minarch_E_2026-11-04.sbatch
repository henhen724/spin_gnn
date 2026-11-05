#!/bin/bash
#SBATCH --job-name=sud2MinArchE
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=07:00:00
#SBATCH --array=0-5%3
#SBATCH --output=%x_%A_%a.out
#
# STAGE E: IS rho_50 ~ 0.76 A BARRIER THAT NO ABLATED MODEL CAN CROSS?
# THE DEPTH ARM. STAGE D's D2 IS THE WIDTH ARM OF THE SAME QUESTION.
#
#   sbatch scripts/slurm/sudoku2_n48_minarch_E_2026-11-04.sbatch
#
# THE OBSERVATION THAT SETS THIS JOB, and it is not one row, it is the whole
# table at once. Every complete rung on disk, L=6 unless stated, N=48 d=5 q=4,
# --fall --shuffle-adj, 3 seeds, 240k x 50 epochs, the same 14-point grid:
#
#   rung                       rho_50   mean_solve   trainable   escapes
#   L=12 full                  0.385      0.8163     9,638,458     3/3
#   L=8  full                  0.471      0.7111     6,479,418     3/3
#   L=6  full  (the hub)       0.592      0.5389     4,899,898     3/3
#   ---------------------------------------------------- the 0.76 rung -------
#   L=3  full                  0.757      0.3102     2,530,618     3/3
#   L=6  d192 full (D1, 2/3)   0.759/0.752 0.3138/0.3266 3,381,562  2/2
#   L=4  full                  0.763      0.3084     3,320,378     3/3
#   L=6  no_mlp                0.763      0.3039     1,746,490     2/3
#   L=6  rf_mlp                0.763      0.3104     3,320,890     2/3
#   L=6  linear                0.765      0.3146     4,899,898     2/3
#   L=4  linear                0.765      0.3086     3,320,378     1/3
#   L=6  ff256 (C's P3)        0.740      0.3320     2,535,994     3/3
#   ---------------------------------------------------------------------------
#   L=2  full                  0.919      0.0807     1,740,858     0/3
#   L=4  no_mlp                0.921      0.0796     1,218,106     0/3
#   L=6  d128 full (C's P1)   CENSORED    0.0273     2,059,834     2/3
#   L=1  full                 CENSORED    0.0211       951,098     0/3
#
# NINE DISTINCT RUNGS SIT AT rho_50 0.740-0.765 AND mean_solve 0.304-0.332.
# They span 1,746,490 to 4,899,898 trainable parameters -- a factor of 2.8 --
# and they were produced by six different interventions: shallower depth, a
# narrower stream, a narrower MLP, a linearised MLP, a frozen random MLP, and
# no MLP at all. The band they land in is 0.025 wide in rho_50. The ACROSS-SEED
# SPREAD ON A SINGLE ONE OF THEM is wider than that: ff256's mean_solve spread
# is 0.1588, no_mlp d128's is 0.2075. So the nine rungs are not merely close,
# they are closer to each other than one rung is to itself across seeds.
#
# THE SAME THING HAPPENS ONE RUNG DOWN. L=2 full (0.919) and L=4 no_mlp (0.921)
# are the same number to three decimals at 1.74M and 1.22M parameters.
#
# TWO READINGS, AND THIS JOB SEPARATES THEM.
#
#   (A) BARRIER. rho_50 ~ 0.76 is a floor that any model missing a piece of the
#       complete architecture hits and cannot pass, no matter how much capacity
#       it is given back. The full model passes THROUGH that level at L=3-4 and
#       keeps descending (0.592, 0.471, 0.385) because it is complete. Under
#       (A) the MLP is load-bearing IN KIND, and so is width, and the right
#       description of the task is that unablated depth buys deduction rounds
#       while capacity in any other form does not.
#
#   (B) CAPACITY. 0.76 is a coincidence of this particular budget: each
#       ablation happened to cost about the same, and enough depth or width
#       given back to an ablated model carries it past 0.76 like any other.
#
# WHY THE EXISTING DATA CANNOT DECIDE IT. Every one of the nine rungs is an
# ablation at L=6 or shallower, i.e. every ablated model in the grid has been
# given AT MOST the hub's depth. Not one ablated model has ever been run deeper
# than the hub. The one place the question has been put -- can capacity be
# given back -- is stage D's D2, which gives an attention-only stack 1.5x the
# WIDTH. This job gives one 1.3x and 2x the DEPTH, which is the axis the full
# model's own ladder shows to be the one that works.
#
# =====================================================================
# THE GRID. 6 cells, all N=48 d=5 q=4, --fall --shuffle-adj, d_model 256,
# ff 1024, nhead 4, pre-norm, no warmup, lr 1e-3, batch 32, 240k instances x
# 50 epochs = 375k gradient steps, n_val 300. Three seeds (0,1,2) per rung.
#
#   rung   layers  arch     seeds  trainable   est h ea  asks
#   E1       8     no_mlp   0,1,2  2,274,874    ~3.6     interpolation
#   E2      12     no_mlp   0,1,2  3,331,642    ~5.2     extrapolation
#
# Trainable counts are arithmetic, not guesses: the attention-only stack is
# measured at 1,746,490 (L=6) and 1,218,106 (L=4), so 264,192 per layer and
# 161,338 fixed. The assertion block reads the real number back out of
# config.json and the write-up quotes THAT, after the stage D header estimated
# d384 no_mlp at "~3.6M, 26% under the hub" when the built model is 3,799,354,
# which is 22% under. Estimates in a header are not results.
#
# ONE KNOB MOVES. Both rungs differ from stage B's no_mlp at L=6 in num_layers
# alone, and from the hub in num_layers and arch_mode -- deliberately two,
# because the question IS whether the second can be paid for by the first, and
# both single-knob reference points are already measured (no_mlp at L=6 is
# stage B; full at L=8 and L=12 are stage A).
#
# THE CONTROL IS NOT IN THIS JOB, for the fifth time and the same reason: the
# comparison targets are stage A's L=6, L=8 and L=12 full rungs and stage B's
# L=6 no_mlp, all run to completion at this exact budget, this exact 14-point
# grid and the same default --seed-start, so they are PAIRED point by point.
#
# WHY L=8 AND L=12 AND NOT L=7 AND L=9. They are the two depths where the FULL
# model has already been measured, so each stage E rung has a paired full-model
# partner at identical depth and identical budget. A win at a depth with no
# partner would need a new full-model rung to interpret.
#
# =====================================================================
# PRE-REGISTERED, written before submission and before stage D's D2 produced a
# single completed cell (51151957 task 1 was 2h54m into its run at the time
# this file was committed; git log is the timestamp).
#
# METRIC OF RECORD: rho_50, the realized hint density where solve rate crosses
# 0.5 on the 14-point x 400-instance grid below, by linear interpolation.
# LOWER IS BETTER. Censored rho_50 is reported as "> 0.95", never as a number.
# Secondary metrics that never censor: mean_solve_rate (unweighted mean over
# the 14 bins) and solve_at_floor (solve rate at rho = 0.3333).
#
# SCORED AGAINST THE FROZEN BAR stages A, B, C and D were scored against:
# s* = 0.1484 in rho_50 and 0.2918 in mean_solve. NOT RECOMPUTED HERE, and this
# job has a specific reason to say so that stages C and D did not: C's and D's
# rungs carry a non-default d_model, ff or warmup, so rung_key() gives them
# their own 6-tuple keys and they stay out of the pooled bar automatically.
# STAGE E's RUNGS CARRY ENTIRELY DEFAULT HYPERPARAMETERS and would pool into
# the registered grid, raising or lowering s* for every verdict already
# published against it. scripts/minarch_analysis.py therefore defines SCAN_E
# but deliberately LEAVES IT OUT of SCAN_DIRS; read it with --scan-dir, and
# score it with the frozen numbers above.
#
#  E1 -- ATTENTION-ONLY AT L=8. The interpolation point: the prediction sits
#      between two measured full-model rungs, so neither outcome requires
#      extrapolating anything.
#      Prediction under (A) BARRIER, which is the one this header bets on at
#      MODERATE confidence: rho_50 median stays in [0.72, 0.80], mean_solve
#      in [0.26, 0.38], i.e. on the 0.76 rung with two more layers than the
#      hub, 54% fewer parameters, and nothing to show for the depth.
#      Prediction under (B) CAPACITY: rho_50 median at or below 0.592 + s* =
#      0.740 -- it joins or beats the hub.
#      These two overlap at exactly 0.740, so the tie-break is stated in
#      advance: 0.740 exactly, or any median inside [0.72, 0.74], is scored as
#      (A) and written as "did not clear the hub", because (B) is the claim
#      that needs the evidence and a boundary value is not evidence. This is
#      written down because stage C's P3 landed on its threshold to three
#      decimals and had to be rescued by hand afterwards.
#
#  E2 -- ATTENTION-ONLY AT L=12. The extrapolation point and the one that can
#      actually kill (A). Twice the hub's depth, 3,331,642 trainable against
#      the hub's 4,899,898 and against full L=12's 9,638,458.
#      Prediction under (A): rho_50 median in [0.72, 0.80] again, and the
#      pair (E1, E2) flat across a 1.5x depth range is what makes "barrier"
#      a claim rather than a one-cell coincidence.
#      Prediction under (B): rho_50 median at or below 0.471 + s* = 0.619,
#      full L=8's rung, since L=12 ablated should behave like something in
#      the L=8-to-12 range.
#      FALSIFIER OF (A), stated so it cannot be moved later: EITHER rung with
#      a median rho_50 below 0.740 on at least 2 of 3 crossing seeds. One such
#      rung ends the barrier reading outright; there is no partial credit and
#      no "but the other rung held".
#
#  E3 -- WHAT THIS SAYS ABOUT STAGE D's D2, REGISTERED BEFORE D2 FINISHED.
#      D2 predicted that 1.5x width buys the MLP back: rho_50(d384, no_mlp)
#      at or below 0.740. Reading (A) predicts D2 FAILS THAT PREDICTION and
#      lands on the 0.76 rung beside d256 no_mlp, because under (A) capacity
#      in any form is not the currency. So this header is on record, before
#      the cell exists, backing D2's FALSIFIER over D2's prediction.
#      If D2 comes in at 0.76 and E1/E2 also come in at 0.76, the barrier is
#      demonstrated on two independent axes and the brief has its answer: the
#      minimal architecture that still does direct deduction is the complete
#      one, and the smallest complete one measured is the hub.
#      If D2 comes in at or below 0.740 while E1 and E2 stay at 0.76, the
#      currency is width specifically and not capacity in general, which is
#      a sharper and more surprising result than either job registered.
#
#  E4 -- OUT-OF-SAMPLE TEST OF THE SUFFICIENCY RESULT. Free, no extra cells.
#      scripts/acc_vs_solve.py fitted the DEPTH LADDER ALONE and currently
#      holds at 13/13 well-determined held-out rungs, 28 cells, worst
#      discriminating miss 0.0091 against a leave-one-out bar of 0.0423.
#      These rungs are not depth cells of the full model, so this is a genuine
#      out-of-sample test. Prediction: both land on the curve within 0.0423.
#      FALSIFIER: either falls off, which would be the grid's first evidence
#      of an architectural change doing something SOLVE-SPECIFIC rather than
#      moving the model along one quality axis, and would be reported ahead of
#      E1 and E2.
#
#  W -- WATCH ITEMS. The first one is not boilerplate here; it is the single
#      most likely way this job produces nothing.
#      TRAINABILITY IS THE REAL RISK ON THIS ROW. The attention-only stack is
#      already marginal on the plateau: no_mlp at L=6 escapes finding 44's 1.64
#      on 2 of 3 seeds (1.631/1.688/1.597) and at L=4 on 0 of 3
#      (1.661/1.702/1.709). Deeper attention-only models are plausibly HARDER
#      to optimise, not easier. If E1 or E2 comes back 0/3 escapes, the cell is
#      written as "did not train in 375k steps", NEVER as evidence for (A) --
#      that is exactly the trap stage C's P2 fell into and was recorded as
#      unscoreable for. The budget is deliberately NOT increased to compensate,
#      because every rung in the grid is scored at 240k x 50 and a rung with a
#      different budget cannot be compared to the ladder it is being read
#      against. A longer-budget rerun is the follow-up cell, not this cell.
#      GRAMMAR vs DEDUCTION. truncated_rate <= 0.01 and well_formed_rate >=
#      0.99 at every rho for every cell with solve > 0.1.
#      LEAK. Every cell trains and evaluates with --shuffle-adj, and the
#      assertion block re-reads shuffle_adj, fall, arch_mode, num_layers,
#      nhead, d_model, dim_feedforward, warmup_epochs and max_len_margin out of
#      each written config.json rather than trusting the command line
#      (findings 16/17/23).
#      SILENT KNOBS. These rungs DO have default d_model/ff/warmup, so unlike
#      stages C and D they key as plain 3-tuples (8,4,'no_mlp') and
#      (12,4,'no_mlp'). They cannot collide with an existing rung -- no_mlp has
#      only ever been run at L=4 and L=6 -- but they CAN enter the pooled bar,
#      which is why SCAN_E stays out of SCAN_DIRS. See the scoring note above.
#
#  SCORING RULE, fixed now so it cannot be chosen afterwards: every comparison
#  is median-vs-median against the frozen s* above. Smaller than s* is "not
#  separable at this budget", never a result. Three seeds per rung and the
#  spread quoted, always (findings 41/42/47).
#
# =====================================================================
# COST. Measured law for FULL at d256: 0.40 h per layer + 0.40 h fixed. The
# attention-only stack drops 2*d*ff = 524,288 multiply-adds per layer and keeps
# attention's 4*d^2 = 262,144, so it should be cheaper per layer, but no
# attention-only cell has been timed above L=6 and the law is not extrapolated
# on a guess: --time 07:00:00 covers full-model cost at L=12 (5.2 h) with 35%
# headroom. 6 cells x ~4.4 h = ~26 GPU-h, ~9 h wall at 3 concurrent.
#
# THROTTLED TO 3 CONCURRENT and submitted while stages C and D still hold 5
# GPUs between them: sganguli showed 8 of 16 idle at submission, so this fills
# idle capacity rather than competing with the group or with its own earlier
# stages.
#
# ENVIRONMENT. Module stack only -- ${SCRATCH}/spin_gnn_venv is long dead to
# the 90-day purge -- and python3 explicitly, since bare `python` is the system
# 2.7. The smoke check turns an environment failure into a 30-second job
# instead of a five-hour one that dies in the eval.

set -euo pipefail

# task: layers:seed
# INTERLEAVED BY SEED, not grouped by rung: at %3 the first wave is tasks 0-2,
# so this order returns a first seed of BOTH rungs in the first wave instead of
# all of E1 and nothing of E2. The grid table above is the authority on what the
# rungs are; this is only the order they run in.
SPECS=(
  "08:0"  "12:0"  "08:1"
  "12:1"  "08:2"  "12:2"
)
TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
IFS=: read -r LAYERS SEED <<< "${SPECS[$TASK]}"
# 10# forces base 10, as in stages C and D: "08" would otherwise read as octal
# and this is the field it would actually bite, since 08 is not a valid octal
# literal and bash would abort with "value too great for base".
LAYERS=$((10#$LAYERS)); SEED=$((10#$SEED))
DMODEL=256
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

OUT="${SCRATCH}/spin_gnn/sudoku_n48_minarch_E_2026-11-04"
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
# The header's arithmetic: 264,192 per attention-only layer + 161,338 fixed.
# Printed beside the real number so a wrong estimate is visible in the log
# rather than surviving into the write-up.
print("arch_mode=%s L=%d nhead=%d d_model=%d ff=%d warmup=%d  trainable %s params "
      "(header predicted %s)  max_len=%d (margin %d)"
      % (c["arch_mode"], c["num_layers"], c["nhead"], c["d_model"], c["dim_feedforward"],
         int(c.get("warmup_epochs", 0) or 0), format(c.get("n_trainable_params", -1), ","),
         format(161338 + 264192 * c["num_layers"], ","),
         c["max_len"], c["max_len_margin"]))
PY

echo
echo "=== the transition curve for this cell (E1-E2) ==="
# Same 14-point grid and the same default --seed-start as stages A-D, so these
# curves are paired with stage A's L=8 and L=12 full rungs and stage B's L=6
# no_mlp point by point. Bottoms out at 0.33 because the fall ensemble's
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
# architecture cannot represent DD; it has shown it did not train in 375k steps.
# On THIS job that distinction is the difference between a result and nothing:
# no_mlp escapes 2/3 at L=6 and 0/3 at L=4, so a deeper attention-only stack
# failing to escape is the expected failure mode, not evidence of a barrier.
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
# E4 is scored off this line together with history.json's accuracy field: the
# sufficiency curve is fitted on validation TOKEN ACCURACY, not on NLL.
print("final val token accuracy = %.4f" % h[-1]["accuracy"])
print("PLATEAU_ESCAPE=%s  (final train_nll %.4f vs finding 44's plateau %.2f)"
      % ("yes" if final_train < PLATEAU else "NO", final_train, PLATEAU))
PY
date +'task end  %F %T'
