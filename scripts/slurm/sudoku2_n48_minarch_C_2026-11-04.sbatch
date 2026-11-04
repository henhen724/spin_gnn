#!/bin/bash
#SBATCH --job-name=sud2MinArchC
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --array=0-14%4
#SBATCH --output=%x_%A_%a.out
#
# STAGE C: NARROW THE MODEL, AND FIND OUT WHETHER THE ONE LOAD-BEARING PIECE
# IS LOAD-BEARING BECAUSE IT IS WIDE.
#
#   sbatch scripts/slurm/sudoku2_n48_minarch_C_2026-11-04.sbatch
#
# WHERE THIS JOB STARTS FROM. Stage A (51032450) walked the depth ladder and
# Stage B (51072446) stripped the block at L=6, the shallowest rung where all
# three seeds still solve. Three seeds each, leak-free, pooled s* = 0.1484 in
# rho_50 and 0.2918 in mean_solve:
#
#     L=6 h4 full  (the hub)   rho_50 0.592 [spread 0.012]   mean_solve 0.5389
#     linear   (no elementwise nonlinearity)   +0.173   mean_solve 0.3146
#     rf_mlp   (linear1 frozen at init)        +0.171   mean_solve 0.3104
#     no_mlp   (attention only)                +0.171   mean_solve 0.3039
#     nhead=1                                  censored 3/3, mean_solve 0.0270
#
# TWO RESULTS CAME OUT OF THAT, AND BOTH SET THIS JOB.
#
# (1) THE MLP IS ALL-OR-NOTHING. The three cuts remove three different
#     amounts -- the nonlinearity only, the training of the first matrix only,
#     the whole MLP write -- and they cost the SAME: mean_solve 0.3146 /
#     0.3104 / 0.3039, a spread of 0.0107 against the hub's own across-seed
#     spread of 0.0141. A linear MLP is worth what no MLP is worth, and random
#     nonlinear features are worth what no MLP is worth. Only the LEARNED
#     nonlinearity buys anything, and what it buys is about half the depth:
#     every stripped L=6 cell lands on the L=3/L=4 rungs of the full ladder.
#     So there is exactly ONE load-bearing nonlinear component. This job asks
#     how WIDE it has to be, which is the natural next question and the one
#     the analytic programme needs answered.
#
# (2) nhead=1 IS THE ONLY "WORSE" VERDICT THAT IS CONFOUNDED. It came back
#     censored at 3/3 -- but with 0/3 plateau escapes (final train_nll 1.657 /
#     1.648 / 1.650 against finding 44's 1.64). By the watch item fixed in
#     Stage B's header that is a TRAINABILITY result at this budget, not a
#     statement about one head, and it must not be written up as one until the
#     cell has been given a fair chance to train. P4 below gives it that.
#
# =====================================================================
# THE GRID. 15 cells, all N=48 d=5 q=4, --fall --shuffle-adj, L=6, pre-norm,
# lr 1e-3, batch 32, 240k instances x 50 epochs = 375k gradient steps,
# n_val 300. Three seeds (0,1,2) on every rung. ONE KNOB MOVES PER ROW.
#
#   tasks  d_model  ff   nhead  arch    warmup  est h ea   asks
#    0-2     128   1024    4    full      0       2.3     P1  width 256->128
#    3-5     128   1024    4    no_mlp    0       2.3     P2  is the MLP gap
#                                                             width-dependent
#    6-8     256    256    4    full      0       2.6     P3  ff 1024->256
#    9-11    256   1024    1    full      5       2.8     P4  trainability
#                                                             control on N5
#   12-14     64   1024    4    full      0       2.0     P5  bracket, 128->64
#
# THE CONTROL IS NOT IN THIS JOB, for the third time and for the same reason:
# L=6 / d256 / ff1024 / nhead4 / full / no warmup is Stage A tasks 6-8, run to
# completion at the same budget, the same 14-point grid and the same default
# --seed-start, so it is PAIRED with every cell here point by point. Every row
# above differs from it in exactly one respect except 3-5, which is
# deliberately TWO changes (narrow AND no MLP) because its registered quantity
# is a DIFFERENCE OF DIFFERENCES and both of its terms are in this job.
#
# ff STAYS AT 1024 ON THE d_model ROWS. Narrowing the residual stream while
# holding the MLP width fixed is what isolates d_model; letting ff follow
# d_model would move two knobs and the result would be unattributable. The ff
# axis gets its own row at the original d_model for the same reason.
#
# Throttled to 4 concurrent: at submission sganguli had 0 of 16 GPUs idle with
# 34 GPU jobs queued, 3 of them Stage B's last cells. %4 queues politely behind
# the group instead of claiming every GPU that frees up.
#
# =====================================================================
# PRE-REGISTERED, written before submission. HYPOTHESIS IN ONE SENTENCE:
# the one load-bearing piece is the LEARNED MLP nonlinearity, not the width it
# is written in, so halving d_model and quartering ff are both free while the
# MLP's contribution stays the same size.
#
# METRIC OF RECORD: rho_50, the realized hint density where solve rate crosses
# 0.5 on the 14-point x 400-instance grid below, by linear interpolation.
# LOWER IS BETTER. Censored rho_50 is reported as "> 0.95", never as a number.
# Secondary metrics that never censor: mean_solve_rate (unweighted mean over
# the 14 bins) and solve_at_floor (solve rate at rho = 0.3333).
#
# SCORED AGAINST THE BAR THAT WAS IN FORCE WHEN STAGE A AND B WERE SCORED:
# pooled s* = 0.1484 in rho_50, over every complete rung in stage A and B. The
# rungs in THIS job also enter the pool and may widen it; if they do,
# scripts/minarch_analysis.py prints the registered bar beside the pooled one
# and says so. A later job widening the bar must never reopen a verdict that
# was already scored against the narrower one.
#
#  P1 -- HALVING d_model IS FREE. Moderate confidence. Findings 13 and 26 both
#      say depth beats width on this task, and at N=24 narrowing the MLP 4x
#      cost nothing.
#      Prediction: rho_50(d128, L=6, full) - 0.592 < s*, i.e. median at or
#      below 0.740, with all three seeds crossing 0.5.
#      FALSIFIER: gap > s*, or any seed censored while all three hub seeds
#      solved. Then the residual stream width is itself load-bearing at N=48
#      and the minimal architecture is bounded below by it.
#
#  P2 -- THE MLP'S CONTRIBUTION IS WIDTH-INDEPENDENT. This is the registered
#      quantity of the job, and it is a DIFFERENCE OF DIFFERENCES:
#
#          gap(d128)  = rho_50(d128, no_mlp) - rho_50(d128, full)
#          gap(d256)  = 0.763 - 0.592 = +0.171   (Stage B, already measured)
#
#      Prediction: | gap(d128) - gap(d256) | < s*. Mechanism: if the MLP's job
#      is a fixed amount of per-token nonlinear work rather than capacity that
#      scales with the stream, halving the stream should not change what
#      removing the MLP costs.
#      FALSIFIER: the two gaps differ by more than s*. If gap(d128) is SMALLER,
#      the MLP is partly redundant with width and a narrow attention-only stack
#      is closer to the full model than the wide one is -- which would make the
#      minimal architecture narrower AND simpler at once. If LARGER, the MLP is
#      compensating for a narrow stream and the two knobs interact, so neither
#      can be reported alone.
#      BOTH TERMS OF gap(d128) ARE IN THIS JOB (tasks 0-2 and 3-5) at the same
#      seeds, so the difference is paired and does not inherit Stage B's noise.
#
#  P3 -- QUARTERING ff IS FREE. Low-to-moderate confidence, and this is the
#      cell that most directly asks how wide the one load-bearing nonlinearity
#      must be. Evidence is split: finding 26 found ff 1024->256 free at N=24,
#      finding 25 found it cost 7.1 points at N=96. N=48 is between them, so
#      this is a genuine measurement rather than a formality.
#      Prediction: rho_50(ff256, L=6, full) - 0.592 < s*.
#      FALSIFIER: gap > s*. Then the MLP's width is part of what it contributes
#      and "one learned nonlinearity" is not a complete description of it --
#      the size of the nonlinear bank matters, and the ff axis needs its own
#      ladder before any analytic claim is made.
#
#  P4 -- NHEAD=1 DID NOT TRAIN; IT WAS NOT SHOWN TO BE WORSE. This is a
#      CONTROL on Stage B's N5, not a new bet, and it exists because N5 is the
#      only "WORSE" verdict in the grid that is confounded by trainability.
#      Intervention: warmup 5 epochs (linear ramp from lr/6 to 1e-3, then hold
#      the constant 1e-3 for the remaining 45), the cheapest fix that has
#      worked here before -- finding 32 took L=12/ff=256 from 0.8100 to 0.8900
#      with exactly this.
#      Prediction: at least 1 of 3 seeds gets below the 1.64 plateau.
#      FALSIFIER: 0/3 escapes again. THAT IS A USEFUL RESULT, not a failed
#      cell: two independent optimizer settings failing the same way is enough
#      to write N5's "WORSE" as a property of the architecture rather than of
#      the schedule, which is the thing that currently cannot be said.
#      If seeds DO escape, N5 is re-scored on them and the old verdict is
#      withdrawn in writing.
#      NOTE this cell is NOT a replicate of Stage B's nhead=1 cells: warmup is
#      in the checkpoint tag and in the cell key, so the two cannot merge.
#
#  P5 -- d64 BRACKETS THE WIDTH LADDER FROM BELOW. Expected to be censored,
#      and that is the point: with 256 / 128 / 64 the width axis can be read as
#      a LADDER against the depth ladder, the way the stripped cells already
#      are, instead of as a single pass/fail. Treating the breaking point as
#      the result is the brief.
#      Prediction: rho_50 censored at 2 or 3 of 3 seeds; mean_solve well below
#      the d128 row.
#      FALSIFIER: d64 ties the hub within s*. Then width is nearly irrelevant
#      on this task and the minimal architecture should be searched at d32 and
#      below before anything else is tried.
#
#  W -- WATCH ITEMS, unchanged from Stage B, repeated because they decide how
#      a failed cell is WRITTEN UP.
#      TRAINABILITY vs CAPACITY. Every cell prints a plateau-escape flag (final
#      train_nll against 1.64, plus the last-5-epoch slope). 0 of 3 escapes is
#      "this did not train at this budget", NEVER "this cannot represent DD".
#      The narrow cells are the ones at risk here, and P4 exists precisely
#      because that distinction was left open at the end of Stage B.
#      GRAMMAR vs DEDUCTION. truncated_rate <= 0.01 and well_formed_rate >=
#      0.99 at every rho for every cell with solve > 0.1. Stage A and B breach
#      this at the margin (max truncated 0.0200, min well_formed 0.9500 over
#      235 live bins) -- recorded, not rounded away, and far too small to carry
#      any gap reported here.
#      LEAK. Every cell trains and evaluates with --shuffle-adj, and the
#      assertion block below re-reads shuffle_adj, fall, arch_mode, num_layers,
#      nhead, d_model, dim_feedforward, warmup_epochs and max_len_margin out of
#      each written config.json rather than trusting the command line. Findings
#      16/17/23, and the narrower the model the more tempting the shortcut.
#      SILENT KNOBS. d_model, dim_feedforward and warmup_epochs became part of
#      the cell key in commit 49be487 before this job was written. Without that
#      every cell here would have landed on a Stage A or Stage B key and moved
#      that rung's median instead of appearing as a new rung -- finding 43's
#      failure mode, and the reason the assertion block reads the width back.
#
#  SCORING RULE, fixed now so it cannot be chosen afterwards: every comparison
#  is median-vs-median against the pooled s* defined above, computed by
#  scripts/minarch_analysis.py and nothing else. Smaller than s* is "not
#  separable at this budget", never a result. Three seeds per rung and the
#  spread quoted, always (findings 41/42/47).
#
# =====================================================================
# COST. Measured law at this budget and d256: 0.40 h per layer + 0.40 h fixed
# (Stage A: L=12 5.2 h, L=8 3:37, L=6 2:48, L=4 2:00; Stage B's L=6 cells came
# in at 2:32-2:47). The narrow cells are cheaper but not proportionally so --
# attention over a 98-token sequence does not shrink with d_model the way the
# matmuls do -- so the estimates above are deliberately conservative. 15 cells
# x ~2.4 h = ~36 GPU-h, ~9 h wall at 4 concurrent. --time 05:00:00 is ~79%
# headroom over the longest expected cell; sganguli allows 7 d.
#
# ENVIRONMENT. Module stack only -- ${SCRATCH}/spin_gnn_venv is long dead to
# the 90-day purge -- and python3 explicitly, since bare `python` is the system
# 2.7. The smoke check turns an environment failure into a 30-second job
# instead of a two-hour one that dies in the eval.

set -euo pipefail

# task: d_model:ff:nhead:arch_mode:warmup:seed   (priority order, see the grid)
SPECS=(
  "128:1024:4:full:0:0"    "128:1024:4:full:0:1"    "128:1024:4:full:0:2"
  "128:1024:4:no_mlp:0:0"  "128:1024:4:no_mlp:0:1"  "128:1024:4:no_mlp:0:2"
  "256:0256:4:full:0:0"    "256:0256:4:full:0:1"    "256:0256:4:full:0:2"
  "256:1024:1:full:5:0"    "256:1024:1:full:5:1"    "256:1024:1:full:5:2"
  "064:1024:4:full:0:0"    "064:1024:4:full:0:1"    "064:1024:4:full:0:2"
)
TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
IFS=: read -r DMODEL FF NHEAD ARCH WARMUP SEED <<< "${SPECS[$TASK]}"
# 10# forces base 10: 064 and 0256 are zero-padded in SPECS so the table above
# lines up, and bash would otherwise read them as octal.
DMODEL=$((10#$DMODEL)); FF=$((10#$FF)); NHEAD=$((10#$NHEAD)); WARMUP=$((10#$WARMUP))
LAYERS=6
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
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n48_minarch_C_2026-11-04"
# One directory PER TASK: sudoku_hparam_sweep appends to <out-dir>/sweep.csv and
# four cells run concurrently, so a shared directory would interleave rows from
# several writers into one file.
cell="${OUT}/t${TT}_d${DD}_ff${FFS}_h${HH}_${ARCH}_w${WARMUP}_s${SEED}"
mkdir -p "$cell"

echo "task=${TASK}  d_model=${DMODEL} ff=${FF} nhead=${NHEAD} arch=${ARCH} warmup=${WARMUP} seed=${SEED}"
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
# d_model, dim_feedforward and warmup_epochs are read back here for the same
# reason arch_mode is (finding 43): they are now part of the cell key, so a
# cell that silently trained at the wrong width would not error anywhere -- it
# would merge into a Stage A or Stage B rung and move that rung's median.
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
print("arch_mode=%s L=%d nhead=%d d_model=%d ff=%d warmup=%d  trainable %s params  "
      "max_len=%d (margin %d)"
      % (c["arch_mode"], c["num_layers"], c["nhead"], c["d_model"], c["dim_feedforward"],
         int(c.get("warmup_epochs", 0) or 0), format(c.get("n_trainable_params", -1), ","),
         c["max_len"], c["max_len_margin"]))
PY

echo
echo "=== the transition curve for this cell (P1-P5) ==="
# Same 14-point grid and the same default --seed-start as Stage A and B, so
# these curves are paired with Stage A's L=6 d256 control point by point.
# Bottoms out at 0.33 because the fall ensemble's rho_min at N=48 is
# 0.332 +- 0.024.
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.33 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== W: plateau escape (trainability, NOT capacity -- see watch items) ==="
# P4 is scored off this block, so it prints the same way for every cell.
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
