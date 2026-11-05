#!/bin/bash
#SBATCH --job-name=sud2MinArchD
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --array=0-5%3
#SBATCH --output=%x_%A_%a.out
#
# STAGE D: THE LOAD-BEARING RESOURCE IS THE RESIDUAL STREAM WIDTH, NOT THE MLP.
# FIND WHERE THE WIDTH CLIFF IS, AND WHETHER WIDTH CAN BUY THE MLP BACK.
#
#   sbatch scripts/slurm/sudoku2_n48_minarch_D_2026-11-04.sbatch
#
# WHERE THIS JOB STARTS FROM. Stage C (51114734) tasks 0-5 are complete and
# they reverse the job's own headline bet. All numbers L=6, N=48 d=5 q=4,
# --fall --shuffle-adj, 3 seeds, same budget and same 14-point grid:
#
#   rung                        rho_50      mean_solve   trainable   train NLL
#   d256 ff1024 full (the hub)  0.592        0.5389      4,899,898   1.524/1.528/1.530
#   d256 ff1024 no_mlp          0.763        0.3039      1,746,490   1.631/1.688/1.597
#   d128 ff1024 full            CENSORED     0.0273      2,059,834   1.623/1.649/1.621
#   d128 ff1024 no_mlp          CENSORED     0.0291        480,058   1.725/1.738/1.706
#
# THREE THINGS COME OUT OF THAT, AND THEY SET THIS JOB.
#
# (1) P1 IS FALSIFIED, AND IT IS NOT A TRAINABILITY ARTEFACT. Its falsifier
#     read "gap > s*, or any seed censored while all three hub seeds solved";
#     all three d128 seeds are censored. 2 of 3 got BELOW finding 44's 1.64
#     plateau, so unlike N5's nhead=1 this is the first decisive "worse" in the
#     grid that the trainability caveat does not touch. Findings 13 and 26 said
#     depth beats width on this task. At N=48 they are wrong about width.
#
# (2) WIDTH BUYS MORE THAN THE MLP DOES, PER PARAMETER. no_mlp at d256 has
#     1,746,490 trainable and solves (rho_50 0.763). full at d128 has
#     2,059,834 -- 18% MORE -- and never solves at any of 14 hint densities on
#     any of 3 seeds. Removing the entire MLP costs 0.235 of mean_solve;
#     halving the stream costs 0.512, more than twice as much.
#
# (3) P2 CANNOT BE SCORED AS REGISTERED, which was foreseen and is recorded
#     rather than dressed up. Its quantity is gap(d128) = rho_50(d128,no_mlp)
#     - rho_50(d128,full), and BOTH terms are censored, so the difference of
#     differences does not exist. The mean_solve fallback compares 0.0291
#     against 0.0273 -- two cells on the floor, the "both cells failed" tie
#     that M3-M7 already produced at L=4. On top of that all three d128 no_mlp
#     seeds are ABOVE the plateau (1.706-1.738), so they are confounded too.
#     The honest reading is that d128 is below the width the task needs and
#     nothing about the MLP can be read there.
#
# =====================================================================
# THE GRID. 6 cells, all N=48 d=5 q=4, --fall --shuffle-adj, L=6, pre-norm,
# nhead 4, ff 1024, no warmup, lr 1e-3, batch 32, 240k instances x 50 epochs
# = 375k gradient steps, n_val 300. Three seeds (0,1,2) per rung.
#
#   rung                 d_model  arch    seeds  est h ea  asks
#   D1  bisect the cliff   192    full    0,1,2    2.6     where width breaks
#   D2  width for the MLP  384    no_mlp  0,1,2    2.7     can width replace it
#
# ONE KNOB MOVES PER ROW, both against the same hub. D1 differs from the hub
# in d_model alone. D2 differs in d_model and arch_mode -- deliberately two
# changes, because the question IS whether the second can be paid for by the
# first, and both of its single-knob reference points are already measured
# (no_mlp at d256 is Stage B; full at d384 is not needed, since D2's claim is
# against the HUB and not against a d384 full model).
#
# THE CONTROL IS NOT IN THIS JOB, for the fourth time and for the same reason:
# L=6 / d256 / ff1024 / nhead4 / full / no warmup is Stage A tasks 6-8, run to
# completion at the same budget, the same 14-point grid and the same default
# --seed-start, so it is PAIRED with every cell here point by point.
#
# WHY d192 AND NOT d160 OR d224. The width ladder is 256 (solves, 0.5389) and
# 128 (never solves, 0.0273) with nothing between, so the first measurement
# worth making is the bisection. 192 is also the only intermediate width that
# divides by nhead=4 into a head dimension (48) matching nothing else in the
# grid, which keeps the head-dimension confound visible instead of hidden.
#
# WHY d384 FOR THE ATTENTION-ONLY ROW. It is the smallest width above the hub
# that leaves the attention-only stack STRICTLY SMALLER than the hub it is
# being asked to match: ~3.6M trainable against 4,899,898. A win at d512 would
# cost more parameters than the model it replaced and would say only that the
# MLP is replaceable in principle; a win at d384 says the minimal architecture
# is attention-only AND smaller, which is the brief.
#
# =====================================================================
# PRE-REGISTERED, written before submission. HYPOTHESIS IN ONE SENTENCE:
# what the MLP contributes is capacity that the residual stream also supplies,
# so the width axis is a ladder with a break between 128 and 256, and an
# attention-only stack given that capacity back as width reaches the hub.
#
# METRIC OF RECORD: rho_50, the realized hint density where solve rate crosses
# 0.5 on the 14-point x 400-instance grid below, by linear interpolation.
# LOWER IS BETTER. Censored rho_50 is reported as "> 0.95", never as a number.
# Secondary metrics that never censor: mean_solve_rate (unweighted mean over
# the 14 bins) and solve_at_floor (solve rate at rho = 0.3333).
#
# SCORED AGAINST THE BAR THAT WAS IN FORCE WHEN STAGES A, B AND C WERE SCORED:
# pooled s* = 0.1484 in rho_50 and 0.2918 in mean_solve, over every complete
# rung in the registered stage A/B grid. These rungs carry a non-default
# d_model, so rung_key() gives them their own keys and they do NOT enter that
# pool; scripts/minarch_analysis.py prints them in its "NOT in the registered
# grid" list. A later job must never widen the bar a published verdict was
# scored against.
#
#  D1 -- THE WIDTH CLIFF IS A GRADIENT, NOT A STEP. Moderate confidence. Every
#      other axis in this grid (depth, the three MLP cuts) turned out to be
#      graded once mean_solve was used instead of rho_50, and the d256/d128
#      pair is the only place left where a 20x jump in mean_solve sits across
#      a single untested interval.
#      Prediction: d192 lands strictly between its neighbours -- mean_solve in
#      [0.10, 0.45] -- with at least 2 of 3 seeds uncensored in rho_50.
#      FALSIFIER: d192 is indistinguishable from EITHER neighbour -- censored
#      3/3 with mean_solve < 0.10 (it is d128), or rho_50 within s* of 0.592
#      (it is the hub). Either makes the width axis a THRESHOLD rather than a
#      ladder, which is the more interesting outcome and the one that would
#      make a bisection at d160 or d224 the obvious next cell.
#
#  D2 -- WIDTH CAN BUY THE MLP BACK. This is the registered question of the
#      job and the one the brief turns on, because an attention-only stack is
#      the analytically tractable object: no elementwise nonlinearity anywhere
#      in the residual path except the attention softmax.
#      Prediction: rho_50(d384, no_mlp) - 0.592 < s*, i.e. median at or below
#      0.740, with at least 2 of 3 seeds crossing 0.5. Mechanism: the MLP's
#      contribution at d256 is worth 0.171 in rho_50, and width is worth more
#      than that per parameter by (2) above, so 1.5x the stream should cover it
#      with room to spare.
#      FALSIFIER: gap > s*, or 0-1 of 3 seeds cross. THAT IS THE STRONGER
#      RESULT, not a failed cell: it would mean the MLP contributes something
#      width cannot supply -- load-bearing in KIND and not merely in capacity
#      -- which is the thing this programme has not been able to say about any
#      component yet. Scored either way, in writing, on the numbers.
#
#  D3 -- OUT-OF-SAMPLE TEST OF THE SUFFICIENCY RESULT. Free: no extra cells.
#      scripts/acc_vs_solve.py (c1cfabc) fitted the DEPTH LADDER ALONE and
#      found that validation token accuracy predicts mean_solve for every
#      held-out rung in stages A, B and C -- 24 cells, worst discriminating
#      miss 0.0091 against the reference's own leave-one-out bar of 0.0423.
#      That fit was made BEFORE these two rungs existed, and neither is a
#      depth cell, so this is a genuine out-of-sample test rather than another
#      in-sample check.
#      Prediction: both rungs land on the curve within 0.0423.
#      FALSIFIER: either rung falls off it. A rung that falls OFF is the first
#      evidence in the whole grid of an architectural change doing something
#      SOLVE-SPECIFIC rather than moving the model along one quality axis, and
#      it would be reported ahead of D1 and D2. D2 is the likeliest place for
#      it: if attention-only at d384 reaches the hub's token accuracy but NOT
#      its solve rate, the sufficiency claim breaks exactly where it matters.
#
#  W -- WATCH ITEMS, unchanged from Stages B and C, repeated because they
#      decide how a failed cell is WRITTEN UP.
#      TRAINABILITY vs CAPACITY. Every cell prints a plateau-escape flag (final
#      train_nll against finding 44's 1.64, plus the last-5-epoch slope). 0 of
#      3 escapes is "this did not train at this budget", NEVER "this cannot
#      represent DD". d384 no_mlp is the cell at risk: Stage B's no_mlp at d256
#      came in at 1.631/1.688/1.597, straddling the plateau already.
#      GRAMMAR vs DEDUCTION. truncated_rate <= 0.01 and well_formed_rate >=
#      0.99 at every rho for every cell with solve > 0.1.
#      LEAK. Every cell trains and evaluates with --shuffle-adj, and the
#      assertion block below re-reads shuffle_adj, fall, arch_mode, num_layers,
#      nhead, d_model, dim_feedforward, warmup_epochs and max_len_margin out of
#      each written config.json rather than trusting the command line
#      (findings 16/17/23).
#      SILENT KNOBS. d_model is part of the cell key since commit 49be487, so
#      these cells cannot merge into a Stage A or B rung and move its median
#      (finding 43). The assertion block reads the width back to prove it.
#
#  SCORING RULE, fixed now so it cannot be chosen afterwards: every comparison
#  is median-vs-median against the pooled s* defined above, computed by
#  scripts/minarch_analysis.py and nothing else. Smaller than s* is "not
#  separable at this budget", never a result. Three seeds per rung and the
#  spread quoted, always (findings 41/42/47).
#
# =====================================================================
# COST. Measured law at this budget and d256: 0.40 h per layer + 0.40 h fixed.
# d192 full is narrower than the hub; d384 no_mlp trades the MLP's 2*d*ff per
# layer for attention's 4*d^2, which at d384 is 590k against the hub's 786k, so
# both rows should come in at or under the hub's 2:48. 6 cells x ~2.7 h = ~16
# GPU-h, ~6 h wall at 3 concurrent. --time 05:00:00 is ~85% headroom.
#
# THROTTLED TO 3 CONCURRENT and submitted while Stage C still holds 4 GPUs:
# sganguli showed 6 of 16 idle at submission, so this fills idle capacity
# rather than competing with the group or with Stage C's remaining tasks.
#
# ENVIRONMENT. Module stack only -- ${SCRATCH}/spin_gnn_venv is long dead to
# the 90-day purge -- and python3 explicitly, since bare `python` is the system
# 2.7. The smoke check turns an environment failure into a 30-second job
# instead of a two-hour one that dies in the eval.

set -euo pipefail

# task: d_model:ff:nhead:arch_mode:warmup:seed
# INTERLEAVED BY SEED, not grouped by rung: at %3 the first wave is tasks 0-2,
# so this order returns a first seed of BOTH rungs in the first ~2.7 h instead
# of all of D1 and nothing of D2. The grid table above is the authority on what
# the rungs are; this is only the order they run in.
SPECS=(
  "192:1024:4:full:0:0"    "384:1024:4:no_mlp:0:0"  "192:1024:4:full:0:1"
  "384:1024:4:no_mlp:0:1"  "192:1024:4:full:0:2"    "384:1024:4:no_mlp:0:2"
)
TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
IFS=: read -r DMODEL FF NHEAD ARCH WARMUP SEED <<< "${SPECS[$TASK]}"
# 10# forces base 10, as in Stage C: zero-padded fields would read as octal.
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

OUT="${SCRATCH}/spin_gnn/sudoku_n48_minarch_D_2026-11-04"
# One directory PER TASK: sudoku_hparam_sweep appends to <out-dir>/sweep.csv and
# three cells run concurrently, so a shared directory would interleave rows
# from several writers into one file.
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
echo "=== the transition curve for this cell (D1-D2) ==="
# Same 14-point grid and the same default --seed-start as Stages A, B and C, so
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
# D3 is scored off this line together with history.json's accuracy field: the
# sufficiency curve is fitted on validation TOKEN ACCURACY, not on NLL.
print("final val token accuracy = %.4f" % h[-1]["accuracy"])
print("PLATEAU_ESCAPE=%s  (final train_nll %.4f vs finding 44's plateau %.2f)"
      % ("yes" if final_train < PLATEAU else "NO", final_train, PLATEAU))
PY
date +'task end  %F %T'
