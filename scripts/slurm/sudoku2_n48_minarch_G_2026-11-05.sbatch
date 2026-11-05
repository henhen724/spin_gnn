#!/bin/bash
#SBATCH --job-name=sud2MinArchG
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-5%3
#SBATCH --output=%x_%A_%a.out
#
# STAGE G: IS THE CHEAP END OF THIS GRID EXCLUDED BY CAPACITY, OR BY THE
# BUDGET? THE TRAINABILITY FLOOR.
#
#   sbatch scripts/slurm/sudoku2_n48_minarch_G_2026-11-05.sbatch
#
# THE OBSERVATION THAT SETS THIS JOB. scripts/pareto_min.py asked the brief's
# question of all 24 rungs at once instead of a few at a time: of the rungs
# with at least two seeds, only TWELVE ever got below finding 44's train-NLL
# plateau of 1.64, and the standing minimum is simply the cheapest of those
# twelve. It is on the frontier BY DEFAULT, not by beating anything. Every
# rung below it is excluded for never escaping the plateau:
#
#   rung                   trainable   val acc   mean_solve   escapes
#   L=4/6 rf_all              14,906   0.109/0.112  0.0000       0/3
#   L=6 no_mlp d128          480,058   0.4558       0.0291       0/3
#   L=6 full d64             934,714   0.4307       0.0013       0/3
#   L=1 full                 951,098   0.3290       0.0211       0/3
#   L=4 no_mlp             1,218,106   0.4699       0.0796       0/3
#   L=2 full               1,740,858   0.4670       0.0807       0/3
#   ------------------------------------------------- standing minimum ------
#   L=6 no_mlp             1,746,490   0.4812       0.3039       2/3
#
# The programme has always written 0/3 as "did not train in 375k steps, never
# evidence in either direction", and minarch_analysis's own docstring says the
# two readings are "separable only by more steps or a better optimizer,
# neither of which is in this job". That has been treated as a dead end for
# five stages. It is not one, and stage E's header named the exit: "A
# longer-budget rerun is the follow-up cell, not this cell." This is that cell.
#
# WHY NOW, AND ON EVIDENCE RATHER THAN TASTE. scripts/escape_timing.py (commit
# 251f35e) reads the per-epoch histories that have been on disk since stage A:
#
#   * 35 escapes exist. TEN of them (29%) happen in the second half of the
#     budget and two in the final fifth. The last escape anywhere in the grid
#     is rf_mlp seed 1 at EPOCH 44 OF 50. Seven rungs have a seed escaping
#     after epoch 15: full L=4 (28/31/31), full d128 (24/30), ff256 (27),
#     no_mlp d256 (31), no_mlp d384 (22/27/29), linear L=4 (41), rf_mlp (44).
#     Escapes are still ARRIVING when training stops, so the 50-epoch budget
#     TRUNCATES the escape distribution rather than exhausting it.
#   * Nothing in the grid is overfitting. 69 of 74 cells have their best
#     validation NLL in the LAST FIFTH of the run, the worst final-minus-best
#     regression anywhere is +0.0069 nats, and all five exceptions are
#     frozen-feature cells whose val NLL moves under 0.02 nats all run. Every
#     run was stopped mid-descent.
#
# So a 0/3 rung is plausibly a rung whose escape time exceeds the budget. That
# does not make the cheap rungs trainable -- it makes 50 epochs unable to tell,
# which is a checkable claim and this job checks it.
#
# =====================================================================
# THE GRID. 6 cells, all N=48 d=5 q=4, --fall --shuffle-adj, L=6, ff 1024,
# nhead 4, pre-norm, no warmup, lr 1e-3, batch 32, 240k instances, n_val 300.
# ONE KNOB MOVES AGAINST STAGES A-F: epochs 50 -> 100, i.e. 750k gradient
# steps instead of 375k. Three seeds (0,1,2) per rung.
#
#   rung  d_model  arch     epochs  seeds  trainable   est h ea  asks
#   G1      128    no_mlp    100    0,1,2    480,058    ~3.1     the floor
#   G2      256    no_mlp    100    0,1,2  1,746,490    ~5.2     the control
#
# Trainable counts are not estimates: both rungs exist at 50 epochs (stage C
# and stage B respectively) and the numbers are read off those cells. The
# assertion block reads them back out of config.json anyway.
#
# EPOCHS 50 -> 100 IS GENUINELY ONE KNOB. --lr-schedule defaults to CONSTANT
# and warmup is 0 on both cells, so there is no schedule whose shape depends
# on the epoch count and nothing is being reshaped. The pool stays at 240k
# instances, so this is more passes over the same data: more optimization and
# nothing else.
#
# WHY THE CONTROL IS IN THIS JOB, which is the OPPOSITE of stages C, D and E.
# There the control was an existing cell, because only the architecture moved.
# Here THE BUDGET moves, and stage E's header is the authority on what that
# costs: "a rung with a different budget cannot be compared to the ladder it
# is being read against". So no existing cell is a valid comparison for G1,
# and exactly two kinds of comparison are legitimate in this job:
#   (i)  G1 vs G2, both at 100 epochs, paired point for point on the same
#        14-point grid and the same --seed-start; and
#   (ii) each rung's PLATEAU ESCAPE, which is a property of that cell's own
#        training curve and makes no reference to the ladder at all.
# Anything else -- in particular G1's rho_50 against the frozen 0.76 band --
# is cross-budget and is NOT scored. Writing this down now because the
# temptation to quote it will be strong once the number exists.
#
# =====================================================================
# PRE-REGISTERED, written and committed before submission.
#
# METRIC OF RECORD for the escape question: whether final train NLL is below
# 1.64. For the solve question, within this job only: rho_50 on the 14-point x
# 400-instance grid below, LOWER IS BETTER, censored reported as "> 0.95";
# plus mean_solve_rate, which never censors.
#
# THE BAR is the frozen s* = 0.1484 in rho_50 and 0.2918 in mean_solve, NOT
# recomputed here -- and stage G has a reason of its own to say so. Its cells
# would key as the bare (6,4,'no_mlp') and (6,4,'no_mlp',128,1024,0), i.e. not
# merely pooling into the registered grid the way stage E's would, but
# COLLIDING with stage B's and stage C's own rungs and moving the median of a
# published verdict by adding three seeds trained twice as long. Commit 251f35e
# fixed that at the source: rung_key() now appends epochs when it is
# non-default, so these cells get 7-tuple keys and CANNOT merge. Verified by
# diffing the full key set over all five scan dirs before and after -- 1036
# rows, 1036 identical keys. SCAN_G still stays out of SCAN_DIRS.
#
#  G1 -- DOES THE CHEAPEST NON-DEGENERATE 0/3 RUNG TRAIN AT TWICE THE BUDGET?
#      no_mlp d128 is 3.6x below the standing minimum and 10.2x below the hub.
#      At 50 epochs it ends at train NLL 1.725 / 1.738 / 1.706 against the 1.64
#      plateau, with tail slopes -0.00069 / -0.00042 / -0.00083 per epoch, and
#      it already reaches validation token accuracy 0.4558 against the standing
#      minimum's 0.4812 while sitting ON the plateau. The plateau is a
#      partial-learning state here, not a dead model: rf_all sits at 0.11.
#      Prediction under TRAINABILITY, the reading this header bets on at LOW
#      TO MODERATE confidence: >= 2 of 3 seeds below 1.64 at 100 epochs.
#      Prediction under CAPACITY: 0 or 1 of 3.
#      THE ASYMMETRY, STATED IN ADVANCE so it is not discovered afterwards:
#      escaping shows the budget was binding. NOT escaping does NOT show
#      capacity is binding, because 100 epochs is also a budget. A null is
#      written "still did not train at 2x the budget" and the gain is that the
#      standing caveat acquires a number instead of being open-ended.
#
#  G2 -- THE PAIRED CONTROL, AND THE CELL THAT MAY MATTER MORE THAN G1.
#      no_mlp d256 at 50 epochs escapes 2/3 (1.631 / 1.688 / 1.597), and the
#      seed that did escape first did so at epoch 31 of 50. It scores rho_50
#      0.763 and mean_solve 0.3039: the standing minimum, and one of the nine
#      rungs piled up in the 0.740-0.765 band.
#      Prediction on escape: 3/3 at 100 epochs.
#      THE INTERESTING NUMBER IS ITS rho_50, and this is reported AHEAD of G1
#      if it happens. If doubling the budget moves G2's median rho_50 below
#      0.763 - s* = 0.615, then the 0.76 band is a BUDGET artifact rather than
#      an architectural floor, and stage E's reading (A) BARRIER is in trouble
#      by a route that has nothing to do with capacity -- a second, independent
#      attack on it, and a sharper one than either D or E mounted.
#      If instead G2's rho_50 stays within s* of 0.763, the band has survived a
#      doubled budget on one of its nine rungs, which is the first positive
#      evidence that the band is about architecture at all.
#
#  G3 -- THE FLOOR CLAIM, registered before the cells exist so that it cannot
#      be assembled from whatever lands. IF G1 escapes on >= 2 of 3 seeds AND
#      median rho_50(G1) is within s* = 0.1484 of median rho_50(G2), THEN the
#      standing minimum drops from 1,746,490 to 480,058 trainable: a 3.6x cut,
#      and 10.2x below the hub. BOTH conditions or neither. A rung that escapes
#      the plateau but does not solve is a trainability result and is written
#      as one; it is not a new minimum.
#
#  G4 -- OUT-OF-SAMPLE SUFFICIENCY, free, no extra cells. acc_vs_solve fits the
#      depth ladder alone and holds at 15/15 well-determined held-out rungs,
#      worst discriminating miss 0.0091 against a leave-one-out bar of 0.0423.
#      These are the FIRST cells in the programme at a different budget, so
#      this is the sharpest test that curve has had: if accuracy-to-solve is a
#      property of the TASK, the budget should not move it; if it is a property
#      of the 50-epoch budget, it will. Prediction: both land within 0.0423.
#      FALSIFIER: either falls off. That would mean the sufficiency result is
#      budget-conditional, and every use of accuracy as the quality axis --
#      including pareto_min's frontier -- would need restating.
#
#  W -- WATCH ITEMS.
#      OVERFITTING IS THE REAL RISK ON THIS JOB, the way trainability was on
#      stage E. The pool stays at 240k, so this is 100 passes over the same
#      data. Measured at 50 epochs there is no sign of it (69/74 cells best at
#      the end, worst regression +0.0069 nats), but absence at 50 is not
#      absence at 100. The assertion block prints final-minus-best validation
#      NLL for every cell. IF G2 OVERFITS, THE CONTROL IS INVALID AND THE JOB
#      IS REPORTED INCONCLUSIVE -- G1 cannot be read against a broken control,
#      and "G1 overfits less because it is smaller" is not a result about
#      deduction.
#      CROSS-BUDGET QUOTING. G1's and G2's rho_50 must not be read against the
#      frozen band or the ladder. See the control note above.
#      KEY COLLISION. Fixed in 251f35e; the assertion block re-reads epochs out
#      of config.json so a cell that silently ran 50 would be caught here
#      rather than after it had merged into a published rung.
#      LEAK. Both cells train and evaluate with --shuffle-adj, re-read out of
#      the written config.json rather than trusted from the command line
#      (findings 16/17/23).
#      GRAMMAR vs DEDUCTION. truncated_rate <= 0.01 and well_formed_rate >=
#      0.99 at every rho for every cell with solve > 0.1.
#
#  SCORING RULE, fixed now so it cannot be chosen afterwards: every comparison
#  is median-vs-median at three seeds against the frozen s* above, and the only
#  comparisons made are the two the control note allows. Smaller than s* is
#  "not separable at this budget", never a result (findings 41/42/47).
#
# =====================================================================
# COST. Measured law for FULL at d256: 0.40 h per layer + 0.40 h fixed, i.e.
# 2.8 h at L=6 for 50 epochs. Stage E timed attention-only at L=8 at 3h18
# against the law's 3.6 h, so no_mlp is ~0.92x full: ~2.6 h for G2 at 50
# epochs, ~5.2 h at 100. d128 is not extrapolated from parameter count, since
# the sequence-length work does not scale with d_model^2; ~60% of G2 is the
# working estimate, ~3.1 h. --time 08:00:00 covers G2 with 54% headroom.
# 6 cells x ~4.2 h = ~25 GPU-h, ~11 h wall at 3 concurrent.
#
# THROTTLED TO 3 CONCURRENT and submitted while stage E still holds 3 GPUs:
# sganguli showed 10 of 16 idle at submission, so this fills idle capacity
# rather than competing with the group or with its own earlier stages.
#
# ENVIRONMENT. Module stack only -- ${SCRATCH}/spin_gnn_venv is long dead to
# the 90-day purge -- and python3 explicitly, since bare `python` is the system
# 2.7. The smoke check turns an environment failure into a 30-second job
# instead of a five-hour one that dies in the eval.

set -euo pipefail

# task: d_model:seed
# INTERLEAVED BY SEED, not grouped by rung: at %3 the first wave is tasks 0-2,
# so this order returns a first seed of BOTH rungs in the first wave instead of
# all of G1 and nothing of G2. The grid table above is the authority on what
# the rungs are; this is only the order they run in.
SPECS=(
  "128:0"  "256:0"  "128:1"
  "256:1"  "128:2"  "256:2"
)
TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
IFS=: read -r DMODEL SEED <<< "${SPECS[$TASK]}"
DMODEL=$((10#$DMODEL)); SEED=$((10#$SEED))
LAYERS=6
FF=1024
NHEAD=4
WARMUP=0
EPOCHS=100
ARCH=no_mlp
# The measured trainable count for this rung at 50 epochs, printed beside the
# real one so a wrong expectation is visible in the log instead of surviving
# into the write-up (stage D's header estimated d384 no_mlp at "~3.6M" when the
# built model is 3,799,354).
case "$DMODEL" in
  128) WANT_TRAINABLE=480058 ;;
  256) WANT_TRAINABLE=1746490 ;;
  *) echo "no measured trainable count for d_model=$DMODEL"; exit 1 ;;
esac
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

OUT="${SCRATCH}/spin_gnn/sudoku_n48_minarch_G_2026-11-05"
# One directory PER TASK: sudoku_hparam_sweep appends to <out-dir>/sweep.csv and
# three cells run concurrently, so a shared directory would interleave rows
# from several writers into one file. It also keeps G2's checkpoint directory
# off stage B's, whose tag is identical -- the tag carries d_model, layers, ff,
# lr, arch and seed, but NOT epochs.
cell="${OUT}/t${TT}_d${DD}_ff${FFS}_h${HH}_${ARCH}_L${LL}_ep${EPOCHS}_s${SEED}"
mkdir -p "$cell"

echo "task=${TASK}  layers=${LAYERS} d_model=${DMODEL} ff=${FF} nhead=${NHEAD} arch=${ARCH} epochs=${EPOCHS} seed=${SEED}"
echo "cell -> ${cell}"
date +'cell start %F %T'

python3 -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 48 --degree 5 --n-colors 4 --shuffle-adj --fall \
    --d-model "$DMODEL" --nhead "$NHEAD" --dim-feedforward "$FF" \
    --num-layers "$LAYERS" --arch-mode "$ARCH" --seed "$SEED" \
    --warmup-epochs "$WARMUP" \
    --n-train-instances 240000 --epochs "$EPOCHS" \
    --n-val-instances 300 --batch-size 32 --n-per-bin 100 \
    --out-dir "$cell"

date +'cell end   %F %T'

# Tag order follows sudoku_hparam_sweep.py exactly: arch, then schedule, then
# warmup, then seed, each suffixed only when non-default. Epochs is NOT in the
# tag, which is why the cell directory above carries it.
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
python3 - "${CKPT}/config.json" "$ARCH" "$LAYERS" "$NHEAD" "$DMODEL" "$FF" "$WARMUP" "$EPOCHS" "$WANT_TRAINABLE" <<'PY'
import json, sys
c = json.load(open(sys.argv[1]))
want_arch, want_L, want_h = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
want_d, want_ff, want_w = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
want_ep, want_tr = int(sys.argv[8]), int(sys.argv[9])
assert c["shuffle_adj"] is True, "LEAK: this cell is not shuffle_adj"
assert c["fall"] is True, "wrong ensemble: fall is not set"
assert c["arch_mode"] == want_arch, (c["arch_mode"], want_arch)
assert c["num_layers"] == want_L and c["nhead"] == want_h
assert c["d_model"] == want_d, (c["d_model"], want_d)
assert c["dim_feedforward"] == want_ff, (c["dim_feedforward"], want_ff)
assert int(c.get("warmup_epochs", 0) or 0) == want_w, (c.get("warmup_epochs"), want_w)
# The whole point of the job. A cell that silently ran 50 epochs would key as
# an EXISTING rung (commit 251f35e makes the key epoch-aware) and would merge
# into a published verdict, so this is an assertion and not a print.
assert int(c["epochs"]) == want_ep, (c["epochs"], want_ep)
assert c["max_len_margin"] == 2 * c["n_nodes"] + 2, c["max_len_margin"]
got = int(c.get("n_trainable_params", -1))
print("arch_mode=%s L=%d nhead=%d d_model=%d ff=%d warmup=%d epochs=%d  trainable %s "
      "(the same rung at 50 epochs measured %s)"
      % (c["arch_mode"], c["num_layers"], c["nhead"], c["d_model"], c["dim_feedforward"],
         int(c.get("warmup_epochs", 0) or 0), int(c["epochs"]), format(got, ","),
         format(want_tr, ",")))
# Architecture must be IDENTICAL to the 50-epoch cell; only the budget moved.
assert got == want_tr, ("trainable count differs from the 50-epoch cell: this "
                        "is supposed to be the same architecture", got, want_tr)
PY

echo
echo "=== the transition curve for this cell (G1-G2) ==="
# Same 14-point grid and the same default --seed-start as stages A-F, so G1 and
# G2 are paired with EACH OTHER point for point. They are NOT paired with the
# 50-epoch ladder: different budget, see the control note in the header.
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.33 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== W: plateau escape (G1) and overfitting (the risk on THIS job) ==="
python3 - "${CKPT}/history.json" <<'PY'
import json, sys
PLATEAU = 1.64
h = json.load(open(sys.argv[1]))
n = len(h)
best = min(range(n), key=lambda i: float(h[i]["nll"]))
final_train = float(h[-1]["train_nll"])
tail = [float(r["train_nll"]) for r in h[-5:]]
slope = (tail[-1] - tail[0]) / max(len(tail) - 1, 1)
# First crossing, not the final value: several 50-epoch seeds dip below 1.64
# and come back, and first-crossing is the quantity escape_timing reports.
esc = next((i for i, r in enumerate(h) if float(r["train_nll"]) < PLATEAU), None)
print("epochs %d  final train_nll %.4f  final val_nll %.4f  best val_nll %.4f at epoch %d"
      % (n, final_train, float(h[-1]["nll"]), float(h[best]["nll"]), best))
print("first epoch below %.2f = %s   (the 50-epoch cell for this rung is in the header)"
      % (PLATEAU, esc if esc is not None else "never"))
print("last-5-epoch train_nll slope = %+.5f/epoch (negative = still descending)" % slope)
print("final val token accuracy = %.4f" % float(h[-1]["accuracy"]))
# W: OVERFITTING. 100 passes over a fixed 240k pool. At 50 epochs the worst
# final-minus-best anywhere in the grid was +0.0069 nats and 69 of 74 cells
# had their best in the last fifth; if that breaks here the control is invalid.
print("OVERFIT: best val at epoch %d of %d (%.2f of the run), final-minus-best %+.4f nats"
      % (best, n - 1, best / (n - 1), float(h[-1]["nll"]) - float(h[best]["nll"])))
print("PLATEAU_ESCAPE=%s  (final train_nll %.4f vs finding 44's plateau %.2f)"
      % ("yes" if final_train < PLATEAU else "NO", final_train, PLATEAU))
PY
date +'task end  %F %T'
