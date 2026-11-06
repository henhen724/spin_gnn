#!/bin/bash
#SBATCH --job-name=sud2MinArchJ
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-5%3
#SBATCH --output=%x_%A_%a.out
#
# STAGE J: SAME PARAMETER BUDGET, TWO SHAPES. DOES SHAPE MATTER AT ALL?
#
#   sbatch scripts/slurm/sudoku2_n48_minarch_J_2026-11-05.sbatch
#
# WHAT H AND I SETTLED, AND WHY THAT MAKES THIS THE NEXT JOB. 51223007 (6/6)
# and 51246202 (3/3) closed this evening, all exit 0:0, and the attention-only
# width ladder at L=12 is now complete and scored against the frozen bar:
#
#   rung   d    trainable   rho_50     mean_solve  spread  esc  cross
#   H1    128     879,418   CENSORED     0.0411    0.2729  1/3   1/3
#   I1    160   1,345,018   0.773        0.3000    0.0164  3/3   3/3
#   H2    192   1,908,922   0.754        0.3220    0.1450  3/3   3/3
#   ----------------------------------------- hub + s* = 0.740 ----------
#   E2    256   3,331,642   0.605        0.5214    0.0480  3/3   3/3
#
# THE THING THAT DECIDES THIS JOB: 160 -> 192 moves rho_50 by 0.019, which is
# 0.13 s* and NOT SEPARABLE, while 192 -> 256 moves it by 0.149, which is
# 1.00 s* and just separable. EVERY SEPARABLE UNIT OF MOVEMENT ON THE WIDTH
# AXIS LIVES IN [192, 256]. Nothing below 192 is worth another cell: I1 and H2
# are the same rung as far as this grid can tell, and H1 did not train.
#
# AND THE WIDTH ANSWER IS NOW CLEAN, WHICH H1 COULD NEVER HAVE MADE IT. H1 is
# inconclusive -- 1 of 3 escapes against a registered >= 2, so stageh_score
# refused to read its censoring as architecture. But I1 and H2 BOTH trained on
# 3 of 3 seeds and BOTH landed back in the [0.72, 0.80] band. A rung that
# trained and still sits in the band is an architecture statement. So (W)
# WIDTH IS A SEPARATE FLOOR is answered yes, by the two rungs that trained,
# and not by the one the prediction was written about.
#
# =====================================================================
# THE QUESTION THIS JOB ASKS, AND IT IS NOT THE NEXT RUNG DOWN. Depth bought
# the MLP back at d256 (0.763 at L=6 -> 0.605 at L=12) and width takes it away
# at L=12 (0.605 at d256 -> 0.754 at d192). Both axes move the same metric in
# the same direction, and the grid has never held one FIXED against the other.
# Stage J builds two rungs at THE SAME PARAMETER COUNT and different shapes:
#
#   rung   shape                 trainable (est)   vs E2    vs hub
#   J1     no_mlp L=12 d224      2,571,130         1.30x    1.91x
#   J2     no_mlp L=16 d192      2,504,890         1.33x    1.96x
#
# 2,571,130 against 2,504,890 is 2.6% apart. That is as close to a matched
# budget as integer widths and depths allow here, and it turns "is depth or
# width the currency" from two ladders that were never compared at a fixed
# price into one controlled contrast.
#
# BOTH RUNGS ARE ALSO ONE KNOB FROM A MEASURED CELL, so neither depends on the
# other to be readable: J1 is d224 against E2's d256 and H2's d192 at the same
# depth; J2 is L=16 against H2's L=12 at the same width.
#
# AND J1 IS THE LAST STANDING-MINIMUM CANDIDATE THE LADDER SUPPORTS. It is the
# bisector of [192, 256], the only segment with separable movement left in it.
#
# Trainable counts from the arithmetic stage H and stage I both confirmed off
# config.json: 4*d^2 + 8*d per layer, 630*d + 58 fixed.
#   J1  141,178 + 202,496*12 = 2,571,130
#   J2  121,018 + 148,992*16 = 2,504,890
# The assertion block reads the real number back out of config.json and the
# write-up quotes THAT.
#
# =====================================================================
# THE GRID. 6 cells, N=48 d=5 q=4, --fall --shuffle-adj, ff 1024 (unused by
# no_mlp but written for key compatibility), nhead 4, pre-norm, no warmup,
# lr 1e-3, batch 32, 240k instances x 50 epochs = 375k gradient steps, n_val
# 300. Seeds 0, 1, 2 per rung. Identical to E2/H/I in every respect except the
# two knobs named above, so every comparison below is paired.
#
# =====================================================================
# PRE-REGISTERED, written and committed before any J cell landed, with the
# complete H and I numbers above already on disk and scored. git log is the
# timestamp.
#
# METRIC OF RECORD: rho_50, the realized hint density where solve rate crosses
# 0.5 on the 14-point x 400-instance grid below, by linear interpolation.
# LOWER IS BETTER. Censored rho_50 is reported as "> 0.95", never as a number.
# Secondary metrics that never censor: mean_solve_rate and solve_at_floor.
#
# SCORED AGAINST THE FROZEN BAR stages A-E were scored against: s* = 0.1484 in
# rho_50 and 0.2918 in mean_solve. NOT RECOMPUTED. Both rungs carry non-default
# d_model, so they key as 6-tuples -- (12,4,'no_mlp',224,1024,0) and
# (16,4,'no_mlp',192,1024,0) -- and cannot enter the pooled bar. Neither can
# collide: no_mlp has never been run at d224, and L=16 has never been run at
# all, at any width. SCAN_J stays OUT of SCAN_DIRS.
#
#  J0 -- THE HEADLINE, AND IT IS THE MATCHED-PARAMETER CONTRAST, reported
#      ahead of J1 and J2 because it is the only thing in this job that no
#      earlier stage could have produced.
#      PREDICTION, and it is the grid's own sufficiency result talking:
#      |rho_50(J1) - rho_50(J2)| <= s* = 0.1484, i.e. NOT SEPARABLE. The
#      accuracy-to-solve fit now holds on 19 of 19 well-determined held-out
#      rungs over 41 cells (worst discriminating miss 0.0222 against the LOO
#      bar 0.0423), which says model quality over this grid is ONE
#      DIMENSIONAL. If quality is one-dimensional and the two rungs cost the
#      same, they should land in the same place, and SHAPE IS NOT A FREE
#      PARAMETER -- only size is.
#      FALSIFIER, AND IT IS THE RESULT WORTH HAVING: a separable difference in
#      either direction. That would be the grid's FIRST evidence that how the
#      parameters are arranged matters independently of how many there are,
#      and it names the winner: (DEPTH) if rho_50(J2) is lower by more than
#      s*, (WIDTH) if rho_50(J1) is lower by more than s*. Reported ahead of
#      everything, and it would make every "smallest architecture" claim in
#      this programme conditional on shape.
#      TIE-BREAK, fixed now: exactly s*, or anything inside it, is NOT
#      SEPARABLE. The falsifier is the claim that needs the evidence.
#
#  J1 -- no_mlp L=12 d224, THE BISECTOR OF THE ONLY SEPARABLE SEGMENT, and a
#      number rather than a direction because both ends are measured.
#      Interpolating H2 (d192, 0.754) and E2 (d256, 0.605) three ways:
#          linear in d            -> 0.680
#          linear in trainable    -> 0.685
#          linear in 1/d          -> 0.669
#      PREDICTION: rho_50 median in [0.665, 0.690], on 3 of 3 crossing seeds
#      and 3 of 3 plateau escapes (both neighbours escape 3/3).
#      FALSIFIER: outside that interval. Above 0.740 in particular would mean
#      the whole 0.149 drop happens inside [224, 256] -- a factor of 1.14 in
#      width -- which is a threshold, not a slope, and by the brief's own
#      instruction that breaking point is the result.
#
#  J2 -- no_mlp L=16 d192, A DEPTH NEVER RUN IN THIS GRID.
#      PREDICTION: rho_50 below H2's 0.754 (more depth at fixed width has
#      lowered rho_50 every time it has been tried on this axis: d256 no_mlp
#      goes 0.763 at L=6, 0.749 at L=8, 0.605 at L=12), and by J0 within s*
#      of J1.
#      TRAINABILITY: >= 2 of 3 escapes. Every d192 cell in the grid escaped
#      3 of 3, and depth has monotonically improved escapes on the no_mlp
#      axis, so both arguments point the same way here for once -- unlike
#      stage H, where the monotone-depth expectation and escape_timing.py's
#      Q3 contradicted each other and Q3 came closer (H1 landed 1/3 against
#      Q3's 0/3 and the header's >= 2/3).
#      NOTE WHAT IS NOT CLAIMED: nothing in this grid sits above L=12, so J2's
#      depth is an extrapolation of the ladder, not a point on it.
#
#  J3 -- THE STANDING-MINIMUM RULE, unchanged from H3 and I3 and NOT softened
#      because this job's rungs are the last candidates. A rung replaces the
#      standing minimum ONLY IF it has 3 of 3 crossing seeds AND a median
#      rho_50 at or below 0.740. Two of three is not enough, and finding 51 is
#      why: stage G showed a 2-of-3 at 50 epochs was ONE UNFINISHED SEED.
#      If J1 lands as predicted it FIRES, moving the standing minimum from E2
#      (3,331,642 at 0.605) to 2,571,130 -- 1.30x smaller and 1.91x below the
#      hub. STATED HONESTLY IN ADVANCE: the predicted rho_50 then RISES from
#      0.605 to about 0.68, which is +0.075, or 0.51 s*, BELOW the
#      separability bar. The claim would therefore be "1.30x smaller at no
#      separable cost in rho_50", never "better". J2 is a candidate under the
#      same rule on the same terms.
#
#  J4 -- OUT-OF-SAMPLE TEST OF THE SUFFICIENCY RESULT. Free, no extra cells.
#      The depth-ladder fit holds on 19 of 19 well-determined held-out rungs
#      over 41 cells, worst discriminating miss 0.0222 against a leave-one-out
#      bar of 0.0423; stage H's and stage I's three rungs all went in tonight
#      and were all EXPLAINED. PREDICTION: both stage J rungs land on the
#      curve within 0.0423.
#      FALSIFIER: either falls off. Note this is the same statistic J0 leans
#      on, so if J0's falsifier fires, THIS is where to look first -- a shape
#      effect that is real should show up as a rung sitting off the curve.
#
#  W -- WATCH ITEMS.
#      SPREAD. H2 came back with a mean_solve spread of 0.1450 across its
#      three seeds, the widest of any rung that trained 3/3, while I1 at
#      0.0164 was the tightest. A 3-of-3 rung can still be noisy, so the
#      spread is quoted for every rung always (findings 41/42/47) and J0's
#      comparison is median-vs-median.
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
#  separable at this budget", never a result.
#
# =====================================================================
# COST. Measured on this exact arch and budget: d128 L=12 ran 3h26m, d192 L=12
# ran 4h18m, d256 L=12 ran 4h50m (stage E). Wall time is close to linear in d
# at fixed depth, not quadratic, so J1 at d224 sits between 4h18m and 4h50m,
# bounded above by a cell already measured. J2 scales with depth instead:
# 16/12 of d192's per-layer cost puts it near 5h30m-6h. --time 08:00:00 covers
# the slower arm with better than 30% headroom and is far inside sganguli's
# 7-day maximum (sh_part). 6 cells x ~5h = ~30 GPU-h, ~10h wall at 3
# concurrent.
#
# THROTTLED TO 3 CONCURRENT. Checked with ListAgents and squeue before
# submitting; sh_part showed sganguli with 2 of 16 GPUs idle and 10 GPU jobs
# queued, so this will wait its turn rather than displace anything, which is
# the right behaviour for an overnight job on the group's own partition.
#
# ENVIRONMENT. Module stack only -- ${SCRATCH}/spin_gnn_venv is long dead to
# the 90-day purge -- and python3 explicitly, since bare `python` is the system
# 2.7. The smoke check turns an environment failure into a 30-second job
# instead of a six-hour one that dies in the eval.

set -euo pipefail

# task: d_model:num_layers:seed
# INTERLEAVED BY SEED, not grouped by rung: at %3 the first wave is tasks 0-2,
# so this order returns a first seed of BOTH rungs in the first wave instead of
# all of J1 and nothing of J2.
SPECS=(
  "224:12:0"  "192:16:0"  "224:12:1"
  "192:16:1"  "224:12:2"  "192:16:2"
)
TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
IFS=: read -r DMODEL LAYERS SEED <<< "${SPECS[$TASK]}"
# 10# forces base 10, as in stages C, D, E, H and I.
DMODEL=$((10#$DMODEL)); LAYERS=$((10#$LAYERS)); SEED=$((10#$SEED))
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

OUT="${SCRATCH}/spin_gnn/sudoku_n48_minarch_J_2026-11-05"
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
# 4*d^2 + 8*d per layer, 630*d + 58 fixed -- confirmed by stage H at d128 and
# d192 and by stage I at d160, in all three cases to the parameter.
d, L = c["d_model"], c["num_layers"]
print("arch_mode=%s L=%d nhead=%d d_model=%d ff=%d warmup=%d  trainable %s params "
      "(header predicted %s)  max_len=%d (margin %d)"
      % (c["arch_mode"], L, c["nhead"], d, c["dim_feedforward"],
         int(c.get("warmup_epochs", 0) or 0), format(c.get("n_trainable_params", -1), ","),
         format((630 * d + 58) + (4 * d * d + 8 * d) * L, ","),
         c["max_len"], c["max_len_margin"]))
# J0 is a MATCHED-BUDGET contrast and it is only matched if the build agrees.
# 2,571,130 vs 2,504,890 is 2.6% apart; anything beyond 5% and the two rungs
# are not the same price and J0 is not the comparison it claims to be.
got = c.get("n_trainable_params", -1)
other = 2504890 if d == 224 else 2571130
print("J0 budget match: this rung %s vs partner %s -> %.2f%% apart%s"
      % (format(got, ","), format(other, ","), 100 * abs(got - other) / other,
         "" if abs(got - other) / other <= 0.05 else "   <-- NOT MATCHED, J0 is void"))
PY

echo
echo "=== the transition curve for this cell (J1-J2) ==="
# Same 14-point grid and the same default --seed-start as stages A-I, so these
# curves are paired with E2 (d256 L=12), H2 (d192 L=12) and I1 (d160 L=12)
# point by point. Bottoms out at 0.33 because the fall ensemble's rho_min at
# N=48 is 0.332 +- 0.024.
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.33 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== W: plateau escape ==="
python3 - "${CKPT}/history.json" <<'PY'
import json, sys
# 1.64 is where finding 44's L=4/ff=1024/240k cell sat for the whole run at
# this exact budget. Registered here: >= 2 of 3 escapes on BOTH rungs. Every
# d192 cell in the grid has escaped 3/3, and depth has monotonically improved
# escapes on the no_mlp axis, so for once the two arguments agree -- unlike
# stage H, where they contradicted and escape_timing.py's Q3 came closer.
PLATEAU = 1.64
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
# J4 is scored off this line together with history.json's accuracy field: the
# sufficiency curve is fitted on validation TOKEN ACCURACY, not on NLL.
print("final val token accuracy = %.4f" % h[-1]["accuracy"])
print("first epoch under the plateau = %s" % (first if first is not None else "never"))
print("PLATEAU_ESCAPE=%s  (final train_nll %.4f vs finding 44's plateau %.2f)"
      % ("yes" if final_train < PLATEAU else "NO", final_train, PLATEAU))
PY
date +'task end  %F %T'
