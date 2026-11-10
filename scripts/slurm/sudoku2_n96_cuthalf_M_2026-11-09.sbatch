#!/bin/bash
#SBATCH --job-name=sud2N96CuthalfM
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-5%6
#SBATCH --output=%x_%A_%a.out
#
# STAGE M: WHICH HALF OF THE STAGE-F CUT IS LOAD-BEARING AT N=96?
#
#   sbatch scripts/slurm/sudoku2_n96_cuthalf_M_2026-11-09.sbatch
#
# WHAT F LEFT OPEN. Stage F (51656823, finding 55) compared the N=48 standing
# minimum against the N=96 complete-model control and falsified the transfer:
#
#   stripped  no_mlp L=16 d192   rho_50 0.850  (0.937 / 0.826 / 0.850)  8,929,514
#   control   full   L=12 d256   rho_50 0.676  (0.679 / 0.564 / 0.676)  9,790,058
#   difference +0.175 = 1.47 s*_96
#
# But that contrast moves TWO things at once: the MLP (full -> no_mlp) and the
# SHAPE (L=12 d256 -> L=16 d192). F can say the pair costs 1.47 yardsticks. It
# cannot say which member of the pair did it, and the two answers point opposite
# ways -- if the MLP is load-bearing, the attention-only cut is what fails to
# transfer and a future N=96 minimum keeps the MLP and cuts width instead; if the
# SHAPE is load-bearing, the attention-only cut DOES transfer and what fails is
# going to d192 at N=96.
#
# THE DESIGN IS A 2x2 AND THREE OF ITS FOUR CORNERS COST NOTHING. The factors are
# ARCH (full / no_mlp) and SHAPE (L=12 d256 / L=16 d192):
#
#                     L=12 d256                 L=16 d192
#     full      A  control, ON DISK (3 seeds)   B  stage M, 3 seeds
#     no_mlp    C  stage M, 3 seeds             D  stripped, ON DISK (3 seeds)
#
# A and D are stage F's two arms, already measured at three seeds each on the
# identical grid. M buys B and C, which completes the square and turns one
# 2-factor contrast into two main effects plus an interaction:
#
#     dMLP   = rho_50(C) - rho_50(A)        the MLP, at the control's shape
#     dSHAPE = rho_50(B) - rho_50(A)        the shape, with the MLP intact
#     inter  = (D - A) - (dMLP + dSHAPE)    what the pair does that the parts do not
#
# WHAT M DOES NOT SEPARATE, said plainly: the SHAPE factor bundles depth (12->16)
# and width (256->192). M splits ARCH from SHAPE, not depth from width. If the
# shape half turns out to be load-bearing, separating those two is the NEXT
# stage, and that is also the only condition under which J1's convexity question
# (the shape of rho_50 on [192, 256]) is worth buying -- it was dropped when F
# failed, and this is the one result that would reopen it.
#
# =====================================================================
# THE GRID. 6 cells, all N=96 d=5 q=4, --fall --shuffle-adj, nhead 4, ff 1024,
# pre-norm, constant lr 1e-3, batch 32, 240k x 50 epochs, n_val 300, NO warmup,
# seeds 0/1/2. Every one of those is stage F's setting and the existing N=96
# control's setting; arch, L and d_model are the only things that differ, which
# is what makes A, B, C, D one square and not four separate runs.
#
#   task  arch     L   d     seed  corner  role                         est h
#     0   no_mlp   12  256    0      C     the MLP alone                 10.0
#     1   no_mlp   12  256    1      C     the MLP alone                 10.0
#     2   no_mlp   12  256    2      C     the MLP alone                 10.0
#     3   full     16  192    0      B     the shape alone               14.5
#     4   full     16  192    1      B     the shape alone               14.5
#     5   full     16  192    2      B     the shape alone               14.5
#
# Corner C runs first: it is the single-variable MLP contrast against the control
# at the control's own shape, it is the cheaper arm, and it is the cell that
# answers the headline question on its own if the square turns out additive.
#
# =====================================================================
# PRE-REGISTRATION. Written and committed before any cell runs (the array is
# submitted with --dependency=afterok on the commit job, so a cell cannot start
# ahead of its own registration). Scored by scripts/stagem_score.py, registered
# in the same commit.
#
# THE THREE YARDSTICKS, FROZEN HERE AND NOT RECOMPUTED. All three are the mean
# |max-min| across the FOUR multi-seed leak-free N=96 rungs that exist today,
# measured read-only by jobs 51743995 and 51744471 before this file was written.
# s* = 0.1484 and mean_solve 0.2918 are N=48 numbers and are NOT reused.
#
#   rho_50              0.138 / 0.115 / 0.105 / 0.111  ->  s*_M        = 0.117
#   mean_solve          0.1766 / 0.1623 / 0.1355 / 0.1598 -> s*_M_ms   = 0.1586
#   val token accuracy  0.0094 / 0.0045 / 0.0070 / 0.0107 -> s*_M_acc  = 0.0079
#
# s*_M = 0.117 is the re-estimate stage F printed and explicitly said "is what a
# LATER stage should freeze its own bar from"; F's own verdicts used 0.119 and
# are not touched by this. THREE bars rather than one because rho_50 CENSORS and
# stage F's worst stripped seed came in at 0.937 against a top bin of 0.95 -- one
# bin of headroom. A corner worse than the stripped arm can censor, and then its
# level is not quotable at all; mean_solve and token accuracy never censor.
#
# M0  THE HEADLINE: THE DECOMPOSITION NAMES ONE HALF. Exactly one of |dMLP| and
#     |dSHAPE| exceeds s*_M = 0.117 on rho_50, and that half carries F's +0.175.
#     All four outcomes are named NOW, so none of them can be a surprise later:
#       (a) MLP ONLY. The MLP is load-bearing at N=96. "Attention-only" is the
#           cut that fails to transfer; the shape change is free; a future N=96
#           minimum keeps the MLP and looks for its savings in width or depth.
#       (b) SHAPE ONLY. The MLP is free at N=96 as it is at N=48 -- the
#           attention-only cut DOES transfer -- and what fails is L=16 d192. The
#           next stage separates depth from width, and J1's convexity question
#           reopens on exactly the condition under which it was dropped.
#       (c) BOTH exceed the bar. The gap is shared, there is no single
#           load-bearing half, and the honest report is both main effects with
#           their sum, not a headline.
#       (d) NEITHER exceeds the bar. THIS IS THE INTERESTING ONE: each single
#           step is free and the pair is not, i.e. the cost is an INTERACTION.
#           Then the F gap does not decompose into main effects at all and the
#           "smallest architecture" question cannot be answered one axis at a
#           time -- the programme's whole ladder method is what that would
#           indict, which is why it is written down in advance.
#     TIE-BREAK, fixed now: exactly s*_M, or anything inside it, is NOT
#     separable. The falsifier is the claim that needs the evidence.
#
# M1  ADDITIVITY. |(dMLP + dSHAPE) - 0.175| <= s*_M = 0.117, i.e. the square's
#     interaction term is smaller than a seed spread and the main effects are the
#     whole story.
#     FALSIFIER: outside. Then the interaction is bigger than the seed spread and
#     NO single-factor statement from this square generalizes to the other
#     corner. The sign is reported explicitly: super-additive (cutting both costs
#     MORE than the sum of the parts) is the reading that most threatens the
#     one-cut-at-a-time ladder, and it is reported as such, not averaged away.
#
# M2  EVERY CORNER TRAINS. 3 of 3 seeds in each new arm produce a MEASURED rho_50
#     -- the solve-rate curve crosses 0.5 somewhere on the 14-point grid rather
#     than censoring at the top.
#     FALSIFIER: any arm censors. Then that corner has NO quotable level, M0 and
#     M1 are read on mean_solve and token accuracy only (both never censor, both
#     with a bar frozen above), and the censor IS that corner's result -- "this
#     corner of the square does not train at this budget" -- reported as the
#     finding, not as a failed run. All three seeds censoring is an architecture
#     statement; one of three is finding 51's seed story.
#
# M3  THE THREE METRICS AGREE ON THE SIGN. On stage F they did, at nearly the
#     same strength -- stripped worse by 1.47 bars on rho_50, 1.52 on mean_solve
#     (0.1811 vs 0.4229), 1.43 on token accuracy (0.4709 vs 0.4822) -- which is
#     why F's verdict is not a rho_50 artifact. Registered: for each new corner,
#     all three metrics agree in SIGN against the control.
#     FALSIFIER: a corner worse on rho_50 but not on accuracy, or the reverse.
#     That corner is then THE result of the stage: the standing non-negotiable
#     says a rung that falls off the accuracy-to-solve relation is the
#     interesting one, and it is reported rather than averaged away.
#     GUARD: acc_vs_solve.py's FITTED curve is an N=48 object and is NOT
#     evaluated against any N=96 cell. The accuracy comparison here is N=96 to
#     N=96 only -- borrowing the N=48 fit would be the same error as reusing
#     s* = 0.1484 at N=96.
#
# M4  THE N=48 STANDING MINIMUM DOES NOT MOVE ON N=96 EVIDENCE. Stated as a
#     registration so no verdict here is later read as one: the standing minimum
#     is attention-only L=16 d192, 2,504,890 trainable at rho_50 0.580 (finding
#     53, rule H3 = I3 = J3 = K2), an N=48 ladder result on the N=48 grid. M can
#     corroborate or undermine it; it can never replace it. Nor can M restore the
#     cross-size reading F removed: after finding 55 every "smallest
#     architecture" claim in this programme is an N=48 claim, whichever half of
#     the cut turns out to be load-bearing.
#
# WATCH ITEMS (not hypotheses; they decide what the numbers MEAN).
#   * NO N=96 PLATEAU LEVEL EXISTS. Finding 44's train-NLL plateau of 1.64 is an
#     N=48 number; every leak-free N=96 cell sits at 1.7456-1.8320 and N=96's own
#     plateau has never been measured. NO ESCAPE COUNT IS QUOTED for any M cell.
#     Final train NLL is printed and compared N=96 to N=96 only.
#   * THE N=96 GRID ENDS AT 0.32, the N=48 grid at 0.33. Never pooled, and no
#     N=96 dir may enter minarch_analysis.SCAN_DIRS -- an N=96 `full` L=12 cell
#     reduces to the same rung_key as the N=48 `full` L=12 rung and would move a
#     frozen bar. The scorer asserts this before reading anything.
#   * n_trainable_params DOES NOT EXIST in the 11-02/11-03 control artifacts (the
#     field was added 11-04). Corner A's size is therefore quoted as TOTAL
#     n_params on all four sides, and both numbers are printed for B and C so
#     neither is mistaken for the other.
#   * TRUNCATION. max_len margin is 2*n_nodes+2 = 194 at N=96; traces run to ~800
#     tokens, truncated_rate should be ~0. Asserted in the smoke check, because
#     this is the axis that silently bound finding 44.
#   * STEPS-BOUND? 240k x 50 is 375k steps, chosen to pair with the control, not
#     because it is enough for N=96. If val_nll is still falling at epoch 50 then
#     every corner is steps-bound and the square compares four undertrained
#     models -- worth saying, and it does not void the pairing, which is the
#     point of pairing.
#
# =====================================================================
# COST, from F's MEASURED wall clock rather than from F's estimate (F predicted
# 7.5 h for the stripped cell and it took 13.0 h; the derived 0.33x per-layer
# factor was wrong because attention's seq^2 term dominates at seq ~800, not the
# FF block). Measured in F: full L=12 d256 11.7 h, no_mlp L=16 d192 13.0 h, each
# including ~2.2 h of CPU instance generation and ~0.5 h of eval. That puts the
# per-layer training cost at ~0.75 h (full d256) and ~0.64 h (no_mlp d192), from
# which the FF block is worth only ~0.13 h per layer at d256:
#
#   corner                   train   gen    eval   total
#   C  no_mlp L=12 d256       7.4 h  2.2 h  0.4 h  10.0 h   x3
#   B  full   L=16 d192      11.8 h  2.2 h  0.5 h  14.5 h   x3
#                                                  73.5 GPU-h
#
# --time 24:00:00 is 65% headroom over the longest cell's estimate. F ran with
# 20:00:00 and finished its longest cell in 13.1 h; B is the first `full` cell at
# d192 anywhere, so its per-layer cost is interpolated, not measured, and the
# extra margin is for that. --array %6 puts 6 of sganguli's 13 idle GPUs to work
# (sh_part, 11-09 22:19: 13 of 16 GPUs idle, 7 d max runtime) and leaves 7 for
# the group; wall clock ~15 h.
#
# ENVIRONMENT. Module stack only -- ${SCRATCH}/spin_gnn_venv is long purged --
# and python3 explicitly, since bare `python` is the system 2.7.

set -euo pipefail

# task: arch:num_layers:d_model:seed
# Corner C first: the single-variable MLP contrast, and the cheaper arm.
SPECS=(
  "no_mlp:12:256:0"
  "no_mlp:12:256:1"
  "no_mlp:12:256:2"
  "full:16:192:0"
  "full:16:192:1"
  "full:16:192:2"
)
TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
IFS=: read -r ARCH LAYERS DMODEL SEED <<< "${SPECS[$TASK]}"
# 10# forces base 10, as in stages C, D, E, H, I, J and F.
LAYERS=$((10#$LAYERS)); DMODEL=$((10#$DMODEL)); SEED=$((10#$SEED))
FF=1024
NHEAD=4
WARMUP=0
NNODES=96
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
from spin_gnn.config import SudokuReasoningConfig, SudokuReasoningTrainConfig
from spin_gnn.train_sudoku_transformer import resolve_max_len_margin
print("torch", torch.__version__, "numpy", numpy.__version__, "numba", numba.__version__)
print("arch modes", ARCH_MODES)
assert "no_mlp" in ARCH_MODES, ARCH_MODES
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))
# The max_len fix is load-bearing at N=96 (traces ~800 tokens) -- assert it is
# present rather than discovering a flush table in truncated_rate 14 h later.
m = resolve_max_len_margin(SudokuReasoningTrainConfig(),
                           SudokuReasoningConfig(n_nodes=96, degree=5, n_colors=4))
assert m == 194, f"expected a 2*96+2 generation margin, got {m}"
print("max_len margin", m)
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n96_cuthalf_M_2026-11-09"
# One directory PER TASK: sudoku_hparam_sweep appends to <out-dir>/sweep.csv and
# six cells run concurrently, so a shared directory would interleave rows from
# several writers into one file.
cell="${OUT}/t${TT}_d${DD}_ff${FFS}_h${HH}_${ARCH}_L${LL}_s${SEED}"
mkdir -p "$cell"

echo "task=${TASK}  n_nodes=${NNODES} arch=${ARCH} layers=${LAYERS} d_model=${DMODEL} ff=${FF} nhead=${NHEAD} seed=${SEED}"
echo "cell -> ${cell}"
date +'cell start %F %T'

python3 -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes "$NNODES" --degree 5 --n-colors 4 --shuffle-adj --fall \
    --d-model "$DMODEL" --nhead "$NHEAD" --dim-feedforward "$FF" \
    --num-layers "$LAYERS" --arch-mode "$ARCH" --seed "$SEED" \
    --warmup-epochs "$WARMUP" \
    --n-train-instances 240000 --epochs 50 \
    --n-val-instances 300 --batch-size 32 --n-per-bin 100 \
    --out-dir "$cell"

date +'cell end   %F %T'

# Tag order follows sudoku_hparam_sweep.py exactly: arch, then schedule, then
# warmup, then seed, each suffixed only when non-default. This is the same
# derivation stage F used, and it is checked against the written config.json
# below rather than trusted.
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
python3 - "${CKPT}/config.json" "$ARCH" "$LAYERS" "$NHEAD" "$DMODEL" "$FF" "$WARMUP" "$NNODES" "${cell}/sweep.csv" <<'PY'
import csv, json, sys
c = json.load(open(sys.argv[1]))
want_arch, want_L, want_h = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
want_d, want_ff, want_w = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
want_n = int(sys.argv[8])
# TOTAL n_params is written to sweep.csv, NOT to config.json; config.json carries
# n_trainable_params only. Corner A's artifacts (11-02/11-03) carry NEITHER in
# config.json and only n_params in sweep.csv, so every cross-corner size
# comparison has to come off sweep.csv on all four sides.
_sw = list(csv.DictReader(open(sys.argv[9])))
n_params_total = int(_sw[0]["n_params"]) if _sw and _sw[0].get("n_params") else -1
assert c["shuffle_adj"] is True, "LEAK: this cell is not shuffle_adj"
assert c["fall"] is True, "wrong ensemble: fall is not set"
assert c["n_nodes"] == want_n, (c["n_nodes"], want_n)
assert c["arch_mode"] == want_arch, (c["arch_mode"], want_arch)
assert c["num_layers"] == want_L and c["nhead"] == want_h
assert c["d_model"] == want_d, (c["d_model"], want_d)
assert c["dim_feedforward"] == want_ff, (c["dim_feedforward"], want_ff)
assert int(c.get("warmup_epochs", 0) or 0) == want_w, (c.get("warmup_epochs"), want_w)
assert c["max_len_margin"] == 2 * c["n_nodes"] + 2, c["max_len_margin"]
# PAIRING ASSERTION: the budget must equal corners A and D exactly, or this cell
# is not part of the same square and M0 is not a paired contrast.
assert int(c.get("n_train_instances") or 0) == 240000, c.get("n_train_instances")
assert int(c.get("epochs") or 0) == 50, c.get("epochs")
assert (c.get("lr_schedule") or "constant") == "constant", c.get("lr_schedule")
d, L = c["d_model"], c["num_layers"]
print("n_nodes=%d arch_mode=%s L=%d nhead=%d d_model=%d ff=%d warmup=%d"
      % (c["n_nodes"], c["arch_mode"], L, c["nhead"], d, c["dim_feedforward"],
         int(c.get("warmup_epochs", 0) or 0)))
print("  trainable %s   TOTAL n_params %s   max_len=%d (margin %d)"
      % (format(c.get("n_trainable_params", -1), ","),
         format(n_params_total, ","),
         c["max_len"], c["max_len_margin"]))
# The square's other three corners, for orientation only. A and D are measured;
# the two M corners are printed so a param count that lands outside this range
# is visible immediately rather than in the scorer 15 h later.
print("  the square's TOTAL n_params: A full L=12 d256 9,790,058   "
      "D no_mlp L=16 d192 8,929,514")
print("  per-layer attention-only term 4d^2+8d = %s  x L = %s  (N=48 part only;"
      % (format(4 * d * d + 8 * d, ","), format((4 * d * d + 8 * d) * L, ",")))
print("   the fixed term is embedding + position and grows with n_nodes, so only")
print("   the per-layer part transfers from the N=48 arithmetic -- printed, not")
print("   predicted.)")
PY

echo
echo "=== the transition curve for this cell (M0-M3) ==="
# IDENTICAL to stage F's grid and to the N=96 fall scan's, including the 0.32
# floor (the fall ensemble's rho_min at N=96 is 0.324, finding 43) and the
# default --seed-start, so all four corners of the square are paired point for
# point. This grid is NOT the N=48 grid, which bottoms out at 0.33. Never pool.
python3 -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.32 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_t${TT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.csv"

echo
echo "=== W: steps-bound? trainability reading (NOT an escape count) ==="
python3 - "${CKPT}/history.json" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
best = min(h, key=lambda r: r["nll"])
print("epochs %d  final train_nll %.4f  final val_nll %.4f  best val_nll %.4f at epoch %d"
      % (len(h), h[-1]["train_nll"], h[-1]["nll"], best["nll"], best["epoch"]))
print("overfit gap (final - best) = %+.4f" % (h[-1]["nll"] - best["nll"]))
# M3's never-censoring metric, printed per cell as it finishes so a corner that
# falls off the accuracy-to-solve relation is visible before the scorer runs.
print("final val TOKEN ACCURACY %.4f   [the square's other corners: A full L=12"
      " d256 0.4822, D no_mlp L=16 d192 0.4709; bar s*_M_acc = 0.0079]"
      % float(h[-1]["accuracy"]))
tail = [r["nll"] for r in h[-5:]]
print("last 5 val_nll:", " ".join("%.4f" % x for x in tail),
      "  drop over them %+.4f" % (tail[-1] - tail[0]))
# The 1.64 plateau is an N=48 number (finding 44) and is NOT applied at N=96.
print("N=96 leak-free control band to date: train_nll 1.7456-1.8320.")
print("This cell: %.4f -- a READING, not a verdict; N=96 has no measured plateau."
      % h[-1]["train_nll"])
print("steps-bound check:",
      "still falling at epoch 50 => steps-bound, and EVERY corner is"
      if tail[-1] < tail[0] else "val_nll flat or rising at the end")
PY
date +'task end  %F %T'
