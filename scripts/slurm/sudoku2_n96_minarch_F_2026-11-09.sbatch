#!/bin/bash
#SBATCH --job-name=sud2N96MinarchF
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --array=0-3%4
#SBATCH --output=%x_%A_%a.out
#
# STAGE F: DOES THE MINIMAL ARCHITECTURE REPLICATE AT A SECOND PROBLEM SIZE?
#
#   sbatch scripts/slurm/sudoku2_n96_minarch_F_2026-11-09.sbatch
#
# WHAT IS BEING ASKED. Stages A-J cut machinery out of the N=48 model and found
# a standing minimum: attention-only (`no_mlp`), L=16, d192, 2,504,890 trainable
# at rho_50 0.580 against the N=48 hub's 4,899,898 at 0.592 -- 1.96x smaller at
# no separable cost. Every one of those cuts was measured at ONE problem size.
# F asks whether the surviving architecture is a fact about direct deduction or
# a fact about N=48, by running it at N=96 against a PAIRED complete-model
# control. This is a REPLICATION question, not another cut: F deliberately does
# not isolate depth, width and the MLP at N=96 (that would be another ladder and
# another 200 GPU-h). It asks the system-level question the N=48 ladder earned
# the right to ask -- "does a 4x smaller MLP-free stack still reach the complete
# model's rho_50" -- and nothing finer.
#
# WHY THIS IS CHEAP NOW. scripts/n96_control.py (11-05) found that the paired
# control ALREADY EXISTS and does not need re-running: eight cells in
# sudoku_n96_fall_scan_2026-11-02 and sudoku_n96_cosine_2026-11-03 were trained
# leak-free (`shuffle_adj` AND `fall`) at the SAME budget the N=48 minarch grid
# uses (240k instances x 50 epochs, batch 32, no warmup, d256 ff1024, constant
# lr 1e-3) and evaluated on the IDENTICAL 14-point x 400-instance rho grid, with
# realized_rho matching point for point. Measured there:
#
#   N=96 full L=12  240k x 50ep   rho_50 0.621   seeds 0.679 / 0.564   9,790,058
#   N=96 full L=8   240k x 50ep   rho_50 0.760   seeds 0.691 / 0.829   6,631,018
#   N=96 full L=16  240k x 50ep   rho_50 0.613   one seed             12,949,098
#   N=96 full L=12  240k x 50 cos rho_50 0.778   seeds 0.830 / 0.726   (finding 47)
#
# So F buys only what is missing: the STRIPPED ARM at three seeds, and the THIRD
# CONTROL SEED. No N=96 rung anywhere has three seeds, and at two the L=12
# control median is a coin flip between 0.564 and 0.679.
#
# THE REFERENCE IS `full` L=12 d256, not L=16. It is the headline cell of the
# N=96 fall scan, it is the rung with the most seeds, and at N=96 depth past 12
# did NOT clearly pay for the complete model (L=16 0.613 vs L=12 0.621, one seed
# against two, well inside the yardstick below). L=16 d256 is left at one seed
# on purpose: a depth-matched control would cost 17 h for a comparison that J0
# just showed is not the axis that matters.
#
# =====================================================================
# THE GRID. 4 cells, all N=96 d=5 q=4, --fall --shuffle-adj, nhead 4, ff 1024,
# pre-norm, constant lr 1e-3, batch 32, 240k x 50 epochs, n_val 300, NO warmup.
# Every one of those is the existing control's setting; n_nodes is the only
# thing that differs from stage J, and arch/d/L the only things that differ
# from the existing N=96 control. The stripped seeds run FIRST so that a
# trainability failure shows up in the first wave rather than after 14 h.
#
#   task  arch     L   d     seed  role                              est h
#     0   no_mlp   16  192    0    the replicate                       7.5
#     1   no_mlp   16  192    1    the replicate                       7.5
#     2   no_mlp   16  192    2    the replicate                       7.5
#     3   full     12  256    2    the MISSING THIRD CONTROL SEED     14.0
#
# =====================================================================
# PRE-REGISTRATION. Written and committed before any cell runs; scored by
# scripts/stagef_score.py, which is registered before the first cell lands.
#
# THE YARDSTICK, FROZEN HERE AND NOT RECOMPUTED. s* = 0.1484 is an N=48 number
# (the pooled seed spread of the N=48 ladder) and is NOT reused at N=96. The
# N=96 yardstick is the mean |max-min| rho_50 over the three two-seed N=96
# rungs that exist today:
#     L=12 fall   0.679 - 0.564 = 0.115
#     L=8  fall   0.829 - 0.691 = 0.138
#     L=12 cosine 0.830 - 0.726 = 0.104
#     s*_96 = 0.119
# It is frozen at 0.119 for every verdict below. It is a TWO-SEED estimate of a
# spread and therefore weak; task 3 improves it, and the re-estimate is reported
# alongside but NEVER substituted into these verdicts after the fact.
#
# F0  THE HEADLINE. The stripped arm reaches the complete model's rho_50 at
#     N=96: |rho_50(no_mlp L=16 d192) - rho_50(full L=12 d256)| <= s*_96 =
#     0.119, with the control median taken over its THREE seeds once task 3
#     lands. Then the minimal architecture is a fact about the task, not about
#     N=48, and the headline becomes "~4x fewer parameters at no separable cost,
#     at two problem sizes".
#     FALSIFIER, AND IT NAMES THE DIRECTION. If the stripped arm is HIGHER by
#     more than 0.119, the N=48 minimum does NOT transfer: the MLP (or the width,
#     or both) is load-bearing at N=96 and every "smallest architecture" claim in
#     this programme is an N=48 claim and must be relabelled as one. If it is
#     LOWER by more than 0.119, do not celebrate -- that is evidence the N=96
#     CONTROL is undertrained at this budget (see the plateau note below), not
#     that stripping helps, and the next move is a step-budget check on the
#     control rather than a victory lap.
#
# F1  IT TRAINS AT ALL. 3 of 3 stripped seeds produce a MEASURED rho_50 -- the
#     solve-rate curve crosses 0.5 somewhere on the 14-point grid rather than
#     censoring at the top.
#     FALSIFIER: any seed censors. Then F0 is unreadable as a level comparison
#     and the result is the BREAKING POINT -- "the attention-only stack that
#     works at N=48 does not train at N=96 at this budget" -- which is a result
#     and is reported as the headline, not as a failed run. Note which: all
#     three censoring is an architecture statement; one of three is the seed
#     story of finding 51 and needs the other two read on mean_solve.
#
# F2  THE CONTROL IS NOT ONE DRAW. The third control seed lands inside the
#     existing [0.564, 0.679] bracket, so the 3-seed control median stays within
#     s*_96 of the 2-seed median 0.621.
#     FALSIFIER: it lands outside. Then the N=96 control is seed-dominated at
#     two seeds, F0 survives only as an ORDERING ("the stripped arm is / is not
#     worse"), and the levels are not quotable. This is the single most likely
#     way F ends up weaker than it looks, which is exactly why task 3 is in the
#     job instead of being assumed.
#
# F3  THE N=48 STANDING MINIMUM DOES NOT MOVE ON N=96 EVIDENCE. Stated as a
#     registration so that no verdict here is later read as one: the standing
#     minimum is an N=48 ladder result (rule H3 = I3 = J3) and N=96 cells are a
#     different problem with a different grid floor. F can CORROBORATE or
#     UNDERMINE the N=48 minimum; it can never replace it.
#
# WATCH ITEMS (not hypotheses; they decide what the numbers MEAN).
#   * NO N=96 PLATEAU LEVEL EXISTS. Finding 44's train-NLL plateau of 1.64 is an
#     N=48 number. Every leak-free N=96 cell to date sits at 1.7456-1.8320,
#     which is NOT eight failures to train -- the plateau at N=96 has never been
#     measured. So NO escape count is quoted for any F cell. Final train NLL is
#     printed and compared ONLY N=96-to-N=96: a stripped cell far above the
#     control's 1.75-1.83 band is the trainability reading, and it is a reading,
#     not a verdict.
#   * STEPS-BOUND? 240k x 50 is 375k steps, chosen to pair with the control, not
#     because it is enough for N=96. If the control's val_nll is still falling at
#     epoch 50 then BOTH arms are steps-bound and F0 compares two undertrained
#     models -- true of the comparison, and worth saying, but it does not void
#     the pairing, which is the point of pairing.
#   * TRUNCATION. max_len margin is 2*n_nodes+2 = 194 at N=96; traces run to
#     ~800 tokens. truncated_rate should be ~0. It is asserted in the smoke
#     check because this is the axis that silently bound finding 44.
#   * n_trainable_params DOES NOT EXIST in the old N=96 control artifacts (the
#     field was added 11-04). The new cells have it. Size comparisons against
#     the old control therefore use TOTAL n_params on both sides, and the
#     assertion below prints both numbers so neither is quoted by mistake.
#
# =====================================================================
# COST. The stripped arm has no measurement to stand on, so this is an estimate
# and --time carries the uncertainty. From the N=96 fall scan's measured law
# (8.8 ms/step/layer at d256 full, batch 32, seq ~780): dropping the FF block
# removes 2*d*ff = 524k of the 786k per-layer linear FLOPs, and d192 scales the
# remainder by (192/256)^2 = 0.56 while attention's seq^2*d term (~25% of cost
# at N=96) scales by 0.75. That puts the stripped layer near 0.33x of a full
# d256 layer, i.e. ~2.9 ms/step/layer, so 16 layers x 375k steps ~ 4.8 h.
#
#   cell                 train   gen    eval   total
#   no_mlp L=16 d192      4.8 h  2.2 h  0.5 h   7.5 h   x3
#   full   L=12 d256     11.4 h  2.2 h  0.4 h  14.0 h        (train MEASURED)
#                                              36.5 GPU-h
#
# Instance generation is 2.2 h of GPU idling on a CPU loop at N=96 (33.4 ms per
# instance, measured 11-02) and is budgeted rather than optimized away, for the
# same reason as the fall scan: parallelizing the generator is a code change
# this run should not also be testing.
#
# --time 20:00:00 is 43% headroom over the one cell whose training time is
# MEASURED and 2.7x the estimated stripped cell, which is the right margin when
# the 0.33x factor is derived rather than observed. --array %4 runs all four at
# once and still leaves 10 of sganguli's 16 GPUs for the rest of the group
# (sh_part, 11-09: 14 of 16 idle, 7 d max runtime); wall clock ~14 h.
#
# ENVIRONMENT. Module stack only -- ${SCRATCH}/spin_gnn_venv is long purged --
# and python3 explicitly, since bare `python` is the system 2.7.

set -euo pipefail

# task: arch:num_layers:d_model:seed
# The three stripped seeds run first so a trainability failure (F1) surfaces in
# hours rather than after the 14 h control cell.
SPECS=(
  "no_mlp:16:192:0"
  "no_mlp:16:192:1"
  "no_mlp:16:192:2"
  "full:12:256:2"
)
TASK="${SLURM_ARRAY_TASK_ID:?this script is a job array; submit it with sbatch}"
IFS=: read -r ARCH LAYERS DMODEL SEED <<< "${SPECS[$TASK]}"
# 10# forces base 10, as in stages C, D, E, H, I and J.
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

OUT="${SCRATCH}/spin_gnn/sudoku_n96_minarch_F_2026-11-09"
# One directory PER TASK: sudoku_hparam_sweep appends to <out-dir>/sweep.csv and
# four cells run concurrently, so a shared directory would interleave rows from
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
# warmup, then seed, each suffixed only when non-default. For task 3 (full,
# warmup 0) this reduces to EXACTLY the fall scan's tag with an _s02 suffix,
# which is what makes it the same rung as the existing two control seeds.
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
# TOTAL n_params is written to sweep.csv, NOT to config.json (checked 11-09);
# config.json carries n_trainable_params only. The old N=96 control artifacts
# carry NEITHER in config.json and only n_params in their sweep.csv, so the
# cross-size size comparison has to come off sweep.csv on both sides.
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
# PAIRING ASSERTION: these must equal the existing N=96 control's budget exactly,
# or task 3 is not a third seed of that rung and F0 is not a paired contrast.
assert int(c.get("n_train_instances") or 0) == 240000, c.get("n_train_instances")
assert int(c.get("epochs") or 0) == 50, c.get("epochs")
assert (c.get("lr_schedule") or "constant") == "constant", c.get("lr_schedule")
d, L = c["d_model"], c["num_layers"]
# The N=48 arithmetic (4*d^2+8*d per layer, 630*d+58 fixed) is NOT asserted
# here: the fixed term is embedding + position and both grow with n_nodes, so
# only the PER-LAYER part transfers. Printed, not predicted.
print("n_nodes=%d arch_mode=%s L=%d nhead=%d d_model=%d ff=%d warmup=%d"
      % (c["n_nodes"], c["arch_mode"], L, c["nhead"], d, c["dim_feedforward"],
         int(c.get("warmup_epochs", 0) or 0)))
print("  trainable %s   TOTAL n_params %s   max_len=%d (margin %d)"
      % (format(c.get("n_trainable_params", -1), ","),
         format(n_params_total, ","),
         c["max_len"], c["max_len_margin"]))
print("  per-layer attention-only term 4d^2+8d = %s  x L = %s  (N=48 part only)"
      % (format(4 * d * d + 8 * d, ","), format((4 * d * d + 8 * d) * L, ",")))
print("  SIZE COMPARISONS AGAINST THE OLD N=96 CONTROL USE TOTAL n_params ON BOTH")
print("  SIDES -- n_trainable_params is absent from the 11-02/11-03 artifacts.")
PY

echo
echo "=== the transition curve for this cell (F0-F2) ==="
# IDENTICAL to the N=96 fall scan's grid, including the 0.32 floor (the fall
# ensemble's rho_min at N=96 is 0.324, finding 43) and the default --seed-start,
# so every F cell is paired point for point with the existing control cells.
# This grid is NOT the N=48 grid: that one bottoms out at 0.33. Never pool them.
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
tail = [r["nll"] for r in h[-5:]]
print("last 5 val_nll:", " ".join("%.4f" % x for x in tail),
      "  drop over them %+.4f" % (tail[-1] - tail[0]))
# The 1.64 plateau is an N=48 number (finding 44) and is NOT applied at N=96.
# The only in-size reference is the existing control band.
print("N=96 leak-free control band to date: train_nll 1.7456-1.8320.")
print("This cell: %.4f -- a READING, not a verdict; N=96 has no measured plateau."
      % h[-1]["train_nll"])
date_note = "still falling at epoch 50 => steps-bound, and BOTH arms are"
print("steps-bound check:", date_note if tail[-1] < tail[0] else "val_nll flat or rising at the end")
PY
date +'task end  %F %T'
