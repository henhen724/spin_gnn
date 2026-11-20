#!/bin/bash
#SBATCH --job-name=sud2N192Ladder
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-1%2
#SBATCH --output=%x_%A_%a.out
#
# THE THIRD RUNG OF THE N LADDER: N=192, matched to the N=48 and N=96 fall
# scans cell for cell.
#
#   sbatch scripts/slurm/sudoku2_n192_ladder_2026-11-20.sbatch
#
# WALLTIME IS MEASURED, NOT EXTRAPOLATED. Job 53061369 (H100 80GB, 3 epochs x
# 4800 instances) timed 42.2 s/epoch to three consistent digits -- 64.8 ->
# 107.0 -> 149.2 s -- which is 35.2 min/epoch at 240000, so ~29.3 h of
# training per cell. Against N=96's 13.6 h that is 2.15x, i.e. the LINEAR
# bound of the sizing probe's 2.0x-3.9x range and not the quadratic one; the
# attention term does not dominate at this length on an H100. Instance
# generation is 19.4 s per 4800 (~16 min at 240k) and the 14x400 eval ran
# 1-1.75 h at N=96, so budget ~33 h per cell. --time=48:00:00 carries ~45%
# headroom, and 2 cells is ~66 GPU-h -- within a thousandth of what finding
# 45's entire six-cell N=96 grid cost (67 GPU-h).
#
# The same probe confirmed the sizing numbers end to end: longest training
# sequence 1598 + 386 margin -> max_len 1984 against the 1976 predicted
# (0.4% out), vocab 202, 10,089,162 trainable parameters.
#
# =====================================================================
# WHY, AND WHAT THE ON-DISK DATA ALREADY SAYS
#
# The brief is to train on graph instances large enough to SEE DD's fast/slow
# transition: T ~ log N above rho_c ~ 0.46, T ~ N^gamma (gamma -> 0.727) below
# it (finding 1+34, Dario's Sudoku_SC slides). Under the fall ensemble the
# slow phase is reachable at every size -- rho_min is 0.333 at N=48 and 0.320
# at N=192 (52ab5eb), both below rho_c. The INSTANCES are not the problem.
#
# THE MODELS ARE. Reading the two scans already on disk, at the SAME cell
# (L=12, d256, ff1024, 240k x 50ep, fall, shuffle_adj, seed 0):
#
#   N=48  t08: solve 0.7550 at rho 0.333 (T=13.79) -- still solving AT THE
#              GRID FLOOR, i.e. this is the one size where a trained model
#              operates BELOW rho_c, inside the slow phase. rho_50 is
#              CENSORED below the floor, which is finding 44's H1 failure.
#   N=96  t00: solve 0.0075 at rho 0.448 (T=12.05) and 0.0000 below.
#              It dies exactly AT rho_c. rho_50 = 0.679.
#   N=96  t04: the seed-1 twin, rho_50 = 0.564. Still above rho_c.
#
# So raising N from 48 to 96 moved the model OUT of the slow phase. That is
# the opposite of what "go bigger to see the transition" assumes.
#
# AND IT IS NOT THE PRODUCT-OF-DEDUCTIONS ARITHMETIC. Whole-instance solve
# rate cannot be compared across N, because solving needs ~N(1-rho)
# consecutive correct deductions, so solve must fall with N at fixed rho even
# at constant per-step skill. Normalising that away with
#
#     a_eff = solve_rate ** (1 / n_blank),     n_blank = round(N*(1-rho))
#
# -- the per-deduction accuracy implied by the solve rate, which is well
# defined because n_blank is constant inside every density bin -- the penalty
# SURVIVES, at matched DD depth:
#
#     T ~ 13.8   N=48  a_eff 0.9913
#     T ~ 12.0   N=96  a_eff 0.9118 (seed 0) / 0.9626 (seed 1)
#     T ~  5.2   N=48  a_eff 0.9999 ; N=96 0.9694 (s0) / 0.9928 (s1)
#
# The N-penalty is real and comparable in size to the seed term, which is why
# this grid buys BOTH seeds and not one.
#
# =====================================================================
# THE CELLS. Only n_nodes changes from finding 45's N=96 headline cell.
#   N=192 d=5 q=4, --fall --shuffle-adj, L=12, d_model 256, nhead 4,
#   ff 1024, pre-norm, lr 1e-3 constant, batch 32, n_val 300, no warmup,
#   240000 instances x 50 epochs (= 375k gradient steps, the protocol the
#   N=48 and N=96 scans both hold fixed).
#
#   task  seed  role
#     0     0   the third rung, paired to N=48 t08 and N=96 t00
#     1     1   the SEED YARDSTICK at this size, paired to N=96 t04
#
# =====================================================================
# PRE-REGISTRATION. Written before either cell runs. This grid is EXPECTED TO
# SHOW THE MODEL GETTING WORSE; that is registered here as the expected
# outcome so it cannot later be reported as a surprise or as a bad run.
#
# N1  THE N-PENALTY AT MATCHED DEPTH CONTINUES. a_eff at dd_mean_T ~ 12 is
#     lower at N=192 than at N=96, i.e. below 0.9118, the WORSE N=96 seed.
#     FALSIFIER: a_eff at matched T lands at or above 0.9626, the BETTER N=96
#     seed. Then the 48->96 drop was not a size effect, and the premise of
#     this whole ladder is wrong.
#
# N2  THE MODEL DOES NOT REACH THE SLOW PHASE AT N=192. rho_50 > rho_c = 0.46
#     for both seeds, continuing 0.679/0.564 at N=96.
#     FALSIFIER: rho_50 <= 0.46 for either seed. Then size does not push the
#     model out of the slow phase and N=96 was the anomaly, not N=48.
#
# N3  THE SEED YARDSTICK DOES NOT SHRINK. s*_192 = |rho_50(s0) - rho_50(s1)|
#     >= s*_96 = 0.119. s* is an N-specific quantity by standing rule (0.1484
#     is N=48's and was not reused at N=96).
#     FALSIFIER: s*_192 < 0.119, which would be welcome but would contradict
#     finding 45's "the seed term has to be REDUCED, not sampled".
#
# N4  TRAINABILITY, STATED SO IT CANNOT BE READ AS AN ESCAPE COUNT. The 1.64
#     plateau is an N=48 number and the 1.7456-1.8320 band is N=96's; N=192's
#     plateau has never been measured, so NO ESCAPE IS COUNTED ANYWHERE in
#     this run. Instead: both cells' val_nll is still descending at epoch 50
#     (the final epoch is the argmin to within 0.005 nats).
#     FALSIFIER: either cell's val_nll turns up -- the first overfitting seen
#     at 240k x 50ep in this programme.
#     BREAKING POINT: if both cells sit flat at their initial plateau for all
#     50 epochs, N=192 IS NOT TRAINABLE AT THIS BUDGET and that is the result,
#     reported as such -- not retried at a larger size or a longer run.
#
# N5  THE GRAMMAR HOLDS AT 2x THE SEQUENCE LENGTH. truncated_rate <= 0.01 in
#     every bin with solve_rate > 0.1, as at N=96 (max 0.0075 over 51 bins).
#     This is the first run to lean on a max_len of ~1976.
#     FALSIFIER: truncated_rate > 0.01 anywhere solve > 0.1 -- then max_len is
#     binding again and every deep-bin number here is a lower bound.
#
# WHAT FOLLOWS IF N1 CONFIRMS. The obstacle to seeing the transition in a
# trained network is the N-penalty, not the instance size, and raising N
# further makes it worse. The cheaper lever is finding 10: (degree, n_colors)
# buys DD depth about 6x more cheaply than N -- at N=60, (d=4,q=3) reaches
# mean depth 14.5, which (d=5,q=4) does not reach until N~333 -- and it is
# cheaper in tokens too, since length is N*(degree+3) + T + 4. That changes
# the ensemble's own rho_c and floor, so it is a DIFFERENT physics question
# from Dario's q=4 d=5 curve and is NOT a substitution to make silently. It
# is the registered follow-up, to be put to hshunt rather than assumed.
#
# =====================================================================
# TRAPS, carried verbatim in force from scripts/n96_control.py:
#
#  1. KEY COLLISION. d256 / ff1024 / warmup 0 are all DEFAULTS, so an N=192
#     cell reduces to `rung_key` (12, 4, 'full') -- THE SAME KEY as the N=48
#     and N=96 full L=12 rungs. n_nodes MUST be in any key that touches these
#     files, and these dirs stay OUT of SCAN_DIRS.
#  2. `n_trainable_params` is ABSENT from every N=96 artifact (added at N=48
#     on 11-04), so any cross-N size comparison uses `n_params` (total,
#     embeddings included) or says nothing.
#  3. The 1.64 plateau is an N=48 number. See N4: no escape count appears.
#  4. THE RHO GRIDS DIFFER BY SIZE -- N=48 ends 0.333, N=96 0.325, N=192
#     ~0.320 -- so the three rungs are NEVER POOLED. Cross-N comparison is on
#     dd_mean_T, or on a_eff at matched T, never on a shared rho axis.
#  5. s* = 0.1484 (N=48) and s*_96 = 0.119 are NOT reused here; N3 measures
#     this size's own yardstick.
#
set -euo pipefail

cd /home/users/hshunt/spin_gnn

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 \
    py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312

SEED=$SLURM_ARRAY_TASK_ID
SCAN=$SCRATCH/spin_gnn/sudoku_n192_ladder_2026-11-20
CELL=$SCAN/t0${SEED}_L12_ff01024_n240000_s${SEED}
mkdir -p "$CELL"

echo "=== commit at submit ==="
git log --oneline -1
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python3 - <<'PY'
import torch
import spin_gnn.sudoku_hparam_sweep  # noqa: F401
import scripts.rho_transition_eval   # noqa: F401
from spin_gnn.config import SudokuReasoningConfig, SudokuReasoningTrainConfig
from spin_gnn.train_sudoku_transformer import resolve_max_len_margin
assert torch.cuda.is_available(), "no GPU visible to torch"
m = resolve_max_len_margin(SudokuReasoningTrainConfig(),
                           SudokuReasoningConfig(n_nodes=192, degree=5, n_colors=4))
assert m == 386, f"expected a 2*192+2 generation margin, got {m}"
print("cuda", torch.cuda.get_device_name(0), "max_len margin", m)
PY

date +'cell start %F %T'
python3 -u -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 192 --degree 5 --n-colors 4 --shuffle-adj --fall \
    --d-model 256 --nhead 4 --dim-feedforward 1024 \
    --num-layers 12 --seed "$SEED" \
    --n-train-instances 240000 --epochs 50 \
    --n-val-instances 300 --batch-size 32 --n-per-bin 100 \
    --out-dir "$CELL"
date +'cell end   %F %T'

TAG="d0256_h04_L12_ff01024_lr1e-03_pre_shuf_fall"
if [ "$SEED" -ne 0 ]; then
    TAG="${TAG}_s$(printf '%02d' "$SEED")"
fi
CKPT="${CELL}/${TAG}"
[ -f "${CKPT}/model.pt" ] || { echo "expected checkpoint missing: ${CKPT}"; ls -1 "$CELL"; exit 1; }

echo
echo "=== the transition curve for this cell (N1, N2, N5) ==="
# Same 14 grid values as N=96, bottoming at 0.32 against this size's measured
# rho_min of 0.320+-0.013, so the floor note fires and labels where the
# ensemble stops instead of leaving a flat tail to be read as a plateau.
# --seed-start left at the default 6e10, so both seeds see the SAME 400
# instances per rho and are paired point for point. Identical grid VALUES do
# NOT license pooling across N -- see trap 4; the cross-N axis is dd_mean_T.
python3 -u -m scripts.rho_transition_eval \
    --checkpoint "$CKPT" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 0.40 0.35 0.32 \
    --n-per-bin 400 \
    --out-csv "$SCAN/rho_curve_t0${SEED}_${SLURM_ARRAY_JOB_ID}_${SEED}.csv"

echo
echo "=== N4: trainability, with NO escape count (N=192's plateau is unmeasured) ==="
python3 - "${CKPT}/history.json" <<'PY'
import json, sys
h = json.load(open(sys.argv[1]))
best = min(h, key=lambda r: r["nll"])
print("epochs %d  final train_nll %.4f  final val_nll %.4f  best val_nll %.4f at epoch %d"
      % (len(h), h[-1]["train_nll"], h[-1]["nll"], best["nll"], best["epoch"]))
print("overfit gap (final - best) = %+.4f   [N4 falsifier: > 0]"
      % (h[-1]["nll"] - best["nll"]))
tail = [r["nll"] for r in h[-5:]]
print("last 5 val_nll:", " ".join("%.4f" % x for x in tail),
      "  drop over them %+.4f" % (tail[-1] - tail[0]))
# N4's BREAKING POINT: flat for all 50 epochs means not trainable at this
# budget, which is the result and is not retried larger.
span = max(r["train_nll"] for r in h) - min(r["train_nll"] for r in h)
print("train_nll span over the whole run %.4f   [breaking point: ~0 = flat]" % span)
PY
date +'task end  %F %T'
