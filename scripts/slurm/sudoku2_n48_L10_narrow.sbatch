#!/bin/bash
#SBATCH --job-name=sud2N48L10
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=13:00:00
#SBATCH --output=%x_%j.out
#
# NARROW THE LEAK-FREE N=48 KNEE: ONE CELL AT L=10.
#
#   sbatch scripts/slurm/sudoku2_n48_L10_narrow.sbatch
#
# WHY. Finding 38 closed the ladder finding 33 left open, but only to a bracket
# three rungs wide. At rho=0.60, six depth bins T=3..8, realized density pinned
# at 0.5836 (spread 0.000488):
#
#     L=6   0/6 bins   pool 0.6153           (50099945, 10-29)
#     L=8   3/6 bins   pool 0.8660   0.986 0.959 0.927 0.867 0.775 0.857
#     L=12  6/6 bins   pool 0.9780   1.000 0.998 1.000 0.986 0.984 0.957
#
# so L*(48, leak-free, ff=1024, rho=0.60) is in (8, 12] and L=9, 10, 11 are
# unmeasured. One cell at the midpoint halves the bracket, and it is the cell
# the hyper-parameter scan needs as its centre: the scan asks whether required
# reasoning depth tracks MLP depth, and that question is only sharp once the
# depth knee itself is known to better than three rungs.
#
# ONE CHANGE. Every argument below is job 50527633's verbatim (N=48, degree 5,
# q=4, d_model 256, nhead 4, ff=1024, pre-norm, lr 1e-3, batch 32, 240000 x
# 100ep, --shuffle-adj); `num_layers` 12 -> 10 is the only difference, and it is
# not an INSTANCE_KEY, so this cell pools with all five existing cells on the
# same n=1500 instances and the same depth bins. The five are re-evaluated here
# rather than quoted, so every contrast below is paired instance by instance.
#
# PRE-REGISTERED, written before the cell was submitted:
#
#  (a) HEADLINE, moderate-LOW confidence: L=10 clears 0.90 in ALL SIX rho=0.60
#      bins, i.e. L* = 10 and the bracket narrows to (8, 10]. The binding bin is
#      T=7, where linear interpolation between L=8 (0.775) and L=12 (0.984)
#      gives 0.880 -- BELOW the gate -- and the prediction rests entirely on
#      per-layer gains decelerating, as 3->4->6 did at this N.
#      FALSIFIER: any bin below 0.90, which puts L* in (10, 12] and says the
#      top two rungs are doing the work. This one is close to a coin flip and
#      is informative in both directions, which is why the cell is worth 9 h.
#
#  (b) POOL: L=10 lands in [0.93, 0.97] on the shared n=1500 pool, strictly
#      between L=8's 0.8660 and L=12's 0.9780 and resolvably above L=8 by
#      paired McNemar (p < 1e-10). The linear midpoint is 0.9220 and
#      deceleration should put it above that.
#      FALSIFIER A: >= 0.978, a tie with the anchor -- then two of the four
#      extra layers buy nothing on the pool whatever the bins say.
#      FALSIFIER B: <= 0.9220, below the linear midpoint, i.e. depth gains
#      ACCELERATE between 8 and 12, which no cell in this project has shown.
#      Prior: my interval predictions here are mostly underestimates (findings
#      23 and 38 both missed on the high side), so a miss above 0.97 would be
#      the likelier of the two failures.
#
#  (c) CONDITIONAL, scored only if (a) is falsified: the bins L=10 misses are
#      the DEEPEST ones and form a contiguous run at the top, as at L=8 (the
#      three deepest) and at finding 22's N=96 L=6.
#      FALSIFIER: a miss at a shallow T together with a clear at a deeper T,
#      which would break the monotone L*(T) reading findings 22/38 rest on.
#
#  (d) LOW CONFIDENCE, the controls. Realized density in the rho=0.60 bins
#      reproduces 0.5836 to within 0.001, and the repeat evaluations of the
#      five reference cells agree instance for instance with 50527633/50527634.
#      FALSIFIER: any disagreement, which would retire the determinism check
#      finding 38 leans on and invalidate the pairing across jobs.
#
# COST. 20 min fixed + 49.9 min per layer, measured from the 10-29 and 10-31
# cell mtimes: L=10 ~8.6 h, plus ~15 min for two evals over six checkpoints.
# --time=13:00:00 is ~40% headroom. sganguli's maximum is 7 d (sh_part, 10-31).
#
# ENVIRONMENT. ${SCRATCH}/spin_gnn_venv is dead (90-day purge), so this is the
# module stack only and calls python3 explicitly: without the venv on PATH,
# bare `python` is the system 2.7. The smoke check turns an environment failure
# into a 30-second job instead of a 9-hour one that dies in the evals.

set -euo pipefail
NNODES=48
NTRAIN=240000
EPOCHS=100
LAYERS=10

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

echo "=== smoke check ==="
python3 - <<'PY'
import torch, numpy, numba
import spin_gnn.sudoku_hparam_sweep, spin_gnn.sudoku_depth_eval, spin_gnn.sudoku_agg_eval
print("torch", torch.__version__, "numpy", numpy.__version__, "numba", numba.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert torch.cuda.is_available(), "no GPU visible to torch"
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n${NNODES}_leakfree_L10_$(date +%Y-%m-%d)"
mkdir -p "$OUT"
echo "N=${NNODES} layers=${LAYERS} n_train=${NTRAIN} epochs=${EPOCHS} -> ${OUT}"

COMMON=(--n-nodes "$NNODES" --degree 5 --n-colors 4 --shuffle-adj
        --d-model 256 --nhead 4 --dim-feedforward 1024
        --n-train-instances "$NTRAIN" --epochs "$EPOCHS"
        --n-val-instances 300 --batch-size 32 --n-per-bin 100)

cell="${OUT}/L10"
echo
echo "=== cell L=${LAYERS} ==="
date +'cell start %F %T'
python3 -m spin_gnn.sudoku_hparam_sweep "${COMMON[@]}" \
    --num-layers "$LAYERS" --out-dir "$cell"
date +'cell end   %F %T'

CKPTS=("${cell}/d0256_h04_L10_ff01024_lr1e-03_pre_shuf")

# Every existing rung of this ladder, re-evaluated so the contrasts are paired.
# Guarded: a missing cell must not fail the evals.
DEEP_DIR="${SCRATCH}/spin_gnn/sudoku_n48_leakfree_lstar_2026-10-31"
REF_DIR="${SCRATCH}/spin_gnn/sudoku_n48_leakfree_lstar_2026-10-29"
for spec in "${DEEP_DIR}:12" "${DEEP_DIR}:08" "${REF_DIR}:06" "${REF_DIR}:04" "${REF_DIR}:03"; do
    d="${spec%:*}"; L="${spec##*:}"
    ref="${d}/L${L}/d0256_h04_L${L}_ff01024_lr1e-03_pre_shuf"
    if [ -d "$ref" ]; then CKPTS+=("$ref"); else echo "skipping missing reference $ref"; fi
done

echo
echo "=== pre-regs (a)+(c)+(d): per-T solve_rate on ONE set of depth bins ==="
python3 -m spin_gnn.sudoku_depth_eval \
    --checkpoint "${CKPTS[@]}" \
    --rho 0.75 0.60 --n-pool 1500 --min-per-bin 60 \
    --out-csv "${OUT}/depth_L10_${SLURM_JOB_ID}.csv"

echo
echo "=== pre-reg (b): the shared n=1500 pool, disjoint from training ==="
python3 -m spin_gnn.sudoku_agg_eval \
    --checkpoint "${CKPTS[@]}" \
    --n-instances 1500 --seed-start 50000000000 \
    --out-csv "${OUT}/sharedpool_n1500_${SLURM_JOB_ID}.csv"

echo
echo "=== summaries ==="
cat "${OUT}/sharedpool_n1500_${SLURM_JOB_ID}.csv"
