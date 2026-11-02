#!/bin/bash
#SBATCH --job-name=sud2RhoBefore
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#
# THE "BEFORE" CURVES: solve rate along rho for the eight existing N=48
# leak-free cells, on the DEFAULT (non-fall) ensemble they were trained on.
#
#   sbatch scripts/slurm/sudoku2_n48_rho_curve_before_2026-11-02.sbatch
#
# WHY THIS IS A SEPARATE, CHEAP JOB. hshunt asked how success rate scales
# across the transition. The fall-colouring fix (commit 5c17d3c) is what makes
# the slow side reachable at all, but it changes the GRAPH ensemble, so no
# fall-trained cell can be compared instance-for-instance against anything
# trained before today. These eight checkpoints already exist and cost nothing
# but ~30 min of GPU to re-read on a rho grid, and they answer the part of the
# question that needs no new training: what does the curve look like on the
# ensemble every result in this project so far was measured on?
#
# THE GRID STOPS AT 0.45 ON PURPOSE. These cells were trained on the
# unconstrained ensemble, whose greedy removal path floors at rho_min = 0.469
# +- 0.049 at N=48 (200 seeds, measured 11-02). Requesting anything below that
# does not produce a lower-density bin -- eval_solve_by_rho silently returns
# the floor again, so the extra points would be ONE density measured several
# times, plotted as though it were a tail. 0.45 is already at the floor for
# most draws and is included precisely so rho_transition_eval's floor note
# fires once and labels where the ensemble stops. The fall scan's grid reaches
# 0.33 because its floor is 0.332 +- 0.024; that difference IS the finding.
#
# PAIRING. --seed-start is left at rho_transition_eval's default 6e10, which
# the fall scan also uses, so within an ensemble every checkpoint sees exactly
# the same 400 instances at each rho and the curves are paired point by point.
# Across the two ensembles they are NOT paired and must not be McNemar'd; the
# honest cross-ensemble comparison is curve shape against dd_mean_T, which is
# why that column is recorded per bin.
#
# 6e10 is far above every training seed ever used here (max 240000) and above
# sudoku_agg_eval's 5e10 pool, so none of these instances has been trained on.
#
# COST. 8 checkpoints x 11 rho x 400 instances = 35,200 generations. Job
# 50728837 did 72,000 at this size in 1:03:50, i.e. ~19/s, so ~31 min plus
# model loading. --time 02:00:00 is ~4x headroom and the job is inside
# sganguli's 7 d limit and 32 GB/core cap (sh_part, 11-02).

set -euo pipefail

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

echo "=== smoke check ==="
python3 - <<'PY'
import torch, numpy
import scripts.rho_transition_eval  # noqa: F401
print("torch", torch.__version__, "numpy", numpy.__version__)
assert torch.cuda.is_available(), "no GPU visible to torch"
print("cuda", torch.cuda.get_device_name(0))
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n48_rho_curves_2026-11-02"
mkdir -p "$OUT"

S="${SCRATCH}/spin_gnn"
CKPTS=(
  "${S}/sudoku_n48_leakfree_lstar_2026-10-31/L12/d0256_h04_L12_ff01024_lr1e-03_pre_shuf"
  "${S}/sudoku_n48_leakfree_L10_2026-10-31/L10/d0256_h04_L10_ff01024_lr1e-03_pre_shuf"
  "${S}/sudoku_n48_leakfree_seed1_2026-11-01/L10/d0256_h04_L10_ff01024_lr1e-03_pre_shuf_s01"
  "${S}/sudoku_n48_leakfree_lstar_2026-10-31/L08/d0256_h04_L08_ff01024_lr1e-03_pre_shuf"
  "${S}/sudoku_n48_leakfree_seed1_2026-11-01/L08/d0256_h04_L08_ff01024_lr1e-03_pre_shuf_s01"
  "${S}/sudoku_n48_leakfree_lstar_2026-10-29/L06/d0256_h04_L06_ff01024_lr1e-03_pre_shuf"
  "${S}/sudoku_n48_leakfree_lstar_2026-10-29/L04/d0256_h04_L04_ff01024_lr1e-03_pre_shuf"
  "${S}/sudoku_n48_leakfree_lstar_2026-10-29/L03/d0256_h04_L03_ff01024_lr1e-03_pre_shuf"
)
for c in "${CKPTS[@]}"; do
    [ -f "${c}/model.pt" ] || { echo "missing checkpoint: ${c}"; exit 1; }
done

date +'start %F %T'
python3 -m scripts.rho_transition_eval \
    --checkpoint "${CKPTS[@]}" \
    --rho-grid 0.95 0.90 0.85 0.80 0.75 0.70 0.65 0.60 0.55 0.50 0.45 \
    --n-per-bin 400 \
    --out-csv "${OUT}/rho_curve_default_ensemble_${SLURM_JOB_ID}.csv"
date +'end   %F %T'
