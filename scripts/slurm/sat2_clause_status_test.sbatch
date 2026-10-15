#!/bin/bash
#SBATCH -J spin_gnn_sat2_cs
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --array=0-1%1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%A_%a.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Hypothesis: explicit [SATISFIED]/[REACTIVATED] clause-status tokens (opt-in
# via SATReasoningConfig.use_clause_status, see ksat2_tokens.Vocab) remove an
# implicit clause-bookkeeping burden from attention and should raise
# solve_rate above the pre-norm baseline at this exact config (d_model=128,
# nhead=2, num_layers=16, lr=3e-4, norm_first=True): 12.3% and 4.3% across
# two existing independent runs (sweep_m020_h02_L16_pre.csv /
# sweep_m020_h02_L16_pre.csv, see notebooks/results_prenorm_scan.ipynb).
# Only ONE thing changes here (use_clause_status=True) -- everything else,
# including --seed (always the default 0, same as both baseline runs: data
# is identical, so the baseline's own 12.3%/4.3% spread was purely from
# unseeded model init, not data). Two array tasks = two independent process
# launches = two independent model-init samples, the same noise source the
# baseline had, run sequentially (%1) since this is a single small job.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
cd "${SLURM_SUBMIT_DIR}"

# Per-task subdirectory: both tasks use identical hyperparameters (including
# --seed 0), so sat2_hparam_sweep.py's tag alone would collide between the
# two array tasks and the second would silently overwrite the first's
# checkpoint -- caught before submitting, not after.
OUT_DIR="${SCRATCH}/spin_gnn/runs/2026-10-14_clause-status-tokens/task${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUT_DIR"

echo "task ${SLURM_ARRAY_TASK_ID}: use_clause_status=True seed=0 (d=128,h=2,L=16,lr=3e-4,norm_first=True)"

python -m spin_gnn.sat2_hparam_sweep \
    --n-vars 16 \
    --alpha 1.25 \
    --use-clause-status \
    --seed 0 \
    --d-model 128 \
    --nhead 2 \
    --num-layers 16 \
    --norm-first \
    --lr 3e-4 \
    --n-train-instances 3000 \
    --n-val-instances 300 \
    --batch-size 64 \
    --epochs 150 \
    --out-dir "$OUT_DIR" \
    "$@"
