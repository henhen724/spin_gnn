#!/bin/bash
#SBATCH -J spin_gnn_sat2_seedfix
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --array=0-1%1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%A_%a.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Hypothesis: with torch.manual_seed(sat_cfg.seed) now called before model
# construction (train_sat_transformer2.py), two runs with seed=0 at the
# current-best config (d_model=128, nhead=2, num_layers=16, lr=3e-4,
# norm_first=True) should show much less variance than the previous
# 12.3%/4.3% spread seen at this exact config (both of those were run
# before this fix existed). A CPU regression test already confirmed
# bit-identical weights/history for same-seed reruns; this checks whether
# that holds up (or at least greatly shrinks the gap) on the real GPU
# training path, where cuDNN can still introduce some nondeterminism this
# fix doesn't pin down.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
cd "${SLURM_SUBMIT_DIR}"

# Per-task subdirectory: both tasks use identical hyperparameters
# (including --seed 0, deliberately, to test the fix), so the checkpoint
# tag alone would collide between the two array tasks.
OUT_DIR="${SCRATCH}/spin_gnn/runs/2026-10-15_seed-fix-validation/task${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUT_DIR"

echo "task ${SLURM_ARRAY_TASK_ID}: seed=0 (d=128,h=2,L=16,lr=3e-4,norm_first=True) -- seed-fix validation"

python -m spin_gnn.sat2_hparam_sweep \
    --n-vars 16 \
    --alpha 1.25 \
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
