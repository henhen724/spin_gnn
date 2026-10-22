#!/bin/bash
#SBATCH --job-name=sud2d256x16
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=%x_%j.out
#
# Does the data-scaling gain continue, and does it reach k0?
#
# ONE change against 48854233 (the d256 run that took agg_solve_rate
# 0.847 -> 0.973): n_train_instances 20000 -> 80000. d_model 256, nhead 4,
# 12 layers, ff 1024, lr 1e-3, batch 32, epochs 300, n_val 300, --n-per-bin
# 100 all held at 48854233's values.
#
# WHY. Training-set size was pinned at 5000 for all 31 runs before today and
# turned out to be the binding constraint: 5000 -> 20000 gave the largest
# single gain in 34 runs. Width was the other win, but 48869017 showed the two
# are only partly additive -- d384 x 20000 reached k0 0.91, short of the 0.95
# I pre-registered and well short of the 0.99 that width alone gave at 5000.
# So the question is whether the remaining weakness at rho=1.0 responds to
# more data or only to more parameters.
#
# PRE-REGISTERED PREDICTION.
#  (a) agg: 0.973 is already near this eval's ceiling and n=300 cannot resolve
#      small gains (SE ~0.009), so I predict a *tie* on agg -- anything inside
#      0.973 +/- 0.02 must be reported as a tie, not an improvement. Any real
#      ranking has to come from re-scoring on 48912134's shared 1500-instance
#      pool, which is where this checkpoint should be sent afterwards.
#  (b) k0 >= 0.85, continuing 0.24 (5000) -> 0.60 (20000). This is the real
#      target and the falsifiable part.
#  (c) mean_o1 >= 0.96, up from 0.918.
# If k0 instead stays near 0.60, data scale does not fix rho=1.0 at all and
# the width result is a different mechanism -- which is more informative than
# a win, and would point the next run at d512 x 20000 rather than more data.
#
# Runtime: 48854233 did 300 epochs over 20000 instances in 1:24, so 80000 is
# ~5.6h plus ~4x the one-off DD pool construction. 8h leaves margin without
# the 140x over-request of 48868696. Memory: 48854233 peaked at 1.05GB RSS for
# 20000 held instances, so 80000 is ~4.3GB; 32G is already generous.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_hparam_sweep_2026-10-21_d256_data16x"
mkdir -p "$OUT_DIR"

echo "d_model=256 nhead=4 num_layers=12 dim_feedforward=1024 epochs=300 n_train=80000"

python -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 24 \
    --degree 5 \
    --n-colors 4 \
    --d-model 256 \
    --nhead 4 \
    --num-layers 12 \
    --dim-feedforward 1024 \
    --n-train-instances 80000 \
    --n-val-instances 300 \
    --batch-size 32 \
    --epochs 300 \
    --n-per-bin 100 \
    --out-dir "$OUT_DIR" \
    "$@"
