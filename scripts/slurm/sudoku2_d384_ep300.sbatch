#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_d384
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH -o %x-%j.out

# Next rung of the width ladder: d_model 256 -> 384.
#
# Changes exactly ONE thing vs the incumbent best (d0256_h04_L12_ff01024_
# lr1e-03_pre @300ep, agg_solve_rate 0.847): d_model. Layers stay at 12, heads
# at 4, dim_feedforward at 1024, lr at 1e-3, epochs at 300, n_train at 5000.
# Holding dim_feedforward fixed while d_model moves is what the existing
# ladder did (d0064/d0128/d0256 are all ff01024), so this continues the same
# protocol rather than starting a new one -- the ff/d_model ratio drops from
# 4x to 2.67x as a consequence, which is a known and accepted part of that
# protocol, not an independent second change.
#
# Why width and why now. scripts/sweep_leaderboard.py over all 31 runs:
#
#   r(agg_solve_rate, n_params)           = +0.713
#   r(agg_solve_rate, epochs)             = +0.634
#   r(agg_solve_rate, final_val_nll)      = +0.169
#   r(agg_solve_rate, n_train_instances)  =  n/a -- constant at 5000, all 31
#
# Parameter count is the strongest thing in the sweep that tracks the metric
# of record, and the width ladder itself is monotone: 0.720 (d0064) -> 0.773
# (d0128) -> 0.847 (d0256) at matched epochs. The epochs correlation is a
# confound (only d0064/d0128 were ever run to 900, and the matched d0128
# 300->900 pair moved 0.773 -> 0.767, i.e. nothing), which is what the
# companion job 48854232 is currently testing directly at d0256.
#
# Hypothesis: agg_solve_rate exceeds 0.847. Confidence is moderate, not high
# -- three points make a monotone trend but not a law, and d0256 is already
# the single worst model in the sweep at rho = 1.0 (solve_rate_k0 = 0.24
# against d0064's 1.00), so width may be buying low-rho performance by
# trading away the trivial regime rather than by getting better overall. The
# k0 column is therefore as much the result here as agg is: if d384 lands
# above 0.847 on agg with k0 falling further, that is evidence the two rho
# ranges are in genuine tension and the "target of record" question has to be
# settled before any more width is bought.
#
# Note on the companion data-size run (48854233). That one's case rested on
# val_nll, and the correlation above undercuts it: across the sweep val_nll
# barely tracks agg_solve_rate (+0.17), so its much better val_nll (1.10 at
# epoch 139 vs d0256's 2.41 at 300) is NOT yet evidence of a better solve
# rate. Both that run and this one are therefore live, and the width axis is
# not a hedge against the data axis -- n_train_instances has never been varied
# at all, so neither result can be predicted from this sweep.
#
# Runtime: d0256 @300ep fit inside 1h30. d384 is ~2.2x the parameters, so
# ~3h; 6h leaves margin. Memory raised 24 -> 32GB for the wider activations.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
# Slurm stdout is block-buffered, so progress lines otherwise stay invisible
# for the whole run and cannot be polled.
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_hparam_sweep_2026-10-21_d384"
mkdir -p "$OUT_DIR"

echo "d_model=384 nhead=4 num_layers=12 dim_feedforward=1024 epochs=300 n_train=5000"

python -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 24 \
    --degree 5 \
    --n-colors 4 \
    --d-model 384 \
    --nhead 4 \
    --num-layers 12 \
    --dim-feedforward 1024 \
    --n-train-instances 5000 \
    --n-val-instances 300 \
    --batch-size 32 \
    --epochs 300 \
    --n-per-bin 100 \
    --out-dir "$OUT_DIR" \
    "$@"
