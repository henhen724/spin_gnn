#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_probe
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out

# Is DD state linearly present in the residual stream, and at which depth?
#
# The headline is a CONTRAST, not any single probe score:
#   known_mask    -- which nodes have a color yet (bookkeeping; trackable by
#                    copying + position alone)
#   frontier_mask -- which nodes are deducible right now. This is the DD
#                    predicate: it needs the graph table joined to the known
#                    set. Linearly present => the model computes DD state.
# Layer 0 is the raw token+position embedding and is the control: a probe
# that already wins there is reading the tokenization, not a computation.
#
# Probes train and test on DISJOINT instance pools, so a probe cannot win by
# memorizing positions.
#
# Run over the two 900-epoch checkpoints (d_model 64 and 128) and the two
# 300-epoch ones. That is four separately-trained models, not a training
# series -- a crude first read on "when"; Slurm (sudoku2_checkpoint_series)
# produces the real within-run curve.

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

EP900="${SCRATCH}/spin_gnn/sudoku_hparam_sweep_2026-10-19_ep900"
EP300="${SCRATCH}/spin_gnn/sudoku_hparam_sweep"
OUT_DIR="${SCRATCH}/spin_gnn/sudoku_probes_2026-10-19"
mkdir -p "$OUT_DIR"

python -m spin_gnn.sudoku_probes \
    --checkpoint \
        "${EP900}/d0128_h04_L12_ff01024_lr1e-03_pre" \
        "${EP900}/d0064_h04_L12_ff01024_lr1e-03_pre" \
        "${EP300}/d0128_h04_L12_ff01024_lr1e-03_pre" \
        "${EP300}/d0064_h04_L12_ff01024_lr1e-03_pre" \
    --n-train-instances 500 \
    --n-test-instances 200 \
    --max-positions-per-instance 20 \
    --probe-epochs 600 \
    --out-csv "${OUT_DIR}/probes.csv"
