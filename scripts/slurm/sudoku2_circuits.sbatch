#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_circ
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out

# Does the model USE the DD forcing rule, or merely represent something a
# probe can read off it?
#
# Counterfactual: node u is forced when its known neighbours cover q-1 colors
# leaving c free. Recolor the one neighbour uniquely holding color a to c --
# now a is the free color, so DD's answer for u flips c -> a. One token
# changes; u's token, the graph table and the reasoning prefix do not.
#
# Hypothesis: clean_correct is high AND corrupt_flipped is well above the
# 1/q = 0.25 chance rate, i.e. the model tracks neighbour colors through the
# forcing rule. Patching then shows recovery concentrated at the graph row
# for u in the middle layers (where the neighbour table is joined to the known
# colors), not at the edited hint alone.
#
# Negative result is informative and handled: if the flip rate sits at chance
# the module stops before patching and records localized=false, because
# localizing a circuit that behavior says is absent would be mapping noise.
#
# Sanity check built into the metric: patching the prediction position at the
# final layer must give recovery = 1.0 by construction (asserted in
# tests/test_sudoku_circuits.py). Treat that cell as a control, not a result.

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
OUT_DIR="${SCRATCH}/spin_gnn/sudoku_circuits_2026-10-19"
mkdir -p "$OUT_DIR"

python -m spin_gnn.sudoku_circuits \
    --checkpoint "${EP900}/d0128_h04_L12_ff01024_lr1e-03_pre" \
    --n-instances 600 \
    --n-cases 150 \
    --out-dir "${OUT_DIR}/d0128_L12"
