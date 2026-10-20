#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_algid
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out

# Which rule does the model follow where candidate algorithms disagree?
#
# The next_node probe hits 0.764 (Slurm 48580216), which looks like evidence
# the model knows the deducible set -- but solve_with_dd_trace emits each
# round as sorted(...), so node ids ascend within a round and a pure ordering
# rule scores well above chance with no deducibility at all. This scores ONLY
# positions where DD and a baseline disagree, where the model's prediction is
# a genuine vote between the two.
#
# Hypothesis, following the probe contrast (known_mask 0.92 vs frontier_mask
# 0.67): the model follows `ascending_after` noticeably more often than DD on
# their disagreements, which would mean "bookkeeping + an ordering convention"
# explains the next_node probe and the model is not computing deducibility.
#
# The reverse -- DD winning the disagreements - would mean the frontier IS
# computed but is not linearly readable, and the probe simply lacked power.
# Either outcome is informative, which is why this is worth running.
#
# most_known_nbrs is the discriminating middle case: it uses the graph and is
# close to DD's forcing condition without being equal to it, so it separates
# "counts known neighbours" from "applies the forcing rule".

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

EP900="${SCRATCH}/spin_gnn/sudoku_hparam_sweep_2026-10-19_ep900"
OUT_DIR="${SCRATCH}/spin_gnn/sudoku_algorithm_id_2026-10-19"
mkdir -p "$OUT_DIR"

for CK in d0128_h04_L12_ff01024_lr1e-03_pre d0064_h04_L12_ff01024_lr1e-03_pre; do
    echo "=== $CK ==="
    python -m spin_gnn.sudoku_algorithm_id \
        --checkpoint "${EP900}/${CK}" \
        --n-instances 400 \
        --out-csv "${OUT_DIR}/${CK}.csv"
done
