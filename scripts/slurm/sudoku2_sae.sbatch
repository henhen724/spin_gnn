#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_sae
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH -o %x-%j.out

# Do the transformer's features correspond to DD's own predicates?
#
# The probes (sudoku2_probes) answer "is DD state linearly decodable". They
# cannot answer "is DD state what the model actually represents", because a
# linear probe can read a predicate off a distributed code the model never
# uses as a unit. An SAE asks the sharper question: when the residual stream
# is factored into sparse directions, do those directions individually mean
# frontier[v], known[v], color[v]=c, next_node[v]?
#
# Hypothesis: a meaningful share of live features match a named DD predicate
# at high F1, and frontier[*] features -- the ones requiring the graph table
# joined to the known set, not just copying -- appear in the middle layers.
#
# Reading the output honestly: feature labels mean nothing unless the
# dictionary is actually sparse. The module defaults (l1_coeff=0.1,
# 1500 epochs, lr=3e-3) were calibrated against a planted sparse code, where
# a weaker setting reconstructed at FVE=0.99 with L0~21/48 -- a dense,
# uninterpretable dictionary that looks healthy on reconstruction alone.
# CHECK THE REPORTED L0 against the number of DD predicates expected before
# trusting any feature labels. score_features is also guarded by a test that
# randomly-firing features must not reach the interpretability threshold.
#
# Layers 4/8/12 of the 12-layer best checkpoint (Slurm 48513926_1,
# agg_solve_rate=0.767).

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
cd "${SLURM_SUBMIT_DIR}"

EP900="${SCRATCH}/spin_gnn/sudoku_hparam_sweep_2026-10-19_ep900"
OUT_DIR="${SCRATCH}/spin_gnn/sudoku_sae_2026-10-19"
mkdir -p "$OUT_DIR"

python -m spin_gnn.sudoku_sae \
    --checkpoint "${EP900}/d0128_h04_L12_ff01024_lr1e-03_pre" \
    --layer 0 2 4 6 8 10 12 \
    --expansion 16 \
    --l1-coeff 0.1 \
    --sae-epochs 1500 \
    --n-instances 1500 \
    --max-positions-per-instance 40 \
    --out-dir "$OUT_DIR"
