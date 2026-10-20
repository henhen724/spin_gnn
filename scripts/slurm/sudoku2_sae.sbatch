#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_sae
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=08:00:00
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
# RERUN of 48581592, which was NOT INTERPRETABLE. That run used a fixed
# l1_coeff=0.1 and produced L0 = 84/176/212/240/261/278 of 2048 features for
# layers 2..12 with ZERO dead features: a dense dictionary, so its feature
# labels (which showed `known` growing 84 -> 592 and `frontier` at 0-2) could
# not be read as semantics. It still reported FVE=0.992, which is exactly why
# reconstruction quality must never be the health check.
#
# Cause: the reconstruction term is a SUM over dimensions, so it scales with
# d_model while the L1 term does not. l1_coeff=0.1 was calibrated at
# d_model=12 and is therefore ~10x too weak at d_model=128.
#
# Fix: --target-l0 searches l1_coeff geometrically until L0 reaches the
# target, starting from an informed 1.0 so it converges in a step or two, and
# records hit_target_l0 in sae_summary.csv so a dictionary that stayed dense
# is visible rather than being reported as a result.
#
# Layers 4/8/12 of the 12-layer best checkpoint (Slurm 48513926_1,
# agg_solve_rate=0.767).

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
OUT_DIR="${SCRATCH}/spin_gnn/sudoku_sae_2026-10-19_sparse"
mkdir -p "$OUT_DIR"

python -m spin_gnn.sudoku_sae \
    --checkpoint "${EP900}/d0128_h04_L12_ff01024_lr1e-03_pre" \
    --layer 0 4 8 12 \
    --expansion 16 \
    --l1-coeff 1.0 \
    --target-l0 32 \
    --sae-epochs 1500 \
    --n-instances 1500 \
    --max-positions-per-instance 40 \
    --out-dir "$OUT_DIR"
