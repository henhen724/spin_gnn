#!/bin/bash
#SBATCH --job-name=sud2ffdeppool
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
#
# Shared-pool scoring for the depth/width trade-off, because the per-run
# 300-instance agg_solve_rate cannot carry the finer half of 49510197.
#
# WHY. 49510197 settled the big question on its own numbers: at 1.65M matched
# parameters, L=4/ff=256 scores 0.9367 against L=2/ff=1024's 0.467. A gap of
# +0.470 is far outside 300-instance noise (Wilson half-width ~0.06 at p=0.5),
# so that one needs nothing further. But two comparisons that matter are the
# same size as the noise:
#   * L=6/ff=256 (0.9800) vs L=3/ff=1024 (0.9567), matched at ~2.44M -- a gap
#     of 0.023 against a half-width of ~0.03. Unresolvable as it stands, and
#     it is the one that would say spending a FIXED budget on depth beats the
#     reference width rather than merely beating a crippled L=2.
#   * Parameter efficiency against the L=*/ff=1024 ladder, where the gaps
#     between adjacent L are 0.01-0.03 at the top end.
# STATUS already retired the per-run 300-instance number as metric of record
# for exactly this reason; this applies that rule to the new cells instead of
# quoting numbers the protocol says not to quote.
#
# PROTOCOL as established by 48999377: spin_gnn.sudoku_agg_eval on the shared
# pool, seed_start=5e10, disjoint from every training set, instance i seeded at
# seed_start+i so that a larger n is a strict superset of a smaller one. n=1500
# gives a Wilson half-width of ~0.010 at p=0.96, and the paired McNemar exact
# test resolved +0.0113 at this n before, so it is sufficient for 0.023.
#
# EIGHT CELLS, all 150 epochs, all N=24, d_model=256, nhead=4, lr=1e-3, pre:
#   from 49433454 (layer scan, ff pinned at 1024):   L=2, L=3, L=4, L=12
#   from 49510197 (trade-off):  L=2/ff=1792, L=2/ff=4096, L=4/ff=256, L=6/ff=256
# The two L=2/ff=1024 and L=4/ff=1024 cells appear in both roles -- reference
# for the matched pairs and rungs of the ff=1024 ladder -- so eight cells cover
# both comparisons with no retraining.
#
# PRE-REGISTERED, to keep this from being a fishing trip:
#  (a) L=4/ff=256 beats L=2/ff=1024 on the shared pool, paired, p < 0.001.
#      This is a formality given +0.470 at n=300; it is listed so that a
#      failure here is read as something wrong with the eval, not as news.
#  (b) L=6/ff=256 >= L=3/ff=1024, paired McNemar p < 0.05, at matched ~2.44M.
#      Falsifier: the pair is statistically tied, in which case the honest
#      claim shrinks to "depth is not WORSE than width at a fixed budget" and
#      the trade-off result rests entirely on the 1.65M pair, where the gap is
#      too large to be in doubt.
#  (c) L=6/ff=256 (2.45M) is within 0.02 of L=12/ff=1024 (9.55M) despite 3.9x
#      fewer parameters. Falsifier: the 9.55M cell is resolvably better, in
#      which case width still buys something real at the top end and the
#      efficiency claim is limited to the knee region rather than general.
#
# Cost: 49430217 scored a comparable eval in 00:25:57. Eight cells at n=1500 is
# ~8x a 300-instance pass but decoding dominates and batches well; budget 4 h.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

LS="${SCRATCH}/spin_gnn/sudoku_layer_scan_2026-10-24_n24"
FD="${SCRATCH}/spin_gnn/sudoku_ff_depth_tradeoff_2026-10-25_n24"
OUT_DIR="${SCRATCH}/spin_gnn/sudoku_ff_depth_sharedpool_2026-10-25"
mkdir -p "$OUT_DIR"

python -m spin_gnn.sudoku_agg_eval \
    --checkpoint \
        "${LS}/d0256_h04_L02_ff01024_lr1e-03_pre" \
        "${LS}/d0256_h04_L03_ff01024_lr1e-03_pre" \
        "${LS}/d0256_h04_L04_ff01024_lr1e-03_pre" \
        "${LS}/d0256_h04_L12_ff01024_lr1e-03_pre" \
        "${FD}/d0256_h04_L02_ff01792_lr1e-03_pre" \
        "${FD}/d0256_h04_L02_ff04096_lr1e-03_pre" \
        "${FD}/d0256_h04_L04_ff00256_lr1e-03_pre" \
        "${FD}/d0256_h04_L06_ff00256_lr1e-03_pre" \
    --n-instances 1500 \
    --seed-start 50000000000 \
    --out-csv "${OUT_DIR}/sharedpool_n1500.csv"
