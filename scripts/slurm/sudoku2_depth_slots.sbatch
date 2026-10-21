#!/bin/bash
#SBATCH --job-name=sud2slots
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
#
# Does the depth cost live in picking the node or in picking the color?
#
# Two findings are in tension. 48861799: the frontier predicate decodes at
# 0.863 AUROC from epoch 5 and never improves. 48863464: solve_rate falls with
# DD round count T at pinned hint density. A model whose frontier knowledge is
# static should not care how many rounds an instance needs, so something else
# must degrade with depth.
#
# The reasoning grammar alternates (node, color), so every scored target in
# that section is either a SELECTION or a COLORING slot, and a teacher-forced
# pass labels each with its round index. Teacher forcing is deliberate: the
# prefix is always the correct chain, so a drop at round r is a failure to use
# a correct r-1-round chain rather than compounding of the model's own errors.
#
# PRE-REGISTRATION, and it is not what I predicted three hours ago. Writing
# the deck I proposed the cost was in chaining colors. A 25-instance CPU smoke
# test at rho=0.60 on the agg leader says that is wrong:
#
#     coloring   overall 0.992   by round: 0.993 1.000 0.966 1.000
#     selection  overall 0.663   by round: 0.618 0.651 0.759 0.800
#
# Coloring is at ceiling and flat; all the variance is in selection, and
# selection accuracy RISES with round index. So I predict: coloring flat near
# 0.99 at every depth and density, and no depth cost attributable to coloring.
# That would falsify the chaining-colors story on the deck and leave the
# tension unresolved rather than explained -- which is the honest outcome to
# report if it holds at n=600.
#
# The marginal selection rise must not be read as skill. Candidate count is
# n_nodes - n_hints - pairs_emitted, and at pinned rho that is a deterministic
# function of reasoning position, so later rounds simply have fewer nodes left.
# `round_effect_within_position` is the number that matters: at n=25 it was
# -0.027 for selection, i.e. the marginal rise reverses sign once position is
# held fixed. Whether that survives n=600 is the actual question this job asks.
#
# Teacher-forced accuracy also does not have to match solve_rate: 0.992
# per-token coloring over ~20 slots still leaves room for generation to fail.
# This job cannot attribute solve_rate failures by itself, only locate the
# weak slot.
#
# Three densities, matching 48863464 so the two can be read side by side.
# rho=0.50 is included for continuity but inherits that job's caveat.
# 600 instances x 3 rho x 2 checkpoints is one forward pass each, no
# autoregressive decoding, so instance construction dominates; 48863464 built
# 7200 instances inside 3h, so 2h is margin for 3600.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_depth_slots_2026-10-21"
mkdir -p "$OUT_DIR"

python -m spin_gnn.sudoku_depth_slots \
    --checkpoint \
        "${SCRATCH}/spin_gnn/sudoku_hparam_sweep/d0256_h04_L12_ff01024_lr1e-03_pre" \
        "${SCRATCH}/spin_gnn/sudoku_hparam_sweep_2026-10-19_ep900/d0128_h04_L12_ff01024_lr1e-03_pre" \
    --rho 0.50 0.60 0.75 \
    --n-instances 600 \
    --min-cell 30 \
    --n-position-bins 6 \
    --out-csv "${OUT_DIR}/depth_slots.csv"
