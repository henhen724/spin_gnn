#!/bin/bash
#SBATCH -J sud2sublayer
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH -o %x-%j.out

# WHICH SUBLAYER WRITES THE DD STEP: block i's MLP, or block i's attention?
#
# hshunt asked, for Friday's talk, which MLP layers run the DD steps. Nothing
# in the repo can answer that yet, and the honest reason is a measurement gap:
#
#   - The SAE pages (findings 19/21/29) train one dictionary on the RESIDUAL
#     STREAM AFTER each block. They locate the forcing-set representation at
#     L8 and the neighbour relation at L8-L10, but the residual stream after
#     block 8 is the sum of everything written up to and including block 8, so
#     it cannot say which sublayer wrote it.
#   - Activation patching (finding 27) splices whole residual-stream cells, so
#     it has the same blind spot: it says the fact is in flight over L6-L10,
#     not which component moves it.
#   - Head ablation (finding 27) is the only sublayer-resolved result, and it
#     only rules things OUT: no single head matters much (the worst single
#     head still retains +0.839 of the effect).
#
# So "which MLP layers" has never been measured, and the deck currently says
# so in a footer. This job measures it.
#
# THE INTERVENTION. These are pre-norm blocks, so the two sublayers are
# separate additive writes into the residual stream:
#     x = x + attn(norm1(x));   x = x + mlp(norm2(x))
# Zeroing linear2's weight AND bias makes block i's MLP write exactly zero;
# zeroing out_proj's weight and bias does the same for its attention. The skip
# connection is untouched in both cases, so this deletes one write rather than
# cutting the block out, and every later block still gets a well-formed
# residual stream. Scored as effect_retained = (ablated counterfactual effect)
# / (intact counterfactual effect), the SAME normalisation run_head_ablation
# uses -- so an MLP number and a head number can be read on one axis.
#
# PRE-REGISTERED PREDICTIONS, with falsifiers.
#
# (a) HEADLINE. At least one block's MLP ablation retains clearly LESS effect
#     than the worst single head did (+0.839) -- I will call <= +0.60 a yes.
#     Reason: the forcing set is a conjunction over a node's known neighbours,
#     the kind of thing an MLP computes after attention has gathered the
#     operands, and the head result already says no single head holds it.
#     FALSIFIER: every MLP retains >= 0.839 as well, which would mean the
#     computation is spread over both sublayers with no single load-bearing
#     write, and the slide has to say "no single sublayer carries it" instead
#     of naming layers. That is a real possible outcome of this run.
#
# (b) The load-bearing MLPs sit in the L6-L10 window, where patching shows the
#     fact in flight and where the SAE finds the forcing-set features.
#     Falsifier: the minimum sits at L0-L2 or at L11, which would divorce the
#     causal localisation from the representational one and make the deck's
#     "encoded exactly where the fact is in flight" line a coincidence.
#
# (c) The LEAK-READER's profile differs from the SOLVER's -- later, shallower,
#     or flat. Falsifier: the two profiles match, which would say the sublayer
#     structure is the architecture rather than the mechanism, and would make
#     this a result about transformers, not about DD.
#
# (d) SANITY CONTROL, read FIRST, and it decides whether (a)-(c) are readable
#     at all: ablated_clean_logit_diff. If removing a sublayer also destroys
#     the model's CLEAN prediction, then losing the counterfactual effect is
#     not localisation, it is a broken network. Any layer whose clean
#     logit_diff collapses toward 0 is reported as uninformative, not as a
#     mechanism.
#
# SAME CASES as the circuits run. --seed-start, --n-instances and --n-cases
# match sudoku2_circuits_leakfree.sbatch exactly, so these numbers sit on the
# same counterfactuals as the patching and head-ablation pages and can be put
# on the same slide. Only the sublayer stage runs; behaviour reruns because it
# is cheap and it is the gate that protects the denominator.
#
# COST. 12 layers x 2 sublayers x 3 forwards per case, on a d256/L12 model at
# ~250 tokens and batch 1: ~25k forwards for arm A, ~18k for arm C. That is
# minutes of GPU; the wall clock is dominated by collect_cases generating 1200
# instances on CPU. 3 h is sized for both arms with room, and arm A is first
# so a truncation loses only the control.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

LF1000="${SCRATCH}/spin_gnn/sudoku_n24_data_2026-10-26/data240k_ep100/d0256_h04_L12_ff01024_lr1e-03_pre_shuf"
LEAKED="${OAK}/hshunt/checkpoints/spin_gnn/sudoku_dd_d0256_L12_ntrain80000_ep300_sharedpool0.9960_job48912548"
OUT="${SCRATCH}/spin_gnn/sudoku_sublayer_2026-10-29"
mkdir -p "$OUT"

echo "=== ARM A: leak-free, shared-pool 1.0000 -- THE SOLVER ==="
python -m spin_gnn.sudoku_circuits \
    --checkpoint "$LF1000" \
    --n-instances 1200 \
    --n-cases 350 \
    --stages sublayer \
    --out-dir "${OUT}/A_leakfree_1000"

echo
echo "=== ARM C: leaked, shared-pool 0.9960 -- negative control ==="
python -m spin_gnn.sudoku_circuits \
    --checkpoint "$LEAKED" \
    --n-instances 1200 \
    --n-cases 250 \
    --stages sublayer \
    --out-dir "${OUT}/C_leaked_0996"
