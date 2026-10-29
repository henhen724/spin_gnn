#!/bin/bash
#SBATCH -J sud2chain
#SBATCH -p sganguli
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH -o %x-%j.out

# DOES IT CHAIN? The round-0 counterfactual, moved onto DEDUCED neighbours.
#
# Job 50075339 settled the elimination rule over GIVEN facts: recolour the one
# hint that is doing the forcing and the leak-free models flip to DD's
# recomputed answer on 350/350 and 250/250 cases, against 0.388 for the leaked
# twin. But every case there is round 0, where all known neighbours are hints.
# DD is iterative, and a node forced in round k>=1 is forced partly by facts the
# model DERIVED. Eliminating against a self-produced fact is the difference
# between applying an inference rule once and running the algorithm.
#
# This job edits the binding neighbour's colour token IN THE REASONING SECTION
# instead of in the hints. Everything else -- the instance stream, the
# one-token-change discipline, the flip/chance accounting -- is identical to
# 50075339, so the two flip rates are directly comparable and any gap between
# them is about the PROVENANCE of the fact, not about the rule.
#
# NO GPU ON PURPOSE. 350 cases x 2 forward passes on a 12-layer d=256 model over
# ~200 tokens is ~700 forwards; a CPU node does that in minutes and the real
# cost is DD trace generation, which is CPU-bound anyway. Both sganguli GPU
# nodes are fully allocated and job 50075499 (the SAE on the 1.0000 checkpoint)
# is pending behind them; this job deliberately does not compete for that GPU.
#
# PRE-REGISTERED PREDICTIONS, with falsifiers.
#
# (a) HEADLINE. corrupt_flipped stays high (>=0.60) on deduced neighbours for
#     both leak-free arms, i.e. roughly the round-0 rate. That is "yes, it
#     chains": the model eliminates against facts it produced itself, which is
#     DD's propagation step and not just DD's rule.
#     FALSIFIER: flip rate collapses toward chance (0.25). Read with
#     `reverts_to_original` -- see the ambiguity note below.
#
# (b) The rate does not decay with the round v was deduced in. A model that
#     chains one step but not three would show round 1 high and rounds 3-5
#     falling off. Falsifier: monotone decay in `by_round`, which would bound
#     the depth of genuine propagation and would be a real and publishable
#     limitation rather than a failure.
#
# (c) The leaked arm stays near chance here too, as it did at round 0. If it
#     instead does WELL on chained cases while failing round-0 ones, that would
#     mean its 0.9960 comes from reading the adjacency-order leak rather than
#     from any elimination, and the contrast would need rewriting.
#
# THE ONE AMBIGUITY, stated because it does not go away. Editing an emitted
# deduction makes the prefix inconsistent with the graph and hints: the model is
# shown a step it would not have written. A model that DETECTS the contradiction
# and keeps its own answer scores as "does not chain" while chaining perfectly in
# free generation. So a LOW flip rate is ambiguous, and `reverts_to_original`
# separates the cases: reverting to the original colour is the detection
# signature; a third colour is neither. A HIGH flip rate carries no such
# ambiguity -- it can only mean the edited fact was used. The 8-case CPU
# validation already returned 1.000 flips with 0.000 reverts out to round 4,
# so the expected outcome is the unambiguous one; this job is the powered
# version with the controls attached.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

LF1000="${SCRATCH}/spin_gnn/sudoku_n24_data_2026-10-26/data240k_ep100/d0256_h04_L12_ff01024_lr1e-03_pre_shuf"
LF0854="${SCRATCH}/spin_gnn/sudoku_n24_leakfree_2026-10-26/d0256_h04_L12_ff01024_lr1e-03_pre_shuf"
LEAKED="${OAK}/hshunt/checkpoints/spin_gnn/sudoku_dd_d0256_L12_ntrain80000_ep300_sharedpool0.9960_job48912548"
OUT="${SCRATCH}/spin_gnn/sudoku_chain_2026-10-28"
mkdir -p "$OUT"

# Arm order matches 50075339: the verdict arm first, then the control that
# sharpens it, then the second leak-free model.
echo "=== ARM A: leak-free 1.0000 (finding 23) -- THE CHAINING VERDICT ==="
python -m spin_gnn.sudoku_chain --checkpoint "$LF1000" \
    --n-instances 400 --n-cases 350 --device cpu \
    --out-dir "${OUT}/A_leakfree_1000"

echo
echo "=== ARM C: leaked 0.9960 -- negative control (prediction c) ==="
python -m spin_gnn.sudoku_chain --checkpoint "$LEAKED" \
    --n-instances 400 --n-cases 250 --device cpu \
    --out-dir "${OUT}/C_leaked_0996"

echo
echo "=== ARM B: leak-free 0.8540 -- the checkpoint findings 19/21 opened ==="
python -m spin_gnn.sudoku_chain --checkpoint "$LF0854" \
    --n-instances 400 --n-cases 250 --device cpu \
    --out-dir "${OUT}/B_leakfree_0854"

echo
echo "=== summaries ==="
for d in "${OUT}"/*/; do
    echo "--- $d"
    python -c "import json,sys; d=json.load(open('${d}chain_summary.json')); print(f\"  flipped {d['corrupt_flipped']:.3f}  clean {d['clean_correct']:.3f}  reverts {d['reverts_to_original']:.3f}  n={d['n_cases']}\")" 2>/dev/null || echo "  (none)"
done
