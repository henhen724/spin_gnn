#!/bin/bash
#SBATCH -J sud2saeAbl
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH -o %x-%j.out

# FROM "REPRESENTED" TO "USED": ablate the justifies_next features.
#
# Findings 19/21 showed the leak-free model represents `justifies_next[u]` --
# "u is a known neighbour of the node whose colour is being emitted" -- at
# pre_color positions, 69 features at L8 against the leaked twin's 15. That is
# a decoding result. This job removes those features from the forward pass and
# asks whether the colour decision actually depends on them.
#
# DEPENDS ON 50075499 (sud2saeSolved): it needs that run's dictionaries for the
# 1.0000 checkpoint, including the norm_layerNN.pt stats and the per-kind
# feature labels, both of which spin_gnn/sudoku_sae.py only started writing on
# 10-28. Do not submit this before that job has written
# sudoku_sae_solved_2026-10-28/leakfree1000_l0_64/.
#
# THE DESIGN, in one line: everything is scored against the `recon` condition
# (full SAE reconstruction spliced back in, nothing zeroed), so reconstruction
# error is netted out of every effect rather than being charged to the ablation.
#
# THE PAIR THAT CARRIES THE CLAIM is zero_jn_bind vs zero_jn_nonbind. Both
# remove the same number of features, from the same family, at the same
# position, in the same layer, on the same cases. They differ only in WHICH
# neighbour the feature refers to: the one whose colour is doing the forcing,
# versus another known neighbour of the same node whose colour is not. No
# reconstruction-error story and no feature-count story can produce a gap
# between those two. Only the model using that neighbour's colour to eliminate
# a candidate can.
#
# PRE-REGISTERED PREDICTIONS, with falsifiers.
#
# (a) HEADLINE. zero_jn_bind drops accuracy on the forced colour well below
#     `recon`, and zero_jn_nonbind does not -- a selective, paired gap at
#     p < 0.05 by exact McNemar. That is "yes, it computes the elimination".
#     FALSIFIER: bind and nonbind are indistinguishable. Then the features are
#     decoration -- present, decodable, unused -- and the verdict is "no, it is
#     doing something else", with findings 19/21 demoted to representational
#     structure exactly as the `next_node` probe was demoted by its hand-rule
#     baseline.
#
# (b) zero_random (count-matched, outside the family) costs LESS than
#     zero_jn_all. Falsifier: random costs as much, which would mean the
#     ablation is measuring generic reconstruction damage and the whole
#     comparison is uninformative. This is the control that decides whether (a)
#     is readable at all, so read it first.
#
# (c) The effect is largest in the MIDDLE layers (L8 over L4 and L12), matching
#     where finding 19 put the best justifies_next F1. Falsifier: flat in
#     layer, or largest at L12, which would put the elimination at the output
#     end and contradict the F1 profile.
#
# (d) At --kind pre_node there is nothing to ablate: finding 21 found ZERO
#     justifies_next features there in both arms, so the module should report
#     "no justifies_next features at this layer" and skip. Falsifier: features
#     exist there, which would be the first sign the solved model moved the
#     edge computation into node SELECTION -- DD-like search, the half the
#     model is currently believed not to do.
#
# TWO SELECTION BREADTHS, both run. `interpretable` is finding 19/21's own
# gate (above min_f1 AND above the position-rolled null). `beats_null` keeps
# every feature above its null, which buys per-node coverage: the selective
# conditions need BOTH the binding and a non-binding neighbour to own features,
# and at one feature per node almost no case qualifies. The headline is the
# `interpretable` pass; `beats_null` is the powered replication.
#
# COST. Six conditions x one forward pass per case per layer: 300 cases x 3
# layers x 6 = 5400 forwards on a 12-layer d=256 model over ~200 tokens, which
# is minutes. The real cost is generating 1200 instances and their DD traces on
# CPU. 3 h is generous; expect well under one.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

SOLVED="${SCRATCH}/spin_gnn/sudoku_n24_data_2026-10-26/data240k_ep100/d0256_h04_L12_ff01024_lr1e-03_pre_shuf"
SAE="${SCRATCH}/spin_gnn/sudoku_sae_solved_2026-10-28/leakfree1000_l0_64"
OUT="${SCRATCH}/spin_gnn/sudoku_sae_ablate_2026-10-29"
mkdir -p "$OUT"

if [ ! -f "${SAE}/norm_layer08.pt" ]; then
    echo "FATAL: ${SAE}/norm_layer08.pt missing -- job 50075499 has not written its"
    echo "dictionaries yet, or ran before the norm-stat fix. Not guessing."
    exit 1
fi

echo "=== pre_color, finding 19/21's own interpretability gate (HEADLINE) ==="
python -m spin_gnn.sudoku_sae_ablate \
    --checkpoint "$SOLVED" --sae-dir "$SAE" \
    --layer 4 8 12 --n-instances 1200 --n-cases 300 \
    --kind pre_color --select interpretable \
    --out-dir "${OUT}/solved_pre_color_interp"

echo
echo "=== pre_color, widened to every feature above its null (POWERED REPLICATION) ==="
python -m spin_gnn.sudoku_sae_ablate \
    --checkpoint "$SOLVED" --sae-dir "$SAE" \
    --layer 4 8 12 --n-instances 1200 --n-cases 300 \
    --kind pre_color --select beats_null \
    --out-dir "${OUT}/solved_pre_color_null"

echo
echo "=== pre_node: expected to find nothing to ablate (prediction d) ==="
python -m spin_gnn.sudoku_sae_ablate \
    --checkpoint "$SOLVED" --sae-dir "$SAE" \
    --layer 4 8 12 --n-instances 1200 --n-cases 300 \
    --kind pre_node --select beats_null \
    --out-dir "${OUT}/solved_pre_node_null"

echo
echo "=== summaries ==="
for d in "${OUT}"/*/; do echo "--- $d"; cat "${d}ablate_summary.csv" 2>/dev/null || echo "(none)"; done
