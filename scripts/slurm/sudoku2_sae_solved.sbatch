#!/bin/bash
#SBATCH -J sud2saeSolved
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH -o %x-%j.out

# THE SAE HAS NEVER BEEN RUN ON THE MODEL THAT ACTUALLY SOLVED THE TASK.
#
# Findings 19 (49729375) and 21 (49744985) are both on the leak-free checkpoint
# that scores 0.8540 on the shared pool. The leak-free model that scores
# 1.0000 with a ZERO memorization gap -- finding 23, job 49705064, cell B at
# 240000 instances x 100 epochs, the declared working budget -- has never been
# opened up. hshunt asked "is the network that actually trained learning DD",
# and the honest answer today is that the interpretability is about its weaker
# predecessor. This job fixes that.
#
# ONE CHANGE from 49744985: the checkpoint. Every SAE hyper-parameter, the
# layer list, target L0, the null-shift count and the seed are byte-identical
# to that job's COMMON (which was itself identical to 49729375's, and
# reproduced its sae_summary.csv byte for byte in both arms). So the new
# numbers drop straight into findings 19/21's tables as a third arm, and
# nothing in the comparison is confounded by SAE settings.
#
# WHAT THE COMPARISON BUYS. At L8/pre_color the measured ladder so far is:
#
#     leaked   0.9960   15 justifies_next features, best F1 0.247
#     leakfree 0.8540   69 justifies_next features, best F1 0.743
#     leakfree 1.0000   ???                                  <- this job
#
# `justifies_next[u]` = "u is a KNOWN neighbour of the node whose colour is
# being emitted", i.e. exactly the input to DD's elimination rule.
#
# PRE-REGISTERED PREDICTIONS, with falsifiers.
#
# (a) HEADLINE. The 1.0000 model has MORE `justifies_next` features at
#     `pre_color` and a HIGHER best F1 at L8 than the 0.854 model, so the
#     ladder is monotone in solve_rate. Reason: if the elimination rule is what
#     the last 0.146 of solve_rate is made of, the predicate it needs should be
#     represented more sharply in the model that has all of it. FALSIFIER: the
#     counts or F1 go DOWN or tie, which would say the perfect score was bought
#     with something other than a cleaner forcing rule -- and would point the
#     verdict toward "it is doing something else", consistent with whatever
#     job 50075339's counterfactual arm A reports.
#
# (b) The `pre_node` / `pre_color` asymmetry from finding 21 survives: ZERO
#     `justifies_next` features at `pre_node` at every layer. Falsifier: any
#     appear there, which would mean the solved model moved the edge
#     computation earlier -- into node selection -- and would be the first
#     evidence of DD-like SEARCH rather than just DD-like inference.
#
# (c) Layer 0 still has ~23 `next_node` features at F1 1.000 at `pre_color` and
#     0 at `pre_node`, reproducing finding 21's "layer-0 features were free" in
#     a third model. This is the reproduction check; a miss here means
#     something about the position masks, not about DD.
#
# (d) The gap to the LEAKED arm widens: (1.0000 arm - leaked arm) at
#     L8/pre_color exceeds the 0.743 - 0.247 = 0.496 already measured.
#     Falsifier: it narrows, which would weaken the "removing the leak CAUSES
#     the edge representation" reading that findings 19/21 rest on.
#
# CAVEAT THAT DOES NOT GO AWAY. best_f1 is a max over 256 predicates on ONE
# dictionary per layer at one seed. That is a multiple-comparisons ceiling and
# it applies to all three arms equally; it is why the causal job (50075339) and
# not this one carries the verdict. This job says what is REPRESENTED; the
# ablation says what is USED.
#
# COST. 49744985 did TWO checkpoints with these exact settings in under 4 h,
# so one is ~2 h. 5 h is sized for the l1 search needing its full 8 tries.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

SOLVED="${SCRATCH}/spin_gnn/sudoku_n24_data_2026-10-26/data240k_ep100/d0256_h04_L12_ff01024_lr1e-03_pre_shuf"
BASE="${SCRATCH}/spin_gnn/sudoku_sae_solved_2026-10-28"
mkdir -p "$BASE"

# Byte-identical to 49744985's COMMON. Do not touch: it is what makes the new
# arm comparable to findings 19/21 without a re-run of those.
COMMON=(--expansion 16 --l1-coeff 0.1 --l1-max-tries 8 --sae-epochs 1500
        --n-instances 3000 --max-positions-per-instance 40
        --alive-rate 1e-3 --n-null-shifts 19 --seed 0)

echo "=== leak-free 1.0000 (finding 23 cell B): all layers, L0=64, split by kind ==="
python -m spin_gnn.sudoku_sae --checkpoint "$SOLVED" \
    --layer 0 2 4 6 8 10 12 --target-l0 64 --split-by-kind \
    "${COMMON[@]}" --out-dir "${BASE}/leakfree1000_l0_64"

echo
echo "=== pooled summary (compare to kindsplit leakfree_l0_64 = the 0.854 arm) ==="
cat "${BASE}/leakfree1000_l0_64/sae_summary.csv"

echo
echo "=== by-kind summary (the pre_node / pre_color split, the result) ==="
cat "${BASE}/leakfree1000_l0_64/sae_summary_by_kind.csv"
