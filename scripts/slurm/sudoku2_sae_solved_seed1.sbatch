#!/bin/bash
#SBATCH -J sud2saeSeed1
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH -o %x-%j.out

# IS THE SAE RESULT A SEED ARTEFACT? THE SECOND DICTIONARY, FINALLY.
#
# Findings 19, 21 and 29 all rest on ONE dictionary per layer at ONE seed, with
# `best_f1` a max over 256 predicates. That caveat is written into all three and
# it is the open item blocking them from being a paper figure. This job is the
# replication: the SOLVER checkpoint (finding 23 cell B, leak-free 1.0000), the
# same seven layers, the same target L0, every SAE hyper-parameter byte-identical
# to job 50075499 -- `--seed 1` is the ONLY change.
#
# WHAT `--seed` ACTUALLY MOVES, stated because it matters for the reading:
# `sudoku_sae.py` uses it for the instance pool (`seed_start = 600000000 + seed`),
# the position sample, the SAE initialisation, the training batch order AND the
# null-shift RNG (lines 431, 433, 449-458). So this is a FULL REPLICATION, not an
# SAE-init-only ablation -- agreement is strong evidence the structure is real;
# disagreement will not say which of the four sources caused it. There is no
# narrower knob exposed, and inventing one would change the code under the
# comparison.
#
# SEED 0, the numbers being replicated (`sae_summary_by_kind.csv`, job 50075499):
#
#   layer        0     2     4     6     8    10    12
#   jn @pre_color  0     0     0     2    17    28     0   (47 total)
#   best_jn_f1  0.000 0.178 0.000 0.416 0.768 0.623 0.231
#   jn @pre_node   0     0     0     0     0     0     0
#   L0 @pre_color: 23 next_node features at best F1 1.000; @pre_node 0 at 0.000
#
# PRE-REGISTERED, written 10-31 before submission:
#
# (a) HEADLINE, high confidence. ZERO `justifies_next` features at `pre_node` at
#     all seven layers. This is the asymmetry findings 21(d) and 29(b) rest on --
#     the forcing set is encoded only while the model picks the COLOUR -- and it
#     has never been checked against a second dictionary.
#     FALSIFIER: any non-zero count at `pre_node`. That would make the asymmetry
#     a seed artefact and it would have to come out of the deck and the paper.
#
# (b) REPRODUCTION CHECK at layer 0: at `pre_color`, `best_next_node_f1` >= 0.99
#     and `n_next_node_features` in [18, 28] (23 +/- 5); at `pre_node`, 0 features
#     and `best_next_node_f1` below its own null (0.107).
#     FALSIFIER: outside those bands, which would point at the position masks
#     rather than at anything about DD.
#
# (c) LOCALISATION, the one that matters for finding 29's headline. At
#     `pre_color`, `best_justifier_f1` at L8 >= 0.70 AND L8 is the argmax over
#     layers. Seed 0: 0.768 at L8 against 0.623 at L10.
#     FALSIFIER: the peak moves off L8 or falls below 0.70. Then "the forcing set
#     enters at L8, inside job 50075339's L6->L10 patching handoff" is not
#     seed-robust, and the deck must quote the layer to +/-2 instead of exactly.
#
# (d) LOW CONFIDENCE, and it decides which statistic the paper prints. Feature
#     COUNTS depend on where the l1 search lands and should be less stable than
#     the F1s: total `justifies_next` at `pre_color` within +/-40% of 47, i.e.
#     28-66. FALSIFIER: outside that, in which case report best F1 only and drop
#     the counts from the figure.
#
# COST. Job 50075499 ran this exact configuration in 01:34:07. 5 h is sized for
# the l1 search needing all 8 tries, as it was there.
#
# ENVIRONMENT. Modules only and explicit `python3`: ${SCRATCH}/spin_gnn_venv is
# dead (the purge took pandas on 10-31, pip and cffi before that), and without it
# on PATH bare `python` is the system 2.7. `sudoku_sae` imports only torch, numpy,
# csv, json and math, all of which the module stack provides.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

echo "=== smoke check ==="
python3 - <<'PY'
import torch, numpy
import spin_gnn.sudoku_sae
print("torch", torch.__version__, "numpy", numpy.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert torch.cuda.is_available(), "no GPU visible to torch"
PY

SOLVED="${SCRATCH}/spin_gnn/sudoku_n24_data_2026-10-26/data240k_ep100/d0256_h04_L12_ff01024_lr1e-03_pre_shuf"
BASE="${SCRATCH}/spin_gnn/sudoku_sae_solved_2026-10-31"
mkdir -p "$BASE"
test -d "$SOLVED"

# Byte-identical to 50075499's COMMON except --seed. Do not touch anything else:
# that is what makes seed 1 a replication rather than a new experiment.
COMMON=(--expansion 16 --l1-coeff 0.1 --l1-max-tries 8 --sae-epochs 1500
        --n-instances 3000 --max-positions-per-instance 40
        --alive-rate 1e-3 --n-null-shifts 19 --seed 1)

echo "=== SOLVER (leak-free 1.0000), all layers, L0=64, split by kind, SEED 1 ==="
python3 -m spin_gnn.sudoku_sae --checkpoint "$SOLVED" \
    --layer 0 2 4 6 8 10 12 --target-l0 64 --split-by-kind \
    "${COMMON[@]}" --out-dir "${BASE}/leakfree1000_l0_64_seed1"

echo
echo "=== pooled summary (seed 1) ==="
cat "${BASE}/leakfree1000_l0_64_seed1/sae_summary.csv"

echo
echo "=== by-kind summary (the pre_node / pre_color split, the result) ==="
cat "${BASE}/leakfree1000_l0_64_seed1/sae_summary_by_kind.csv"
