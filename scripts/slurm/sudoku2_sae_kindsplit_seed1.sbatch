#!/bin/bash
#SBATCH -J sud2saeKSseed1
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH -o %x-%j.out
#
# A SECOND DICTIONARY ON THE TWO CONTRAST ARMS (LEAK-READER AND 0.854 TWIN).
#
# WHY. Findings 19/21 rest on a between-arm contrast measured at pre_color,
# layer 8, on ONE SAE seed per arm (job 49744985, seed 0):
#
#            arm                      justifies_next feats @ L8   best_justifier_f1
#   LEAK-READER (0.9960, job 48912548)            15                    0.247
#   0.854 TWIN  (leak-free, --shuffle-adj)        69                    0.743
#
# Job 50536365 has since re-fit the SOLVER arm at seed 1 and shown that the
# STRUCTURE replicates (zero justifies_next at pre_node at all 7 layers; layer-0
# next_node F1 1.000; total forcing-set count 47 -> 53, inside the pre-registered
# [28,66]) but the PEAK LAYER does not: best_justifier_f1 at pre_color peaks at
# L10 (0.839) in seed 1 against L8 (0.768) in seed 0. One rung of movement is
# enough to matter for a number read off a single layer, which is exactly how the
# contrast above is read. This job re-fits the other two arms at seed 1 so the
# contrast has an error bar on both sides of it instead of one.
#
# HYPOTHESIS. The between-arm gap is a property of the two MODELS, not of the
# dictionaries: the leak-reader never has to represent which neighbours are
# known, so there is nothing for a forcing-set feature to latch onto, whatever
# dictionary is fit on it. A second seed should therefore reproduce the ORDERING
# and the rough MAGNITUDE of the gap even if, as on the solver, the peak layer
# moves.
#
# PRE-REGISTERED, before looking at any seed-1 output from these two arms:
#
#  (a) HEADLINE, the one that matters. At pre_color, max over layers of
#      best_justifier_f1 is at least 0.30 higher for the 0.854 TWIN than for the
#      LEAK-READER. Seed 0 gave 0.743 - 0.247 = 0.496 at L8, and the seed-to-seed
#      movement measured on the solver at L8 was 0.022, a 23x ratio, so 0.30 is a
#      wide margin and a miss would be informative.
#      FALSIFIER: gap < 0.30. That would mean findings 19/21 are seed-sensitive
#      and the caveat has to be strengthened rather than lifted.
#
#  (b) ASYMMETRY, the third and fourth dictionaries to test it. Zero
#      justifies_next features at pre_node at all seven layers in BOTH arms.
#      This has now held in three dictionaries for three different checkpoints
#      (all three arms at seed 0) and in the solver at seed 1.
#      FALSIFIER: any non-zero count at pre_node.
#
#  (c) LAYER 0. At pre_color both arms have best_next_node_f1 >= 0.99 and zero
#      next_node features at pre_node. Seed 0: twin 23 features at F1 1.000,
#      leak-reader 24 at ~1.000. COUNTS ARE DELIBERATELY NOT PRE-REGISTERED here:
#      the layer-0 dictionary is degenerate at this target (hit_target_l0 = 0 in
#      both solver seeds, l0 7.1 and 8.8 against a target of 64, ~4000 of 4096
#      features dead), and the solver's layer-0 count moved 23 -> 56 between seeds
#      while its F1 held at 1.000. F1 is the stable statistic there; the count is
#      not, and pre-registering it would be scoring noise.
#
#  (d) LOW CONFIDENCE, LOCALISATION. The LEAK-READER's justifies_next features
#      stay late -- concentrated at L10/L12, with none at L6 or L8. Seed 0: 13 at
#      L10, 2 at L12, zero at L6 and L8, i.e. entirely AFTER the leak-free arm's
#      L6->L10 patching handoff window (finding 27).
#      FALSIFIER: leaked features at L6 or L8, which would weaken finding 29(e).
#
# Everything except --seed is byte-identical to sudoku2_sae_kindsplit.sbatch
# (job 49744985), which is the point: one thing changed at a time. The env block
# is the module-only stack, because $SCRATCH/spin_gnn_venv was purged.
#
# COST. 2 arms x 7 layers, each an l1 search of up to 8 SAE fits at 1500 epochs.
# The one-arm seed-1 solver job (50536365) took 30 min on an H100, so ~1-1.5 h
# here, against a 6 h limit that holds even if every layer uses all 8 tries.

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

LEAKFREE="${SCRATCH}/spin_gnn/sudoku_n24_leakfree_2026-10-26/d0256_h04_L12_ff01024_lr1e-03_pre_shuf"
LEAKED="${OAK}/hshunt/checkpoints/spin_gnn/sudoku_dd_d0256_L12_ntrain80000_ep300_sharedpool0.9960_job48912548"
BASE="${SCRATCH}/spin_gnn/sudoku_sae_kindsplit_2026-10-31_seed1"
mkdir -p "$BASE"

COMMON=(--expansion 16 --l1-coeff 0.1 --l1-max-tries 8 --sae-epochs 1500
        --n-instances 3000 --max-positions-per-instance 40
        --alive-rate 1e-3 --n-null-shifts 19 --seed 1)

for arm in leakfree:"$LEAKFREE" leaked:"$LEAKED"; do
    name="${arm%%:*}"; ckpt="${arm#*:}"
    echo
    echo "=== ${name}: all layers at target L0=64, split by position kind, seed 1 ==="
    python3 -m spin_gnn.sudoku_sae --checkpoint "$ckpt" \
        --layer 0 2 4 6 8 10 12 --target-l0 64 --split-by-kind \
        "${COMMON[@]}" --out-dir "${BASE}/${name}_l0_64_seed1"
done

echo
echo "=== summaries ==="
for d in "${BASE}"/*/; do echo "--- $d"; cat "${d}sae_summary_by_kind.csv"; done
