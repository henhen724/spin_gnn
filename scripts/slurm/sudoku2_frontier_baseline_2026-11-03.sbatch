#!/bin/bash
#SBATCH --job-name=sud2FrntBase
#SBATCH --partition=sganguli
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
#
# HOW MUCH OF THE N=96 FRONTIER PROBE IS FREE, AND WHAT IS THE CEILING?
#
# 50979059 probed the six N=96 fall cells and the headline turns entirely on
# one comparison: the frontier_mask AUROC the model reaches (0.731-0.772)
# against the best score obtainable from DD's labels alone. That free score
# was computed ONCE, at one instance pool, at 300 probe epochs, inside the
# probe job -- a point estimate on one side of a comparison whose other side
# carries three seeds. This job replaces it.
#
# No model is involved anywhere here. Everything is computed from DD's own
# ground-truth labels, so it measures a property of the TASK.
#
# WHAT CHANGED IN THE SCRIPT (and why this job exists at all):
#   * degree_aug gave the baseline probe n_known_nbrs, which is NOT what DD
#     thresholds -- a node is forced when its known neighbours cover q-1
#     DISTINCT colours, and at degree 5 with q=4 the known neighbours usually
#     repeat. color_aug now hands it n_known_colors, the real quantity.
#   * rule_on_visible computes DD's rule instead of fitting it. It was written
#     with an assert at 0.999 and the assert FIRED at 0.832, which is the most
#     useful thing this job has produced so far: frontier_mask is the ROUND's
#     set, fixed at the round boundary, while the state visible at a probe
#     position has already moved on within the round. So the ceiling for any
#     reader scoring the visible state is ~0.83, not 1.0, and the model's 0.77
#     has to be read against that.
#
# PRE-REGISTERED.
#   H1 (the margin is real) The free score -- best of not_known / known_probe /
#      degree_aug on AUROC -- varies by <= 0.01 across three instance pools. The
#      model's margin over it is 0.058-0.099, so a pool spread near 0.01 leaves
#      the margin standing and a spread near 0.06 destroys it.
#   H2 (the committed baseline was not underfit) 600 probe epochs move the free
#      score by <= 0.01 against 300. This is a real risk and it cuts against the
#      result I want: the 0.6734 in the probe job was fit for half as long as the
#      MODEL probes it is compared with, and an underfit baseline understates the
#      free score and flatters the model. If 600 epochs lifts it materially, the
#      P4 margin shrinks by exactly that much and the deck says so.
#   H3 (there is headroom, and the model is inside it) color_aug lands above
#      every free score and near rule_on_visible. If instead color_aug sits at
#      the free score, then a LINEAR probe cannot use n_known_colors either, the
#      bracket collapses, and no comparison on this axis means anything.
#   H4 (N=24 becomes comparable) the same bracket at N=24, so findings 19/21's
#      N=24 probe numbers can be re-read on AUROC against a like-for-like
#      baseline instead of against the N=24 free score quoted at balanced
#      accuracy.
#
# COST. CPU only: fit_linear_probe defaults to device="cpu" and this script
# never overrides it, so the GPU the previous baseline ran beside was never
# used. Timed on a login node at N=96 fall, 2 threads: 50 train / 30 test
# instances give 3662 / 2070 labelled positions and four probes at 100 epochs
# run in 31 s wall including imports and instance generation. 250/150 is 5x
# the positions and 600 epochs is 6x the steps, so ~4 min per run on 8
# threads. Nine runs, ~25 min expected, --time 2:00:00 as margin.

set -euo pipefail

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_frontier_baseline_2026-11-03"
mkdir -p "$OUT_DIR"

# N=96 fall, the ensemble the six probed cells were trained on: three pools at
# the committed 300 epochs, three at the 600 the model probes used.
for EP in 300 600; do
    for OFF in 0 1 2; do
        echo
        echo "=== N=96 fall, pool offset ${OFF}, ${EP} probe epochs ==="
        python3 -m scripts.frontier_from_known_baseline \
            --n-nodes 96 --degree 5 --n-colors 4 --fall --shuffle-adj \
            --n-train 250 --n-test 150 --probe-epochs "${EP}" \
            --pool-offset "${OFF}" \
            --out-json "${OUT_DIR}/n96_fall_ep${EP}_off${OFF}.json"
    done
done

# N=24, the ensemble every mechanistic finding before 11-03 was measured on.
for OFF in 0 1 2; do
    echo
    echo "=== N=24, pool offset ${OFF}, 600 probe epochs ==="
    python3 -m scripts.frontier_from_known_baseline \
        --n-nodes 24 --degree 5 --n-colors 4 --shuffle-adj \
        --n-train 250 --n-test 150 --probe-epochs 600 \
        --pool-offset "${OFF}" \
        --out-json "${OUT_DIR}/n24_ep600_off${OFF}.json"
done

echo
echo "wrote:"
ls -1 "${OUT_DIR}"
