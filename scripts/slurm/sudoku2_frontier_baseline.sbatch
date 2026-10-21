#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_frontbase
#SBATCH -p sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out

# Is the frontier_mask probe signal anything more than bookkeeping?
#
# The probe sweep (48580216) reports frontier_mask at 0.679 against a layer-0
# control of 0.571, and STATUS has been carrying that gap as "weak but nonzero
# trace of the DD predicate, unclear whether real structure or leakage from
# known_mask". This settles it.
#
# frontier is a SUBSET of not-known -- a node that already has a color is never
# deducible -- so a reader that recovers only known_mask (which the model does,
# at 0.919) scores on frontier_mask for free, with no notion of deducibility.
#
# Hypothesis: the trivial rule `frontier = 1 - known` already EXCEEDS 0.679,
# which would mean the frontier probe is not weak evidence of DD but no
# evidence at all -- the model's frontier readout is below what its own
# bookkeeping implies. A 20-instance smoke test on a login node put not_known
# at 0.762, i.e. 0.08 above the model, so this is the expected outcome; this
# run is for a properly sized sample and a converged known_probe ceiling.
#
# No GPU and no model: everything is computed from DD's ground-truth labels,
# so it measures the TASK, not a network. The shuffled-known control must land
# at ~0.50 or the pooling is wrong.

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

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_frontier_baseline_2026-10-21"
mkdir -p "$OUT_DIR"

python scripts/frontier_from_known_baseline.py \
    --n-train 250 \
    --n-test 150 \
    --probe-epochs 400 \
    --out-json "${OUT_DIR}/frontier_from_known.json"
