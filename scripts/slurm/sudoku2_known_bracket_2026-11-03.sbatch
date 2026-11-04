#!/bin/bash
#SBATCH --job-name=sud2KnownBrkt
#SBATCH --partition=sganguli
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
#
# WHAT IS THE FREE SCORE FOR known_mask?
#
# Finding 46 quotes a known_mask collapse -- AUROC 0.998 at N=24 against 0.907
# at N=96 -- and has nothing behind it, which is precisely the gap 51009233
# closed for frontier_mask and left open here. An unbracketed AUROC across two
# ensembles compares two different tasks.
#
# The known_mask bracket has a different shape from the frontier one, and the
# difference is the finding:
#   hints_only        the hint mask, constant down the trace
#   depth_probe       only (round_index, emitted-this-round) -- a token count
#   hints_plus_depth  both; THE FREE SCORE
#   exact             hints union the emissions so far, which IS known_mask,
#                     so the ceiling is exactly 1.0 -- not frontier's 0.83.
# known_mask is printed verbatim in the context. Any shortfall is a copying
# failure, not a limit of the label.
#
# A login-node spot check at N=24 with 25/15 instances and 60 probe epochs put
# hints_only alone at AUROC 0.8944, with known_mask 79% positive. If that
# holds at full size, most of the model's 0.998 at N=24 is free and the
# "bookkeeping is near-perfect" reading of finding 46 needs rewriting.
#
# PRE-REGISTERED.
#   K1 the free score is well above 0.5 at both N -- the spot check says ~0.89
#      at N=24 -- so the raw 0.998/0.907 comparison overstates what the models
#      do. The falsifier is a free score near chance, which would mean the raw
#      numbers were fine all along.
#   K2 the N=24 cells close MORE of the known bracket than the N=96 fall cells,
#      i.e. the collapse survives normalisation the way the frontier one did.
#      The interesting failure is the opposite: a free score that is itself
#      much higher at N=24 (longer traces, different rho realisation) would
#      make the collapse an artefact of the ensembles.
#   K3 hints_plus_depth >= hints_only at 600 epochs on 250 instances. The spot
#      check had it BELOW (0.7232 vs 0.8944), which is the same underfit-BCE
#      pathology H2 tested on the frontier side; if it is still below at full
#      size, the fitted free scores are floors and only hints_only can be
#      quoted.
#
# The frontier rows are recomputed unchanged at the same pool offsets and the
# same seeds, so this job doubles as a reproducibility check on 51009233: the
# frontier free score and ceiling must come back identical, and if they do not
# something about the task generation is not deterministic.
#
# COST. Same nine runs as 51009233 (~25 min there) plus two extra multilabel
# probes per run, so ~40 min expected. --time 2:00:00 as margin. CPU only:
# fit_linear_probe defaults to device="cpu" and no model is loaded anywhere.

set -euo pipefail

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
cd "${SLURM_SUBMIT_DIR}"

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_known_bracket_2026-11-03"
mkdir -p "$OUT_DIR"

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
