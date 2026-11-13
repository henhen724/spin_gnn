#!/bin/bash
#SBATCH --job-name=chainEval
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/chainEval_%j.out
#
# THE CHAINED TEST, registered at 968c355 before this was submitted.
#
# Finding 69 is round-0 only: every fact it edited was a HINT. This moves all
# four edits into the reasoning section and requires the binding neighbour to
# be a DEDUCTION the model emitted earlier, so the question becomes whether the
# elimination rule runs over the model's own output.
#
# NOTHING IS TRAINED. Six existing checkpoints, four forward passes per case.
#
# ONE JOB, five steps, because each is a couple of minutes and Scheduler
# etiquette asks short tasks to be aggregated:
#   0  the six checkpoint paths, read from the scorer's own table
#   1  census: how many chained cases 1200 instances yield, and at what depths.
#      Costs a minute; a thin yield stops the job before six cells pay for it.
#   2  smoke: one cell at a small size, so every assert in build_quad_chain
#      fires on real data before the measurement starts
#   3  the six cells
#   4  the scorer, which is also the registration
#
# NO --deterministic: these are forward passes under no_grad, paired per case
# between two networks, so kernel noise has nowhere to enter. Finding 67's
# pinning is for TRAINING runs.

set -uo pipefail
cd /scratch/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
date +'start %F %T'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
            py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
            py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR=/scratch/users/hshunt/numba_cache
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

RUN=/scratch/users/hshunt/spin_gnn/runs/2026-11-13_chain
N_INST=1200
N_CASES=300
mkdir -p "$RUN"

echo
echo "=== 0. the six cells, from scripts.ddguard_score.CKPT ============="
python3 -c "
from scripts.ddguard_score import ARMS, CKPT, SEEDS
for a in ARMS:
    for s in SEEDS:
        print('%s %d %s' % (a, s, CKPT[a][s]))
" > "$RUN/cells.txt" || exit 1
cat "$RUN/cells.txt" | sed 's/^/  /'
CK0=$(awk 'NR==1 {print $3}' "$RUN/cells.txt")

echo
echo "=== 1. census: chained-case yield at $N_INST instances ============"
python3 -u -m spin_gnn.sudoku_chain4 --checkpoint "$CK0" \
    --n-instances "$N_INST" --n-cases "$N_CASES" --cases-only \
    | tee "$RUN/census.txt" || exit 1
# Match the sentence, not the first line: anything torch decides to print on
# stdout first would otherwise make YIELD non-numeric, and a non-numeric test
# fails OPEN in bash -- it would run all six cells on a thin yield.
YIELD=$(grep -oE '^[0-9]+ chained cases' "$RUN/census.txt" | awk '{print $1}')
if ! [ "${YIELD:-x}" -eq "${YIELD:-x}" ] 2>/dev/null; then
    echo "could not read the case yield from the census output"; exit 1
fi
if [ "$YIELD" -lt "$N_CASES" ]; then
    echo "THIN YIELD: $YIELD chained cases < $N_CASES requested."
    echo "Stopping before the six cells. Raise --n-instances and resubmit."
    exit 1
fi
echo "  yield $YIELD >= $N_CASES, proceeding"

echo
echo "=== 2. smoke, small, into job-local scratch ======================="
python3 -u -m spin_gnn.sudoku_chain4 --checkpoint "$CK0" \
    --n-instances 150 --n-cases 20 \
    --out-dir "${L_SCRATCH_JOB:-/tmp}/chain_smoke" || {
        echo "SMOKE FAILED -- construction is wrong, nothing measured"; exit 1; }

echo
echo "=== 3. the six cells ==============================================="
while read -r ARM SEED CK; do
    echo
    echo "--- $ARM s$SEED ---"
    python3 -u -m spin_gnn.sudoku_chain4 --checkpoint "$CK" \
        --n-instances "$N_INST" --n-cases "$N_CASES" \
        --out-dir "$RUN/${ARM}_s${SEED}" || { echo "CELL FAILED"; exit 1; }
done < "$RUN/cells.txt"

echo
echo "=== 4. score ======================================================"
python3 -u scripts/chain_score.py || { echo "SCORING FAILED"; exit 1; }

date +'end %F %T'
