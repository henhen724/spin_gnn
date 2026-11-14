#!/bin/bash
#SBATCH --job-name=provEval
#SBATCH --partition=sganguli
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/provEval_%j.out
#
# THE PROVENANCE-PAIRED GUARD TEST, registered in scripts/prov_score.py and
# committed before this was submitted.
#
# Finding 70's read-back noticed, POST HOC, that J2's guard rate is 0.9903 /
# 0.8619 / 0.9830 on chained cases against 0.0000 / 0.0000 / 0.8444 on round-0
# ones. That comparison is cell-level and confounded three ways: provenance,
# depth and clash. This pairs the two inside ONE case -- the same v, broken
# either through a hint or through one of the model's own deductions -- so depth
# is identical by construction and clash is matched.
#
# NOTHING IS TRAINED. Six existing checkpoints, six forward passes per case.
#
# ONE JOB, five steps, because each is a couple of minutes and Scheduler
# etiquette asks short tasks to be aggregated:
#   0  the six checkpoint paths, read from the scorer's own table
#   1  census: how many provenance-paired cases the instances yield, at what
#      depths, and how often the two edits can be clash-matched. Costs a minute;
#      a thin yield stops the job before six cells pay for it. The chained probe
#      had this guard return 0 and stop the job in 31 s, which is how the clash
#      theorem was found.
#   2  smoke: one cell at a small size, so every assert in build_variants fires
#      on real data before the measurement starts
#   3  the six cells
#   4  the scorer, which is also the registration
#
# NO --deterministic: these are forward passes under no_grad, paired per case
# within one network, so kernel noise has nowhere to enter. Finding 67's pinning
# is for TRAINING runs.

set -uo pipefail
cd /scratch/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
date +'start %F %T'
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse --short HEAD

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
            py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
            py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR=/scratch/users/hshunt/numba_cache
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

RUN=/scratch/users/hshunt/spin_gnn/runs/2026-11-13_prov
N_INST=1200
N_CASES=400
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
echo "=== 1. census: paired-case yield at $N_INST instances ============="
python3 -u -m spin_gnn.sudoku_prov --checkpoint "$CK0" \
    --n-instances "$N_INST" --n-cases "$N_CASES" --cases-only \
    | tee "$RUN/census.txt" || exit 1
# Match the sentence, not the first line: anything torch decides to print on
# stdout first would otherwise make YIELD non-numeric, and a non-numeric test
# fails OPEN in bash -- it would run all six cells on a thin yield.
YIELD=$(grep -oE '^[0-9]+ provenance-paired cases' "$RUN/census.txt" | awk '{print $1}')
if ! [ "${YIELD:-x}" -eq "${YIELD:-x}" ] 2>/dev/null; then
    echo "could not read the case yield from the census output"; exit 1
fi
if [ "$YIELD" -lt "$N_CASES" ]; then
    echo "THIN YIELD: $YIELD paired cases < $N_CASES requested."
    echo "Stopping before the six cells. Raise --n-instances and resubmit."
    exit 1
fi
echo "  yield $YIELD >= $N_CASES, proceeding"

echo
echo "=== 2. smoke, small, into job-local scratch ======================="
python3 -u -m spin_gnn.sudoku_prov --checkpoint "$CK0" \
    --n-instances 150 --n-cases 20 \
    --out-dir "${L_SCRATCH_JOB:-/tmp}/prov_smoke" || {
        echo "SMOKE FAILED -- construction is wrong, nothing measured"; exit 1; }

echo
echo "=== 3. the six cells ==============================================="
while read -r ARM SEED CK; do
    echo
    echo "--- $ARM s$SEED ---"
    python3 -u -m spin_gnn.sudoku_prov --checkpoint "$CK" \
        --n-instances "$N_INST" --n-cases "$N_CASES" \
        --out-dir "$RUN/${ARM}_s${SEED}" || { echo "CELL FAILED"; exit 1; }
done < "$RUN/cells.txt"

echo
echo "=== 4. score ======================================================"
python3 -u scripts/prov_score.py || { echo "SCORING FAILED"; exit 1; }

date +'end %F %T'
