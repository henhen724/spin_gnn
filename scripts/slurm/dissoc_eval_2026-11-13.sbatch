#!/bin/bash
#SBATCH --job-name=dissocEval
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/dissocEval_%j.out
#
# THE SAME-EDIT DISSOCIATION, registered at a29fbc2 (scripts/dissoc_score.py's
# docstring is the registration, committed before this job was submitted).
#
# NOTHING IS TRAINED. Six existing checkpoints, read four times each over the 350
# round-0 cases finding 68 published on -- the identical case list, from
# sudoku_skip's own finder, rng and instance stream, imported rather than copied.
#
# Measurement, scoring and the smoke cell are ONE job on purpose. Each piece is
# two to three minutes, and Scheduler etiquette in /etc/agents/AGENTS.md asks for
# short tasks to be aggregated rather than submitted as a handful of tiny jobs.
# The steps fail closed in order: a construction assert in the smoke cell costs
# twenty seconds, the same assert after six cells costs the job.
#
# No --deterministic and no CUBLAS_WORKSPACE_CONFIG: finding 67's pinning is for
# TRAINING. These are forward passes under torch.no_grad(), compared per case
# between variants of one sequence through one network, so kernel noise has
# nowhere to enter the comparison. H5 checks that assumption rather than assuming
# it, by reproducing finding 68's argmax case by case.
#
# The checkpoint paths are NOT retyped here. They come from ddguard_score.CKPT,
# the module that owns them, so this job and the scorer cannot disagree about
# which checkpoint is which cell.

set -uo pipefail
cd /scratch/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date +'start %F %T'

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
            py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
            py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR=/scratch/users/hshunt/numba_cache
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

RUN=/scratch/users/hshunt/spin_gnn/runs/2026-11-13_dissoc
mkdir -p "$RUN"

echo
echo "=== 0. the cell table, from ddguard_score.CKPT ===================="
python3 -c "
import sys; sys.path.insert(0, '.')
from scripts.ddguard_score import CKPT, ARMS, SEEDS
for a in ARMS:
    for s in SEEDS:
        print('%s %d %s' % (a, s, CKPT[a][s]))
" > "$RUN/cells.txt" || { echo "cannot read CKPT"; exit 1; }
cat "$RUN/cells.txt" | sed 's/^/  /'
NCELL=$(wc -l < "$RUN/cells.txt")
if [ "$NCELL" -ne 6 ]; then echo "expected 6 cells, got $NCELL"; exit 1; fi

echo
echo "=== 1. smoke: one cell, 20 instances =============================="
# Buys every assert in build_quad and rule_edit_valid for twenty seconds.
SMOKE_CK=$(head -1 "$RUN/cells.txt" | cut -d' ' -f3)
python3 -u -m spin_gnn.sudoku_dissoc --checkpoint "$SMOKE_CK" \
    --n-instances 20 --n-cases 15 --out-dir "$RUN/_smoke" || {
    echo "SMOKE FAILED -- the quad construction is wrong, no cells run"; exit 1; }

echo
echo "=== 2. six cells, 400 instances -> 350 cases ======================"
while read -r ARM SEED CK; do
    OUT="$RUN/${ARM}_s${SEED}"
    echo
    echo "---- $ARM s$SEED ----"
    python3 -u -m spin_gnn.sudoku_dissoc --checkpoint "$CK" \
        --n-instances 400 --n-cases 350 --out-dir "$OUT" || {
        echo "CELL $ARM s$SEED FAILED"; exit 1; }
done < "$RUN/cells.txt"

echo
echo "=== 3. score ======================================================"
# Guards 1-3 fail closed before any hypothesis is read; guard 3 asserts this case
# list IS finding 68's, row for row, which is what makes the numbers paired.
python3 -u scripts/dissoc_score.py || { echo "SCORING FAILED"; exit 1; }

echo
date +'end %F %T'
