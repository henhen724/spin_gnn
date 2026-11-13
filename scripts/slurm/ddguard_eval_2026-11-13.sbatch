#!/bin/bash
#SBATCH --job-name=ddgEval
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/ddgEval_%j.out
#
# THE FIRE-AND-GUARD PROBE. Submitted --dependency=afterok on
# scripts/slurm/ddguard_register_2026-11-13.sbatch, whose docstring-commit is the
# registration and whose guard 1 verifies the six checkpoints -- so this job
# cannot start against an unregistered hypothesis or a mislabelled cell.
#
# NOTHING IS TRAINED. Twelve evaluations of six checkpoints that already exist.
# No knob moves; no published number can change. The hypotheses, bars,
# falsifiers and caveats are in scripts/ddguard_score.py and were committed
# before this ran.
#
# WHY A GPU AND NOT CPU: finding 28's N=24 arms ran on CPU, but N=48 is a longer
# sequence and this is 12 cells rather than 3. A GPU on sganguli starts fast and
# turns an hour into minutes. Determinism is NOT pinned here and does not need to
# be (finding 67): every cell is model.eval() under torch.no_grad() with no
# optimizer, so the only arithmetic is a forward pass, and the comparison is
# PAIRED PER CASE between two different networks rather than between two runs of
# one network. Run-to-run kernel nondeterminism has nowhere to enter the
# statistic. What the probe does NOT remove is the seed-to-seed basin lottery,
# which is exactly why all three seeds of each arm are run.
#
# --n-instances 800 for both tests, --n-cases 350 as findings 27/28 used. The
# instance pool is a pure function of (config, --seed-start, --n-instances) and
# those are identical across all six cells, so the case lists match row for row;
# the scorer's guard 2 asserts that and exits non-zero if they ever do not.

set -uo pipefail
cd /home/users/hshunt/spin_gnn || exit 1

echo "host=$(hostname)  job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
date +'start %F %T'

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 \
            py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 \
            py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR=/scratch/users/hshunt/numba_cache
export PYTHONUNBUFFERED=1

RUN=/scratch/users/hshunt/spin_gnn/runs/2026-11-13_ddguard
SEED_START=600000000
NINST=800
NCASES=350
mkdir -p "$RUN"

# The six checkpoints come from the scorer, so the launcher cannot drift from the
# module that defines the arms.
mapfile -t CELLS < <(python3 scripts/ddguard_score.py --paths)
printf '  cell %s\n' "${CELLS[@]}"
if [ "${#CELLS[@]}" -ne 6 ]; then
    echo "expected 6 cells, got ${#CELLS[@]}"; exit 1
fi

echo
echo "=== 0. SMOKE: does a TIED checkpoint load and find cases at all? ==="
# The one real unknown. Both modules read len(encoder.layers) for per-layer work,
# which is 1 on a tied block and 16 on its untied twin -- the bug fixed in the
# 2026-11-10 stageop commit, found because stage P was the first tied checkpoint
# anyone pointed these at. Behaviour-only does not touch that path, but a tiny
# cell costs a minute and a wrong answer here would otherwise cost twelve.
SMOKE="${RUN}/_smoke"
read -r SARM SSEED SCKPT <<<"${CELLS[0]}"
echo "smoking ${SARM} s${SSEED}"
python3 -u -m spin_gnn.sudoku_circuits --checkpoint "$SCKPT" --stages none \
    --n-instances 40 --n-cases 6 --seed-start "$SEED_START" \
    --out-dir "${SMOKE}/rule" || { echo "SMOKE rule FAILED"; exit 1; }
python3 -u -m spin_gnn.sudoku_skip --checkpoint "$SCKPT" \
    --n-instances 40 --n-cases 6 --seed-start "$SEED_START" \
    --out-dir "${SMOKE}/guard" || { echo "SMOKE guard FAILED"; exit 1; }
echo "smoke: PASS -- the tied checkpoint loads and both case finders fire"

echo
echo "=== 1. THE RULE (fire): does it follow a redirected forcing set? ==="
for cell in "${CELLS[@]}"; do
    read -r ARM SEED CKPT <<<"$cell"
    echo
    echo "--- rule ${ARM} s${SEED} ---"
    python3 -u -m spin_gnn.sudoku_circuits --checkpoint "$CKPT" --stages none \
        --n-instances "$NINST" --n-cases "$NCASES" --seed-start "$SEED_START" \
        --out-dir "${RUN}/rule/${ARM}_s${SEED}"
    echo "rule ${ARM} s${SEED} rc=$?"
done

echo
echo "=== 2. THE GUARD (withhold): does it stop naming an undetermined node? ==="
for cell in "${CELLS[@]}"; do
    read -r ARM SEED CKPT <<<"$cell"
    echo
    echo "--- guard ${ARM} s${SEED} ---"
    python3 -u -m spin_gnn.sudoku_skip --checkpoint "$CKPT" \
        --n-instances "$NINST" --n-cases "$NCASES" --seed-start "$SEED_START" \
        --out-dir "${RUN}/guard/${ARM}_s${SEED}"
    echo "guard ${ARM} s${SEED} rc=$?"
done

echo
echo "=== 3. SCORE against the registration ============================="
python3 -u scripts/ddguard_score.py --run-dir "$RUN"
echo "score rc=$?"

echo
echo "NOT PUBLISHED -- a deck's figures still have to be read back twice."
date +'end %F %T'
