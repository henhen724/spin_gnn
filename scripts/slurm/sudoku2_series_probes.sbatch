#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_serprobe
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out

# WHEN during training does DD structure appear?
#
# Submit with a dependency on the checkpoint-series job:
#   sbatch --dependency=afterok:<series_jobid> scripts/slurm/sudoku2_series_probes.sbatch
#
# Probes every snapshot of ONE training run, so the emergence of DD structure
# is measured within a single run rather than across separately-seeded models.
# series.csv carries solve_rate at each snapshot on a fixed instance pool, so
# the two curves are directly comparable.
#
# Hypothesis, given the end-of-training probe result (Slurm 48580216): the
# known_mask (bookkeeping) probe rises early and saturates near 0.92, while
# frontier_mask (the DD deducibility predicate, which needs the graph table
# joined to the known set) stays weak throughout, ending at 0.67 against a
# layer-0 control of 0.57. If so, the interesting question is whether
# solve_rate climbs while frontier stays flat -- that would say the model
# gains capability WITHOUT ever representing DD's frontier linearly, i.e. it
# is not executing DD as written.
#
# The alternative worth watching for: frontier_mask jumps at some epoch and
# solve_rate follows. That would be DD being learned, and would date it.

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

SERIES_DIR="${SCRATCH}/spin_gnn/sudoku_checkpoint_series_2026-10-19"
OUT_DIR="${SCRATCH}/spin_gnn/sudoku_series_probes_2026-10-19"
mkdir -p "$OUT_DIR"

# Every snapshot, in epoch order (zero-padded names sort correctly).
mapfile -t SNAPSHOTS < <(find "$SERIES_DIR" -maxdepth 1 -type d -name 'epoch*' | sort)
if [ "${#SNAPSHOTS[@]}" -eq 0 ]; then
    echo "no epoch* snapshots under $SERIES_DIR -- did the series job write them?" >&2
    exit 1
fi
echo "probing ${#SNAPSHOTS[@]} snapshots"

python -m spin_gnn.sudoku_probes \
    --checkpoint "${SNAPSHOTS[@]}" \
    --n-train-instances 400 \
    --n-test-instances 160 \
    --max-positions-per-instance 20 \
    --probe-epochs 600 \
    --out-csv "${OUT_DIR}/series_probes.csv"
