#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_probeauc
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%j.out

# Re-probe the checkpoint series on a THRESHOLD-FREE metric, with seed spread.
#
# Why this rerun exists. The series probes (48582169) scored every target with
# balanced accuracy, and the frontier_mask number that came out -- 0.679 at the
# peak layer, against a 0.571 layer-0 control -- turned out not to be
# interpretable. The label-only baseline (48825742) showed why: `1 - known`,
# which contains no notion of deducibility at all, scores 0.734 on the same
# metric (ABOVE the model), while a probe trained by the same procedure on the
# known bits alone scores 0.588 (BELOW it). Both cannot be the right reference.
# The cause is the metric, not the probes: balanced accuracy thresholds a
# BCE-trained probe's logits at 0, and on a ~15%-positive target that threshold
# is badly placed, so the metric ranks procedures by how freely they predict
# positives rather than by what they separate. A high-recall hand rule is
# flattered and a calibrated probe is penalised.
#
# So: auroc and average_precision are now computed in fit_linear_probe
# (spin_gnn/sudoku_probes.py; tests/test_sudoku_probes.py pins the identity
# that a 0/1 rule's AUROC equals its balanced accuracy, which is what keeps
# the hand baselines comparable across the change). Nothing about the probes,
# the activations, the checkpoints or the splits changes -- only which numbers
# come out, so this is a rescore and not a new experiment.
#
# Second change, independent of the first: --seed 0 1 2 instead of a single
# seed. The seed shifts both instance pools and the position sampling, so the
# spread across seeds is the measurement noise. This matters because the
# headline reading of the series -- that known_mask/next_node/remaining_mask
# reach 94-100% of their range by epoch 5 while next_color tracks competence
# (r=0.963) and frontier_mask stays flat (range 0.108) -- currently rests on
# one trajectory with no error bars. A tier split whose gaps are smaller than
# the seed spread is not a finding.
#
# Hypotheses:
# (1) frontier_mask's AUROC comes in ABOVE 0.679 -- a threshold-free metric
#     should recover signal that a misplaced threshold discards. The question
#     the rescore settles is whether it clears `not_known`'s 0.734 once that
#     baseline is also scored threshold-free (for a 0/1 rule the two metrics
#     coincide, so 0.734 is its AUROC unchanged). Above it: the model has
#     something beyond bookkeeping. Below: it does not, and this time the
#     comparison is sound.
# (2) The three-tier split survives: the epoch-1 -> epoch-5 jumps on
#     known_mask (0.620 -> 0.925) and next_color (0.327 -> 0.327, i.e. none)
#     are 10-100x the seed spread. If the spread turns out comparable to
#     frontier_mask's whole 0.108 range, then "flat throughout" is the only
#     defensible claim about it and the layer-0 lift must be dropped.
#
# Runtime: the single-seed sweep over these 16 snapshots took 5m38s on a GPU
# (48582169), so three seeds is ~17 min; the label-only baseline rerun that
# follows takes well under a minute. 1h30 is margin, and the two steps are
# aggregated into one job rather than submitted as two short ones.

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
OUT_DIR="${SCRATCH}/spin_gnn/sudoku_series_probes_auc_2026-10-21"
mkdir -p "$OUT_DIR"

# Same snapshots, same order, as the run being rescored.
mapfile -t SNAPSHOTS < <(find "$SERIES_DIR" -maxdepth 1 -type d -name 'epoch*' | sort)
if [ "${#SNAPSHOTS[@]}" -eq 0 ]; then
    echo "no epoch* snapshots under $SERIES_DIR -- did the series job write them?" >&2
    exit 1
fi
echo "probing ${#SNAPSHOTS[@]} snapshots x 3 seeds"

python -m spin_gnn.sudoku_probes \
    --checkpoint "${SNAPSHOTS[@]}" \
    --n-train-instances 400 \
    --n-test-instances 160 \
    --max-positions-per-instance 20 \
    --probe-epochs 600 \
    --seed 0 1 2 \
    --out-csv "${OUT_DIR}/series_probes_auc.csv"

echo
echo "=== label-only frontier baselines, rescored on the same metrics ==="
python scripts/frontier_from_known_baseline.py \
    --n-train 250 --n-test 150 --probe-epochs 300 \
    --out-json "${OUT_DIR}/frontier_baseline_auc.json"
