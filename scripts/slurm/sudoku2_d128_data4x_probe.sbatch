#!/bin/bash
#SBATCH --job-name=sud2d128x4
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=2:30:00
#SBATCH --output=%x_%j.out
#
# Is the 0.863 -> 0.907 frontier-AUROC gap width or data?
#
# Finding 3 used to read "the frontier is represented from epoch 5 and never
# develops, peak AUROC 0.863". The competence ladder (48913017) found
# 0.907-0.938 instead, rising with competence -- but 0.863 came from the
# d0128 checkpoint series (48861799) and every ladder checkpoint is d0256 or
# d0384, so width and training-set size are confounded in exactly the
# comparison the revision rests on. Within a width the ladder is clean (d256
# 0.907 -> 0.916, d384 0.916 -> 0.938), and those effects are small; the big
# step is the one that crosses widths. This run fills the missing cell of the
# 2x2 so the two effects can be separated:
#
#               n_train=5000     n_train=20000
#   d0128          0.863            THIS RUN
#   d0256          0.907             0.916
#   d0384          0.916             0.938
#
# ONE change against the d0128 series (sudoku_checkpoint_series_2026-10-19,
# probed by 48861799): n_train_instances 5000 -> 20000. d_model 128, nhead 4,
# 12 layers, ff 1024, lr 1e-3, batch 32 all held; epochs 300, which is the
# series checkpoint the comparison uses (epoch00300), not its 900.
#
# The job then does three things in one allocation rather than three queue
# waits, which also keeps it well past the 10-minute aggregation guideline:
#   1. train d0128 x 20000 for 300 epochs;
#   2. probe BOTH the new checkpoint and the series' epoch00300 (d0128 x 5000)
#      with probe hyperparameters identical to 48861799 and 48913017, so the
#      2x2's bottom-left corner is re-measured by the same code path and the
#      same splits as its new right-hand neighbour rather than quoted from a
#      run three days old;
#   3. score both on 48912134's shared 1500-instance pool, with the current
#      best (d0256 x 20000) as the paired reference, so the two new models land
#      on the same competence axis as everything else and finding 8's claim
#      (next_node peak accuracy reproduces the shared-pool ranking) gets two
#      more points to be tested against instead of the four it was read off.
#
# PRE-REGISTERED PREDICTIONS.
#  (a) frontier_mask peak AUROC in [0.855, 0.895], i.e. clearly SHORT of
#      d0256 x 5000's 0.907. Reasoning: within-width data effects measured so
#      far are small (+0.009 at d256, +0.022 at d384), so 0.863 + ~0.015. If it
#      instead reaches >= 0.907 the 0.863 -> 0.907 gap was data and not width
#      at all, finding 3's revision is understated, and the whole ladder has to
#      be re-read as a data-scaling curve.
#  (b) next_node peak accuracy >= 0.93, joining the two other n_train=20000
#      models (0.957, 0.964) rather than the 5000-instance ones (0.789-0.810).
#      This is the sharper test of finding 8: it predicts a STEP with data,
#      not a slope with competence, at a width that has never been tested.
#  (c) shared-pool solve_rate between 0.90 and 0.96, and NOT better than
#      d0256 x 20000's 0.9847 (negative paired delta). A win here would mean
#      width has been the wrong axis all along.
#  (d) the d0128 x 5000 re-probe reproduces 0.863 +/- 0.015. If it does not,
#      something drifted between 48861799 and now and (a) is uninterpretable --
#      this is the control, and it is checked first.
#
# Runtime: 48854233 (d256 x 20000) trained 300 epochs in 1:24 and d128 is
# roughly a quarter of the parameters, so ~1:10 with margin; the ladder probed
# 5 checkpoints x 3 seeds in 5:50 and 48912134 scored 4 checkpoints x 1500
# instances in 5:31, so stages 2 and 3 are ~10 minutes together. 2:30 is the
# smallest request that covers a slow GPU without repeating the 140x
# over-request of 48868696.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

S="${SCRATCH}/spin_gnn"
OUT_DIR="${S}/sudoku_hparam_sweep_2026-10-21_d128_data4x"
EVAL_DIR="${S}/sudoku_d128_data4x_2026-10-21"
mkdir -p "$OUT_DIR" "$EVAL_DIR"

NEW_CKPT="${OUT_DIR}/d0128_h04_L12_ff01024_lr1e-03_pre"
OLD_CKPT="${S}/sudoku_checkpoint_series_2026-10-19/epoch00300"
BEST_CKPT="${S}/sudoku_hparam_sweep_2026-10-21_d256_data4x/d0256_h04_L12_ff01024_lr1e-03_pre"

echo "=== 1/3 train: d_model=128 nhead=4 L=12 ff=1024 epochs=300 n_train=20000 ==="
python -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 24 \
    --degree 5 \
    --n-colors 4 \
    --d-model 128 \
    --nhead 4 \
    --num-layers 12 \
    --dim-feedforward 1024 \
    --n-train-instances 20000 \
    --n-val-instances 300 \
    --batch-size 32 \
    --epochs 300 \
    --n-per-bin 100 \
    --out-dir "$OUT_DIR"

echo
echo "=== 2/3 probes: the new d128 x 20000 and the d128 x 5000 control ==="
python -m spin_gnn.sudoku_probes \
    --checkpoint "$OLD_CKPT" "$NEW_CKPT" \
    --n-train-instances 400 \
    --n-test-instances 160 \
    --max-positions-per-instance 20 \
    --probe-epochs 600 \
    --seed 0 1 2 \
    --out-csv "${EVAL_DIR}/probes_d128_2x2.csv"

echo
echo "=== 3/3 shared-pool paired eval against the current best ==="
python -m spin_gnn.sudoku_agg_eval \
    --checkpoint "$BEST_CKPT" "$NEW_CKPT" "$OLD_CKPT" \
    --n-instances 1500 \
    --out-csv "${EVAL_DIR}/agg_paired_d128.csv"
