#!/bin/bash
#SBATCH --job-name=sud2n96warm
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=28:00:00
#SBATCH --output=%x_%j.out
#
# IS L*(96, ff=256) > 12 A FACT ABOUT COMPUTATION, OR ABOUT TRAINING?
#
# HYPOTHESIS, stated before the run. It is about training. The deep-thin cell
# at N=96 fails the depth gate because it never learned the output GRAMMAR, not
# because it cannot deduce, and a 5-epoch linear lr warmup is enough to fix
# that. If so, L=12/ff=256 clears bins it currently misses and the width factor
# at N=96 stops being super-multiplicative.
#
# THE EVIDENCE THAT POINTS AT TRAINING (finding 25, job 49816217, N=96, ff=256,
# rho=0.60 depth bins):
#
#                      well_formed_rate        solve | well_formed
#     L=8  ff=256      0.971 - 1.000           0.976 -> 0.836 over T=4..7 (r=-1.00)
#     L=12 ff=256      0.853 (min), 6 bins     0.923 -> 0.983 over T=4..9 (r=+0.72)
#
# The L=12 cell's CONDITIONAL solve rate RISES with reasoning depth and sits at
# the wide knee's level: conditioned on emitting a parseable answer it deduces
# fine, and 15% of its outputs are simply not parseable. That is an
# early-training failure signature, not a capacity one -- a model short of
# serial depth degrades on the HARD bins, which is exactly what L=8/ff=256 does
# and L=12/ff=256 does not. The second hint is that final val_nll ranks the
# three N=96 cells in REVERSE of their pooled solve rate, i.e. the loss the
# optimizer sees is not tracking the behaviour being measured.
#
# ONE CHANGE from job 49816217: --warmup-epochs 5. lr stays 1e-3, the schedule
# stays "constant" (so the ramp is followed by a hold, not a decay -- cosine
# would be a second change and gets its own job if this one needs a follow-up),
# and the data budget, epochs, batch, norm, seed and instance distribution are
# 49816217's. shuffle_adj stays False, so these are DECODING scores in the same
# leaked-order task as findings 11, 13, 14, 22 and 25.
#
# The warmup code is new (config.warmup_epochs, _build_lr_schedule, and a
# --warmup-epochs flag). Validated on CPU before submitting: with
# warmup_epochs=5, epochs=150, lr=1e-3 the per-epoch lr is
# 1.667e-4, 3.333e-4, 5.000e-4, 6.667e-4, 8.333e-4, then 1.000e-3 held; with
# warmup_epochs=0 the "constant" path still returns None, so every earlier run
# is byte-identical.
#
# L=8 IS THE CONTROL, NOT A SECOND TREATMENT. Its cold well_formed_rate is
# already 0.971-1.000, so there is no grammar failure for warmup to repair; if
# warmup is doing what the hypothesis says, L=8 should barely move while L=12
# recovers. If BOTH jump, the effect is not about the grammar at all.
#
# PRE-REGISTERED:
#  (a) L=12/ff=256 + warmup reaches well_formed_rate >= 0.98 in every rho=0.60
#      depth bin (cold: 0.853 at the worst bin). Falsifier: it stays <= 0.90,
#      which kills the trainability story outright -- a thin MLP at depth 12
#      then simply cannot hold the output grammar at N=96, and that becomes the
#      finding.
#  (b) Given (a), L=12/ff=256 clears the 0.90 gate on the rho=0.60 bins, so
#      L*(96, ff=256) <= 12 and the MLP-width factor on L* is 1.5 at both N=48
#      and N=96 rather than growing past it. Falsifier: well_formedness
#      recovers but the bins still miss -- then the knee really is above 12,
#      the factor IS super-multiplicative, and finding 25 stands as written
#      with one confound removed.
#  (c) L=8/ff=256 + warmup ties its cold self on the n=1500 shared pool
#      (McNemar p > 0.05 against 0.8407). Falsifier: it improves resolvably,
#      which means warmup lifts the whole thin column and the 7.1-point width
#      gap in finding 25 was measured against an undertrained competitor -- the
#      honest response then is that the wide cells need re-running with warmup
#      too before any width number at N=96 is quoted, and this job cannot
#      settle that.
#  (d) With warmup, final val_nll stops ranking the thin cells in reverse of
#      their pooled solve rate. Falsifier: the inversion survives, in which
#      case val_nll is not a usable model-selection signal at N=96 and the open
#      "no best-val checkpoint selection" item should be closed as moot rather
#      than implemented.
#
# COST. Job 49816217 ran exactly these two trainings plus a 2-checkpoint depth
# eval and a 3-checkpoint n=1500 pool in 15:50:15. This adds two checkpoints to
# the pool (the cold cells, so cold-vs-warm is paired on the same 1500
# instances) -- call it 17-18 h against a 28 h limit. Resources are 49816217's.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

WIDE="${SCRATCH}/spin_gnn/sudoku_n96_layer_scan_2026-10-26/d0256_h04_L08_ff01024_lr1e-03_pre"
COLD="${SCRATCH}/spin_gnn/sudoku_n96_ff256_2026-10-27"
OUT="${SCRATCH}/spin_gnn/sudoku_n96_warmup_2026-10-28"
mkdir -p "$OUT"

# L=12 first: it carries the hypothesis, so if the job runs out of time the
# expensive question is already answered.
for L in 12 8; do
    echo
    echo "=== N=96 num_layers=${L}, ff=256, warmup 5 (one change from 49816217) ==="
    python -m spin_gnn.sudoku_hparam_sweep \
        --n-nodes 96 --degree 5 --n-colors 4 \
        --d-model 256 --nhead 4 --num-layers "$L" --dim-feedforward 256 \
        --n-train-instances 80000 --n-val-instances 300 --batch-size 32 \
        --epochs 150 --warmup-epochs 5 --n-per-bin 100 \
        --out-dir "$OUT"
done

echo
echo "=== the same depth bins finding 25 was read off ==="
python -m spin_gnn.sudoku_depth_eval \
    --checkpoint "${OUT}/d0256_h04_L12_ff00256_lr1e-03_pre_w05" \
                 "${OUT}/d0256_h04_L08_ff00256_lr1e-03_pre_w05" \
    --rho 0.75 0.60 \
    --n-pool 600 \
    --min-per-bin 40 \
    --out-csv "${OUT}/depth_n96_ff256_warmup.csv"

echo
echo "=== metric of record: warm vs cold vs the wide knee, one pool ==="
# The wide knee goes first so vs_reference reads against L*(96, ff=1024) = 8,
# as it does in findings 22 and 25. The two cold ff=256 cells are 49816217's
# checkpoints, included so warm-vs-cold is paired instance-for-instance.
python -m spin_gnn.sudoku_agg_eval \
    --checkpoint "$WIDE" \
                 "${OUT}/d0256_h04_L12_ff00256_lr1e-03_pre_w05" \
                 "${OUT}/d0256_h04_L08_ff00256_lr1e-03_pre_w05" \
                 "${COLD}/d0256_h04_L12_ff00256_lr1e-03_pre" \
                 "${COLD}/d0256_h04_L08_ff00256_lr1e-03_pre" \
    --n-instances 1500 \
    --seed-start 50000000000 \
    --out-csv "${OUT}/sharedpool_n1500.csv"
