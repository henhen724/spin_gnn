#!/bin/bash
#SBATCH --job-name=sud2n48ffd
#SBATCH --partition=sganguli
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
#
# HYPOTHESIS. Finding 13 (thin-deep beats wide-shallow at matched parameters)
# is a statement about the REQUIRED reasoning depth L*, not a blanket
# architecture preference. It was measured only at N=24, where L*=3. At N=48
# L*=4 (finding 14, 49523504). If depth is what the extra parameters have to
# buy, the matched-parameter winner at N=48 should still be the deeper cell,
# and the knee in a thin (ff=256) column should sit at the SAME L*=4 as in the
# wide (ff=1024) column. If instead MLP width can substitute for depth once the
# problem is bigger, the thin column's knee should fall DEEPER than 4.
#
# This is the "hyper-parameter scan at each N" step of the standing plan: it
# relates required reasoning depth to MLP depth at a second problem size.
#
# WHAT IS ALREADY MEASURED, and why only three cells are needed. The ff=1024
# column at N=48 exists in full on the metric of record (49523504's pool,
# seed_start 5e10, n=1500): L=1 0.0040, L=2 0.1787, L=3 0.8293, L=4 0.9093,
# L=8 0.9533, L=12 0.9393. So the wide side of every matched pair is done; only
# the thin side is missing.
#
# THE MATCHED PAIRS. At d_model=256 a layer costs 789,760 params at ff=1024 and
# 396,544 at ff=256 (computed from the scan's own n_params column, base 134,970),
# a ratio of 1.992 -- so 2L thin layers match L wide ones to within 0.4%:
#   L=4/ff=256  1,721,146  vs  L=2/ff=1024  1,714,490   (+0.39%)
#   L=6/ff=256  2,514,234  vs  L=3/ff=1024  2,504,250   (+0.40%)
#   L=8/ff=256  3,307,322  vs  L=4/ff=1024  3,294,010   (+0.40%)  <- the knee
# The 0.4% excess favours the thin side, same as in finding 13's N=24 pairs, so
# a thin win of that order would not be meaningful; the N=24 margins were
# +0.4860 and +0.0267, i.e. 100x and 7x the parameter slack.
#
# ONE CHANGE. Everything except dim_feedforward and num_layers is byte-identical
# to the layer scan: N=48, degree=5, n_colors=4, rho=0.3, rho_max=0.95, removal
# hints, d_model=256, nhead=4, pre-norm, lr=1e-3 CONSTANT, batch=32, 150 epochs,
# n_train=20000, seed=0. In particular the lr schedule stays constant even
# though cosine landed today (38d503a): the comparison here is the ff/depth
# split against an existing constant-lr column, and changing both at once would
# confound it. Cosine is 49565433's variable, not this job's.
#
# PRE-REGISTERED:
#  (a) L=8/ff=256 beats L=4/ff=1024 (0.9093) on the shared pool, paired p<0.05.
#      This is the load-bearing one: it is finding 13 tested AT the N=48 knee.
#      Falsifier: a tie or a loss means the thin-deep advantage does not survive
#      at a size where more depth is actually required, and finding 13 should be
#      restated as N=24-specific rather than a general claim about depth.
#  (b) L=4/ff=256 beats L=2/ff=1024 (0.1787) by > 0.30. This replicates finding
#      13's strongest N=24 pair (+0.4860, p=9.9e-204) at the new size. Falsifier:
#      no gain, which would mean something about N=48 inverts the trade-off
#      wholesale and (a) must not be read as a knee result.
#  (c) The thin column's knee is at L=4, i.e. L=4/ff=256 clears 0.90 while
#      L=2/ff=1024 does not. Falsifier: the thin knee is at 6 or 8, which is the
#      interesting negative -- it would say MLP width partly substitutes for
#      depth at larger N, the opposite of finding 13's reading, and would make
#      "L* is a property of the problem" false as stated.
#  (d) L*(T) in the thin column is FLAT in T, as it is in every series measured
#      so far (15 bins, 0 rising). Falsifier: a rising L*(T) at ff=256 would be
#      the first, and would mean the flatness was an artifact of wide MLPs
#      masking a serial cost.
#
# COST. 49523504 did one N=48 training cell plus a 6-cell depth eval plus a
# 7-checkpoint n=1500 pool in 00:35:25 at 32G/2 cores with 1.1G peak RSS. This
# is three cells (each CHEAPER than that one: ff=256 halves the per-layer cost)
# plus a 3-cell depth eval plus a 6-checkpoint pool, so ~1.5 h expected. 4 h is
# the margin; 32G/2 cores is proven sufficient for this exact shape of work.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

SCAN="${SCRATCH}/spin_gnn/sudoku_layer_scan_2026-10-24_n48"
OUT="${SCRATCH}/spin_gnn/sudoku_ff_depth_tradeoff_2026-10-25_n48"
mkdir -p "$OUT"

for L in 4 6 8; do
    echo
    echo "=== thin cell: num_layers=${L}, dim_feedforward=256, N=48 ==="
    python -m spin_gnn.sudoku_hparam_sweep \
        --n-nodes 48 --degree 5 --n-colors 4 \
        --d-model 256 --nhead 4 --num-layers "$L" --dim-feedforward 256 \
        --n-train-instances 20000 --n-val-instances 300 --batch-size 32 \
        --epochs 150 --n-per-bin 100 \
        --out-dir "$OUT"
done

echo
echo "=== depth bins in the thin column: is L*(T) flat at ff=256 too? ==="
python -m spin_gnn.sudoku_depth_eval \
    --checkpoint "${OUT}"/d0256_h04_L0[468]_ff00256_lr1e-03_pre \
    --rho 0.75 0.60 \
    --n-pool 600 \
    --min-per-bin 40 \
    --out-csv "${OUT}/depth_by_ff_n48.csv"

echo
echo "=== metric of record: three matched pairs on one pool of 1500 ==="
# The knee goes first so the CSV's vs_reference deltas read against it; every
# other pairing comes out of the per-instance npz exactly.
python -m spin_gnn.sudoku_agg_eval \
    --checkpoint "${SCAN}/d0256_h04_L04_ff01024_lr1e-03_pre" \
                 "${SCAN}/d0256_h04_L02_ff01024_lr1e-03_pre" \
                 "${SCAN}/d0256_h04_L03_ff01024_lr1e-03_pre" \
                 "${OUT}/d0256_h04_L04_ff00256_lr1e-03_pre" \
                 "${OUT}/d0256_h04_L06_ff00256_lr1e-03_pre" \
                 "${OUT}/d0256_h04_L08_ff00256_lr1e-03_pre" \
    --n-instances 1500 \
    --seed-start 50000000000 \
    --out-csv "${OUT}/sharedpool_n1500.csv"
