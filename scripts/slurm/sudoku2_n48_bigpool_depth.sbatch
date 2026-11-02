#!/bin/bash
#SBATCH --job-name=sud2N48bigpool
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=%x_%j.out
#
# PUT AN ERROR BAR ON L* ITSELF: RE-EVALUATE THE N=48 LEAK-FREE LADDER ON A 4x
# LARGER DEPTH POOL. NO RETRAINING -- the six seed-0 checkpoints are final.
#
#   sbatch scripts/slurm/sudoku2_n48_bigpool_depth.sbatch
#
# HYPOTHESIS, stated before the run: H1 -- L*(48, leak-free, rho=0.60) = 12 is
# real, and it fails the 95% test today only because the binding bin has n=70.
# Quadrupling the pool should make it clear the gate at a simultaneous 95%
# level without moving the point estimate.
#
# WHY. L* is not a paired contrast. Every other number in this project
# (findings 38/40, McNemar on the shared n=1500 pool) compares two models on
# the SAME instances and gets enormous power from the pairing. L* is an
# ABSOLUTE threshold on ONE cell -- "solve_rate >= 0.90 in every depth bin" --
# so it gets none of that help, and its precision is set purely by the bin
# sizes. Those are small at the deep end, and nothing in STATUS says so.
#
# $SCRATCH/lstar_fragility.py audited every L* on disk (2026-11-01, CPU only,
# no new training). Per-bin Wilson intervals, Bonferroni across the bins the
# gate quantifies over, worst bin not best:
#
#   ladder                       L*   worst-bin 95% lo   verdict at 95%
#   N=48 leak-free rho=0.75       8        0.9455        HOLDS
#   N=48 leak-free rho=0.60      12        0.8422        point estimate only
#   N=96 ff=1024   rho=0.75       4        0.7967        smallest safe rung is 8
#   N=96 ff=1024   rho=0.60       8        0.7605        point estimate only
#   N=96 ff=256    rho=0.75       8        0.9013        HOLDS (barely)
#
# One of five survives. Two specifics worth stating plainly:
#   * N=96, rho=0.75, ff=1024: the L=4 cell clears its binding bin 155/172 =
#     0.9012. At 154/172 = 0.8953 it does not, and L* jumps 4 -> 8. ONE
#     INSTANCE decides a number STATUS quotes flat, and finding 26's "ff
#     matters more at larger N" ratio (8/4 = 2.0x) is 1.0x at 95%.
#   * N=96, rho=0.60: finding 31's L=7 misses by 0.0029 -- 8% of one bin SE.
# These are re-readings of CSVs already on disk, not new claims about the
# models, and they do not depend on how 50691430/50691432 come out.
#
# WHAT SIZE FIXES IT. The design table in the same script inverts the Wilson
# bound: the smallest n at which a cell whose TRUE bin rate is p clears 0.90 at
# a simultaneous 95% level over 6 bins is
#
#       p     0.94  0.95  0.96  0.97  0.98  1.00
#       n      373   244   161   115    99    63
#
# The cells that actually decide L* sit at 0.93-0.96 -- that is what being at
# the knee means -- so they need n ~ 160-380 per bin. The current deep-end bins
# are n=70 (N=48, T=8), n=57 and n=43 (N=96). At n=70 nothing below a true rate
# of 0.98 can ever clear, which is why the audit reads the way it does.
#
# This is on the critical path for the scan hshunt asked for (does required
# reasoning depth track MLP depth?): L* is that scan's DEPENDENT VARIABLE, and
# at current bin sizes it carries +-2 to 4 rungs. Measuring it better is worth
# more than another rung measured the old way.
#
# ONE CHANGE. --n-pool 1500 -> 6000 and --min-per-bin 60 -> 150. Same six
# checkpoints, same --rho, same --seed-start default (30e9), same code. The
# pool NESTS exactly: build_instances draws instance i from seed_start + i
# alone (train_sudoku_transformer.py, default_rng(seed_start + i + k) for every
# component), so instances 0..1499 of the 6000-pool ARE the 1500-pool and every
# old bin is a subset of its new one. Nothing here supersedes the n=1500
# numbers; it extends them.
#
# min-per-bin 150 is picked from the table above, not for roundness: below it a
# reported bin cannot support the statement the gate is making, so a bin that
# does not reach 150 should be dropped rather than gated on.
#
# PRE-REGISTERED, written before submission:
#
#  (A) HEADLINE, moderate confidence. On the SAME six bins the n=1500 run
#      gated at rho=0.60 (apples to apples), L=12's worst-bin Bonferroni 95%
#      lower bound reaches >= 0.90, i.e. L*=12 becomes a 95% statement and not
#      just a point estimate. Expected n at T=8 is ~280 and the point rate is
#      0.9571, which gives a lower bound near 0.925.
#      FALSIFIER: the bound stays below 0.90 with n >= 250 in that bin. Then
#      the T=8 rate is genuinely nearer 0.93 than 0.96, L*(48, rho=0.60) is NOT
#      12, and the ladder has no clearing rung at all -- 12 layers would not be
#      enough for rho=0.60 at N=48, which is a bigger result than the one I am
#      predicting and would make the deeper rungs the priority over the scan.
#
#  (B) BIN SET, the confound, genuinely uncertain. A larger pool admits DEEPER
#      T bins that n=1500 dropped, and the gate quantifies over every admitted
#      bin, so the test gets strictly harder at the same time as the intervals
#      get tighter. These must not be conflated. Prediction: at rho=0.60 at
#      least one new bin (T >= 9) reaches 150, and L=12's point rate there is
#      below its T=8 rate (solve_rate is monotone decreasing in T in every
#      ladder measured so far).
#      Scoring rule, fixed now: (A) is scored ONLY on the old six bins. If a
#      new deep bin drops L=12 below the gate, that is reported separately as
#      "L*(48, rho=0.60) is undefined under a stricter bin set", NOT as (A)
#      failing. The two questions have different answers and both are wanted.
#
#  (C) CONTROL, high confidence. For every (checkpoint, rho, T) bin present in
#      both runs, n_6000 / n_1500 lies in [3.0, 5.0] (nesting plus Poisson:
#      the smallest bin, n=70, has a ratio sd of ~0.48, so this is ~4 sd), and
#      every bin's solve_rate agrees with the n=1500 value within 3 binomial SE
#      of the smaller bin.
#      FALSIFIER: a systematic shift in solve_rate between the two pools, which
#      would mean the pool is not nested the way the generator says it is and
#      would retire the determinism argument that the cross-job pairing in
#      findings 38 and 40 rests on.
#
#  (D) rho=0.75 UNCHANGED. L*=8 still clears, and at n ~ 780 in its binding bin
#      its lower bound rises from 0.9455 toward 0.97.
#      FALSIFIER: L*(rho=0.75) != 8.
#
# COST. Measured from 50607549's own output mtimes: depth_eval over 6
# checkpoints x 2 rho at n-pool 1500 took 15 min 29 s, and it rebuilds the pool
# once per (checkpoint, rho) -- 12 builds. Generation dominates at ~52 ms per
# instance, so 4x the pool is ~62 min, plus decoding that scales the same way.
# --time 03:00:00 is ~2.7x headroom. sganguli (sh_part, 11-01): max 7 d, 32 GB
# per core, so 4 cpus and 64 GB are inside the limits.
#
# DELIBERATELY NOT DONE HERE. sudoku_depth_eval rebuilds the pool for every
# checkpoint; memoizing it on (instance-config, n_pool, seed_start) would cut
# this job roughly 6x and help every future depth eval. Not touched, because
# 50691430 and 50691432 are mid-training and will IMPORT that module when they
# reach their eval stage -- editing it now would change the code that produces
# their pre-registered numbers while they are in flight. It is a follow-up for
# after they land, not a saving worth taking today.
#
# No agg_eval here on purpose: the paired shared-pool contrasts already have
# all the power they need (finding 40's McNemar p = 5.4e-09 at n=1500). This
# job changes exactly one thing, the depth-bin sample size.

set -euo pipefail
NNODES=48
NPOOL=6000
MINPERBIN=150
DATESTAMP=2026-11-01

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

echo "=== smoke check ==="
python3 - <<'PY'
import torch, numpy, numba
import spin_gnn.sudoku_depth_eval
print("torch", torch.__version__, "numpy", numpy.__version__, "numba", numba.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert torch.cuda.is_available(), "no GPU visible to torch"
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n${NNODES}_leakfree_bigpool_${DATESTAMP}"
mkdir -p "$OUT"

# The six seed-0 rungs, exactly the set 50691430/50691432 re-evaluate. Guarded
# so a missing cell cannot fail the job.
DEEP_DIR="${SCRATCH}/spin_gnn/sudoku_n48_leakfree_lstar_2026-10-31"
L10_DIR="${SCRATCH}/spin_gnn/sudoku_n48_leakfree_L10_2026-10-31"
REF_DIR="${SCRATCH}/spin_gnn/sudoku_n48_leakfree_lstar_2026-10-29"
CKPTS=()
for spec in "${DEEP_DIR}:12" "${L10_DIR}:10" "${DEEP_DIR}:08" \
            "${REF_DIR}:06" "${REF_DIR}:04" "${REF_DIR}:03"; do
    d="${spec%:*}"; L="${spec##*:}"
    ref="${d}/L${L}/d0256_h04_L${L}_ff01024_lr1e-03_pre_shuf"
    if [ -d "$ref" ]; then CKPTS+=("$ref"); else echo "skipping missing reference $ref"; fi
done
echo "N=${NNODES} n_pool=${NPOOL} min_per_bin=${MINPERBIN} ckpts=${#CKPTS[@]} -> ${OUT}"

echo
echo "=== pre-regs (A)+(B)+(C)+(D): depth bins at n_pool=${NPOOL} ==="
date +'depth start %F %T'
python3 -m spin_gnn.sudoku_depth_eval \
    --checkpoint "${CKPTS[@]}" \
    --rho 0.75 0.60 --n-pool "$NPOOL" --min-per-bin "$MINPERBIN" \
    --out-csv "${OUT}/depth_bigpool_${SLURM_JOB_ID}.csv"
date +'depth end   %F %T'

echo
echo "=== bin sizes, old vs new (pre-reg C) ==="
python3 - "${OUT}/depth_bigpool_${SLURM_JOB_ID}.csv" <<'PY'
import glob, os, sys
import pandas as pd

new = pd.read_csv(sys.argv[1])
S = os.environ["SCRATCH"] + "/spin_gnn"
old = pd.concat([pd.read_csv(p) for g in ("sudoku_n48_leakfree_lstar_2026-10-29/depth_lstar_*.csv",
                                          "sudoku_n48_leakfree_lstar_2026-10-31/depth_lstar_*.csv",
                                          "sudoku_n48_leakfree_L10_2026-10-31/depth_L10_*.csv")
                for p in sorted(glob.glob(f"{S}/{g}"))], ignore_index=True)
old = old.drop_duplicates(subset=["checkpoint", "rho_requested", "T"])
key = ["checkpoint", "rho_requested", "T"]
m = new.merge(old, on=key, suffixes=("_new", "_old"))
m["ratio"] = m.n_new / m.n_old
m["d_rate"] = m.solve_rate_new - m.solve_rate_old
m["se_old"] = (m.solve_rate_old * (1 - m.solve_rate_old) / m.n_old) ** 0.5
m["z"] = m.d_rate / m.se_old.replace(0, float("nan"))
print(m[["checkpoint", "rho_requested", "T", "n_old", "n_new", "ratio",
         "solve_rate_old", "solve_rate_new", "z"]].round(4).to_string(index=False))
print(f"\nratio range {m.ratio.min():.2f}-{m.ratio.max():.2f} (pre-reg C wants [3.0, 5.0])")
print(f"max |z| vs old bins {m.z.abs().max():.2f} (pre-reg C wants <= 3)")
bins_new = new[new.rho_requested == 0.60].groupby("T").n.max()
print(f"\nrho=0.60 bins now admitted (T: n): {dict(bins_new)}")
PY
