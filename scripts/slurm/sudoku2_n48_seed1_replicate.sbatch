#!/bin/bash
#SBATCH --job-name=sud2N48s1
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=13:00:00
#SBATCH --output=%x_%j.out
#
# PUT AN ERROR BAR ON THE N=48 LEAK-FREE DEPTH LADDER: SEED 1 AT L=10 AND L=8.
#
#   sbatch --job-name=sud2N48L10s1 --time=13:00:00 \
#       scripts/slurm/sudoku2_n48_seed1_replicate.sbatch 10
#   sbatch --job-name=sud2N48L08s1 --time=10:00:00 \
#       scripts/slurm/sudoku2_n48_seed1_replicate.sbatch 8
#
# HYPOTHESIS, stated before the runs: H1 -- finding 40's inversion was
# OPTIMIZATION NOISE, and a second seed at L=10 will not reproduce it.
#
# WHY. Finding 40 put L=10 BELOW L=8 on the shared n=1500 pool:
#
#     L=3    0.0093      L=8    0.8660   3/6 bins at rho=0.60
#     L=4    0.3153      L=10   0.8007   2/6 bins      <-- the inversion
#     L=6    0.6153      L=12   0.9780   6/6 bins
#
# paired L=10 - L=8 = -0.0653, discordant 92-190, McNemar exact p = 5.41e-09.
# That p compares two CHECKPOINTS on the same instances. It says nothing about
# two TRAINING PROCEDURES, and there is exactly one seed per rung on this
# ladder -- no rung in findings 11/14/22/33/38/40 has ever been replicated. So
# H1 (optimization noise) and H2 (a real non-monotonicity in depth) are both
# live and the existing data cannot separate them.
#
# H1 is quantitatively coherent, which is why it leads. Near this knee solve
# rate is extraordinarily steep in val_nll: L=12 beats L=8 by 0.1120 on a
# val_nll gap of 0.0040, and L=8 beats L=10 by 0.0653 on a gap of 0.0026 --
# the same ~25-28 solve-rate points per 0.001 nll. A seed that moves final
# val_nll by ~0.002, which is unremarkable, therefore moves solve rate by
# ~0.05. The inversion is the size optimization noise should be.
#
# This blocks the hyper-parameter scan hshunt asked for (does required
# reasoning depth track MLP depth?). Its centre was the L=10 cell, and a scan
# whose rungs carry an unmeasured spread of >= 0.065 cannot resolve the effect
# it is looking for. These two cells are the cheapest thing that unblocks it.
#
# ONE CHANGE. Every argument below is job 50607549's verbatim, which was
# 50527633's verbatim (N=48, degree 5, q=4, d_model 256, nhead 4, ff=1024,
# pre-norm, lr 1e-3, batch 32, 240000 x 100ep, --shuffle-adj); `--seed 1` is
# the only difference. Read from the trainer rather than assumed, it moves
# three things together:
#   * torch.manual_seed(seed) before model construction -> the weight init;
#   * np.random.default_rng(seed) -> the batch order;
#   * build_instances(cfg, n_train, seed_start=seed) -> the training set shifts
#     by ONE index, so 239,999 of the 240,000 instances are shared with seed 0.
# The third is the point, not a flaw: every rung of the existing ladder trained
# on seed 0's instances, so the ONLY thing that varies between those rungs is
# init and batch order, and that is precisely what this measures. One honest
# asterisk, finding 23's: `model_max_len` is derived from the realized data, so
# the one swapped instance can change it by a token or two and with it a
# handful of positional-embedding parameters.
#
# `seed` is NOT in sudoku_agg_eval.INSTANCE_KEYS, so a seed-1 checkpoint is
# scored on exactly the same n=1500 instances at seed_start 5e10 as every
# seed-0 cell, and every contrast below -- within a job, across the two jobs,
# and against findings 38 and 40 -- is paired instance by instance.
#
# PRE-REGISTERED, written before either cell was submitted:
#
#  (a) HEADLINE, the H1/H2 discriminator, moderate confidence. Write
#      s_L = |pool(L, seed 1) - pool(L, seed 0)| on the shared n=1500 pool.
#      Prediction: max(s_8, s_10) >= 0.040. Under H1 the per-cell spread must
#      be comparable to the 0.0653 inversion it is supposed to explain, so a
#      large spread is what H1 commits to and what the scan needs quantified.
#      FALSIFIER: s_8 < 0.040 AND s_10 < 0.040 -- the procedure reproduces to
#      better than the inversion, H1 fails on its own terms, and finding 40's
#      non-monotonicity has to be read as real (H2), which would mean every
#      L* on this ladder needs re-reading rung by rung.
#      If instead either s exceeds 0.065, the spread is as large as the depth
#      effects this ladder measures and EVERY single-seed L* statement in
#      STATUS (findings 11/14/22/33/38/40) is uninterpretable without
#      replicates. That is the most consequential outcome of the two and it is
#      not the one I expect.
#
#  (b) ORDERING, near a coin flip even under H1, and informative either way.
#      Write D_1 = pool(L=10, seed 1) - pool(L=8, seed 1).
#      Prediction: D_1 > 0, i.e. the ladder is monotone across {8, 10} at the
#      second seed and seed 0's ordering was the unlucky draw.
#      FALSIFIER: D_1 < 0 with McNemar p < 0.05 -- two independent
#      initializations both put ten layers below eight, which is the strongest
#      evidence for H2 available at this cost.
#      Scoring note, fixed now so it cannot be chosen later: D_1 < 0 together
#      with a large s from (a) does NOT establish H2. It establishes that
#      single-seed rungs are not separable here, and the honest statement
#      would be "L=8 and L=10 are indistinguishable at this budget", not "the
#      ladder is non-monotone".
#
#  (c) MECHANISM, high confidence. Finding 40 found val_nll reproducing the
#      pool inversion (1.4993 / 1.5033 / 1.5059 / 1.5129 at L=12/8/10/6),
#      placing it in the trained weights rather than in free generation.
#      Prediction, two clauses: (i) within seed 1, sign(nll_10 - nll_8) is the
#      opposite of sign(D_1) -- val_nll and the pool agree on the ordering;
#      (ii) the implied sensitivity |D pool| / |D val_nll| stays in [10, 60]
#      solve-rate points per 0.001 nll, the band the seed-0 pairs give (28 and
#      25).
#      FALSIFIER: a seed-1 pair whose val_nll ordering contradicts its pool
#      ordering, which would move the effect into free generation and kill
#      val_nll even as a same-architecture concordance signal here.
#      Worth it because if (ii) holds it licenses a general rule for this
#      ladder: no two cells separated by less than ~0.002 val_nll can be
#      ranked at one seed, which is a cheap pre-filter for the scan.
#      val_nll remains disqualified as a cross-model ranking statistic
#      (findings 4/20/25); this is concordance, not a ranking.
#
#  (d) CONTROLS, low confidence that anything moves. Realized density in the
#      rho=0.60 bins reproduces 0.5836 to within 0.001; the repeat evaluations
#      of the six seed-0 reference cells agree INSTANCE FOR INSTANCE with
#      50607549 / 50527633 / 50527634 / 50099945 (these will be repeats 12
#      onward, all eleven so far identical); `truncated_rate` <= 0.002 and
#      `outcome_accuracy` >= 0.99 for both new cells, so any gap measured here
#      is deduction and not grammar (finding 25's failure mode).
#      FALSIFIER: any disagreement in the repeats, which retires the
#      determinism check that the whole cross-job pairing rests on.
#
# COST, measured from the three cells' own elapsed times (50527634 L=8
# 07:12:09, 50607549 L=10 08:58:51, 50527633 L=12 10:31:58): 50.0 min per
# layer plus ~32 min fixed. L=10 ~8.9 h, L=8 ~7.2 h, plus ~10 min for two
# evals over seven checkpoints. --time 13:00:00 and 10:00:00 are the limits
# the identical seed-0 jobs ran under, ~45% and ~39% headroom. sganguli's
# maximum is 7 d and its mem/core cap is 32 GB (sh_part, 11-01), so 4 cpus and
# 64 GB are inside the limits. The two cells are independent and submitted
# concurrently: ~9 h of wall clock rather than ~16 h.
#
# TWO CODE CHANGES, both about telling the two seeds apart and neither
# touching any number already measured:
#   * sudoku_hparam_sweep tags a non-default seed `_s01`, by the same
#     only-when-non-default rule as `_cos` and `_w05`, so no tag written
#     before today changes and the checkpoint path itself now carries the seed;
#   * sudoku_depth_eval falls back to the parent directory on a duplicate
#     checkpoint basename, the expression sudoku_agg_eval already used, so a
#     future genuine collision cannot silently drop half the rows.
#
# ENVIRONMENT. ${SCRATCH}/spin_gnn_venv is dead (90-day purge), so this is the
# module stack only and calls python3 explicitly: without the venv on PATH,
# bare `python` is the system 2.7. The smoke check turns an environment failure
# into a 30-second job instead of a 9-hour one that dies in the evals.

set -euo pipefail
LAYERS="${1:?usage: sbatch [--job-name=... --time=...] $0 <num_layers>}"
LL=$(printf '%02d' "$LAYERS")
SEED=1
NNODES=48
NTRAIN=240000
EPOCHS=100
# Fixed by hand to the SUBMIT date rather than $(date) so that two cells
# submitted together land in one directory however long each waits in the
# queue. 50607549 stamped itself with its start date and finished the day
# after, which is why its directory says 10-31.
DATESTAMP=2026-11-01

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
cd "${SLURM_SUBMIT_DIR}"

echo "=== smoke check ==="
python3 - <<'PY'
import torch, numpy, numba
import spin_gnn.sudoku_hparam_sweep, spin_gnn.sudoku_depth_eval, spin_gnn.sudoku_agg_eval
print("torch", torch.__version__, "numpy", numpy.__version__, "numba", numba.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
assert torch.cuda.is_available(), "no GPU visible to torch"
PY

OUT="${SCRATCH}/spin_gnn/sudoku_n${NNODES}_leakfree_seed${SEED}_${DATESTAMP}"
mkdir -p "$OUT"
echo "N=${NNODES} layers=${LAYERS} seed=${SEED} n_train=${NTRAIN} epochs=${EPOCHS} -> ${OUT}"

COMMON=(--n-nodes "$NNODES" --degree 5 --n-colors 4 --shuffle-adj
        --d-model 256 --nhead 4 --dim-feedforward 1024
        --n-train-instances "$NTRAIN" --epochs "$EPOCHS"
        --n-val-instances 300 --batch-size 32 --n-per-bin 100)

cell="${OUT}/L${LL}"
echo
echo "=== cell L=${LAYERS} seed=${SEED} ==="
date +'cell start %F %T'
python3 -m spin_gnn.sudoku_hparam_sweep "${COMMON[@]}" \
    --num-layers "$LAYERS" --seed "$SEED" --out-dir "$cell"
date +'cell end   %F %T'

CKPTS=("${cell}/d0256_h04_L${LL}_ff01024_lr1e-03_pre_shuf_s$(printf '%02d' "$SEED")")

# Every seed-0 rung of this ladder, re-evaluated so the contrasts are paired
# inside this job as well as across jobs. Guarded: a missing cell must not fail
# the evals. The OTHER seed-1 cell is deliberately absent -- it is training
# concurrently and does not exist yet; the two are paired afterwards through
# the shared pool, which eleven repeat evaluations have shown to be
# deterministic instance for instance.
DEEP_DIR="${SCRATCH}/spin_gnn/sudoku_n48_leakfree_lstar_2026-10-31"
L10_DIR="${SCRATCH}/spin_gnn/sudoku_n48_leakfree_L10_2026-10-31"
REF_DIR="${SCRATCH}/spin_gnn/sudoku_n48_leakfree_lstar_2026-10-29"
for spec in "${DEEP_DIR}:12" "${L10_DIR}:10" "${DEEP_DIR}:08" \
            "${REF_DIR}:06" "${REF_DIR}:04" "${REF_DIR}:03"; do
    d="${spec%:*}"; L="${spec##*:}"
    ref="${d}/L${L}/d0256_h04_L${L}_ff01024_lr1e-03_pre_shuf"
    if [ -d "$ref" ]; then CKPTS+=("$ref"); else echo "skipping missing reference $ref"; fi
done

echo
echo "=== pre-regs (a)+(d): per-T solve_rate on ONE set of depth bins ==="
python3 -m spin_gnn.sudoku_depth_eval \
    --checkpoint "${CKPTS[@]}" \
    --rho 0.75 0.60 --n-pool 1500 --min-per-bin 60 \
    --out-csv "${OUT}/depth_seed1_L${LL}_${SLURM_JOB_ID}.csv"

echo
echo "=== pre-regs (a)+(b)+(d): the shared n=1500 pool, disjoint from training ==="
python3 -m spin_gnn.sudoku_agg_eval \
    --checkpoint "${CKPTS[@]}" \
    --n-instances 1500 --seed-start 50000000000 \
    --out-csv "${OUT}/sharedpool_n1500_${SLURM_JOB_ID}.csv"

echo
echo "=== summaries ==="
cat "${OUT}/sharedpool_n1500_${SLURM_JOB_ID}.csv"
echo
echo "=== pre-reg (c): final val_nll of the new cell ==="
python3 - "$cell" <<'PY'
import json, sys, glob, os
for h in sorted(glob.glob(os.path.join(sys.argv[1], "*", "history.json"))):
    last = json.load(open(h))[-1]
    print(os.path.basename(os.path.dirname(h)),
          "val_nll %.4f  token_acc %.4f  epoch %d" % (last["nll"], last["accuracy"], last["epoch"]))
PY
