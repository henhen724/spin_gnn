#!/bin/bash
#SBATCH --job-name=scale111
#SBATCH --output=/home/users/hshunt/spin_gnn/scale111_%A_%a.out
#SBATCH --partition=sganguli
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#
# STAGE SCALE -- does the cost of a DEDUCTION token grow with N, or only the
# cost of a GRAPH token?  INFERENCE ONLY: nothing is trained, no checkpoint is
# written, no closed knob is touched, no GPU is requested.
#
# WHY NOW.  Finding 109 measured that the `val_nll` floor RISES with N (1.371 /
# 1.625 / 1.861 at N=48/96/192) and attributed the whole rise to the graph
# section, f_graph = (5/16) log N, on an analytic argument: the graph slots name
# cells from an alphabet that doubles with N, the reasoning slots do not have to
# pay for that twice.  Stage SECTION (finding 110) built the instrument that can
# check this -- a per-section decomposition of `val_nll` -- but ran it at ONE N,
# so 109's split has never been MEASURED.  It matters immediately: the N=192
# ladder (job 53062385) is mid-training and its loss is only readable against a
# floor whose composition is known.  If the reasoning section's cost is
# N-invariant, the ladder's 2.19 is mostly graph tokens and the comparison
# across N is a category error; if it is not, the N-penalty is in the deduction
# itself and that is a far more interesting result than the floor.
#
# BOTH ANSWERS ARE LIVE and I am not predicting the mechanism, only the sign.
# Two accounts, both consistent with everything measured so far:
#   ACCUMULATION -- a longer trace means more rounds to get through before the
#     late cells, so per-step difficulty is flat and the apparent N-penalty is
#     really a DEPTH penalty.  This is what 108's matched-depth comparison
#     hinted at (a_eff 0.9940 at N=48 vs 0.9122/0.9628 at N=96 at T=12.0, with
#     the SEED term 0.0506 EXCEEDING the N-penalty 0.0312).
#   PER-STEP -- each individual deduction is harder at N=96 because the
#     neighbourhood has to be located in a bigger graph the model is holding in
#     context.  Then depth does NOT absorb the gap.
# C2 below is the discriminator and it is the only prediction here I cannot
# already guess.
#
# WHY ACCURACY AND NOT NLL IS THE CROSS-N STATISTIC.  `sudoku_trace.py:83`
# appends `sorted(...)` per round, so within-round emission order is sorted by
# node index and, given perfect deduction, the next cell is the smallest
# not-yet-emitted forced cell.  `reason_node`'s entropy floor is therefore ZERO
# at every N -- there is no log-N ordering entropy to subtract -- which makes
# teacher-forced `reason_node` ACCURACY comparable across N as it stands.  NLL
# is not: the cell alphabet doubles 48 -> 96, so chance alone costs log 2 in
# nats.  (Finding 94's "the within-round order is arbitrary" means arbitrary as
# a GENERATOR CHOICE; the choice made was `sorted`.  I had this wrong earlier
# this session and corrected it before designing the stage.)
#
# SCOPE, AND THE JUDGEMENT IS MINE AND IS FLAGGED RATHER THAN HIDDEN.  Like
# stage HEADS and stage SECTION I judged this inside standing scope: it trains
# nothing, touches no closed knob, is orthogonal to the four registered
# questions waiting on hshunt, and reuses checkpoints that already exist.
# Teacher-forced accuracy is NOT solve rate and nothing here clears or moves the
# 0.740 rho_50 gate; it prices a loss, it does not rank an architecture.  The
# same probe runs unchanged on N=192 when the ladder lands, because N and the
# instance family both come from each checkpoint's own config.json.
#
# ---------------------------------------------------------------------------
# WHAT THE SMOKE ALREADY SHOWED, DISCLOSED IN FULL BEFORE REGISTERING.
# Stage HEADS disclosed H1's sign from a 3-instance smoke and stage SECTION did
# the same; this is that move again, and the bands below are set AROUND what was
# seen, not against it.  4 instances per bin, so these are indicative only.
#
#   N=48, t08 L12 s0:  rho 0.90  total nll 1.4634  reason_node acc 1.0000  meanT 1.25
#                      rho 0.60  total nll 1.5027  reason_node acc 1.0000  meanT 4.50
#   N=96, t00 L12 s0:  rho 0.90  total nll 1.7079  reason_node acc 1.0000  meanT 1.75
#                      rho 0.60  total nll 1.7608  reason_node acc 0.9038  meanT 6.50
#                      rho 0.35  total nll 1.7555  reason_node acc 0.7778  meanT 20.00
#
#   N=96, rho 0.60, per section:
#     graph_nbr_novel   4.1064 nats   acc 0.0187
#     graph_nbr_copy    0.9404 nats   acc 0.5812
#     round_mark        0.2681 nats   acc 0.8846
#     reason_node       0.2887 nats   acc 0.9038
#     reason_color      0.0000 nats   acc 1.0000
#
#   by_T, N=96: rho 0.60 {"4":[33,39],"5":[36,39],"6":[36,39],"11":[36,39]}
#               rho 0.35 {"17":[98,126],"20":[47,63],"26":[51,63]}
#
#   Diagnostics on the smoke: d1_bad 0, d3_bad 0, n_skip 0 on EVERY bin --
#   the label walk, the analytic length and max_len all already hold at N=96.
#
# So C1's DIRECTION is near-certain (1.0000 -> 0.9038 is ~0.096 on ~154 tokens,
# se ~ 0.024) and only its magnitude is open; C3 and C4 are likewise disclosed
# (SECTION's N=48 graph_nbr_novel median was 3.4269, against 4.1064 here, a rise
# of 0.68 against log 2 = 0.6931).  C2 IS UNTOUCHED BY THE SMOKE: N=48 reached
# T 1.25/4.50 and N=96 reached 1.75/6.50/20.00, so there is NO overlapping depth
# band at all and the `by_T` column exists precisely to build one at full size.
# C6 is new this registration and is the sharp structural one.
#
# ---------------------------------------------------------------------------
# REGISTERED PREDICTIONS.  Scored by scripts/stagescale_fit.py, which is
# committed in the same commit as this file and BEFORE any cell lands.
#
# C1  MATCHED DENSITY (direction disclosed, magnitude live).  At each of the
#     three cells that exist at both N (L=8 s0, L=12 s0, L=8 s1 -- architecture,
#     width, recipe, 240k tokens and seed all fixed, only N varying) and at
#     rho = 0.60, teacher-forced `reason_node` accuracy is LOWER at N=96, with
#     the median drop over the three pairs in [0.03, 0.20].
#     FALSIFIER: a drop below 0.01 or above 0.30 on 2 of 3 pairs.
#
# C2  MATCHED DEPTH -- THE LIVE ONE.  Pooling the matched pairs and every
#     density and binning `reason_node` tokens by the instance's DD depth T, at
#     the deepest band where both N have >= 200 tokens, N=96's accuracy is
#     within 0.05 of N=48's.  That is the ACCUMULATION account: depth, not N,
#     explains C1's drop.
#     FALSIFIER: a gap above 0.15 at that band, which is the PER-STEP account --
#     N carries a penalty beyond depth, and 109's attribution of the whole floor
#     rise to the graph section is wrong.
#     If no band clears 200 tokens at both N, the scorer reports C2 UNSCOREABLE
#     rather than reading a verdict off a thin band.
#
# C3  DEDUCTION SHARE (disclosed near-certain).  (reason_node + reason_color)
#     NLL as a fraction of total NLL is lower at N=96 on all three pairs.
#     FALSIFIER: higher on 2 or more.
#
# C4  THE GRAPH SLOT PAYS FOR THE ALPHABET.  `graph_nbr_novel` NLL rises by
#     0.40-0.75 nats from N=48 to N=96 on every matched pair; log 2 = 0.6931 is
#     the uniform-alphabet increment.
#     FALSIFIER: a change outside [0.25, 0.95] on any pair.
#
# C6  AND THE COPY SLOT DOES NOT.  `graph_nbr_copy` NLL changes by less than
#     0.15 nats in EITHER direction on every matched pair.  Reason: every
#     undirected edge is written twice, so a copy slot's cost is which of the
#     already-named neighbours comes next -- within-row ORDERING entropy,
#     bounded by log(degree!)/degree and independent of N -- not cell identity.
#     C4 and C6 together are the mechanism behind 109's f_graph = (5/16) log N:
#     5 of the 16 slots per row are novel and pay log N, the rest do not.
#     FALSIFIER: a change above 0.35 on any pair.
#
# C5  THE COLOUR STEP IS LOCAL.  `reason_color` accuracy >= 0.99 at N=96 on
#     every matched cell at rho >= 0.60 -- picking the colour of an already
#     named cell reads q=4 neighbours and cannot care about N.
#     FALSIFIER: below 0.95 anywhere.
#
# DIAGNOSTICS, and any failure VOIDS the C above it rather than being reported
# beside it:
#   D1  the section label walk agrees with encode_instance on every instance.
#   D2  the per-section NLL sums reconstruct the published total to 1e-6.
#   D3  every instance has length exactly N(degree+3) + T + 4, at BOTH N.
#
# COST.  9 runs x 5 densities x 200 instances, 4 tasks, CPU only.  Priced off
# the smoke: ~0.15 s/instance at N=48 and ~0.45 s at N=96 including the forward
# pass, plus one-off instance generation for 10 cached bins shared across tasks.
# Worst task (three N=96 runs) ~25 min; 2 h is the ceiling with headroom for a
# cold NFS read of the checkpoints.

set -e
cd /home/users/hshunt/spin_gnn

module purge
module load devel math python/3.12.1
module load py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
module load py-numba/0.60.0_py312 py-scipy/1.12.0_py312

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python3 scripts/stagescale_probe.py \
  --task "${SLURM_ARRAY_TASK_ID}" \
  --ntasks "${SLURM_ARRAY_TASK_COUNT}"

echo "task ${SLURM_ARRAY_TASK_ID} done"
