#!/bin/bash
#SBATCH --job-name=width112
#SBATCH --output=/home/users/hshunt/spin_gnn/width112_%A_%a.out
#SBATCH --partition=sganguli
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#
# STAGE WIDTH -- finding 111 showed the per-step deduction penalty survives
# matching on DD depth.  What mediates it?  INFERENCE ONLY: nothing is trained,
# no checkpoint is written, no closed knob is touched, no GPU is requested.
#
# WHY NOW.  Stage SCALE falsified C2: at matched depth T, N=96's per-step
# `reason_node` accuracy plateaus at 0.7604 against N=48's 0.9129.  That ruled
# out the accumulation account but did not say what about a bigger graph costs
# the accuracy.  There is a candidate that is NOT graph size as such:
#
#   n_blank = N(1 - rho) cells are spread over T rounds, so at matched (T, rho)
#   doubling N doubles the cells forced PER ROUND.
#
# and `sudoku_trace.py:83` emits each round SORTED by cell index, so naming the
# j-th cell of a round means naming the smallest forced cell not among the j-1
# already named.  If the gap closes once round width is matched, finding 111's
# penalty is a round-width effect and N was a proxy.  If it survives, the
# penalty really is about locating a neighbourhood in a larger graph.  Both
# matter for the fast-slow regime, where T AND round width grow with N.
#
# SAME INSTANCES AS STAGE SCALE.  This probe imports stage SCALE's
# `instance_cache`, `run_path` and `section_labels` rather than copying them,
# and reuses the cached 200-per-bin puzzles.  Stage SCALE produced published
# numbers and is NOT edited here.  Because the puzzles and checkpoints are
# identical, DW3 below is a genuine cross-stage consistency check and is free.
#
# SCOPE, AND THE JUDGEMENT IS MINE AND IS FLAGGED RATHER THAN HIDDEN.  As with
# HEADS, SECTION and SCALE: trains nothing, touches no closed knob, orthogonal
# to the four registered questions waiting on hshunt, reuses existing
# checkpoints.  Teacher-forced accuracy is NOT solve rate and nothing here
# clears or moves the 0.740 rho_50 gate.
#
# ---------------------------------------------------------------------------
# WHAT THE SMOKE SHOWED, DISCLOSED IN FULL -- INCLUDING THAT IT REVERSED THE
# PREDICTION I WAS ABOUT TO REGISTER.
#
# I intended to register "accuracy falls with within-round position j", on the
# reasoning that later positions carry more bookkeeping.  One full bin
# (`L12_s0`, rho 0.45, 200 instances, not a 4-instance smoke) says that is
# FALSE, and that the live variable is round WIDTH instead:
#
#   N=96, rho 0.45, L12_s0   (overall 0.8283, mean round width 4.43)
#     by position j:  j0 0.808  j1 0.805  j2 0.770  j3 0.782  j4 0.797
#                     j5 0.793  j6 0.826  j7 0.853          <- FLAT, if anything rising
#     by width w:     w1 0.874  w2 0.838  w3 0.797  w4 0.754  w5 0.727
#                     w6 0.703  w7 0.689  w8 0.695          <- MONOTONE DECLINE
#
#   N=48, rho 0.45, L12_s0   (overall 0.9951, mean round width 2.99)
#     by position j:  j0 0.994  j1 0.993  j2 0.995  j3 0.994  j4 1.000 ...
#     by width w:     w1 0.991  w2 0.991  w3 0.992  w4 0.993  w5 1.000 ...
#                                                            <- FLAT IN BOTH
#
#   w_bad 0 on every bin.
#
# So W2/W3/W5's directions are disclosed and the bands are set around what was
# seen.  The mechanism this suggests: to emit the smallest forced cell the model
# must determine WHICH cells are forced; a round of width w has w of them, so a
# wider round is a bigger set to determine, uniformly at every position in it.
# Position j does not matter because the already-named cells are visible in
# context, so the task at each position is the same.
#
# W4 IS THE LIVE ONE and the smoke speaks against the tidy answer: at rho 0.45
# the N gap at w=1 is 0.117 and at w=8 is 0.305, i.e. matching width does NOT
# close it -- N=96 is instead MORE SENSITIVE to width.  One cell at one density
# is not three pairs at five, and L12 at N=48 is at ceiling (0.995), so W5 is
# scored on the L08 cells which sit at 0.93-0.94 and can actually show
# structure.
#
# ---------------------------------------------------------------------------
# REGISTERED PREDICTIONS.  Scored by scripts/stagewidth_fit.py, committed in
# the same commit as this file and BEFORE any cell lands.
#
# W1  MECHANISM CHECK (direction mechanical, magnitude disclosed).  At rho 0.60
#     mean round width is 1.3-1.8x larger at N=96 on all three matched pairs.
#     FALSIFIER: a ratio outside [1.1, 2.5] on any pair.
#
# W2  WIDTH IS A REAL AXIS AT N=96 (direction disclosed).  Pooled over the three
#     pairs and all five densities, N=96 accuracy falls across w = 1..6 by at
#     least 0.08, with at most one inversion.
#     FALSIFIER: a drop under 0.03, or more than one inversion.
#
# W3  POSITION IS NOT (direction disclosed, and this REPLACES the prediction I
#     had intended).  At N=96 the accuracy spread across w = 1..6 is at least
#     2x the spread across j = 0..5.
#     FALSIFIER: the position spread equals or exceeds the width spread.
#
# W4  THE LIVE ONE -- DOES MATCHED WIDTH CLOSE THE N GAP?  Pooled over all
#     densities, at every width band with >= 200 tokens at both N, the median
#     N=48-minus-N=96 gap EXCEEDS 0.05, i.e. round width does NOT mediate
#     finding 111 and the size penalty is not reducible to it.
#     FALSIFIER: a median gap below 0.05, which would mean width fully explains
#     111 and `N` was a proxy for round width all along.
#
# W5  THE WIDTH SENSITIVITY IS SPECIFIC TO THE LARGER GRAPH.  At N=48, scored on
#     the L08_s0 and L08_s1 cells ONLY -- L12 sits at 0.9995 and a ceiling
#     cannot show structure -- the accuracy spread across w = 1..6 is at most
#     0.03.
#     FALSIFIER: a spread above 0.08.
#
# DIAGNOSTICS, and any failure VOIDS the W above it:
#   DW1  the round walk consumes the trace exactly (w_bad = 0 on every bin).
#   DW2  by_w, by_j and by_tj each total exactly n_tokens.
#   DW3  the overall `reason_node` accuracy agrees with stage SCALE's to 1e-6 --
#        same checkpoints, same cached instances, so any disagreement means one
#        of the two stages mislabelled its tokens.
#
# COST.  9 runs x 5 densities, instances already cached by stage SCALE, 4 tasks,
# CPU only.  Measured on the smoke: ~2 min per density for an N=96 run, so the
# worst task (three N=96 runs) is ~30 min.  2 h is the ceiling.

set -e
cd /home/users/hshunt/spin_gnn

module purge
module load devel math python/3.12.1
module load py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
module load py-numba/0.60.0_py312 py-scipy/1.12.0_py312

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

python3 scripts/stagewidth_probe.py \
  --task "${SLURM_ARRAY_TASK_ID}" \
  --ntasks "${SLURM_ARRAY_TASK_COUNT}"

echo "task ${SLURM_ARRAY_TASK_ID} done"
