#!/bin/bash
#
# STAGE LOGIC: TEACHER-FORCED *LOGICAL* ACCURACY, AND WHETHER A FREE-GENERATION
# FIRST ERROR NAMES AN UNFORCED NODE. NOTHING IS TRAINED.
#
# THE QUESTION. From stage POS's measurement A, hshunt's fit makes the solve
# rate a clean power of the teacher-forced reasoning-trace token accuracy p,
# S = C p^k with C ~ 1 for rho >= 0.65 and k ~ 0.73 n (free power fit n^1.07,
# so LINEAR in n). The coefficient is 0.73 and not 1.0 because many
# teacher-forced node errors are harmless: DD deductions inside a round are
# order-free, so naming a different FORCED node than the trace writes next is
# scored wrong by A although it is a legal DD move. If every logical error were
# fatal and only logical errors counted, k would have to be n exactly with
# prefactor 1. The gap 0.73 -> 1 measures how much of A's error budget is
# bookkeeping rather than reasoning.
#
# WHAT RUNS. Measurement L re-scores the SAME teacher-forced forward pass as A
# against the set of tokens DD PERMITS: at a node position, the forced set F of
# the current round (hints plus completed earlier rounds, same-round siblings
# not counted as evidence) minus what this round already emitted; at a colour
# position, the DD colour of the node in the true prefix, unchanged from A; at
# a boundary, BEGIN_ROUND or the terminator. Every node-position argmax is also
# classified in_F / premature / recolor / nonnode -- "premature" (an uncoloured
# node outside F) is a guard failure and is the thing worth counting.
# Measurement F replays the greedy rollout and asks, at each instance's FIRST
# error, whether the named node was in F computed from the model's OWN prefix.
# B did not save which node was named, so this needs the replay; it is the same
# decode on the same cached instances and the replayed solve rate must equal
# B's to the digit, which is how the replay is verified rather than trusted.
#
# THE EVAL IS NOT REBUILT. LOGIC_INST points at stage POS's instance cache, so
# an L curve and a stage POS hazard curve are the same 5,600 instances per run
# and the 14-point grid, 400 per bin, seed start 60,000,000,000 are unchanged.
#
# POPULATION 79 + 1. Every (cell, seed) with nonzero solve rate at rho = 0.50,
# which stagepos_cells REFUSES to run if it is not exactly 79, plus
# full L=2 d256 s0 flagged in79=False. Fits use the 79; the L2 run is measured
# and plotted, never pooled.
#
# PREDICTIONS AND FALSIFIERS ARE REGISTERED IN
# runs/stagelogic_2026-11-17/REGISTERED.md and in the scorer's docstring, both
# committed before this was submitted: L1 the exponent against n lands in
# [0.85, 1.20] with prefactor in [0.5, 2.0] (falsified outside [0.7, 1.4]);
# L2 p_logic >= p_strict everywhere, an identity whose violation is a bug;
# L3 the median k_logic/k_strict lands in [1.1, 1.6]; L4 the median fraction of
# first errors naming an unforced node exceeds 0.5 (falsified below); L5 the
# premature-rate confound, reported either way.
#
# SHARDING IS ROUND ROBIN. stagepos_cells sorts by depth then width, so
# contiguous task 9 would draw eight L=16 cells and set the array's wall clock
# on its own. LOGIC_SHARD="i/10" takes every tenth run: 8 runs per task, each
# spanning the depth range.
#
# DRY CHECK IN EVERY TASK, seconds of CPU, because four things here can be
# wrong without crashing: the forced-set identity that lets L read F off the
# trace instead of recomputing DD, the position schema, the guarantee that the
# TRUE node is always in F (without which the logical rule could mark a correct
# argmax wrong and p_logic >= p_strict would stop being an identity), and the
# tracker's ability to certify a ground-truth trace as fully forced. It passed
# on CPU before submission on 141 rounds across rho = 0.85, 0.60, 0.40.
#
# TIME. Stage POS did A+B on 7 runs in 8:55 on one sganguli GPU; L+F is the
# same two passes plus a few seconds of CPU forced-set work per run, so 8 runs
# is ~15 min. 3 hours is heavy margin so a slow node cannot truncate a shard,
# and %5 leaves the partition usable by the rest of the group.
#
# NOTHING IS APPLIED AND NO VERDICT IS PRINTED HERE; scoring is
# scripts/stagelogic_fit.py, run afterwards.
#
#SBATCH --job-name=stageLogic
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-9%5
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%A_%a.out

set -uo pipefail
cd /home/users/hshunt/spin_gnn

export LOGIC_OUT="/scratch/users/hshunt/spin_gnn/runs/stagelogic_2026-11-17"
export LOGIC_INST="/scratch/users/hshunt/spin_gnn/runs/stagepos_2026-11-17/instances"
export LOGIC_RUNS="all"
export LOGIC_SHARD="${SLURM_ARRAY_TASK_ID}/10"
export LOGIC_MEASURE="LF"
export LOGIC_BATCH="64"

echo "host=$(hostname)  job=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}  shard=${LOGIC_SHARD}"
echo "start $(date '+%F %T')"
git rev-parse --short HEAD

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo
echo "################ DRY CHECK  $(date '+%T') ################"
python3 scripts/_stagelogic_dry_check.py
rc=$?
echo "exit=${rc}"
if [ "${rc}" -ne 0 ]; then
  echo "DRY CHECK FAILED -- not measuring anything"
  exit "${rc}"
fi

echo
echo "################ SHARD ${LOGIC_SHARD}, L and F  $(date '+%T') ################"
python3 scripts/stagelogic_score.py
echo "exit=$?"

echo
echo "end $(date '+%F %T')"
