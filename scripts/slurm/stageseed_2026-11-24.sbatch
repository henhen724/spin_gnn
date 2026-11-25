#!/bin/bash
# Stage SEED -- the two d192 seeds that were trained but never evaluated.
#
# WHAT THIS IS. hshunt approved 2 more TRAINING seeds of
# `tied_no_mlp L=16 d192` (270,010 params), the ladder's smallest object and
# its worst rule-breaker (h_rule 0.08106, u 0.4359 at r68). Those two seeds
# ALREADY EXIST: stage P (array 51815311, 2026-11-10) trained all three and
# all three model.pt files survive. So this job TRAINS NOTHING. It runs the
# existing stage LOGIC measurements L and F on the two checkpoints that were
# never evaluated.
#
# WHY THEY WERE NEVER EVALUATED. Not an oversight -- a selection.
# stagepos_cells.successful_runs admits a cell only if its solve rate at
# rho = 0.50 is strictly positive, and at rho = 0.50 seed 0 solved 1 of 400
# while seeds 1 and 2 solved 0 of 400. Seed 0 is in the population by a single
# instance, and is also the best of the three on val token accuracy (0.4715 vs
# 0.4561, 0.4567). The published d192 rule number is therefore the best of
# three seeds, not a typical one.
#
# POPULATION SAFETY. The two cells enter through the new POS_EXTRA hook, which
# appends them with in79=False and in_seven=False AFTER the seven, so r00..r79
# keep their ids and "the seven" keeps its size. With POS_EXTRA unset
# successful_runs returns exactly what it returned before -- checked by hand
# before submission. stagerule_fit.load() drops non-in79 rows, so no existing
# analysis can absorb these two by accident, and the shards land in this
# stage's own directory under the tag `r80-r81`, which does not match the
# `*of10` glob the published fits read.
#
# INSTANCES ARE REUSED, NOT REBUILT. LOGIC_INST points at stage POS's cache, so
# all three seeds are scored on the same 5,600 instances per run, the same
# 14-point grid at 400/bin, the same eval seed start 60,000,000,000.
#
# MEASURE. LF, not F alone. F carries the registered arms (h_rule and
# u = 1 - s_F(0)), but L costs about the same again on two runs, makes the new
# seeds comparable to r68 on the logic arm too, and brings the work in this job
# up to roughly ten minutes rather than three.
#
# TIME. Stage POS did A+B on 7 runs in 8:55 on one sganguli GPU and stage LOGIC
# put L+F on 8 runs at ~15 min, so two runs is ~5 min plus the dry check. One
# hour is heavy margin so a slow node cannot truncate the measurement.
#
# NOTHING IS APPLIED AND NO VERDICT IS PRINTED HERE. The registered predictions
# S1-S5 and the diagnostics D1-D5 are in
# runs/stageseed_2026-11-24/REGISTERED.md, written before this was submitted,
# and are scored afterwards by scripts/stageseed_fit.py.
#
#SBATCH --job-name=stageSeed
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%j.out

set -uo pipefail
cd /home/users/hshunt/spin_gnn

export POS_EXTRA="sudoku_n48_tied_P_2026-11-10/t01_d0192_ff01024_h04_tied_no_mlp_L16_s1,sudoku_n48_tied_P_2026-11-10/t02_d0192_ff01024_h04_tied_no_mlp_L16_s2"

export LOGIC_OUT="/scratch/users/hshunt/spin_gnn/runs/stageseed_2026-11-24"
export LOGIC_INST="/scratch/users/hshunt/spin_gnn/runs/stagepos_2026-11-17/instances"
export LOGIC_RUNS="r80,r81"
export LOGIC_MEASURE="LF"
export LOGIC_BATCH="64"

echo "host=$(hostname)  job=${SLURM_JOB_ID}  runs=${LOGIC_RUNS}"
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
echo "################ r80 and r81, L and F  $(date '+%T') ################"
python3 scripts/stagelogic_score.py
echo "exit=$?"

echo
echo "end $(date '+%F %T')"
