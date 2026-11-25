#!/bin/bash
# Stage SIB -- every gate-dropped sibling seed of every single-seed cell.
#
# WHAT THIS IS. Finding 135 (read-only, commit 4370f01) audited the six cells
# that are single-seed inside the 79 and found that NONE of them is single-seed
# because of a training budget: all three seeds were trained, all have a
# model.pt on disk, all were scanned at rho = 0.50, and the siblings were
# dropped because stagepos_cells.successful_runs() admits a cell only on
#
#     if len(at50) == 1 and float(at50[0]["solve_rate"]) > 0:
#
# The admitted seed clears that gate by between 1 and 14 solved instances of
# 400 (r68 1, r23 4, r08 5, r44 11, r09 13, r78 14) while every dropped sibling
# reads exactly 0/400. Asked whether to score the 10 remaining siblings on the
# rule arm -- no training, ~20 GPU-minutes -- hshunt answered "Yes, all 10".
#
# SO THIS JOB TRAINS NOTHING. It runs the existing stage LOGIC measurements L
# and F on checkpoints that have been on $SCRATCH since 2026-11-04, 11-05 and
# 11-10, against stage POS's CACHED instances, so every sibling is scored on the
# same 5,600 instances as the admitted seed it is compared with.
#
# WHY THIS IS NOT JUST 134 AGAIN. Stage SEED seeded d192, a cell that was
# ALREADY a rule-breaker (h_rule 0.08106), so it could not separate "the gate
# hides bad seeds" from "d192 happens to be unstable". Four of the six cells
# here are published as rule-INTACT -- r44 0.00004, r78 0/26,949, r08 0/24,955,
# r09 0.00008, i.e. 125x to infinitely under the 0.01 bar. If their hidden
# siblings break the bar, then "the rule is a property of the architecture"
# fails on rows that are currently published as clean. That is S2 and it is the
# reason to spend the twenty minutes.
#
# TWELVE CHECKPOINTS, NOT TEN. r80 and r81 are the two d192 siblings stage SEED
# already measured, re-run deliberately. (1) It removes a run-id collision: with
# only the ten new entries, THIS stage's r80/r81 would be no_mlp L=4 seeds 1 and
# 2, so the same ids would name different checkpoints in two published stages.
# (2) D5 then becomes a real end-to-end check that this pipeline reproduces what
# 134 published (0.5656 = 3,694/6,531 and 0.5641 = 3,805/6,745, u 0.2882 and
# 0.2655) instead of assuming it. Four GPU-minutes for both.
#
# POPULATION SAFETY. The twelve enter through POS_EXTRA, which appends them with
# in79=False and in_seven=False AFTER the seven, so r00..r79 keep their ids and
# "the seven" keeps its size. With POS_EXTRA unset successful_runs() returns
# exactly what it returned before. stagerule_fit.load() drops non-in79 rows, and
# the shards land in this stage's own directory tagged r80-r91, which the
# published `*of10` glob cannot match -- so no published fit can absorb these
# rows even with POS_EXTRA exported. Nothing here is re-admitted to the 79.
#
# TIME. Stage SEED did 2 runs of L+F in 3:39 including the dry check. Twelve
# runs is ~22 min at that rate; d448 is the widest object here and the L=4 cells
# the shallowest, so the mix should average out. Two hours is heavy margin so a
# slow node cannot truncate the measurement half way through the twelve.
#
# NO VERDICT IS PRINTED HERE. The registered predictions S1-S5 and the
# diagnostics D1-D6 are in runs/stagesib_2026-11-24/REGISTERED.md, written and
# committed BEFORE this was submitted, and are scored afterwards by
# scripts/stagesib_fit.py.
#
#SBATCH --job-name=stageSib
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%j.out

set -uo pipefail
cd /home/users/hshunt/spin_gnn

# Order matters: POS_EXTRA is appended in this order, so these take r80..r91.
# d192 first, so r80/r81 mean the same checkpoints here as in stage SEED.
export POS_EXTRA="sudoku_n48_tied_P_2026-11-10/t01_d0192_ff01024_h04_tied_no_mlp_L16_s1,sudoku_n48_tied_P_2026-11-10/t02_d0192_ff01024_h04_tied_no_mlp_L16_s2,sudoku_n48_minarch_2026-11-04/t22_L04_h04_no_mlp_s1,sudoku_n48_minarch_2026-11-04/t23_L04_h04_no_mlp_s2,sudoku_n48_minarch_2026-11-04/t15_L04_h04_rf_mlp_s0,sudoku_n48_minarch_2026-11-04/t17_L04_h04_rf_mlp_s2,sudoku_n48_minarch_G_2026-11-05/t00_d0128_ff01024_h04_no_mlp_L06_ep100_s0,sudoku_n48_minarch_G_2026-11-05/t02_d0128_ff01024_h04_no_mlp_L06_ep100_s1,sudoku_n48_minarch_H_2026-11-05/t00_d0128_ff01024_h04_no_mlp_L12_s0,sudoku_n48_minarch_H_2026-11-05/t02_d0128_ff01024_h04_no_mlp_L12_s1,sudoku_n48_tiedwidth_S_2026-11-10/t04_d0448_ff01024_h04_tied_no_mlp_L16_s1,sudoku_n48_tiedwidth_S_2026-11-10/t05_d0448_ff01024_h04_tied_no_mlp_L16_s2"

export LOGIC_OUT="/scratch/users/hshunt/spin_gnn/runs/stagesib_2026-11-24"
export LOGIC_INST="/scratch/users/hshunt/spin_gnn/runs/stagepos_2026-11-17/instances"
export LOGIC_RUNS="r80,r81,r82,r83,r84,r85,r86,r87,r88,r89,r90,r91"
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

# The dry check imports successful_runs(), so a POS_EXTRA suffix that resolves
# to zero or to more than one cell raises SystemExit here, in seconds, instead
# of after the first checkpoint has been scored.
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
echo "################ r80-r91, L and F  $(date '+%T') ################"
python3 scripts/stagelogic_score.py
echo "exit=$?"

echo
echo "end $(date '+%F %T')"
