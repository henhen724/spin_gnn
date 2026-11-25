#!/bin/bash
# Stage SIB, continuation -- r91 only, the one run the wall limit truncated.
#
# WHAT HAPPENED. Job 53631778 measured the twelve gate-dropped sibling
# checkpoints and hit TIMEOUT at 02:00:21 against its --time=02:00:00 with r91
# (tied_no_mlp L=16 d448 seed 2) at 8 of 14 densities. r80-r90 are COMPLETE at
# 14 densities each and are not re-measured here. Nothing is deleted: the
# truncated 8 rows stay in
#   runs/stagesib_2026-11-24/F_hazard_r80-r81-...-r91.csv
# and scripts/stagesib_fit.py drops r91 from that shard and takes it only from
# this job's own F_hazard_r91.csv, so the partial rows can never be pooled with
# the complete ones.
#
# WHY TWO HOURS WAS WRONG. The estimate came from stage SEED, where two d192
# checkpoints took 3:39 for L+F together. It was wrong by ~5x and for a reason
# worth recording: the printed per-run time at rho=0.33 was 15s for the L=4
# cells, 49s for d192 and 271s for d448 -- and those are only the visible pass.
# Total wall was ~10 min per run. The two d448 siblings are the slowest objects
# in the set because they are the WORST models in it (val acc 0.0412 and 0.0308
# against ~0.456 for every other sibling): a model that cannot colour anything
# emits long traces that never reach SOLVED, so every rollout runs to the cap.
# BAD CHECKPOINTS ARE EXPENSIVE TO EVALUATE, and an eval-only job sized from
# good checkpoints will under-budget. Four hours is ~6x r90's observed cost.
#
# No training. No new approval needed: hshunt approved all 10 siblings, and r91
# is one of them -- this finishes an approved measurement the clock cut short.
#
#SBATCH --job-name=stageSib91
#SBATCH --partition=sganguli
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%j.out

set -uo pipefail
cd /home/users/hshunt/spin_gnn

# IDENTICAL to 53631778, and it must stay identical: POS_EXTRA is positional, so
# r91 only means "tiedwidth_S t05 d0448 seed 2" while all twelve entries are
# present in this order. Only LOGIC_RUNS differs.
export POS_EXTRA="sudoku_n48_tied_P_2026-11-10/t01_d0192_ff01024_h04_tied_no_mlp_L16_s1,sudoku_n48_tied_P_2026-11-10/t02_d0192_ff01024_h04_tied_no_mlp_L16_s2,sudoku_n48_minarch_2026-11-04/t22_L04_h04_no_mlp_s1,sudoku_n48_minarch_2026-11-04/t23_L04_h04_no_mlp_s2,sudoku_n48_minarch_2026-11-04/t15_L04_h04_rf_mlp_s0,sudoku_n48_minarch_2026-11-04/t17_L04_h04_rf_mlp_s2,sudoku_n48_minarch_G_2026-11-05/t00_d0128_ff01024_h04_no_mlp_L06_ep100_s0,sudoku_n48_minarch_G_2026-11-05/t02_d0128_ff01024_h04_no_mlp_L06_ep100_s1,sudoku_n48_minarch_H_2026-11-05/t00_d0128_ff01024_h04_no_mlp_L12_s0,sudoku_n48_minarch_H_2026-11-05/t02_d0128_ff01024_h04_no_mlp_L12_s1,sudoku_n48_tiedwidth_S_2026-11-10/t04_d0448_ff01024_h04_tied_no_mlp_L16_s1,sudoku_n48_tiedwidth_S_2026-11-10/t05_d0448_ff01024_h04_tied_no_mlp_L16_s2"

export LOGIC_OUT="/scratch/users/hshunt/spin_gnn/runs/stagesib_2026-11-24"
export LOGIC_INST="/scratch/users/hshunt/spin_gnn/runs/stagepos_2026-11-17/instances"
export LOGIC_RUNS="r91"
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
echo "################ r91, L and F  $(date '+%T') ################"
python3 scripts/stagelogic_score.py
echo "exit=$?"

echo
echo "end $(date '+%F %T')"
