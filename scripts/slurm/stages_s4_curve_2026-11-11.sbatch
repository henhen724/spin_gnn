#!/bin/bash
#SBATCH --job-name=stagesS4
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stagesS4_%j.out
#
# S4, exactly as stages_score.py registered it and printed the command for: do
# the tied cells land on the N=48 accuracy-to-solve curve within the LOO bar
# 0.0423?
#
# WHY THIS IS THE FIRST THING RUN AFTER THE d320 NUMBERS LANDED. Tied T=16 d320
# reaches val token accuracy 0.4830 against the untied twin J2's 0.4940 -- a
# 0.011 gap -- but rho_50 0.757 against J2's 0.580, which is a large gap in
# solve terms. Those two facts are only consistent if the accuracy-to-solve
# curve is steep here, and S4 is the test of whether the tied cells sit ON that
# curve at all:
#   ON the curve  -> tying costs ACCURACY and nothing else; the whole rho_50 gap
#                    is what that 0.011 buys at this part of the curve, and the
#                    object to improve is the optimization, not the architecture.
#   OFF the curve -> tying changes the accuracy-to-solve RELATION, which the
#                    registration calls the headline over S0, because it is a
#                    different and more interesting claim than any level.
# Costs no GPU, needs no new cell, and is registered -- so it runs before
# anything is submitted on the strength of the d320 read.
#
# N=48 only (guard 1): this curve is an N=48 object and is never applied to N=96.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

SCR=/scratch/users/hshunt/spin_gnn

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"

python3 -u -m scripts.acc_vs_solve --scan-dir \
    "$SCR/sudoku_n48_minarch_2026-11-04" \
    "$SCR/sudoku_n48_minarch_L6_2026-11-04" \
    "$SCR/sudoku_n48_minarch_C_2026-11-04" \
    "$SCR/sudoku_n48_minarch_D_2026-11-04" \
    "$SCR/sudoku_n48_minarch_E_2026-11-04" \
    "$SCR/sudoku_n48_minarch_G_2026-11-05" \
    "$SCR/sudoku_n48_minarch_H_2026-11-05" \
    "$SCR/sudoku_n48_minarch_I_2026-11-05" \
    "$SCR/sudoku_n48_minarch_J_2026-11-05" \
    "$SCR/sudoku_n48_minarch_K_2026-11-09" \
    "$SCR/sudoku_n48_kbrep_N_2026-11-10" \
    "$SCR/sudoku_n48_sizeshape_O_2026-11-10" \
    "$SCR/sudoku_n48_tied_P_2026-11-10" \
    "$SCR/sudoku_n48_tiedwidth_S_2026-11-10"
echo "curve rc=$?"

echo
echo "end $(date '+%F %T')"
