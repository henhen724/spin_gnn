#!/bin/bash
#SBATCH --job-name=stageuCurve
#SBATCH --partition=sganguli
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/hshunt/spin_gnn/stageuCurve_%j.out
#
# U4, which the scorer only printed the command for. This is the interpretive
# crux of stage U and not a footnote, because of what the per-seed numbers show:
#
#   k=2  seeds  0.752 / 0.753 / 0.603      train NLL 1.5832 / 1.5836 / 1.5371
#   k=4  seeds  0.755 / 0.762 / 0.579      train NLL 1.5797 / 1.5816 / 1.5306
#   k=1  seeds  0.754 / 0.757 / 0.765      train NLL 1.5851 / 1.5915 / 1.5864
#
# ONE seed in each k > 1 cell landed somewhere k=1 never went in three seeds: a
# distinctly lower train NLL (about 0.05 below, five times U1's flat band) with a
# much lower rho_50. The bound-aware MEDIAN the gate is read from hides it
# completely, which is why U0 reads as a flat negative.
#
# HYPOTHESIS: nothing architectural happened. The good seeds simply reached
# J2-level token accuracy (0.4929 and 0.4948 against J2's 0.4940) and collected
# J2-level rho_50 (0.603 and 0.579 against J2's 0.580) because that is what the
# N=48 accuracy-to-solve curve pays at that accuracy. If their residuals sit
# INSIDE the 0.0423 LOO bar, the binding constraint is which basin the optimizer
# finds -- not the architecture, and not the role count -- which is U1's "recipe,
# not roles" reading sharpened to a mechanism.
#
# The alternative, and it would be the bigger result: a good seed OFF the curve
# would mean cycling k blocks changes the accuracy-to-solve RELATION rather than
# the level, and U4 outranks U0 by registration if that happens.
#
# Read-only. No commit, no deck, no STATUS.md edit.

set -u
cd /home/users/hshunt/spin_gnn || exit 1

module purge
module load devel math viz python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-pandas/2.2.1_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
export PYTHONUNBUFFERED=1

S=/scratch/users/hshunt/spin_gnn

echo "host=$(hostname)  job=$SLURM_JOB_ID"
echo "start $(date '+%F %T')"
echo

python3 -u -m scripts.acc_vs_solve --scan-dir \
    "$S/sudoku_n48_minarch_2026-11-04" \
    "$S/sudoku_n48_minarch_L6_2026-11-04" \
    "$S/sudoku_n48_minarch_C_2026-11-04" \
    "$S/sudoku_n48_minarch_D_2026-11-04" \
    "$S/sudoku_n48_minarch_E_2026-11-04" \
    "$S/sudoku_n48_minarch_G_2026-11-05" \
    "$S/sudoku_n48_minarch_H_2026-11-05" \
    "$S/sudoku_n48_minarch_I_2026-11-05" \
    "$S/sudoku_n48_minarch_J_2026-11-05" \
    "$S/sudoku_n48_minarch_K_2026-11-09" \
    "$S/sudoku_n48_kbrep_N_2026-11-10" \
    "$S/sudoku_n48_sizeshape_O_2026-11-10" \
    "$S/sudoku_n48_tied_P_2026-11-10" \
    "$S/sudoku_n48_tiedwidth_S_2026-11-10" \
    "$S/sudoku_n48_cycled_U_2026-11-11"
echo "acc_vs_solve rc=$?"

echo
echo "end $(date '+%F %T')"
