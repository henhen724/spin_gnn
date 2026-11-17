#!/bin/bash
#SBATCH --job-name=stageLogicFit
#SBATCH --partition=sganguli
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/%x_%j.out

# Scores the stage LOGIC predictions and draws the ten registered figures,
# once the stageLogic array has written every shard.  Submitted with
#   sbatch --dependency=afterok:<arrayjobid> scripts/slurm/stagelogic_fit_2026-11-17.sbatch
# so nothing has to poll.  CPU only -- this is pandas-free CSV reading,
# a dozen least-squares fits and matplotlib.

set -euo pipefail

module purge
module load devel math viz python/3.12.1
module load py-numpy/1.26.3_py312 py-scipy/1.12.0_py312 py-matplotlib/3.8.3_py312

export LOGIC_OUT=/scratch/users/hshunt/spin_gnn/runs/stagelogic_2026-11-17
export POS_OUT=/scratch/users/hshunt/spin_gnn/runs/stagepos_2026-11-17
export LOGIC_INST=/scratch/users/hshunt/spin_gnn/runs/stagepos_2026-11-17/instances
export MPLCONFIGDIR="${L_SCRATCH_JOB:-/tmp}/mpl"

cd /home/users/hshunt/spin_gnn
python3 scripts/stagelogic_fit.py
