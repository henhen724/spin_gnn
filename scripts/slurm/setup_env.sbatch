#!/bin/bash
#SBATCH -J spin_gnn_setup_env
#SBATCH -p normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH -o %x-%j.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# One-off job to build the venv (scripts/setup_env.sh) without needing an
# interactive sh_dev session -- CPU-only partition, no GPU needed for pip
# installs. Chain scale_sweep.sbatch behind this with
# `sbatch --dependency=afterok:<this_jobid> scale_sweep.sbatch`.

set -euo pipefail
# Slurm copies batch scripts into a spool dir before running them, so $0
# does NOT point back into the repo here (that's what broke this originally
# -- it resolved to a nonexistent /var/spool/slurmd/.../setup_env.sh).
# SLURM_SUBMIT_DIR is the directory `sbatch` was invoked from.
bash "${SLURM_SUBMIT_DIR}/scripts/setup_env.sh"
