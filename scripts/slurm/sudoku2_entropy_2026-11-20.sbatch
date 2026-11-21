#!/bin/bash
#SBATCH --job-name=sud2Entropy
#SBATCH --partition=sganguli
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/users/hshunt/spin_gnn/sud2Entropy_%A_%a.out
#
# Stage ENTROPY.  Inference only -- nothing is trained, no checkpoint is
# written, no closed knob is touched.  Predictions E1-E7 and diagnostics
# DE1-DE3 were committed in 946a7ec BEFORE this ran; the DE3 tolerance
# amendment is recorded in the probe docstring.
#
# CPU, not GPU: this is nine small checkpoints over cached instances, and the
# SCALE instance cache already exists so nothing is regenerated.
set -u
cd /home/users/hshunt/spin_gnn
module purge --force
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 \
    py-numpy/1.26.3_py312 py-numba/0.60.0_py312 py-scipy/1.12.0_py312
export OMP_NUM_THREADS=8
echo "task ${SLURM_ARRAY_TASK_ID} start $(date -Iseconds)"
python3 scripts/stageentropy_probe.py \
    --task "${SLURM_ARRAY_TASK_ID}" --ntasks 3 --limit 100
echo "task ${SLURM_ARRAY_TASK_ID} end $(date -Iseconds)"
