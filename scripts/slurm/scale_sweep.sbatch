#!/bin/bash
#SBATCH -J spin_gnn_scale
#SBATCH -p sganguli
#SBATCH --nice=1000000
#SBATCH --array=0-39
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH -o %x-%A_%a.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Data/context-length scaling sweep for the 32-spin model with power-law
# lag correlations Gamma_{k,k'} = |k-k'|^-gamma_param (--gamma-param below).
#
# -p sganguli, not the shared gpu/normal partitions: this is the lab's own
# allocation. --nice=1000000 deliberately deprioritizes these jobs (default
# nice is 0) so any other lab member's job queued on this partition is
# scheduled ahead of these -- per request, this sweep should never compete
# with anyone else's work here.
#
# No GPU requested: at n_spins=32 the model/graphs are tiny (a handful of KB
# per batch); GPU kernel-launch overhead would not pay for itself, and it
# avoids tying up the partition's scarce GPUs (14/16 idle at last check) for
# a workload that doesn't need them. If future sweeps scale n_spins up
# substantially, revisit and add -G 1.
#
# Array index -> (context_length, n_train_sequences) via the lookup table
# below: 5 context lengths x 8 training-set sizes = 40 grid points.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"

K_VALUES=(2 4 8 16 32)
NSEQ_VALUES=(1 2 4 8 16 32 64 128)
NUM_NSEQ=${#NSEQ_VALUES[@]}

IDX=${SLURM_ARRAY_TASK_ID}
K_IDX=$(( IDX / NUM_NSEQ ))
N_IDX=$(( IDX % NUM_NSEQ ))
K=${K_VALUES[$K_IDX]}
NSEQ=${NSEQ_VALUES[$N_IDX]}

OUT_DIR="${SCRATCH}/spin_gnn/scale_sweep/power1"
mkdir -p "$OUT_DIR"

echo "task ${IDX}: context_length=${K} n_train_sequences=${NSEQ}"

python -m spin_gnn.scale_sweep \
    --n-spins 32 \
    --gamma-kind power_law \
    --gamma-param 1.0 \
    --context-length "$K" \
    --n-train-sequences "$NSEQ" \
    --n-test-realizations 8 \
    --trajectory-length 512 \
    --epochs 50 \
    --hidden-dim 32 \
    --num-layers 2 \
    --out-dir "$OUT_DIR" \
    "$@"
