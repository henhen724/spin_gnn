#!/bin/bash
#SBATCH -J spin_gnn_sudoku2_mlp2
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --array=0-5
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%A_%a.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Round 2, following sudoku2_mlp_sweep.sbatch (Slurm 48157903): that 16-config
# grid (num_layers in {2,4,6,8} x dim_feedforward in {128,256,512,1024} at
# d_model=64, nhead=4) found num_layers=8/dim_feedforward=1024 best
# (mean_o1_solve_rate=0.904, 1.2M params), clearly rising with both depth and
# width together (neither alone was enough -- L=8/ff=128 only reached 0.698).
# But every run finished in 4-6 min against a 3h budget, and the best run's
# val_accuracy was still climbing at the last epoch (150) rather than
# plateaued (0.521 -> 0.561 -> 0.571 at epochs 50/100/149) -- undertrained,
# not capacity-saturated. This round pushes num_layers further {8,10,12} x
# dim_feedforward {1024,2048} at the same d_model=64/nhead=4, and doubles
# epochs 150->300 to let these larger configs actually converge.
# Goal: find the smallest config reaching mean_o1_solve_rate ~1.0.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312 py-numba/0.60.0_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"
export NUMBA_CACHE_DIR="${SCRATCH}/numba_cache"
cd "${SLURM_SUBMIT_DIR}"

L_VALUES=(8 10 12)
FF_VALUES=(1024 2048)
NUM_FF=${#FF_VALUES[@]}

IDX=${SLURM_ARRAY_TASK_ID}
L_IDX=$(( IDX / NUM_FF ))
FF_IDX=$(( IDX % NUM_FF ))
NUM_LAYERS=${L_VALUES[$L_IDX]}
DIM_FF=${FF_VALUES[$FF_IDX]}
D_MODEL=64
NHEAD=4

OUT_DIR="${SCRATCH}/spin_gnn/sudoku_hparam_sweep"
mkdir -p "$OUT_DIR"

echo "task ${IDX}: d_model=${D_MODEL} nhead=${NHEAD} num_layers=${NUM_LAYERS} dim_feedforward=${DIM_FF} epochs=300"

python -m spin_gnn.sudoku_hparam_sweep \
    --n-nodes 24 \
    --degree 5 \
    --n-colors 4 \
    --d-model "$D_MODEL" \
    --nhead "$NHEAD" \
    --num-layers "$NUM_LAYERS" \
    --dim-feedforward "$DIM_FF" \
    --n-train-instances 5000 \
    --n-val-instances 300 \
    --batch-size 32 \
    --epochs 300 \
    --n-per-bin 100 \
    --out-dir "$OUT_DIR" \
    "$@"
