#!/bin/bash
#SBATCH -J spin_gnn_sat_mlp
#SBATCH -p sganguli
#SBATCH -G 1
#SBATCH --array=0-15
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH -o %x-%A_%a.out
#SBATCH --mail-user=hshunt@stanford.edu
#SBATCH --mail-type=END,FAIL

# Follow-up to sat_nhead_sweep.sbatch: that scan fixed d_model=256 and swept
# nhead, finding nhead=8 the best on raw accuracy/NLL (93.7%, the sweep-wide
# best) while nhead=16/32 were the only configs with ANY nonzero
# val_satisfying_completion_rate, at worse accuracy -- but every nhead>=8
# config's best_epoch landed on the very last training epoch, suggesting
# under-training rather than a settled capacity ceiling.
#
# This sweep fixes d_model=256, nhead=8 (the best all-around point so far)
# and scans two capacity axes that were never varied in any prior sweep:
# num_layers (number of stacked transformer blocks, i.e. "number of MLPs",
# each block = one attention sub-layer + one feedforward/MLP sub-layer) and
# dim_feedforward (the hidden width of each block's own MLP, previously
# always implicitly 4*d_model=1024 and never exposed). 4 x 4 = 16 grid
# points; same 10,000 train / 500 val instances, 15 epochs, max_trace_tokens=2000
# as every prior sweep, for direct comparability.
#
# Same partition/resource convention as sat_nhead_sweep.sbatch (sganguli,
# no --nice -- this lab's own GPU allocation, not backgrounded). Writes into
# the SAME sat_hparam_sweep directory; sat_hparam_sweep.py's tag now encodes
# all four hyperparameters, so there's no collision with the earlier grids.

set -euo pipefail
module purge
module load devel math python/3.12.1 py-pytorch/2.4.1_py312 py-numpy/1.26.3_py312
source "${SCRATCH}/spin_gnn_venv/bin/activate"
export PYTHONPATH="${SCRATCH}/spin_gnn_venv/lib/python3.12/site-packages:${PYTHONPATH:-}"

L_VALUES=(1 2 4 6)
FF_VALUES=(256 512 1024 2048)
NUM_FF=${#FF_VALUES[@]}

IDX=${SLURM_ARRAY_TASK_ID}
L_IDX=$(( IDX / NUM_FF ))
FF_IDX=$(( IDX % NUM_FF ))
NUM_LAYERS=${L_VALUES[$L_IDX]}
DIM_FF=${FF_VALUES[$FF_IDX]}
D_MODEL=256
NHEAD=8

OUT_DIR="${SCRATCH}/spin_gnn/sat_hparam_sweep"
mkdir -p "$OUT_DIR"

echo "task ${IDX}: d_model=${D_MODEL} nhead=${NHEAD} num_layers=${NUM_LAYERS} dim_feedforward=${DIM_FF}"

python -m spin_gnn.sat_hparam_sweep \
    --n-vars 24 \
    --k 3 \
    --alpha 4.2 \
    --d-model "$D_MODEL" \
    --nhead "$NHEAD" \
    --num-layers "$NUM_LAYERS" \
    --dim-feedforward "$DIM_FF" \
    --n-train-instances 10000 \
    --n-val-instances 500 \
    --batch-size 8 \
    --epochs 15 \
    --max-trace-tokens 2000 \
    --out-dir "$OUT_DIR" \
    "$@"
